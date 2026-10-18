"""Forecasting: a seasonal series is predicted several steps ahead."""

import json

import numpy as np
import pandas as pd
import pytest
import support

from scml.contracts import errors

HORIZON = 4
PERIOD = 7


def series(rows=700, seed=0, name="shop"):
    rng = np.random.default_rng(seed)
    steps = np.arange(rows)
    value = (
        20
        + 0.03 * steps
        + 4 * np.sin(2 * np.pi * steps / PERIOD)
        + rng.normal(0, 0.4, rows)
    )
    return pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=rows, freq="D"),
            "sales": value,
        }
    ).assign(shop=name)


def run(tmp_path, frame, **kwargs):
    return support.make_project(
        frame.drop(columns="shop"),
        "sales",
        task="forecast",
        horizon=HORIZON,
        time_column="date",
        seasonal_period=PERIOD,
        time_budget=300,
        output_dir=tmp_path,
        **kwargs,
    ).run()


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return run(tmp_path_factory.mktemp("forecast"), series())


def test_the_model_beats_no_change_and_the_seasonal_baseline_at_every_step(
    result,
):
    steps = result.evaluation["forecast_backtest"]["steps"]

    assert len(steps) == HORIZON
    for step in steps:
        assert step["model_mae"] < step["seasonal_naive_mae"]
        assert step["model_mae"] < step["last_value_mae"]
    assert result.evaluation["test_metrics"]["mase"] < 0.6


def test_each_rolling_origin_is_reported_per_step(result):
    origins = result.evaluation["forecast_backtest"]["origins"]

    assert len(origins) >= 2
    for origin in origins:
        assert len(origin["model_mae"]) == HORIZON
        assert len(origin["last_value_mae"]) == HORIZON
    path = result.run_dir / "diagnostics" / "forecast_backtest.json"
    assert json.loads(path.read_text())["origins"]


def test_training_targets_never_reach_into_the_locked_period(result):
    frame = series()
    manifest = json.loads(
        (result.run_dir / "validation_manifest.json").read_text()
    )
    usable = frame.iloc[: len(frame) - HORIZON].reset_index(drop=True)
    last_training = usable["date"].iloc[manifest["train_ids"]].max()
    first_test = usable["date"].iloc[manifest["test_ids"]].min()

    assert result.evaluation["split_strategy"] == "temporal"
    assert last_training + pd.Timedelta(days=HORIZON) < first_test


def test_a_series_never_seen_in_training_is_forecast_from_its_history(result):
    other = series(120, seed=11, name="new_shop")
    history, future = other.iloc[:100], other.iloc[100:104]

    forecast = result.bundle.forecast(history.drop(columns="shop"))

    assert list(forecast["step"]) == [1, 2, 3, 4]
    assert list(forecast["date"]) == list(future["date"])
    error = np.abs(forecast["forecast"].to_numpy() - future["sales"].to_numpy())
    naive = np.abs(history["sales"].iloc[-1] - future["sales"].to_numpy())
    assert error.mean() < naive.mean()


def test_asking_beyond_the_trained_horizon_is_refused(result):
    with pytest.raises(errors.ArtifactError, match="1 to 4"):
        result.bundle.forecast(series(60).drop(columns="shop"), steps=9)


def test_forecasts_need_a_time_column_and_a_horizon(tmp_path):
    frame = series().drop(columns="shop")
    with pytest.raises(ValueError, match="time_column"):
        support.make_project(
            frame, "sales", task="forecast", horizon=3, output_dir=tmp_path
        )
    with pytest.raises(ValueError, match="horizon"):
        support.make_project(
            frame,
            "sales",
            task="regression",
            horizon=3,
            time_column="date",
            output_dir=tmp_path,
        )


def test_many_series_are_forecast_together(tmp_path):
    frame = pd.concat(
        [series(400, seed=1, name="a"), series(400, seed=2, name="b")]
    )

    result = support.make_project(
        frame,
        "sales",
        task="forecast",
        horizon=3,
        time_column="date",
        group_column="shop",
        seasonal_period=PERIOD,
        time_budget=300,
        output_dir=tmp_path,
    ).run()
    forecast = result.bundle.forecast(frame.drop(columns=[]))

    assert set(forecast["shop"]) == {"a", "b"}
    assert len(forecast) == 2 * 3


def test_the_backtest_works_under_hard_isolation_where_locked_rows_are_sealed(
    tmp_path,
):
    result = run(tmp_path, series(300), enforcement="hard")

    steps = result.evaluation["forecast_backtest"]["steps"]
    assert len(steps) == HORIZON
    assert all(s["seasonal_naive_mae"] is not None for s in steps)
