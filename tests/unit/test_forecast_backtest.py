"""Rolling-origin, horizon-aware evaluation of forecast families on panels."""

import dataclasses
import json

import numpy
import pandas
import pytest

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.evaluation import forecasting
from scmle.models.forecast import families
from scmle.models.forecast import panel


def line(length=30):
    """One series rising by 2 each row."""
    frame = pandas.DataFrame(
        {"t": range(length), "v": 2.0 * numpy.arange(length)}
    )
    return panel.Panel.build(frame, "t", "v")


def stores(entities=6, length=120, noise=0.5, seed=0, slope=0.0):
    """Weekly-seasonal daily stores at different levels, rising by `slope`."""
    rng = numpy.random.default_rng(seed)
    parts = []
    for entity in range(entities):
        steps = numpy.arange(length)
        values = (
            40.0 * entity
            + slope * steps
            + 10.0 * numpy.sin(2 * numpy.pi * steps / 7)
        )
        parts.append(
            pandas.DataFrame(
                {
                    "store": f"s{entity}",
                    "day": pandas.date_range("2024-01-01", periods=length),
                    "sales": values + rng.normal(scale=noise, size=length),
                }
            )
        )
    return pandas.concat(parts, ignore_index=True)


def test_origins_step_back_by_the_horizon_from_the_end():
    assert forecasting.origins(line(), 3, 4).tolist() == [17, 20, 23, 26]
    assert forecasting.origins(line(), 3, 2, stride=1).tolist() == [25, 26]


def test_origins_refuse_a_panel_too_short_for_them():
    with pytest.raises(errors.BlockedError, match="needs 23 time points"):
        forecasting.origins(line(20), 3, 7)


def test_a_trend_backtest_scores_each_step_exactly():
    report = forecasting.backtest(line(), 2, [families.Seasonal()], count=3)
    (score,) = report.scores
    assert report.origins == ("23", "25", "27")
    assert score.mae == (2.0, 4.0)
    assert score.mase == (1.0, 2.0)
    assert score.overall == 1.5
    assert report.counts == (3, 3)
    assert report.undefined == 0
    assert score.pvalue is None
    assert score.beats is None


def test_too_few_pairs_leave_the_comparison_unassessed():
    report = forecasting.backtest(
        line(), 2, [families.Seasonal(), families.Smoothing()], count=3
    )
    smoothing = report.scores[1]
    assert smoothing.name == "exponential_smoothing"
    assert smoothing.mae == pytest.approx((2.0, 4.0), abs=1e-2)
    assert smoothing.pvalue is None
    assert smoothing.beats is None


def seasonal(frame):
    """Seasonal naive against boosting, both with the weekly period."""
    built = panel.Panel.build(frame, "day", "sales", "store")
    return forecasting.backtest(
        built,
        7,
        [families.Seasonal(7), families.Boosted(period=7, seed=0)],
        count=5,
        period=7,
    )


def test_boosting_beats_seasonal_naive_on_trending_seasonal_stores():
    # Seasonal naive repeats last week, so it trails a trend by a week's
    # rise at every step; boosting learns the rise from the lag changes.
    report = seasonal(stores(slope=0.5))
    naive, boosted = report.scores
    assert report.baseline == "seasonal_naive"
    assert report.entities == 6
    assert report.counts == (30,) * 7
    assert naive.overall == pytest.approx(0.993, abs=1e-3)
    assert boosted.overall == pytest.approx(0.156, abs=1e-3)
    assert boosted.pvalue is not None
    assert boosted.pvalue < forecasting.ALPHA
    assert boosted.beats is True
    assert report.best() == "lag_boosting"


def test_boosting_is_not_shown_to_beat_seasonal_naive_on_a_pure_season():
    # Without a trend, last week is already the best forecast up to noise.
    report = seasonal(stores())
    naive, boosted = report.scores
    assert naive.overall == pytest.approx(0.959, abs=1e-3)
    assert boosted.overall == pytest.approx(0.941, abs=1e-3)
    assert boosted.pvalue == pytest.approx(0.444, abs=1e-3)
    assert boosted.beats is False


def test_a_series_too_short_for_boosting_is_blocked_not_a_data_error():
    frame = pandas.DataFrame(
        {
            "day": pandas.date_range("2024-01-01", periods=30),
            "sales": numpy.arange(30.0),
        }
    )
    built = panel.Panel.build(frame, "day", "sales")
    assert len(forecasting.origins(built, 3, 3)) == 3
    with pytest.raises(errors.BlockedError, match="step 1 has 20 row"):
        forecasting.backtest(
            built, 3, [families.Seasonal(), families.Boosted()], count=3
        )


def test_constant_histories_are_counted_not_dropped():
    frame = pandas.DataFrame(
        {
            "g": ["a"] * 12 + ["b"] * 12,
            "t": list(range(12)) * 2,
            "v": [5.0] * 12 + list(numpy.arange(12.0)),
        }
    )
    built = panel.Panel.build(frame, "t", "v", "g")
    report = forecasting.backtest(built, 1, [families.Seasonal()], count=2)
    assert report.undefined == 2
    assert report.counts == (4,)
    assert report.scores[0].mae == (0.5,)
    assert report.scores[0].mase == (1.0,)


def test_family_names_must_be_distinct():
    with pytest.raises(errors.InputError, match="distinct"):
        forecasting.backtest(
            line(), 2, [families.Seasonal(), families.Seasonal(4)], count=2
        )


def test_the_baseline_must_be_one_of_the_families():
    with pytest.raises(errors.InputError, match="baseline 'x'"):
        forecasting.backtest(
            line(), 2, [families.Seasonal()], count=2, baseline="x"
        )


def test_the_report_serialises_to_json():
    report = forecasting.backtest(line(), 2, [families.Seasonal()], count=2)
    record = json.loads(json.dumps(dataclasses.asdict(report)))
    assert record["horizon"] == 2
    assert record["scores"][0]["name"] == "seasonal_naive"


def configured(**validation):
    return tasks.Config(
        task="forecast",
        data=tasks.Dataset(
            uri="m",
            target="sales",
            time_column="day",
            group_column="store",
            horizon=3,
            seasonal_period=7,
            forecast_lags=[1, 7],
            forecast_windows=[7],
        ),
        validation=tasks.Validation(strategy="rolling", seed=1, **validation),
    )


def test_a_project_configuration_runs_the_backtest():
    spec = configured(folds=4)
    report = forecasting.run(spec, stores())
    assert report.horizon == 3
    assert report.origins == (
        "2024-04-17T00:00:00.000000",
        "2024-04-20T00:00:00.000000",
        "2024-04-23T00:00:00.000000",
        "2024-04-26T00:00:00.000000",
    )
    assert [s.name for s in report.scores] == [
        "seasonal_naive",
        "exponential_smoothing",
        "lag_boosting",
    ]
    assert report.baseline == "seasonal_naive"


def test_the_configuration_reaches_every_family():
    naive, smoothing, boosted = forecasting.contenders(configured(folds=4))
    assert isinstance(naive, families.Seasonal)
    assert naive.period == 7
    assert isinstance(smoothing, families.Smoothing)
    assert isinstance(boosted, families.Boosted)
    assert boosted.lags == (1, 7)
    assert boosted.windows == (7,)
    assert boosted.period == 7
    assert boosted.seed == 1


def test_the_panel_is_built_from_the_value_column_not_the_target():
    spec = configured(folds=4)
    assert spec.data.value_column == "sales"
    assert spec.data.target == "sales__h1"
    frame = stores()
    # The target holds no values: only the value column can be read.
    frame["sales__h1"] = None
    report = forecasting.run(spec, frame)
    built = panel.Panel.build(stores(), "day", "sales", "store")
    direct = forecasting.backtest(
        built, 3, forecasting.contenders(spec), count=4, period=7
    )
    assert report == direct


def test_horizon_rows_space_the_origins_in_time_points_per_series():
    # 30 rows over 6 stores are 5 days.
    spec = configured(folds=3, horizon_rows=30)
    report = forecasting.run(spec, stores())
    assert report.origins == (
        "2024-04-16T00:00:00.000000",
        "2024-04-21T00:00:00.000000",
        "2024-04-26T00:00:00.000000",
    )


def test_a_project_without_a_horizon_is_refused():
    spec = tasks.Config(
        data=tasks.Dataset(uri="m", target="sales", time_column="day"),
    )
    with pytest.raises(errors.InputError, match=r"data\.horizon"):
        forecasting.run(spec, stores())
