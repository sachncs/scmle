"""Time-to-event modelling with right-censored rows."""

import numpy as np
import pandas as pd
import pytest
import support

import scmle
from scmle.contracts import errors


def censored_frame(rows=1200, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    noise = rng.normal(size=rows)
    rate = np.exp(1.0 * x1 - 0.7 * x2)
    happens = rng.exponential(1 / rate)
    stops = rng.exponential(1.5, rows)
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "noise": noise,
            "time": np.minimum(happens, stops),
            "event": (happens <= stops).astype(int),
        }
    )


def run(tmp_path, frame, **kwargs):
    return support.make_project(
        frame,
        "time",
        task="survival",
        event_column="event",
        time_budget=300,
        output_dir=tmp_path,
        **kwargs,
    ).run()


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return run(tmp_path_factory.mktemp("survival"), censored_frame())


def test_the_selected_model_orders_risk_well_beyond_chance(result):
    assert result.evaluation["test_metrics"]["c_index"] > 0.7
    baseline = next(t for t in result.trials if t.model_name == "dummy")
    assert baseline.score_mean == pytest.approx(0.5)


def test_the_event_column_is_never_a_feature(result):
    assert "event" not in result.bundle.features
    assert "time" not in result.bundle.features


def test_risk_scores_rise_with_the_true_hazard(result):
    rows = pd.DataFrame(
        {"x1": [-2.0, 2.0], "x2": [0.0, 0.0], "noise": [0.0, 0.0]}
    )

    low, high = result.bundle.predict(rows)

    assert high > low


def test_survival_curves_fall_over_time_and_start_near_one(result):
    rows = pd.DataFrame(
        {"x1": [-1.0, 1.0], "x2": [0.0, 0.0], "noise": [0.0, 0.0]}
    )

    curves = result.bundle.predict_survival(rows, [0.05, 0.5, 2.0])

    assert curves.shape == (2, 3)
    assert (np.diff(curves, axis=1) <= 1e-9).all()
    assert (curves[:, 0] > 0.8).all()
    assert curves[0, 1] > curves[1, 1]  # lower hazard survives longer


def test_survival_needs_an_event_column(tmp_path):
    with pytest.raises(ValueError, match="event_column"):
        support.make_project(
            censored_frame(100), "time", task="survival", output_dir=tmp_path
        )


def test_too_few_events_are_refused(tmp_path):
    frame = censored_frame()
    frame["event"] = 0
    frame.loc[:3, "event"] = 1

    with pytest.raises(errors.BlockedError, match="event"):
        run(tmp_path, frame)


def test_curves_survive_saving_and_loading(result, tmp_path):
    rows = pd.DataFrame(
        {"x1": [-1.0, 1.0], "x2": [0.0, 0.0], "noise": [0.0, 0.0]}
    )
    expected = result.bundle.predict_survival(rows, [0.1, 1.0])

    result.bundle.save(tmp_path / "copy")
    loaded = scmle.ModelBundle.load(tmp_path / "copy", trusted=True)

    np.testing.assert_allclose(
        loaded.predict_survival(rows, [0.1, 1.0]), expected
    )


def test_other_tasks_refuse_survival_curves(tmp_path):
    frame = censored_frame(300)
    frame["target"] = (frame.x1 > 0).astype(int)
    classified = support.make_project(
        frame.drop(columns=["time", "event"]),
        "target",
        time_budget=300,
        output_dir=tmp_path,
    ).run()

    with pytest.raises(errors.ArtifactError, match="not a survival"):
        classified.bundle.predict_survival(frame[["x1", "x2", "noise"]], [1.0])


def test_survival_works_with_a_weight_column(tmp_path):
    frame = censored_frame(800)
    frame["w"] = np.random.default_rng(2).uniform(0.5, 2.0, len(frame))

    result = run(tmp_path, frame, weight_column="w")

    assert result.evaluation["test_metrics"]["c_index"] > 0.65
    assert "w" not in result.bundle.features


def test_time_dependent_metrics_are_reported_on_the_locked_rows(result):
    metrics = result.evaluation["test_metrics"]

    assert {"c_index", "ibs", "td_auc"} <= set(metrics)
    assert metrics["td_auc"] > 0.65
    assert 0.0 < metrics["ibs"] < 0.25


def test_the_selected_model_has_a_lower_brier_score_than_kaplan_meier(result):
    from scmle.evaluation import survival as survival_metrics
    from scmle.models.survival import estimators

    train = censored_frame(1200, seed=0)
    fresh = censored_frame(1500, seed=5)
    baseline = estimators.MarginalSurvival().fit(
        train[["x1"]], train.time, train.event
    )
    times = survival_metrics.time_grid(
        fresh.time.to_numpy(), fresh.event.to_numpy()
    )
    features = fresh[["x1", "x2", "noise"]]

    model_curves = result.bundle.predict_survival(features, times)
    base_curves = baseline.predict_survival(features, times)

    def score(curves):
        return survival_metrics.integrated_brier(
            fresh.time.to_numpy(), fresh.event.to_numpy(), curves, times
        )

    assert score(model_curves) < score(base_curves)
