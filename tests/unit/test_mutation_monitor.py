"""Exact verdicts of the monitor: thresholds, advisors, drift, performance."""

import types

import numpy
import pandas
import pytest
from sklearn import linear_model

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.monitoring import monitor

ROWS = 40


class Profile:
    """A reference profile stub that reports a fixed drift.

    Attributes:
        value: The noise-adjusted PSI it reports.
        missing_rate: The training missing share.
        unseen: The unseen-category share it reports.
    """

    def __init__(self, value: float, unseen: float = 0.0) -> None:
        self.value = value
        self.missing_rate = 0.25
        self.unseen = unseen

    def adjust(self, values: pandas.Series) -> float:
        """Returns the fixed adjusted PSI of non-empty values."""
        assert len(values)
        return self.value

    def psi(self, values: pandas.Series) -> float:
        """Returns the fixed raw PSI, a little above the adjusted one."""
        assert len(values)
        return self.value + 0.01

    def unseen_rate(self, values: pandas.Series) -> float:
        """Returns the fixed unseen-category share."""
        assert len(values)
        return self.unseen


class Fixed(monitor.Advisor):
    """An advisor that always returns the same verdict."""

    def __init__(self, verdict: tuple[str, str] | None) -> None:
        self.verdict = verdict

    def evaluate(self, drift, performance, thresholds):
        """Returns the fixed verdict."""
        return self.verdict


def make_bundle(
    features: dict[str, Profile] | None = None,
    output: float = 0.0,
    reference_metrics: dict[str, float] | None = None,
) -> types.SimpleNamespace:
    """A regression bundle whose model is `y = 2 a` exactly."""
    train = pandas.DataFrame({"a": numpy.arange(float(ROWS))})
    model = linear_model.LinearRegression().fit(train, 2 * train["a"])
    reference = types.SimpleNamespace(
        features=features or {"a": Profile(0.0)},
        output=Profile(output),
        reference_metrics=reference_metrics or {},
    )
    return types.SimpleNamespace(
        reference=reference,
        schema=types.SimpleNamespace(task=tasks.Task.REGRESSION.value),
        info=types.SimpleNamespace(metric="r2"),
        pipeline=model,
        prepare=lambda data: data[list(reference.features)],
        predict=lambda frame: model.predict(frame[["a"]]),
    )


def make_data() -> pandas.DataFrame:
    return pandas.DataFrame(
        {"a": numpy.arange(float(ROWS)), "b": numpy.ones(ROWS)},
        index=numpy.arange(ROWS) + 100,
    )


def outcomes() -> pandas.Series:
    return pandas.Series(2 * numpy.arange(float(ROWS)))


def make_drift(
    features: list[monitor.FeatureDrift] | None = None,
    output: str = monitor.STABLE,
) -> monitor.Drift:
    return monitor.Drift(10, features or [], 0.3, output)


def feature(name: str, psi: float, unseen: float = 0.0):
    status = monitor.Thresholds().status(psi)
    return monitor.FeatureDrift(name, psi, status, 0.0, 0.0, unseen)


def performance(status: str, rise: dict[str, float]) -> monitor.Performance:
    return monitor.Performance(50, 40, 0.8, {}, {}, rise, status)


@pytest.mark.parametrize(
    ("psi", "expected"),
    [(0.05, "stable"), (0.1, "moderate"), (0.2, "moderate"), (0.25, "high")],
)
def test_status_classifies_psi_at_the_bounds(psi, expected) -> None:
    assert monitor.Thresholds().status(psi) == expected


def test_advise_without_a_verdict_is_none() -> None:
    found = monitor.advise(make_drift(), None, monitor.Thresholds())
    assert found == monitor.Recommendation("none", [])


def test_advise_takes_the_strongest_action_and_keeps_every_reason() -> None:
    rules = (Fixed(("investigate", "first")), Fixed(("retrain", "second")))
    found = monitor.advise(make_drift(), None, monitor.Thresholds(), rules)
    assert found == monitor.Recommendation("retrain", ["first", "second"])


def test_advise_ranks_investigate_above_none() -> None:
    rules = (Fixed(None), Fixed(("investigate", "only")))
    found = monitor.advise(make_drift(), None, monitor.Thresholds(), rules)
    assert found == monitor.Recommendation("investigate", ["only"])


def test_degradation_ignores_missing_performance() -> None:
    verdict = monitor.Degradation().evaluate(
        make_drift(), None, monitor.Thresholds()
    )
    assert verdict is None


def test_degradation_names_the_worst_metric() -> None:
    found = performance("degraded", {"r2": -0.2, "mae": -0.5})
    verdict = monitor.Degradation().evaluate(
        make_drift(), found, monitor.Thresholds()
    )
    assert verdict == (
        "retrain",
        "mae degraded 50% vs training on 40 labeled rows",
    )


def test_degradation_ignores_good_performance() -> None:
    found = performance("ok", {"r2": -0.5})
    verdict = monitor.Degradation().evaluate(
        make_drift(), found, monitor.Thresholds()
    )
    assert verdict is None


def test_feature_drift_names_only_the_strongly_drifted_features() -> None:
    drift = make_drift([feature("a", 0.5), feature("b", 0.15)])
    verdict = monitor.FeatureDriftAdvisor().evaluate(
        drift, None, monitor.Thresholds()
    )
    assert verdict == (
        "retrain",
        "50% of features drifted strongly: 'a' (PSI 0.50)",
    )


def test_feature_drift_investigates_the_worst_moderate_feature() -> None:
    drift = make_drift(
        [feature("c", 0.0), feature("a", 0.2), feature("b", 0.15)]
        + [feature(str(n), 0.0) for n in range(3)]
    )
    verdict = monitor.FeatureDriftAdvisor().evaluate(
        drift, None, monitor.Thresholds()
    )
    assert verdict == ("investigate", "feature 'a' drifted (PSI 0.20)")


@pytest.mark.parametrize(
    ("given", "action"),
    [(None, "retrain"), (performance("ok", {}), "investigate")],
)
def test_output_drift_retrains_only_without_performance(given, action) -> None:
    verdict = monitor.OutputDriftAdvisor().evaluate(
        make_drift(output=monitor.HIGH), given, monitor.Thresholds()
    )
    assert verdict == (action, "model output drifted strongly (PSI 0.30)")


def test_novelty_fires_at_the_threshold() -> None:
    drift = make_drift([feature("a", 0.0, 0.1), feature("b", 0.0, 0.2)])
    verdict = monitor.Novelty().evaluate(drift, None, monitor.Thresholds())
    assert verdict == (
        "investigate",
        "20% of 'b' values are categories unseen in training",
    )


def test_watcher_refuses_a_bundle_without_a_reference() -> None:
    bare = types.SimpleNamespace(reference=None)
    with pytest.raises(
        errors.ArtifactError,
        match=r"^this bundle has no monitoring reference profile$",
    ):
        monitor.Watcher(bare)


def test_watcher_defaults_its_thresholds() -> None:
    assert monitor.Watcher(make_bundle()).thresholds == monitor.Thresholds()


def test_infer_gives_the_positive_probability_for_a_binary_task() -> None:
    bundle = make_bundle()
    bundle.schema.task = tasks.Task.BINARY.value
    bundle.predict_proba = lambda frame: numpy.column_stack(
        [numpy.zeros(len(frame)), numpy.full(len(frame), 0.75)]
    )
    found = monitor.Watcher(bundle).infer(make_data()[["a"]])
    assert found.tolist() == [0.75] * ROWS


def test_drift_reports_every_feature_worst_first() -> None:
    bundle = make_bundle(
        features={"a": Profile(0.12), "b": Profile(0.3, 0.5)}, output=0.05
    )
    found = monitor.Watcher(bundle).drift(make_data())
    assert found == monitor.Drift(
        ROWS,
        [
            monitor.FeatureDrift("b", 0.3, "high", 0.0, 0.25, 0.5, 0.31),
            monitor.FeatureDrift("a", 0.12, "moderate", 0.0, 0.25, 0.0, 0.13),
        ],
        0.05,
        "stable",
        0.060000000000000005,
    )


def test_performance_with_too_few_labels_is_insufficient() -> None:
    bundle = make_bundle(reference_metrics={"r2": 1.0})
    known = outcomes().where(numpy.arange(ROWS) < 29)
    found = monitor.Watcher(bundle).performance(make_data(), known)
    assert found == monitor.Performance(
        ROWS, 29, 29 / ROWS, {}, {"r2": 1.0}, {}, "insufficient_labels"
    )


def test_performance_orients_the_change_of_every_metric() -> None:
    bundle = make_bundle(reference_metrics={"mae": 1.0, "r2": 2.0})
    found = monitor.Watcher(bundle).performance(make_data(), outcomes())
    assert (found.rows, found.labeled_rows, found.labeled_fraction) == (
        ROWS,
        ROWS,
        1.0,
    )
    assert found.metrics == pytest.approx({"mae": 0, "rmse": 0, "r2": 1})
    assert found.rise == pytest.approx({"mae": 1.0, "r2": -0.5})
    assert found.status == "degraded"


def test_performance_weights_labels_by_inverse_propensity() -> None:
    bundle = make_bundle()
    noisy = outcomes() + numpy.where(numpy.arange(ROWS) < ROWS // 2, 0, 100)
    propensity = pandas.Series(
        numpy.where(numpy.arange(ROWS) < ROWS // 2, 1e-6, 1.0)
    )
    found = monitor.Watcher(bundle).performance(make_data(), noisy, propensity)
    assert found.metrics["mae"] < 1.0


def test_assess_judges_output_drift_with_the_performance() -> None:
    bundle = make_bundle(output=0.5)
    found = monitor.Watcher(bundle).assess(make_data(), outcomes())
    assert found.performance is not None
    assert found.performance.status == "ok"
    assert found.recommendation == monitor.Recommendation(
        "investigate", ["model output drifted strongly (PSI 0.50)"]
    )


def test_assess_without_outcomes_has_no_performance() -> None:
    found = monitor.Watcher(make_bundle(output=0.5)).assess(make_data())
    assert found.performance is None
    assert found.recommendation.action == "retrain"
