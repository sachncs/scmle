"""Tests for split-conformal calibration."""

import math

import numpy
import pytest

from scmle.evaluation import conformal


def noisy_regression(rows, rng):
    truth = rng.uniform(0, 10, rows)
    return truth + rng.normal(0, 1.0, rows), truth


def test_regression_intervals_cover_at_the_requested_rate():
    rng = numpy.random.default_rng(0)
    predicted, observed = noisy_regression(2000, rng)
    calibrator = conformal.ConformalCalibrator.for_regression(
        observed, predicted
    )

    new_predicted, new_observed = noisy_regression(20000, rng)
    low, high = calibrator.interval(new_predicted, coverage=0.9)
    covered = ((new_observed >= low) & (new_observed <= high)).mean()

    assert 0.88 < covered < 0.92
    assert calibrator.margin(0.9) == pytest.approx(1.645, abs=0.12)


def test_any_coverage_level_can_be_asked_for_later():
    rng = numpy.random.default_rng(6)
    predicted, observed = noisy_regression(3000, rng)
    calibrator = conformal.ConformalCalibrator.for_regression(
        observed, predicted
    )

    assert calibrator.margin(0.5) < calibrator.margin(0.8)
    assert calibrator.margin(0.8) < calibrator.margin(0.95)


def test_classification_sets_cover_and_shrink_when_the_model_is_sure():
    rng = numpy.random.default_rng(1)
    classes = ["a", "b", "c"]
    labels = rng.choice(classes, 3000)
    logits = rng.normal(0, 1, (3000, 3))
    logits[numpy.arange(3000), [classes.index(x) for x in labels]] += 2.0
    probabilities = numpy.exp(logits) / numpy.exp(logits).sum(
        axis=1, keepdims=True
    )
    calibrator = conformal.ConformalCalibrator.for_classification(
        labels, probabilities, classes
    )

    sets = calibrator.sets(probabilities, coverage=0.9)
    covered = numpy.mean(
        [label in s for label, s in zip(labels, sets, strict=True)]
    )

    assert covered >= 0.88
    assert numpy.mean([len(s) for s in sets]) < 3.0


def test_too_few_rows_or_bad_coverage_is_rejected():
    values = numpy.arange(10.0)
    with pytest.raises(ValueError, match="at least"):
        conformal.ConformalCalibrator.for_regression(values, values + 1)
    calibrator = conformal.ConformalCalibrator.for_regression(
        numpy.arange(50.0), numpy.arange(50.0) + 1
    )
    with pytest.raises(ValueError, match="strictly between"):
        calibrator.margin(1.0)


def test_demanding_coverage_with_few_rows_gives_an_infinite_margin():
    scores = numpy.arange(1.0, 21.0)
    assert math.isinf(conformal.finite_sample_quantile(scores, 0.99))
    assert conformal.finite_sample_quantile(scores, 0.5) == 11.0


def test_round_trip_and_wrong_kind_errors():
    rng = numpy.random.default_rng(2)
    predicted, observed = noisy_regression(200, rng)
    calibrator = conformal.ConformalCalibrator.for_regression(
        observed, predicted
    )

    restored = conformal.ConformalCalibrator.from_dict(calibrator.to_dict())

    assert restored == calibrator
    with pytest.raises(ValueError, match="classification"):
        restored.sets(numpy.ones((2, 2)))


class RecordingContext:
    """Collects the events a stage emits."""

    def __init__(self):
        self.events = []

    def emit(self, kind, reason="", artifacts=(), data=None):
        self.events.append((kind, reason, data))


def test_a_skipped_calibration_is_recorded_as_an_event():
    from scmle.core import stages

    context = RecordingContext()

    stages.PackageStage().note_conformal_skipped(context, "time budget spent")

    kind, reason, data = context.events[0]
    assert kind == "conformal.skipped"
    assert "time budget spent" in reason and data == {
        "reason": "time budget spent"
    }


def test_asking_for_intervals_without_calibration_points_at_the_event():
    from scmle.artifacts import bundle
    from scmle.contracts import errors

    model = bundle.ModelBundle.__new__(bundle.ModelBundle)
    model.conformal = None

    with pytest.raises(errors.ArtifactError, match=r"conformal\.skipped"):
        model.require_conformal("regression")


def test_sensitivity_below_the_minimum_rows_is_skipped_not_raised():
    from types import SimpleNamespace

    from scmle.contracts import tasks
    from scmle.core import stages

    context = SimpleNamespace(
        spec=SimpleNamespace(selection=SimpleNamespace(sensitivity_repeats=3)),
        task=tasks.TaskType.REGRESSION,
        resource_governor=SimpleNamespace(skip_reason=lambda first: None),
        split=SimpleNamespace(dev_target=numpy.arange(10.0)),
    )

    record = stages.PackageStage().sensitivity(context, None)

    assert "skipped" in record
