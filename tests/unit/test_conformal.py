"""Tests for split-conformal calibration."""

import math

import numpy as np
import pytest

from mml.evaluation import conformal


def noisy_regression(rows, rng):
    truth = rng.uniform(0, 10, rows)
    return truth + rng.normal(0, 1.0, rows), truth


def test_regression_intervals_cover_at_the_requested_rate():
    rng = np.random.default_rng(0)
    predicted, observed = noisy_regression(2000, rng)
    calibrator = conformal.ConformalCalibrator.for_regression(
        observed, predicted, coverage=0.9
    )

    new_predicted, new_observed = noisy_regression(20000, rng)
    low, high = calibrator.interval(new_predicted)
    covered = ((new_observed >= low) & (new_observed <= high)).mean()

    assert 0.88 < covered < 0.92
    assert calibrator.margin == pytest.approx(1.645, abs=0.12)


def test_classification_sets_cover_and_shrink_when_the_model_is_sure():
    rng = np.random.default_rng(1)
    classes = ["a", "b", "c"]
    labels = rng.choice(classes, 3000)
    logits = rng.normal(0, 1, (3000, 3))
    logits[np.arange(3000), [classes.index(x) for x in labels]] += 2.0
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    calibrator = conformal.ConformalCalibrator.for_classification(
        labels, probabilities, classes, coverage=0.9
    )

    sets = calibrator.sets(probabilities)
    covered = np.mean(
        [label in s for label, s in zip(labels, sets, strict=True)]
    )

    assert covered >= 0.88
    assert np.mean([len(s) for s in sets]) < 3.0


def test_too_few_rows_or_bad_coverage_is_rejected():
    values = np.arange(10.0)
    with pytest.raises(ValueError, match="at least"):
        conformal.ConformalCalibrator.for_regression(values, values + 1)
    with pytest.raises(ValueError, match="strictly between"):
        conformal.ConformalCalibrator.for_regression(
            np.arange(50.0), np.arange(50.0), coverage=1.0
        )


def test_demanding_coverage_with_few_rows_gives_an_infinite_margin():
    scores = np.arange(1.0, 21.0)
    assert math.isinf(conformal.finite_sample_quantile(scores, 0.99))
    assert conformal.finite_sample_quantile(scores, 0.5) == 11.0


def test_round_trip_and_wrong_kind_errors():
    rng = np.random.default_rng(2)
    predicted, observed = noisy_regression(200, rng)
    calibrator = conformal.ConformalCalibrator.for_regression(
        observed, predicted
    )

    restored = conformal.ConformalCalibrator.from_dict(calibrator.to_dict())

    assert restored == calibrator
    with pytest.raises(ValueError, match="classification"):
        restored.sets(np.ones((2, 2)))
