"""Tests for threshold optimization."""

import numpy as np
import pandas as pd
import pytest
from sklearn import base
from sklearn import linear_model

from scml.models import thresholds


def labelled(rows=2000, positive_rate=0.1, seed=0):
    rng = np.random.default_rng(seed)
    actual = rng.random(rows) < positive_rate
    probability = np.clip(
        np.where(actual, rng.beta(4, 2, rows), rng.beta(1.5, 6, rows)), 0, 1
    )
    return probability, actual


def test_f1_optimum_is_not_the_default_cutoff_on_imbalanced_data():
    probability, actual = labelled()
    best, score = thresholds.optimize_threshold(
        probability, actual, thresholds.F1Objective()
    )
    default = thresholds.F1Objective().score(actual, probability >= 0.5)
    assert best != 0.5 and score >= default


def test_cost_objective_moves_the_threshold_with_the_costs():
    probability, actual = labelled()
    cheap_miss = thresholds.CostObjective(
        false_positive_cost=5.0, false_negative_cost=1.0
    )
    costly_miss = thresholds.CostObjective(
        false_positive_cost=1.0, false_negative_cost=10.0
    )
    high = thresholds.optimize_threshold(probability, actual, cheap_miss)[0]
    low = thresholds.optimize_threshold(probability, actual, costly_miss)[0]
    assert low < high


def test_objectives_on_known_decisions():
    actual = np.array([True, True, False, False])
    flagged = np.array([True, False, True, False])
    assert thresholds.F1Objective().score(actual, flagged) == pytest.approx(0.5)
    assert thresholds.AccuracyObjective().score(actual, flagged) == 0.5
    assert thresholds.BalancedAccuracyObjective().score(actual, flagged) == 0.5
    assert thresholds.CostObjective(2.0, 3.0).score(
        actual, flagged
    ) == pytest.approx(-1.25)
    assert (
        thresholds.F1Objective().score(actual, np.zeros(4, dtype=bool)) == 0.0
    )


def test_ties_prefer_thresholds_near_one_half():
    probability = np.array([0.1, 0.9])
    actual = np.array([False, True])
    best = thresholds.optimize_threshold(
        probability, actual, thresholds.AccuracyObjective()
    )[0]
    assert 0.1 < best <= 0.9 and abs(best - 0.5) <= 0.4


def test_threshold_classifier_applies_cutoff_and_keeps_probabilities():
    rng = np.random.default_rng(1)
    features = pd.DataFrame({"a": rng.normal(size=300)})
    target = pd.Series(
        np.where(features.a + rng.normal(size=300) > 0, "yes", "no")
    )
    model = thresholds.ThresholdClassifier(
        linear_model.LogisticRegression(), 0.8
    ).fit(features, target)
    proba = model.predict_proba(features)[:, 1]
    expected = np.where(proba >= 0.8, "yes", "no")
    np.testing.assert_array_equal(model.predict(features), expected)
    assert (model.predict(features) == "yes").mean() < 0.5
    assert list(model.classes_) == ["no", "yes"]


def test_around_fitted_does_not_refit_and_clone_resets():
    features = pd.DataFrame({"a": np.arange(100.0)})
    target = pd.Series([0] * 50 + [1] * 50)
    fitted = linear_model.LogisticRegression().fit(features, target)
    wrapper = thresholds.ThresholdClassifier.around_fitted(fitted, 0.3)
    assert wrapper.estimator_ is fitted and wrapper.predict(features).shape == (
        100,
    )
    twin = base.clone(wrapper)
    assert not hasattr(twin, "estimator_") and twin.threshold == 0.3


def test_out_of_fold_probabilities_cover_validation_rows_only_once():
    rng = np.random.default_rng(2)
    features = pd.DataFrame({"a": rng.normal(size=120)})
    target = pd.Series((features.a > 0).astype(int))
    folds = [
        (list(range(0, 60)), list(range(60, 90))),
        (list(range(0, 90)), list(range(90, 120))),
    ]
    probability, positions = thresholds.out_of_fold_probabilities(
        linear_model.LogisticRegression, features, target, folds
    )
    assert sorted(positions) == list(range(60, 120)) and len(probability) == 60
