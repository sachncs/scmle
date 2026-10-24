"""Ordinal estimator and metrics."""

import numpy as np
import pandas as pd
import pytest
from sklearn import linear_model

from scmle.evaluation import ordinal
from scmle.models.ordinal import estimators


def ladder(rows=900, seed=0, levels=4):
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(
        {"a": rng.normal(size=rows), "b": rng.normal(size=rows)}
    )
    latent = features.a * 1.5 - features.b + rng.logistic(size=rows)
    cuts = np.quantile(latent, np.linspace(0, 1, levels + 1)[1:-1])
    return features, np.searchsorted(cuts, latent)


def test_probabilities_follow_the_level_order_and_sum_to_one():
    features, ranks = ladder()
    model = estimators.CumulativeThresholdRegressor(
        linear_model.Ridge(), levels=4
    ).fit(features, ranks)

    probabilities = model.predict_proba(features)

    assert probabilities.shape == (len(features), 4)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    low_row = features.a.idxmin()
    high_row = features.a.idxmax()
    assert probabilities[low_row].argmax() <= probabilities[high_row].argmax()


def test_predictions_are_valid_ranks_and_beat_a_constant_guess():
    features, ranks = ladder()
    model = estimators.CumulativeThresholdRegressor(
        linear_model.Ridge(), levels=4
    ).fit(features[:600], ranks[:600])

    guess = model.predict(features[600:])

    assert set(np.unique(guess)) <= {0, 1, 2, 3}
    constant = np.full_like(guess, np.median(ranks[:600]))
    assert ordinal.mean_absolute_class_error(
        ranks[600:], guess
    ) < ordinal.mean_absolute_class_error(ranks[600:], constant)


def test_a_missing_top_level_still_gives_a_full_probability_row():
    features, ranks = ladder()
    keep = ranks < 3

    model = estimators.CumulativeThresholdRegressor(
        linear_model.Ridge(), levels=4
    ).fit(features[keep], ranks[keep])

    assert model.predict_proba(features.head(5)).shape == (5, 4)


def test_quadratic_kappa_punishes_far_misses_more_than_near_ones():
    truth = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    near = np.array([1, 1, 2, 3, 0, 1, 2, 2])
    far = np.array([3, 1, 2, 0, 0, 1, 2, 3])

    assert ordinal.quadratic_weighted_kappa(truth, near) > (
        ordinal.quadratic_weighted_kappa(truth, far)
    )
    assert ordinal.quadratic_weighted_kappa(truth, truth) == pytest.approx(1.0)


def test_mean_absolute_class_error_counts_levels():
    assert ordinal.mean_absolute_class_error(
        np.array([0, 2, 3]), np.array([1, 2, 1])
    ) == pytest.approx(1.0)
