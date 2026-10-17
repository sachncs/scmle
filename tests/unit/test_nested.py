"""Nested validation scores the whole tuning procedure on untouched rows."""

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection

from mml.contracts import search
from mml.evaluation import nested

SPACE = search.SearchSpace((search.FloatParameter("C", 1e-3, 100.0, log=True),))


def noise(rows=300, seed=0):
    """Features unrelated to the labels: any score above chance is optimism."""
    rng = np.random.default_rng(seed)
    return (
        pd.DataFrame(rng.normal(size=(rows, 20))),
        pd.Series(rng.integers(0, 2, rows)),
    )


def build(params):
    return linear_model.LogisticRegression(max_iter=300, **params)


def folds(rows):
    splitter = model_selection.KFold(4, shuffle=True, random_state=0)
    return list(splitter.split(np.arange(rows)))


def test_the_honest_estimate_on_pure_noise_is_near_chance_and_optimism_shows():
    features, target = noise()

    result = nested.nested_score(
        build,
        SPACE,
        features,
        target,
        folds(len(target)),
        metrics.get_scorer("roc_auc"),
        True,
        configurations=6,
    )

    assert result is not None and result.folds == 4
    assert abs(result.mean - 0.5) < 0.1
    assert result.optimism > 0  # The search found "winners" in noise.


def test_a_real_signal_survives_the_nested_check():
    features, target = noise()
    features[0] = target * 2.0 + np.random.default_rng(1).normal(
        0, 0.5, len(target)
    )

    result = nested.nested_score(
        build,
        SPACE,
        features,
        target,
        folds(len(target)),
        metrics.get_scorer("roc_auc"),
        True,
    )

    assert result is not None and result.mean > 0.9


def test_stopping_early_returns_nothing():
    features, target = noise()

    result = nested.nested_score(
        build,
        SPACE,
        features,
        target,
        folds(len(target)),
        metrics.get_scorer("roc_auc"),
        True,
        should_stop=lambda: True,
    )

    assert result is None
