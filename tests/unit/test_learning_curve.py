"""Tests for the learning curve."""

import numpy as np
import pandas as pd
from sklearn import dummy
from sklearn import linear_model
from sklearn import metrics

from mml.evaluation import learning_curve

SCORER = metrics.get_scorer("r2")


def problem(rows, noise, seed=0):
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(rng.normal(size=(rows, 6)))
    weights = np.array([1.0, -2.0, 0.5, 0.0, 3.0, 1.5])
    target = pd.Series(
        features.to_numpy() @ weights + rng.normal(0, noise, rows)
    )
    return features, target


def run(rows, noise, estimator=None):
    features, target = problem(rows, noise)
    split = int(rows * 0.8)
    return learning_curve.compute(
        estimator or linear_model.LinearRegression(),
        features,
        target,
        range(split),
        range(split, rows),
        SCORER,
        True,
    )


def test_a_model_that_needs_more_data_keeps_climbing():
    curve = run(
        250,
        noise=3.0,
        estimator=linear_model.Ridge(alpha=1e-6),
    )

    assert curve.sizes[-1] > curve.sizes[0]
    assert curve.scores[-1] > curve.scores[0]


def test_a_simple_problem_plateaus_early():
    curve = run(2000, noise=0.1)

    assert curve.verdict == "plateau"
    assert curve.scores[-1] > 0.99


def test_a_constant_model_is_unclear_not_hopeful():
    curve = run(800, noise=1.0, estimator=dummy.DummyRegressor())

    assert curve.verdict in {"unclear", "plateau"}


def test_judging_curves_directly():
    assert learning_curve.judge([0.5, 0.6, 0.7, 0.8, 0.9])[0] == (
        "more data likely helps"
    )
    assert learning_curve.judge([0.5, 0.8, 0.88, 0.89, 0.89])[0] == "plateau"
    assert learning_curve.judge([0.8, 0.79, 0.78, 0.77, 0.76])[0] == "unclear"


def test_too_few_rows_or_a_stop_request_gives_no_curve():
    features, target = problem(60, 1.0)

    assert (
        learning_curve.compute(
            linear_model.LinearRegression(),
            features,
            target,
            range(40),
            range(40, 60),
            SCORER,
            True,
        )
        is None
    )
    features, target = problem(400, 1.0)
    assert (
        learning_curve.compute(
            linear_model.LinearRegression(),
            features,
            target,
            range(300),
            range(300, 400),
            SCORER,
            True,
            should_stop=lambda: True,
        )
        is None
    )


def test_error_metrics_are_reported_in_natural_orientation():
    features, target = problem(600, 1.0)
    scorer = metrics.get_scorer("neg_root_mean_squared_error")

    curve = learning_curve.compute(
        linear_model.LinearRegression(),
        features,
        target,
        range(450),
        range(450, 600),
        scorer,
        False,
    )

    assert all(score > 0 for score in curve.scores)
