"""Stored fold models for CV+: partition rule, intervals, validation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from scmle.evaluation import conformal
from scmle.evaluation import cv_plus


def test_disjoint_folds_give_the_fold_of_each_row_in_score_order():
    folds = cv_plus.partition_folds([[0, 3], [1, 4], [2]], 5)
    assert folds.tolist() == [0, 0, 1, 1, 2]


@pytest.mark.parametrize("held", [[[0, 1], [1, 2]], [[0, 1], [5]], [[0], [-1]]])
def test_overlapping_or_foreign_positions_give_no_partition(held):
    assert cv_plus.partition_folds(held, 3) is None


def fitted(shift):
    x = np.arange(10, dtype=float).reshape(-1, 1)
    return LinearRegression().fit(x, 2 * x[:, 0] + shift)


def test_interval_matches_the_bounds_function_on_the_same_inputs():
    models = (fitted(0.0), fitted(1.0), fitted(-1.0))
    fold_of_row = np.arange(30) % 3
    residuals = np.linspace(0.1, 3.0, 30)
    ensemble = cv_plus.CvPlusEnsemble(models, fold_of_row, residuals)
    frame = pd.DataFrame({0: [1.5, 4.0]})
    lower, upper = ensemble.interval(frame, 0.8)
    predictions = np.column_stack([m.predict(frame.to_numpy()) for m in models])
    want = conformal.cv_plus_bounds(predictions, fold_of_row, residuals, 0.8)
    assert np.allclose(lower, want[0]) and np.allclose(upper, want[1])
    assert (lower < upper).all()


def test_too_few_rows_or_an_unknown_fold_is_refused():
    models = (fitted(0.0),)
    with pytest.raises(ValueError, match="at least"):
        cv_plus.CvPlusEnsemble(models, np.zeros(5, int), np.ones(5))
    with pytest.raises(ValueError, match="without a model"):
        cv_plus.CvPlusEnsemble(models, np.ones(30, int), np.ones(30))
    with pytest.raises(ValueError, match="differ"):
        cv_plus.CvPlusEnsemble(models, np.zeros(30, int), np.ones(29))


def test_serialized_size_is_positive():
    assert cv_plus.serialized_size([fitted(0.0)]) > 0
