"""Stored fold models for CV+: partition rule, intervals, validation."""

import numpy
import pandas
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
    x = numpy.arange(10, dtype=float).reshape(-1, 1)
    return LinearRegression().fit(x, 2 * x[:, 0] + shift)


def test_interval_matches_the_bounds_function_on_the_same_inputs():
    models = (fitted(0.0), fitted(1.0), fitted(-1.0))
    fold_of_row = numpy.arange(30) % 3
    residuals = numpy.linspace(0.1, 3.0, 30)
    ensemble = cv_plus.CvPlusEnsemble(models, fold_of_row, residuals)
    frame = pandas.DataFrame({0: [1.5, 4.0]})
    lower, upper = ensemble.interval(frame, 0.8)
    predictions = numpy.column_stack(
        [m.predict(frame.to_numpy()) for m in models]
    )
    want = conformal.cv_plus_bounds(predictions, fold_of_row, residuals, 0.8)
    assert numpy.allclose(lower, want[0]) and numpy.allclose(upper, want[1])
    assert (lower < upper).all()


def test_too_few_rows_or_an_unknown_fold_is_refused():
    models = (fitted(0.0),)
    with pytest.raises(ValueError, match="at least"):
        cv_plus.CvPlusEnsemble(models, numpy.zeros(5, int), numpy.ones(5))
    with pytest.raises(ValueError, match="without a model"):
        cv_plus.CvPlusEnsemble(models, numpy.ones(30, int), numpy.ones(30))
    with pytest.raises(ValueError, match="differ"):
        cv_plus.CvPlusEnsemble(models, numpy.zeros(30, int), numpy.ones(29))


def test_serialized_size_is_positive():
    assert cv_plus.serialized_size([fitted(0.0)]) > 0


def test_from_a_thousand_rows_the_scores_are_pooled_out_of_fold():
    held = [list(range(i, 1200, 4)) for i in range(4)]
    plan = cv_plus.plan_scores(held, 1200, 1200, [object()] * 4)

    assert plan.mode == cv_plus.POOLED
    assert plan.folds is None
    assert str(cv_plus.MIN_ROWS_FOR_SPLIT) in plan.reason


def test_below_a_thousand_rows_with_disjoint_folds_the_fold_models_are_kept():
    held = [list(range(i, 200, 4)) for i in range(4)]
    plan = cv_plus.plan_scores(held, 200, 200, [object()] * 4)

    assert plan.mode == cv_plus.FOLD_MODELS
    assert plan.folds.tolist() == [
        i % 4 for i in range(4) for ignored in held[i]
    ]
    assert plan.reason is None


def test_overlapping_folds_fall_back_to_the_pooled_scores():
    plan = cv_plus.plan_scores([[0, 1], [1, 2]], 3, 4, [])
    assert plan.mode == cv_plus.POOLED
    assert "overlap" in plan.reason


def test_a_model_set_above_the_cap_is_not_kept_and_the_cap_is_a_parameter():
    held = [list(range(i, 200, 4)) for i in range(4)]
    models = [fitted(float(i)) for i in range(4)]
    size = cv_plus.serialized_size(models)

    small = cv_plus.plan_scores(held, 200, 200, models, max_bytes=size - 1)
    large = cv_plus.plan_scores(held, 200, 200, models, max_bytes=size)

    assert small.mode == cv_plus.POOLED and str(size - 1) in small.reason
    assert large.mode == cv_plus.FOLD_MODELS
