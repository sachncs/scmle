"""The stack over complete member pipelines and the blenders' arithmetic."""

import math

import numpy
import pytest
from sklearn import ensemble
from sklearn import linear_model

from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.models import ensembles
import scmle.models.registry

PLAN = data.Blueprint(("x",), ("c",))
MEMBERS = {
    "members": [
        {"model": "decision_tree", "params": {"max_depth": 2}},
        {"model": "extra_trees"},
    ]
}


def stacking() -> ensembles.Stacking:
    return scmle.models.registry.Catalog.stock().get("stacking")


def test_a_classifier_stack_runs_its_members_on_the_raw_frame() -> None:
    stack = stacking().assemble(PLAN, tasks.Task.BINARY, 0, MEMBERS)
    assert isinstance(stack, ensemble.StackingClassifier)
    assert [member[0] for member in stack.estimators] == [
        "m0_decision_tree",
        "m1_extra_trees",
    ]
    first = stack.estimators[0][1]
    assert first.steps[-1][1].get_params()["max_depth"] == 2
    assert stack.stack_method == "auto"
    assert isinstance(stack.final_estimator, linear_model.LogisticRegression)


def test_a_regressor_stack_combines_with_a_ridge() -> None:
    stack = stacking().assemble(PLAN, tasks.Task.REGRESSION, 0, MEMBERS)
    assert isinstance(stack, ensemble.StackingRegressor)
    assert isinstance(stack.final_estimator, linear_model.RidgeCV)


def test_a_stack_of_one_member_is_refused() -> None:
    with pytest.raises(
        errors.ConfigError, match=r"^stacking needs at least two members$"
    ):
        stacking().assemble(
            PLAN, tasks.Task.BINARY, 0, {"members": [{"model": "dummy"}]}
        )


def test_a_zero_probability_costs_a_bounded_log_loss() -> None:
    loss = ensembles.ClassifierBlender().loss(
        numpy.array([[0.0, 1.0]]), numpy.array([[1.0, 0.0]])
    )
    assert loss == pytest.approx(-math.log(1e-12))


def test_the_regression_truth_is_a_float_column() -> None:
    truth = ensembles.RegressorBlender().prepare(
        numpy.zeros((3, 2, 1)), [1, 2, 3]
    )[1]
    assert truth.dtype == numpy.float64
    assert truth.shape == (3, 1)
