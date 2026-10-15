"""Built-in scikit-learn model adapters.

Concrete adapters only declare data (estimator classes, parameters,
capabilities); construction logic lives once in `SklearnAdapter.build`.
"""

import inspect
import types
from typing import Any, ClassVar

from sklearn import dummy
from sklearn import ensemble
from sklearn import linear_model
from sklearn import tree

from mml.contracts import models
from mml.contracts import tasks

BINARY = tasks.TaskType.BINARY
MULTICLASS = tasks.TaskType.MULTICLASS
REGRESSION = tasks.TaskType.REGRESSION
CLASSIFICATION = frozenset({BINARY, MULTICLASS})
REGRESSION_ONLY = frozenset({REGRESSION})
ALL_TASKS = CLASSIFICATION | REGRESSION_ONLY
NO_PARAMS: types.MappingProxyType = types.MappingProxyType({})


class SklearnAdapter(models.ModelAdapter):
    """Adapter whose estimators are scikit-learn classes.

    Attributes:
        classifier: Estimator class used for classification tasks.
        regressor: Estimator class used for regression tasks.
        params: Fixed constructor parameters.
    """

    classifier: ClassVar[type | None] = None
    regressor: ClassVar[type | None] = None
    params: ClassVar[types.MappingProxyType] = NO_PARAMS

    def build(self, task: tasks.TaskType, seed: int) -> models.Estimator:
        """Instantiates the class matching `task`, seeding it if supported."""
        estimator_class = (
            self.classifier if task.is_classification else self.regressor
        )
        kwargs: dict[str, Any] = dict(self.params)
        if "random_state" in inspect.signature(estimator_class).parameters:
            kwargs["random_state"] = seed
        return estimator_class(**kwargs)


class DummyAdapter(models.ModelAdapter):
    """Prior / mean predictor used as the mandatory baseline."""

    name = "dummy"
    family = "baseline"
    capabilities = models.Capabilities(
        ALL_TASKS, sparse_input=True, handles_missing=True
    )

    def build(self, task: tasks.TaskType, seed: int) -> models.Estimator:
        """Returns a prior classifier or a mean regressor."""
        if task.is_classification:
            return dummy.DummyClassifier(strategy="prior")
        return dummy.DummyRegressor(strategy="mean")


class LogisticRegressionAdapter(SklearnAdapter):
    """Regularized logistic regression."""

    name = "logistic_regression"
    family = "linear"
    capabilities = models.Capabilities(
        CLASSIFICATION, sparse_input=True, needs_scaling=True
    )
    classifier = linear_model.LogisticRegression
    params = types.MappingProxyType({"max_iter": 1000})


class RidgeAdapter(SklearnAdapter):
    """Ridge regression."""

    name = "ridge"
    family = "linear"
    capabilities = models.Capabilities(
        REGRESSION_ONLY, sparse_input=True, needs_scaling=True
    )
    regressor = linear_model.Ridge


class ElasticNetAdapter(SklearnAdapter):
    """Elastic-net regression."""

    name = "elastic_net"
    family = "linear"
    capabilities = models.Capabilities(
        REGRESSION_ONLY, sparse_input=True, needs_scaling=True, cost=2
    )
    regressor = linear_model.ElasticNet
    params = types.MappingProxyType({"alpha": 0.1, "max_iter": 5000})


class DecisionTreeAdapter(SklearnAdapter):
    """Single decision tree."""

    name = "decision_tree"
    family = "tree"
    capabilities = models.Capabilities(ALL_TASKS, handles_missing=True)
    classifier = tree.DecisionTreeClassifier
    regressor = tree.DecisionTreeRegressor


class RandomForestAdapter(SklearnAdapter):
    """Random forest."""

    name = "random_forest"
    family = "tree"
    capabilities = models.Capabilities(ALL_TASKS, cost=3)
    classifier = ensemble.RandomForestClassifier
    regressor = ensemble.RandomForestRegressor


class ExtraTreesAdapter(SklearnAdapter):
    """Extremely randomized trees."""

    name = "extra_trees"
    family = "tree"
    capabilities = models.Capabilities(ALL_TASKS, cost=3)
    classifier = ensemble.ExtraTreesClassifier
    regressor = ensemble.ExtraTreesRegressor


class HistGradientBoostingAdapter(SklearnAdapter):
    """Histogram-based gradient boosting."""

    name = "hist_gradient_boosting"
    family = "boosting"
    capabilities = models.Capabilities(
        ALL_TASKS,
        handles_missing=True,
        native_categorical=True,
        early_stopping=True,
        cost=3,
    )
    classifier = ensemble.HistGradientBoostingClassifier
    regressor = ensemble.HistGradientBoostingRegressor


BUILTIN_ADAPTERS: tuple[type[models.ModelAdapter], ...] = (
    DummyAdapter,
    LogisticRegressionAdapter,
    RidgeAdapter,
    ElasticNetAdapter,
    DecisionTreeAdapter,
    RandomForestAdapter,
    ExtraTreesAdapter,
    HistGradientBoostingAdapter,
)
