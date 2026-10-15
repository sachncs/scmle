"""Built-in scikit-learn model adapters.

Concrete adapters only declare data (estimator classes, parameters,
capabilities); construction logic lives once in `SklearnAdapter.build`.
"""

from collections.abc import Mapping
import inspect
import types
from typing import Any
from typing import ClassVar

from sklearn import dummy
from sklearn import ensemble
from sklearn import linear_model
from sklearn import neighbors
from sklearn import svm
from sklearn import tree

from mml.contracts import errors
from mml.contracts import models
from mml.contracts import search
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
        fixed_params: Constructor parameters applied to every build.
    """

    classifier: ClassVar[type | None] = None
    regressor: ClassVar[type | None] = None
    fixed_params: ClassVar[types.MappingProxyType] = NO_PARAMS

    def build(
        self,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Instantiates the class matching `task`, seeding it if supported.

        Args:
            task: Resolved task type.
            seed: Random seed.
            params: Search-space overrides applied over `fixed_params`.

        Returns:
            An unfitted estimator.

        Raises:
            ConfigError: If the adapter declares no class for the task.
        """
        estimator_class = (
            self.classifier if task.is_classification else self.regressor
        )
        if estimator_class is None:
            raise errors.ConfigError(
                f"{self.name}: no estimator class for {task.value}"
            )
        kwargs: dict[str, Any] = {**self.fixed_params, **(params or {})}
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

    def build(
        self,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
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
    fixed_params = types.MappingProxyType({"max_iter": 1000})
    search_space = search.SearchSpace(
        (search.FloatParameter("C", 1e-3, 1e2, log=True),)
    )


class RidgeAdapter(SklearnAdapter):
    """Ridge regression."""

    name = "ridge"
    family = "linear"
    capabilities = models.Capabilities(
        REGRESSION_ONLY, sparse_input=True, needs_scaling=True
    )
    regressor = linear_model.Ridge
    search_space = search.SearchSpace(
        (search.FloatParameter("alpha", 1e-3, 1e3, log=True),)
    )


class ElasticNetAdapter(SklearnAdapter):
    """Elastic-net regression."""

    name = "elastic_net"
    family = "linear"
    capabilities = models.Capabilities(
        REGRESSION_ONLY, sparse_input=True, needs_scaling=True, cost=2
    )
    regressor = linear_model.ElasticNet
    fixed_params = types.MappingProxyType({"alpha": 0.1, "max_iter": 5000})
    search_space = search.SearchSpace(
        (
            search.FloatParameter("alpha", 1e-3, 10.0, log=True),
            search.FloatParameter("l1_ratio", 0.05, 0.95),
        )
    )


class DecisionTreeAdapter(SklearnAdapter):
    """Single decision tree."""

    name = "decision_tree"
    family = "tree"
    capabilities = models.Capabilities(ALL_TASKS, handles_missing=True)
    classifier = tree.DecisionTreeClassifier
    regressor = tree.DecisionTreeRegressor
    search_space = search.SearchSpace(
        (
            search.IntParameter("max_depth", 2, 20),
            search.IntParameter("min_samples_leaf", 1, 30, log=True),
        )
    )


class RandomForestAdapter(SklearnAdapter):
    """Random forest."""

    name = "random_forest"
    family = "tree"
    capabilities = models.Capabilities(ALL_TASKS, cost=3)
    classifier = ensemble.RandomForestClassifier
    regressor = ensemble.RandomForestRegressor
    search_space = search.SearchSpace(
        (
            search.IntParameter("n_estimators", 50, 300),
            search.IntParameter("max_depth", 3, 25),
            search.IntParameter("min_samples_leaf", 1, 20, log=True),
            search.ChoiceParameter("max_features", ("sqrt", 0.5, 1.0)),
        )
    )


class ExtraTreesAdapter(SklearnAdapter):
    """Extremely randomized trees."""

    name = "extra_trees"
    family = "tree"
    capabilities = models.Capabilities(ALL_TASKS, cost=3)
    classifier = ensemble.ExtraTreesClassifier
    regressor = ensemble.ExtraTreesRegressor
    search_space = search.SearchSpace(
        (
            search.IntParameter("n_estimators", 50, 300),
            search.IntParameter("max_depth", 3, 25),
            search.IntParameter("min_samples_leaf", 1, 20, log=True),
            search.ChoiceParameter("max_features", ("sqrt", 0.5, 1.0)),
        )
    )


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
    search_space = search.SearchSpace(
        (
            search.FloatParameter("learning_rate", 0.02, 0.3, log=True),
            search.IntParameter("max_leaf_nodes", 8, 64),
            search.IntParameter("min_samples_leaf", 5, 50),
            search.FloatParameter("l2_regularization", 1e-4, 1.0, log=True),
        )
    )


class SupportVectorAdapter(SklearnAdapter):
    """Support vector machine (classification with calibrated probabilities)."""

    name = "svm"
    family = "kernel"
    capabilities = models.Capabilities(
        ALL_TASKS,
        sparse_input=True,
        needs_scaling=True,
        max_rows=20_000,
        cost=4,
    )
    classifier = svm.SVC
    regressor = svm.SVR
    fixed_params = types.MappingProxyType({"probability": True})
    search_space = search.SearchSpace(
        (
            search.FloatParameter("C", 1e-2, 1e2, log=True),
            search.FloatParameter("gamma", 1e-4, 1.0, log=True),
        )
    )

    def build(
        self,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Builds an SVC / SVR; `probability` only exists for classifiers."""
        estimator_class = (
            self.classifier if task.is_classification else self.regressor
        )
        kwargs: dict[str, Any] = dict(params or {})
        if task.is_classification:
            kwargs.update(self.fixed_params)
            kwargs["random_state"] = seed
        return estimator_class(**kwargs)


class KnnAdapter(SklearnAdapter):
    """k-nearest neighbours."""

    name = "knn"
    family = "neighbors"
    capabilities = models.Capabilities(
        ALL_TASKS,
        sparse_input=True,
        needs_scaling=True,
        max_rows=50_000,
        cost=2,
    )
    classifier = neighbors.KNeighborsClassifier
    regressor = neighbors.KNeighborsRegressor
    search_space = search.SearchSpace(
        (
            search.IntParameter("n_neighbors", 3, 50),
            search.ChoiceParameter("weights", ("uniform", "distance")),
        )
    )


BUILTIN_ADAPTERS: tuple[type[models.ModelAdapter], ...] = (
    DummyAdapter,
    LogisticRegressionAdapter,
    RidgeAdapter,
    ElasticNetAdapter,
    DecisionTreeAdapter,
    RandomForestAdapter,
    ExtraTreesAdapter,
    HistGradientBoostingAdapter,
    SupportVectorAdapter,
    KnnAdapter,
)
