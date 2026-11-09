"""Built-in scikit-learn model adapters.

Concrete adapters only declare data (estimator classes, parameters,
capabilities); construction logic lives once in `SklearnAdapter.build`.
"""

from collections.abc import Mapping
import inspect
import types
from typing import Any
from typing import ClassVar

from sklearn import calibration
from sklearn import dummy
from sklearn import ensemble
from sklearn import linear_model
from sklearn import neighbors
from sklearn import neural_network
from sklearn import svm
from sklearn import tree

from scmle.contracts import errors
from scmle.contracts import models
from scmle.contracts import search
from scmle.contracts import tasks
import scmle.models.multitarget.estimators
import scmle.models.ordinal.estimators
import scmle.models.ranking.estimators
import scmle.models.survival.estimators
import scmle.models.uplift.estimators

BINARY = tasks.TaskType.BINARY
MULTICLASS = tasks.TaskType.MULTICLASS
REGRESSION = tasks.TaskType.REGRESSION
CLASSIFICATION = frozenset({BINARY, MULTICLASS})
REGRESSION_ONLY = frozenset({REGRESSION})
ALL_TASKS = CLASSIFICATION | REGRESSION_ONLY
CALIBRATION_FOLDS = 3
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
        ALL_TASKS
        | frozenset(
            {
                tasks.TaskType.SURVIVAL,
                tasks.TaskType.ORDINAL,
                tasks.TaskType.RANKING,
                tasks.TaskType.MULTILABEL,
                tasks.TaskType.MULTIOUTPUT,
                tasks.TaskType.FORECAST,
                tasks.TaskType.UPLIFT,
            }
        ),
        sparse_input=True,
        handles_missing=True,
    )

    def build(
        self,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Returns a prior classifier, a mean regressor or a marginal curve.

        Other tasks get their matching baseline: zero effect for uplift, a
        constant score for ranking, per-target priors or means for multi-target
        tasks, and a median regressor behind thresholds for ordinal.

        Args:
            task: Resolved task type, which selects the baseline.
            seed: Random seed (used by the ordinal baseline).
            params: Only the `levels` entry is read, for ordinal tasks.

        Returns:
            An unfitted baseline estimator for the task.
        """
        if task is tasks.TaskType.SURVIVAL:
            return scmle.models.survival.estimators.MarginalSurvival()
        if task is tasks.TaskType.UPLIFT:
            return scmle.models.uplift.estimators.ZeroEffect()
        if task in (
            tasks.TaskType.MULTILABEL,
            tasks.TaskType.MULTIOUTPUT,
            tasks.TaskType.FORECAST,
        ):
            classify = task is tasks.TaskType.MULTILABEL
            return scmle.models.multitarget.estimators.PerTarget(
                dummy.DummyClassifier(strategy="prior")
                if classify
                else dummy.DummyRegressor(strategy="mean"),
                classify,
            )
        if task is tasks.TaskType.RANKING:
            return scmle.models.ranking.estimators.ConstantRanker()
        if task is tasks.TaskType.ORDINAL:
            return scmle.models.ordinal.estimators.CumulativeThresholdRegressor(
                dummy.DummyRegressor(strategy="median"),
                (params or {}).get("levels"),
                seed,
            )
        if task.is_classification:
            return dummy.DummyClassifier(strategy="prior")
        return dummy.DummyRegressor(strategy="mean")


class LogisticRegressionAdapter(SklearnAdapter):
    """Regularized logistic regression."""

    name = "logistic_regression"
    family = "linear"
    capabilities = models.Capabilities(
        CLASSIFICATION, sparse_input=True, needs_scaling=True, memory_factor=2.0
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
        REGRESSION_ONLY,
        sparse_input=True,
        needs_scaling=True,
        memory_factor=2.0,
    )
    regressor = linear_model.Ridge
    search_space = search.SearchSpace(
        (search.FloatParameter("alpha", 1e-3, 1e3, log=True),)
    )


class SgdLogisticAdapter(SklearnAdapter):
    """Logistic regression trained by stochastic gradient descent."""

    name = "sgd_logistic"
    family = "linear"
    capabilities = models.Capabilities(
        CLASSIFICATION,
        sparse_input=True,
        needs_scaling=True,
        incremental=True,
        stream_only=True,
        memory_factor=1.5,
    )
    classifier = linear_model.SGDClassifier
    fixed_params = types.MappingProxyType(
        {"loss": "log_loss", "max_iter": 50, "tol": 1e-4}
    )
    search_space = search.SearchSpace(
        (search.FloatParameter("alpha", 1e-7, 1e-1, log=True),)
    )


class SgdRegressionAdapter(SklearnAdapter):
    """Linear regression trained by stochastic gradient descent."""

    name = "sgd_regression"
    family = "linear"
    capabilities = models.Capabilities(
        REGRESSION_ONLY,
        sparse_input=True,
        needs_scaling=True,
        incremental=True,
        stream_only=True,
        memory_factor=1.5,
    )
    regressor = linear_model.SGDRegressor
    fixed_params = types.MappingProxyType({"max_iter": 50, "tol": 1e-4})
    search_space = search.SearchSpace(
        (search.FloatParameter("alpha", 1e-7, 1e-1, log=True),)
    )


class ElasticNetAdapter(SklearnAdapter):
    """Elastic-net regression."""

    name = "elastic_net"
    family = "linear"
    capabilities = models.Capabilities(
        REGRESSION_ONLY,
        sparse_input=True,
        needs_scaling=True,
        cost=2,
        memory_factor=2.0,
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
    capabilities = models.Capabilities(
        ALL_TASKS, handles_missing=True, memory_factor=2.0
    )
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
    capabilities = models.Capabilities(ALL_TASKS, cost=3, memory_factor=6.0)
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
    capabilities = models.Capabilities(ALL_TASKS, cost=3, memory_factor=6.0)
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
        memory_factor=3.0,
    )
    classifier = ensemble.HistGradientBoostingClassifier
    regressor = ensemble.HistGradientBoostingRegressor
    monotonic_param = "monotonic_cst"

    def categorical_params(self, indices: list[int]) -> dict[str, Any]:
        """Marks the integer-coded columns as categorical.

        Args:
            indices: Positions of the integer-coded categorical columns.

        Returns:
            Constructor parameters naming those columns, as
            `{"categorical_features": [...]}`.
        """
        return {"categorical_features": list(indices)}

    search_space = search.SearchSpace(
        (
            search.FloatParameter("learning_rate", 0.02, 0.3, log=True),
            search.IntParameter("max_leaf_nodes", 8, 64),
            search.IntParameter("min_samples_leaf", 5, 50),
            search.FloatParameter("l2_regularization", 1e-4, 1.0, log=True),
        )
    )


class SupportVectorAdapter(SklearnAdapter):
    """Support vector machine with sigmoid-calibrated probabilities.

    scikit-learn deprecated `SVC(probability=True)` in 1.9; the supported
    equivalent is a cross-fitted `CalibratedClassifierCV(SVC(),
    ensemble=False)`, which is also considerably faster.
    """

    name = "svm"
    family = "kernel"
    capabilities = models.Capabilities(
        ALL_TASKS,
        sparse_input=True,
        needs_scaling=True,
        max_rows=20_000,
        cost=4,
        memory_factor=4.0,
    )
    classifier = svm.SVC
    regressor = svm.SVR
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
        """Builds an SVR, or a calibrated SVC for classification.

        Args:
            task: Resolved task type.
            seed: Random seed of the classifier.
            params: Search-space overrides for the SVC or SVR.

        Returns:
            An unfitted SVR for regression, or a sigmoid-calibrated SVC with
            cross-fitted calibration for classification.
        """
        kwargs: dict[str, Any] = dict(params or {})
        if not task.is_classification:
            return svm.SVR(**kwargs)
        return calibration.CalibratedClassifierCV(
            svm.SVC(random_state=seed, **kwargs),
            method="sigmoid",
            ensemble=False,
            cv=CALIBRATION_FOLDS,
        )


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
        memory_factor=3.0,
    )
    classifier = neighbors.KNeighborsClassifier
    regressor = neighbors.KNeighborsRegressor
    search_space = search.SearchSpace(
        (
            search.IntParameter("n_neighbors", 3, 50),
            search.ChoiceParameter("weights", ("uniform", "distance")),
        )
    )


class NeuralNetworkAdapter(SklearnAdapter):
    """Multi-layer perceptron: the deep-learning family plugin.

    Deep architectures register exactly like this: a `ModelAdapter` with
    declared capabilities and a search space. Larger frameworks (PyTorch,
    Keras) plug in through the same contract and the `scmle.models` entry point.
    """

    name = "neural_network"
    family = "neural"
    capabilities = models.Capabilities(
        ALL_TASKS,
        needs_scaling=True,
        early_stopping=True,
        max_rows=200_000,
        cost=4,
        memory_factor=4.0,
    )
    classifier = neural_network.MLPClassifier
    regressor = neural_network.MLPRegressor
    fixed_params = types.MappingProxyType(
        {"max_iter": 300, "early_stopping": True, "n_iter_no_change": 10}
    )
    search_space = search.SearchSpace(
        (
            search.ChoiceParameter(
                "hidden_layer_sizes", ((32,), (64,), (64, 32), (128, 64))
            ),
            search.FloatParameter("alpha", 1e-5, 1e-1, log=True),
            search.FloatParameter("learning_rate_init", 1e-4, 1e-2, log=True),
        )
    )


BUILTIN_ADAPTERS: tuple[type[models.ModelAdapter], ...] = (
    DummyAdapter,
    LogisticRegressionAdapter,
    RidgeAdapter,
    SgdLogisticAdapter,
    SgdRegressionAdapter,
    ElasticNetAdapter,
    DecisionTreeAdapter,
    RandomForestAdapter,
    ExtraTreesAdapter,
    HistGradientBoostingAdapter,
    SupportVectorAdapter,
    KnnAdapter,
    NeuralNetworkAdapter,
)
