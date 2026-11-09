"""Built-in scikit-learn model adapters.

Pattern: classes carry behaviour, instances carry data. `Learner` holds the
construction logic once and takes everything that merely describes a model
(name, family, capabilities, estimator classes, fixed parameters, search
space) as keyword arguments, so a model that only declares data is one
configured instance, such as `RIDGE`, not a subclass. A model that really
behaves differently stays a subclass that overrides a method (`SupportVector`
overrides `build`, `HistGradientBoosting` overrides `categorical_params`) and
is configured the same way. Instances are registered like any adapter and
must be treated as immutable. Adapters written as subclasses that declare
class attributes, including user adapters registered through
`register_estimator`, keep working: `Learner` only sets an attribute when it
is given, so class-level declarations are left alone.
"""

from collections.abc import Mapping
import inspect
import types
from typing import Any

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

CLASSIFICATION = frozenset({tasks.Task.BINARY, tasks.Task.MULTICLASS})
REGRESSION_ONLY = frozenset({tasks.Task.REGRESSION})
ALL_TASKS = CLASSIFICATION | REGRESSION_ONLY
CALIBRATION_FOLDS = 3
NO_PARAMS: Mapping[str, Any] = types.MappingProxyType({})


class Learner(models.Model):
    """Adapter whose estimators are scikit-learn classes.

    Configure it with keyword arguments, or subclass it and declare the same
    names as class attributes. An argument left out keeps the class-level
    value, so both styles work.

    Attributes:
        classifier: Estimator class used for classification tasks.
        regressor: Estimator class used for regression tasks.
        fixed_params: Constructor parameters applied to every build.
    """

    classifier: type | None = None
    regressor: type | None = None
    fixed_params: Mapping[str, Any] = NO_PARAMS

    def __init__(
        self,
        *,
        name: str | None = None,
        family: str | None = None,
        capabilities: models.Capabilities | None = None,
        classifier: type | None = None,
        regressor: type | None = None,
        fixed_params: Mapping[str, Any] | None = None,
        search_space: search.SearchSpace | None = None,
        monotonic_param: str | None = None,
    ) -> None:
        """Sets the declared data on this instance.

        Args:
            name: Unique registry name.
            family: Coarse family used for reporting.
            capabilities: Declared capabilities.
            classifier: Estimator class used for classification tasks.
            regressor: Estimator class used for regression tasks.
            fixed_params: Constructor parameters applied to every build.
            search_space: Tunable hyperparameters.
            monotonic_param: Name of the estimator parameter that takes one
                monotonic sign per input column.
        """
        given: dict[str, Any] = {
            "name": name,
            "family": family,
            "capabilities": capabilities,
            "classifier": classifier,
            "regressor": regressor,
            "fixed_params": fixed_params,
            "search_space": search_space,
            "monotonic_param": monotonic_param,
        }
        for attribute, value in given.items():
            if value is not None:
                setattr(self, attribute, value)

    def build(
        self,
        task: tasks.Task,
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


class Dummy(models.Model):
    """Prior / mean predictor used as the mandatory baseline."""

    name = "dummy"
    family = "baseline"
    capabilities = models.Capabilities(
        ALL_TASKS
        | frozenset(
            {
                tasks.Task.SURVIVAL,
                tasks.Task.ORDINAL,
                tasks.Task.RANKING,
                tasks.Task.MULTILABEL,
                tasks.Task.MULTIOUTPUT,
                tasks.Task.FORECAST,
                tasks.Task.UPLIFT,
            }
        ),
        sparse_input=True,
        handles_missing=True,
    )

    def build(
        self,
        task: tasks.Task,
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
        if task is tasks.Task.SURVIVAL:
            return scmle.models.survival.estimators.MarginalSurvival()
        if task is tasks.Task.UPLIFT:
            return scmle.models.uplift.estimators.ZeroEffect()
        if task in (
            tasks.Task.MULTILABEL,
            tasks.Task.MULTIOUTPUT,
            tasks.Task.FORECAST,
        ):
            classify = task is tasks.Task.MULTILABEL
            return scmle.models.multitarget.estimators.PerTarget(
                dummy.DummyClassifier(strategy="prior")
                if classify
                else dummy.DummyRegressor(strategy="mean"),
                classify,
            )
        if task is tasks.Task.RANKING:
            return scmle.models.ranking.estimators.ConstantRanker()
        if task is tasks.Task.ORDINAL:
            return scmle.models.ordinal.estimators.CumulativeThresholdRegressor(
                dummy.DummyRegressor(strategy="median"),
                (params or {}).get("levels"),
                seed,
            )
        if task.is_classification:
            return dummy.DummyClassifier(strategy="prior")
        return dummy.DummyRegressor(strategy="mean")


class HistGradientBoosting(Learner):
    """Histogram-based gradient boosting, which takes native categoricals."""

    def categorical_params(self, indices: list[int]) -> dict[str, Any]:
        """Marks the integer-coded columns as categorical.

        Args:
            indices: Positions of the integer-coded categorical columns.

        Returns:
            Constructor parameters naming those columns, as
            `{"categorical_features": [...]}`.
        """
        return {"categorical_features": list(indices)}


class SupportVector(Learner):
    """Support vector machine with sigmoid-calibrated probabilities.

    scikit-learn deprecated `SVC(probability=True)` in 1.9; the supported
    equivalent is a cross-fitted `CalibratedClassifierCV(SVC(),
    ensemble=False)`, which is also considerably faster.
    """

    def build(
        self,
        task: tasks.Task,
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


TREE_ENSEMBLE_SPACE = search.SearchSpace(
    (
        search.IntParameter("n_estimators", 50, 300),
        search.IntParameter("max_depth", 3, 25),
        search.IntParameter("min_samples_leaf", 1, 20, log=True),
        search.ChoiceParameter("max_features", ("sqrt", 0.5, 1.0)),
    )
)

LOGISTIC_REGRESSION = Learner(
    name="logistic_regression",
    family="linear",
    capabilities=models.Capabilities(
        CLASSIFICATION, sparse_input=True, needs_scaling=True, memory_factor=2.0
    ),
    classifier=linear_model.LogisticRegression,
    fixed_params={"max_iter": 1000},
    search_space=search.SearchSpace(
        (search.FloatParameter("C", 1e-3, 1e2, log=True),)
    ),
)

RIDGE = Learner(
    name="ridge",
    family="linear",
    capabilities=models.Capabilities(
        REGRESSION_ONLY,
        sparse_input=True,
        needs_scaling=True,
        memory_factor=2.0,
    ),
    regressor=linear_model.Ridge,
    search_space=search.SearchSpace(
        (search.FloatParameter("alpha", 1e-3, 1e3, log=True),)
    ),
)

SGD_LOGISTIC = Learner(
    name="sgd_logistic",
    family="linear",
    capabilities=models.Capabilities(
        CLASSIFICATION,
        sparse_input=True,
        needs_scaling=True,
        incremental=True,
        stream_only=True,
        memory_factor=1.5,
    ),
    classifier=linear_model.SGDClassifier,
    fixed_params={"loss": "log_loss", "max_iter": 50, "tol": 1e-4},
    search_space=search.SearchSpace(
        (search.FloatParameter("alpha", 1e-7, 1e-1, log=True),)
    ),
)

SGD_REGRESSION = Learner(
    name="sgd_regression",
    family="linear",
    capabilities=models.Capabilities(
        REGRESSION_ONLY,
        sparse_input=True,
        needs_scaling=True,
        incremental=True,
        stream_only=True,
        memory_factor=1.5,
    ),
    regressor=linear_model.SGDRegressor,
    fixed_params={"max_iter": 50, "tol": 1e-4},
    search_space=search.SearchSpace(
        (search.FloatParameter("alpha", 1e-7, 1e-1, log=True),)
    ),
)

ELASTIC_NET = Learner(
    name="elastic_net",
    family="linear",
    capabilities=models.Capabilities(
        REGRESSION_ONLY,
        sparse_input=True,
        needs_scaling=True,
        cost=2,
        memory_factor=2.0,
    ),
    regressor=linear_model.ElasticNet,
    fixed_params={"alpha": 0.1, "max_iter": 5000},
    search_space=search.SearchSpace(
        (
            search.FloatParameter("alpha", 1e-3, 10.0, log=True),
            search.FloatParameter("l1_ratio", 0.05, 0.95),
        )
    ),
)

DECISION_TREE = Learner(
    name="decision_tree",
    family="tree",
    capabilities=models.Capabilities(
        ALL_TASKS, handles_missing=True, memory_factor=2.0
    ),
    classifier=tree.DecisionTreeClassifier,
    regressor=tree.DecisionTreeRegressor,
    search_space=search.SearchSpace(
        (
            search.IntParameter("max_depth", 2, 20),
            search.IntParameter("min_samples_leaf", 1, 30, log=True),
        )
    ),
)

RANDOM_FOREST = Learner(
    name="random_forest",
    family="tree",
    capabilities=models.Capabilities(ALL_TASKS, cost=3, memory_factor=6.0),
    classifier=ensemble.RandomForestClassifier,
    regressor=ensemble.RandomForestRegressor,
    search_space=TREE_ENSEMBLE_SPACE,
)

EXTRA_TREES = Learner(
    name="extra_trees",
    family="tree",
    capabilities=models.Capabilities(ALL_TASKS, cost=3, memory_factor=6.0),
    classifier=ensemble.ExtraTreesClassifier,
    regressor=ensemble.ExtraTreesRegressor,
    search_space=TREE_ENSEMBLE_SPACE,
)

HIST_GRADIENT_BOOSTING = HistGradientBoosting(
    name="hist_gradient_boosting",
    family="boosting",
    capabilities=models.Capabilities(
        ALL_TASKS,
        handles_missing=True,
        native_categorical=True,
        early_stopping=True,
        cost=3,
        memory_factor=3.0,
    ),
    classifier=ensemble.HistGradientBoostingClassifier,
    regressor=ensemble.HistGradientBoostingRegressor,
    monotonic_param="monotonic_cst",
    search_space=search.SearchSpace(
        (
            search.FloatParameter("learning_rate", 0.02, 0.3, log=True),
            search.IntParameter("max_leaf_nodes", 8, 64),
            search.IntParameter("min_samples_leaf", 5, 50),
            search.FloatParameter("l2_regularization", 1e-4, 1.0, log=True),
        )
    ),
)

SUPPORT_VECTOR = SupportVector(
    name="svm",
    family="kernel",
    capabilities=models.Capabilities(
        ALL_TASKS,
        sparse_input=True,
        needs_scaling=True,
        max_rows=20_000,
        cost=4,
        memory_factor=4.0,
    ),
    classifier=svm.SVC,
    regressor=svm.SVR,
    search_space=search.SearchSpace(
        (
            search.FloatParameter("C", 1e-2, 1e2, log=True),
            search.FloatParameter("gamma", 1e-4, 1.0, log=True),
        )
    ),
)

KNN = Learner(
    name="knn",
    family="neighbors",
    capabilities=models.Capabilities(
        ALL_TASKS,
        sparse_input=True,
        needs_scaling=True,
        max_rows=50_000,
        cost=2,
        memory_factor=3.0,
    ),
    classifier=neighbors.KNeighborsClassifier,
    regressor=neighbors.KNeighborsRegressor,
    search_space=search.SearchSpace(
        (
            search.IntParameter("n_neighbors", 3, 50),
            search.ChoiceParameter("weights", ("uniform", "distance")),
        )
    ),
)

# Multi-layer perceptron: the deep-learning family plugin. Deep architectures
# register exactly like this: a `Model` with declared capabilities and a
# search space. Larger frameworks (PyTorch, Keras) plug in through the same
# contract and the `scmle.models` entry point.
NEURAL_NETWORK = Learner(
    name="neural_network",
    family="neural",
    capabilities=models.Capabilities(
        ALL_TASKS,
        needs_scaling=True,
        early_stopping=True,
        max_rows=200_000,
        cost=4,
        memory_factor=4.0,
    ),
    classifier=neural_network.MLPClassifier,
    regressor=neural_network.MLPRegressor,
    fixed_params={
        "max_iter": 300,
        "early_stopping": True,
        "n_iter_no_change": 10,
    },
    search_space=search.SearchSpace(
        (
            search.ChoiceParameter(
                "hidden_layer_sizes", ((32,), (64,), (64, 32), (128, 64))
            ),
            search.FloatParameter("alpha", 1e-5, 1e-1, log=True),
            search.FloatParameter("learning_rate_init", 1e-4, 1e-2, log=True),
        )
    ),
)

BUILTIN_ADAPTERS: tuple[models.Model, ...] = (
    Dummy(),
    LOGISTIC_REGRESSION,
    RIDGE,
    SGD_LOGISTIC,
    SGD_REGRESSION,
    ELASTIC_NET,
    DECISION_TREE,
    RANDOM_FOREST,
    EXTRA_TREES,
    HIST_GRADIENT_BOOSTING,
    SUPPORT_VECTOR,
    KNN,
    NEURAL_NETWORK,
)
"""Built-in adapter instances, in registration order."""
