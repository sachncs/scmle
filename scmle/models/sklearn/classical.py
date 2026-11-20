"""Built-in scikit-learn model adapters.

Pattern: classes carry behaviour, instances carry data. `Learner` holds the
construction logic once and takes everything that merely describes a model
(name, family, capabilities, estimator classes, fixed parameters, search
space) as keyword arguments, so a model that only declares data is one
configured instance, such as `RIDGE`, not a subclass. A model that really
behaves differently stays a subclass that overrides a method (`Svm`
overrides `build`, `Histogram` overrides `nominals`) and
is configured the same way. Instances are registered like any adapter and
must be treated as immutable. Adapters written as subclasses that declare
class attributes, including user adapters registered through
`custom.register`, keep working: `Learner` only sets an attribute when it
is given, so class-level declarations are left alone.
"""

from collections.abc import Mapping
import inspect
import types
from typing import Any

import numpy
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
from scmle.data.splitting import inner_cv
import scmle.models.multitarget.estimators
import scmle.models.ordinal.estimators
import scmle.models.ranking.estimators
from scmle.models.sklearn import trees
import scmle.models.survival.estimators
import scmle.models.uplift.estimators

CLASSIFICATION = frozenset({tasks.Task.BINARY, tasks.Task.MULTICLASS})
REGRESSION_ONLY = frozenset({tasks.Task.REGRESSION})
ALL_TASKS = CLASSIFICATION | REGRESSION_ONLY
CALIBRATION_FOLDS = 3
NO_PARAMS: Mapping[str, Any] = types.MappingProxyType({})
# Boosting stops on a held-out tenth of the training rows (scikit-learn's own
# `validation_fraction`) after `PATIENCE` rounds without improvement, at most
# `ROUNDS` rounds. Patience 20 at learning rate 0.1 tolerates a plateau of
# two shrinkage-weighted trees' worth of loss; `ROUNDS` = 500 lets a 0.05
# rate reach the 100 rounds scikit-learn runs at 0.1, five times over.
ROUNDS = 500
PATIENCE = 20
HOLDOUT = 0.1
# The untuned learning rate. Smaller shrinkage generalises better when
# enough rounds are allowed (Friedman 2001, "Greedy function approximation",
# section 5); early stopping bounds the cost, and at 0.05 a fit needs about
# twice the 100 rounds scikit-learn's 0.1 runs, inside `ROUNDS`. The search
# still ranges over 0.02 to 0.3.
SHRINKAGE = 0.05
# A plain dict: trials receive the adapter pickled, and a mappingproxy is not.
STOPPING: Mapping[str, Any] = {
    "learning_rate": SHRINKAGE,
    "max_iter": ROUNDS,
    "early_stopping": True,
    "validation_fraction": HOLDOUT,
    "n_iter_no_change": PATIENCE,
}
# A stratified stopping split puts one row of each class on both sides.
STRATIFIED = 2


class Learner(models.Model):
    """Adapter whose estimators are scikit-learn classes.

    Configure it with keyword arguments, or subclass it and declare the same
    names as class attributes. An argument left out keeps the class-level
    value, so both styles work.

    Attributes:
        classifier: Estimator class used for classification tasks.
        regressor: Estimator class used for regression tasks.
        fixed: Constructor parameters applied to every build.
    """

    classifier: type | None = None
    regressor: type | None = None
    fixed: Mapping[str, Any] = NO_PARAMS

    def __init__(
        self,
        *,
        name: str | None = None,
        family: str | None = None,
        capabilities: models.Capabilities | None = None,
        classifier: type | None = None,
        regressor: type | None = None,
        fixed: Mapping[str, Any] | None = None,
        search_space: search.Space | None = None,
        monotonic_param: str | None = None,
    ) -> None:
        """Sets the declared data on this instance.

        Args:
            name: Unique registry name.
            family: Coarse family used for reporting.
            capabilities: Declared capabilities.
            classifier: Estimator class used for classification tasks.
            regressor: Estimator class used for regression tasks.
            fixed: Constructor parameters applied to every build.
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
            "fixed": fixed,
            "space": search_space,
            "param": monotonic_param,
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
            params: Search-space overrides applied over `fixed`.

        Returns:
            An unfitted estimator.

        Raises:
            ConfigError: If the adapter declares no class for the task.
        """
        estimator_class = (
            self.classifier if task.classification else self.regressor
        )
        if estimator_class is None:
            raise errors.ConfigError(
                f"{self.name}: no estimator class for {task.value}"
            )
        kwargs: dict[str, Any] = {**self.fixed, **(params or {})}
        if "random_state" in inspect.signature(estimator_class).parameters:
            kwargs["random_state"] = seed
        built: models.Estimator = estimator_class(**kwargs)
        return built


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
        sparse=True,
        missing=True,
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
            return scmle.models.survival.estimators.Marginal()
        if task is tasks.Task.UPLIFT:
            return scmle.models.uplift.estimators.Inert()
        if task in (
            tasks.Task.MULTILABEL,
            tasks.Task.MULTIOUTPUT,
            tasks.Task.FORECAST,
        ):
            classify = task is tasks.Task.MULTILABEL
            return scmle.models.multitarget.estimators.Panel(
                dummy.DummyClassifier(strategy="prior")
                if classify
                else dummy.DummyRegressor(strategy="mean"),
                classify,
            )
        if task is tasks.Task.RANKING:
            return scmle.models.ranking.estimators.Constant()
        if task is tasks.Task.ORDINAL:
            return scmle.models.ordinal.estimators.Threshold(
                dummy.DummyRegressor(strategy="median"),
                (params or {}).get("levels"),
                seed,
            )
        baseline: models.Estimator = (
            dummy.DummyClassifier(strategy="prior")
            if task.classification
            else dummy.DummyRegressor(strategy="mean")
        )
        return baseline


class Boosted(ensemble.HistGradientBoostingClassifier):  # type: ignore[misc]  # scikit-learn ships no type information, so its base class is Any
    """Boosted classifier whose early stopping follows the rarest class.

    scikit-learn draws the stopping rows stratified and refuses a class with
    a single training row, which a small fold or learning-curve slice of a
    real table can hold. Such a fit runs all `max_iter` rounds instead; the
    configured `early_stopping` is left as it was, and whether this fit
    was allowed to stop on held-out rows is recorded as `stopped_`.

    Attributes:
        stopped_: Whether the last fit was allowed to stop early.
    """

    stopped_: bool

    def fit(  # pyright: ignore[reportIncompatibleMethodOverride]  # the base names these parameters X and y; scikit-learn passes them positionally
        self,
        features: Any,
        labels: Any,
        sample_weight: Any = None,
        **params: Any,
    ) -> "Boosted":
        """Fits, stopping early only when every class has `STRATIFIED` rows.

        Args:
            features: Training features.
            labels: Training labels.
            sample_weight: Optional row weights.
            **params: Passed to scikit-learn's `fit`.

        Returns:
            This fitted classifier.
        """
        configured: Any = self.get_params()["early_stopping"]
        counts = numpy.unique(numpy.asarray(labels), return_counts=True)[1]
        rarest = int(numpy.min(counts))
        self.stopped_ = bool(configured) and rarest >= STRATIFIED
        if not self.stopped_:
            self.early_stopping = False
        try:
            super().fit(features, labels, sample_weight, **params)
        finally:
            self.early_stopping = configured
        return self


class Histogram(Learner):
    """Histogram-based gradient boosting, which takes native categoricals."""

    def nominals(self, indices: list[int]) -> dict[str, Any]:
        """Marks the integer-coded columns as categorical.

        Args:
            indices: Positions of the integer-coded categorical columns.

        Returns:
            Constructor parameters naming those columns, as
            `{"categorical_features": [...]}`.
        """
        return {"categorical_features": list(indices)}


class Svm(Learner):
    """Support vector machine with sigmoid-calibrated probabilities.

    scikit-learn deprecated `SVC(probability=True)` in 1.9; the supported
    equivalent is a cross-fitted `CalibratedClassifierCV(SVC(),
    ensemble=False)`, which is also considerably faster. `inner_cv.Calibrated`
    is that calibrator with its folds lowered for a rare class.
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
        support: models.Estimator
        if not task.classification:
            support = svm.SVR(**kwargs)
        else:
            support = inner_cv.Calibrated(
                svm.SVC(random_state=seed, **kwargs),
                method="sigmoid",
                ensemble=False,  # pyright: ignore[reportArgumentType]  # sklearn infers str from its "auto" default; it also accepts a bool
                cv=CALIBRATION_FOLDS,
            )
        return support


TREE_ENSEMBLE_SPACE = search.Space(
    (
        search.Integer("n_estimators", 50, 300),
        search.Integer("max_depth", 3, 25),
        search.Integer("min_samples_leaf", 1, 20, log=True),
        search.Choice("max_features", ("sqrt", 0.5, 1.0)),
    )
)

LOGISTIC_REGRESSION = Learner(
    name="logistic_regression",
    family="linear",
    capabilities=models.Capabilities(
        CLASSIFICATION, sparse=True, scaling=True, factor=2.0
    ),
    classifier=linear_model.LogisticRegression,
    fixed={"max_iter": 1000},
    search_space=search.Space((search.Real("C", 1e-3, 1e2, log=True),)),
)

RIDGE = Learner(
    name="ridge",
    family="linear",
    capabilities=models.Capabilities(
        REGRESSION_ONLY,
        sparse=True,
        scaling=True,
        factor=2.0,
    ),
    regressor=linear_model.Ridge,
    search_space=search.Space((search.Real("alpha", 1e-3, 1e3, log=True),)),
)

SGD_LOGISTIC = Learner(
    name="sgd_logistic",
    family="linear",
    capabilities=models.Capabilities(
        CLASSIFICATION,
        sparse=True,
        scaling=True,
        incremental=True,
        streaming=True,
        factor=1.5,
    ),
    classifier=linear_model.SGDClassifier,
    fixed={"loss": "log_loss", "max_iter": 50, "tol": 1e-4},
    search_space=search.Space((search.Real("alpha", 1e-7, 1e-1, log=True),)),
)

SGD_REGRESSION = Learner(
    name="sgd_regression",
    family="linear",
    capabilities=models.Capabilities(
        REGRESSION_ONLY,
        sparse=True,
        scaling=True,
        incremental=True,
        streaming=True,
        factor=1.5,
    ),
    regressor=linear_model.SGDRegressor,
    fixed={"max_iter": 50, "tol": 1e-4},
    search_space=search.Space((search.Real("alpha", 1e-7, 1e-1, log=True),)),
)

ELASTIC_NET = Learner(
    name="elastic_net",
    family="linear",
    capabilities=models.Capabilities(
        REGRESSION_ONLY,
        sparse=True,
        scaling=True,
        cost=2,
        factor=2.0,
    ),
    regressor=linear_model.ElasticNet,
    fixed={"alpha": 0.1, "max_iter": 5000},
    search_space=search.Space(
        (
            search.Real("alpha", 1e-3, 10.0, log=True),
            search.Real("l1_ratio", 0.05, 0.95),
        )
    ),
)

DECISION_TREE = Learner(
    name="decision_tree",
    family="tree",
    capabilities=models.Capabilities(ALL_TASKS, missing=True, factor=2.0),
    classifier=trees.Tree,
    regressor=tree.DecisionTreeRegressor,
    search_space=search.Space(
        (
            search.Integer("max_depth", 2, 20),
            search.Integer("min_samples_leaf", 1, 30, log=True),
        )
    ),
)

RANDOM_FOREST = Learner(
    name="random_forest",
    family="tree",
    capabilities=models.Capabilities(ALL_TASKS, cost=3, factor=6.0),
    classifier=trees.Forest,
    regressor=ensemble.RandomForestRegressor,
    search_space=TREE_ENSEMBLE_SPACE,
)

EXTRA_TREES = Learner(
    name="extra_trees",
    family="tree",
    capabilities=models.Capabilities(ALL_TASKS, cost=3, factor=6.0),
    classifier=trees.Extra,
    regressor=ensemble.ExtraTreesRegressor,
    search_space=TREE_ENSEMBLE_SPACE,
)

HIST_GRADIENT_BOOSTING = Histogram(
    name="hist_gradient_boosting",
    family="boosting",
    capabilities=models.Capabilities(
        ALL_TASKS,
        missing=True,
        categorical=True,
        early_stopping=True,
        cost=3,
        factor=3.0,
    ),
    classifier=Boosted,
    regressor=ensemble.HistGradientBoostingRegressor,
    fixed=STOPPING,
    monotonic_param="monotonic_cst",
    search_space=search.Space(
        (
            search.Real("learning_rate", 0.02, 0.3, log=True),
            search.Integer("max_leaf_nodes", 8, 64),
            search.Integer("min_samples_leaf", 5, 50),
            search.Real("l2_regularization", 1e-4, 1.0, log=True),
        )
    ),
)

SUPPORT_VECTOR = Svm(
    name="svm",
    family="kernel",
    capabilities=models.Capabilities(
        ALL_TASKS,
        sparse=True,
        scaling=True,
        scale=20_000,
        cost=4,
        factor=4.0,
    ),
    classifier=svm.SVC,
    regressor=svm.SVR,
    search_space=search.Space(
        (
            search.Real("C", 1e-2, 1e2, log=True),
            search.Real("gamma", 1e-4, 1.0, log=True),
        )
    ),
)

KNN = Learner(
    name="knn",
    family="neighbors",
    capabilities=models.Capabilities(
        ALL_TASKS,
        sparse=True,
        scaling=True,
        scale=50_000,
        cost=2,
        factor=3.0,
    ),
    classifier=neighbors.KNeighborsClassifier,
    regressor=neighbors.KNeighborsRegressor,
    search_space=search.Space(
        (
            search.Integer("n_neighbors", 3, 50),
            search.Choice("weights", ("uniform", "distance")),
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
        scaling=True,
        early_stopping=True,
        scale=200_000,
        cost=4,
        factor=4.0,
    ),
    classifier=neural_network.MLPClassifier,
    regressor=neural_network.MLPRegressor,
    fixed={
        "max_iter": 300,
        "early_stopping": True,
        "n_iter_no_change": 10,
    },
    search_space=search.Space(
        (
            search.Choice(
                "hidden_layer_sizes", ((32,), (64,), (64, 32), (128, 64))
            ),
            search.Real("alpha", 1e-5, 1e-1, log=True),
            search.Real("learning_rate_init", 1e-4, 1e-2, log=True),
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
