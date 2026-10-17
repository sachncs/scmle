"""Adapters for multi-label and multi-output tasks."""

from collections.abc import Mapping
from typing import Any

from sklearn import ensemble
from sklearn import linear_model

from mml.contracts import models
from mml.contracts import search
from mml.contracts import tasks
from mml.models.multitarget import estimators

MULTI = frozenset(
    {
        tasks.TaskType.MULTILABEL,
        tasks.TaskType.MULTIOUTPUT,
        tasks.TaskType.FORECAST,
    }
)
REGRESSION_LIKE = frozenset(
    {tasks.TaskType.MULTIOUTPUT, tasks.TaskType.FORECAST}
)


class MultiTargetAdapter(models.ModelAdapter):
    """Base: one estimator class per kind of target.

    Attributes:
        classifier: Class fitted to each label (multi-label).
        regressor: Class fitted to each quantity (multi-output).
    """

    classifier: type | None = None
    regressor: type | None = None

    def build(
        self,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Wraps the per-target estimator for the task."""
        classify = task is tasks.TaskType.MULTILABEL
        kind = self.classifier if classify else self.regressor
        assert kind is not None  # nosec B101 - capability guard
        settings = dict(params or {})
        try:
            inner = kind(random_state=seed, **settings)
        except TypeError:
            inner = kind(**settings)
        return estimators.PerTarget(inner, classify)


class MultilabelLogisticAdapter(MultiTargetAdapter):
    """Logistic regression for every label."""

    name = "multilabel_logistic"
    family = "linear"
    capabilities = models.Capabilities(
        frozenset({tasks.TaskType.MULTILABEL}), needs_scaling=True, cost=2
    )
    search_space = search.SearchSpace(
        (search.FloatParameter("C", 1e-3, 100.0, log=True),)
    )

    def build(
        self,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """One regularised logistic regression per label."""
        return estimators.PerTarget(
            linear_model.LogisticRegression(
                max_iter=1000, random_state=seed, **dict(params or {})
            ),
            True,
        )


class MultioutputRidgeAdapter(MultiTargetAdapter):
    """Ridge regression for every quantity."""

    name = "multioutput_ridge"
    family = "linear"
    capabilities = models.Capabilities(
        REGRESSION_LIKE, needs_scaling=True, cost=1
    )
    search_space = search.SearchSpace(
        (search.FloatParameter("alpha", 1e-3, 100.0, log=True),)
    )

    def build(
        self,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """One ridge regression per quantity."""
        return estimators.PerTarget(
            linear_model.Ridge(**dict(params or {})), False
        )


class MultiBoostingAdapter(MultiTargetAdapter):
    """Histogram gradient boosting for every target."""

    name = "multi_hist_gradient_boosting"
    family = "boosting"
    capabilities = models.Capabilities(
        MULTI, handles_missing=True, cost=3, memory_factor=3.0
    )
    classifier = ensemble.HistGradientBoostingClassifier
    regressor = ensemble.HistGradientBoostingRegressor
    search_space = search.SearchSpace(
        (
            search.FloatParameter("learning_rate", 0.02, 0.3, log=True),
            search.IntParameter("max_leaf_nodes", 8, 64),
        )
    )


class MultiForestAdapter(models.ModelAdapter):
    """One random forest over all targets jointly."""

    name = "multi_random_forest"
    family = "tree"
    capabilities = models.Capabilities(MULTI, cost=3, memory_factor=6.0)

    def build(
        self,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """The forest for the task's kind of target."""
        return estimators.forest(task is tasks.TaskType.MULTILABEL, seed)


MULTITARGET_ADAPTERS: tuple[type[models.ModelAdapter], ...] = (
    MultilabelLogisticAdapter,
    MultioutputRidgeAdapter,
    MultiBoostingAdapter,
    MultiForestAdapter,
)
