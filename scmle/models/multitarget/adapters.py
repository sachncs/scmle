"""Adapters for multi-label and multi-output tasks."""

from collections.abc import Mapping
from typing import Any

from sklearn import ensemble
from sklearn import linear_model

from scmle.contracts import models
from scmle.contracts import search
from scmle.contracts import tasks
from scmle.models.multitarget import estimators

MULTI = frozenset(
    {
        tasks.Task.MULTILABEL,
        tasks.Task.MULTIOUTPUT,
        tasks.Task.FORECAST,
    }
)
REGRESSION_LIKE = frozenset({tasks.Task.MULTIOUTPUT, tasks.Task.FORECAST})


class Multitarget(models.Model):
    """Base: one estimator class per kind of target.

    Attributes:
        classifier: Class fitted to each label (multi-label).
        regressor: Class fitted to each quantity (multi-output).
    """

    classifier: type | None = None
    regressor: type | None = None

    def __init__(
        self,
        *,
        name: str | None = None,
        family: str | None = None,
        capabilities: models.Capabilities | None = None,
        classifier: type | None = None,
        regressor: type | None = None,
        search_space: search.SearchSpace | None = None,
    ) -> None:
        """Sets the declared data on this instance.

        An argument left out keeps the class-level value, so a subclass that
        declares these names as class attributes keeps working.

        Args:
            name: Unique registry name.
            family: Coarse family used for reporting.
            capabilities: Declared capabilities.
            classifier: Class fitted to each label (multi-label).
            regressor: Class fitted to each quantity (multi-output).
            search_space: Tunable hyperparameters.
        """
        given: dict[str, Any] = {
            "name": name,
            "family": family,
            "capabilities": capabilities,
            "classifier": classifier,
            "regressor": regressor,
            "search_space": search_space,
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
        """Wraps the per-target estimator for the task.

        Args:
            task: Resolved task type; multi-label builds the classifier class,
                any other task the regressor class.
            seed: Random seed, passed as `random_state` when the class takes it.
            params: Constructor overrides for the per-target estimator.

        Returns:
            An unfitted estimator that fits one clone per target column.
        """
        classify = task is tasks.Task.MULTILABEL
        kind = self.classifier if classify else self.regressor
        assert kind is not None  # noqa: S101 -- capability guard, unreachable  # nosec B101
        settings = dict(params or {})
        try:
            inner = kind(random_state=seed, **settings)
        except TypeError:
            inner = kind(**settings)
        return estimators.PerTarget(inner, classify=classify)


class MultilabelLogistic(Multitarget):
    """Logistic regression for every label."""

    name = "multilabel_logistic"
    family = "linear"
    capabilities = models.Capabilities(
        frozenset({tasks.Task.MULTILABEL}), needs_scaling=True, cost=2
    )
    search_space = search.SearchSpace(
        (search.FloatParameter("C", 1e-3, 100.0, log=True),)
    )

    def build(
        self,
        task: tasks.Task,  # noqa: ARG002 -- required by Model.build
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """One regularised logistic regression per label.

        Args:
            task: Resolved task type.
            seed: Random seed.
            params: Search-space overrides.

        Returns:
            An unfitted estimator with one logistic regression per label.
        """
        return estimators.PerTarget(
            linear_model.LogisticRegression(
                max_iter=1000, random_state=seed, **dict(params or {})
            ),
            classify=True,
        )


class MultioutputRidge(Multitarget):
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
        task: tasks.Task,  # noqa: ARG002 -- required by Model.build
        seed: int,  # noqa: ARG002 -- required by Model.build
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """One ridge regression per quantity.

        Args:
            task: Resolved task type.
            seed: Random seed.
            params: Search-space overrides.

        Returns:
            An unfitted estimator with one ridge regression per quantity.
        """
        return estimators.PerTarget(
            linear_model.Ridge(**dict(params or {})), classify=False
        )


MULTI_BOOSTING = Multitarget(
    name="multi_hist_gradient_boosting",
    family="boosting",
    capabilities=models.Capabilities(
        MULTI, handles_missing=True, cost=3, memory_factor=3.0
    ),
    classifier=ensemble.HistGradientBoostingClassifier,
    regressor=ensemble.HistGradientBoostingRegressor,
    search_space=search.SearchSpace(
        (
            search.FloatParameter("learning_rate", 0.02, 0.3, log=True),
            search.IntParameter("max_leaf_nodes", 8, 64),
        )
    ),
)


class MultiForest(models.Model):
    """One random forest over all targets jointly."""

    name = "multi_random_forest"
    family = "tree"
    capabilities = models.Capabilities(MULTI, cost=3, memory_factor=6.0)

    def build(
        self,
        task: tasks.Task,
        seed: int,
        params: Mapping[str, Any] | None = None,  # noqa: ARG002 -- required by Model.build
    ) -> models.Estimator:
        """The forest for the task's kind of target.

        Args:
            task: Resolved task type; multi-label builds a classifier forest,
                any other task a regressor forest.
            seed: Random seed.
            params: Ignored.

        Returns:
            An unfitted forest that models all targets jointly.
        """
        return estimators.forest(task is tasks.Task.MULTILABEL, seed)


MULTITARGET_ADAPTERS: tuple[models.Model, ...] = (
    MultilabelLogistic(),
    MultioutputRidge(),
    MULTI_BOOSTING,
    MultiForest(),
)
"""Multi-target adapter instances, in registration order."""
