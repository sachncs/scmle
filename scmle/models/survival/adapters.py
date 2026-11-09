"""Survival model adapters."""

from collections.abc import Mapping
from typing import Any

from scmle.contracts import models
from scmle.contracts import search
from scmle.contracts import tasks
from scmle.models.survival import estimators

SURVIVAL = frozenset({tasks.Task.SURVIVAL})


class Cox(models.Model):
    """Penalised Cox proportional hazards."""

    name = "cox_ph"
    family = "linear"
    capabilities = models.Capabilities(SURVIVAL, needs_scaling=True, cost=2)
    search_space = search.SearchSpace(
        (search.FloatParameter("penalty", 1e-3, 100.0, log=True),)
    )

    def build(
        self,
        task: tasks.Task,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Returns an unfitted Cox model.

        Args:
            task: Resolved task type.
            seed: Random seed.
            params: Search-space overrides passed to the model constructor.

        Returns:
            An unfitted penalised Cox regressor; `seed` is not used.
        """
        return estimators.CoxRegressor(**dict(params or {}))


class DiscreteTime(models.Model):
    """Boosted trees on a discrete-time hazard."""

    name = "discrete_time_boosting"
    family = "boosting"
    capabilities = models.Capabilities(
        SURVIVAL, handles_missing=False, cost=4, memory_factor=4.0
    )
    search_space = search.SearchSpace(
        (
            search.FloatParameter("learning_rate", 0.02, 0.3, log=True),
            search.IntParameter("max_leaf_nodes", 4, 32),
        )
    )

    def build(
        self,
        task: tasks.Task,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Returns an unfitted discrete-time boosted hazard model.

        Args:
            task: Resolved task type.
            seed: Random seed.
            params: Search-space overrides passed to the model constructor.

        Returns:
            An unfitted discrete-time boosting hazard model.
        """
        return estimators.DiscreteTimeBoosting(
            random_state=seed, **dict(params or {})
        )


SURVIVAL_ADAPTERS: tuple[type[models.Model], ...] = (
    Cox,
    DiscreteTime,
)
