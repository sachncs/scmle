"""Survival model adapters."""

from collections.abc import Mapping
from typing import Any

from mml.contracts import models
from mml.contracts import search
from mml.contracts import tasks
from mml.models.survival import estimators

SURVIVAL = frozenset({tasks.TaskType.SURVIVAL})


class CoxAdapter(models.ModelAdapter):
    """Penalised Cox proportional hazards."""

    name = "cox_ph"
    family = "linear"
    capabilities = models.Capabilities(SURVIVAL, needs_scaling=True, cost=2)
    search_space = search.SearchSpace(
        (search.FloatParameter("penalty", 1e-3, 100.0, log=True),)
    )

    def build(
        self,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Returns an unfitted Cox model."""
        return estimators.CoxRegressor(**dict(params or {}))


class DiscreteTimeAdapter(models.ModelAdapter):
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
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Returns an unfitted discrete-time boosted hazard model."""
        return estimators.DiscreteTimeBoosting(
            random_state=seed, **dict(params or {})
        )


SURVIVAL_ADAPTERS: tuple[type[models.ModelAdapter], ...] = (
    CoxAdapter,
    DiscreteTimeAdapter,
)
