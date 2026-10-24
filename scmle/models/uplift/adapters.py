"""Uplift model adapters."""

from collections.abc import Mapping
from typing import Any

from scmle.contracts import models
from scmle.contracts import tasks
from scmle.models.uplift import estimators

UPLIFT = frozenset({tasks.TaskType.UPLIFT})


class UpliftAdapter(models.ModelAdapter):
    """Base: a meta-learner built from boosted outcome models.

    Attributes:
        learner: The estimator class.
    """

    learner: type = estimators.TwoModelLearner
    family = "causal"

    def build(
        self,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Builds the learner; the outcome kind is set by the wrapper."""
        settings = dict(params or {})
        return self.learner(
            classify=settings.pop("classify", True), random_state=seed
        )


class TwoModelAdapter(UpliftAdapter):
    """T-learner."""

    name = "uplift_t_learner"
    capabilities = models.Capabilities(
        UPLIFT, handles_missing=True, cost=3, memory_factor=4.0
    )
    learner = estimators.TwoModelLearner


class SingleModelAdapter(UpliftAdapter):
    """S-learner."""

    name = "uplift_s_learner"
    capabilities = models.Capabilities(
        UPLIFT, handles_missing=True, cost=3, memory_factor=3.0
    )
    learner = estimators.SingleModelLearner


class CrossAdapter(UpliftAdapter):
    """X-learner."""

    name = "uplift_x_learner"
    capabilities = models.Capabilities(
        UPLIFT, handles_missing=True, cost=4, memory_factor=5.0
    )
    learner = estimators.CrossLearner


UPLIFT_ADAPTERS: tuple[type[models.ModelAdapter], ...] = (
    TwoModelAdapter,
    SingleModelAdapter,
    CrossAdapter,
)
