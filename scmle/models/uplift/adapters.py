"""Uplift model adapters."""

from collections.abc import Mapping
from typing import Any

from scmle.contracts import models
from scmle.contracts import tasks
from scmle.models.uplift import estimators

UPLIFT = frozenset({tasks.Task.UPLIFT})


class Uplift(models.Model):
    """A meta-learner built from boosted outcome models.

    Configure it with keyword arguments, or subclass it and declare the same
    names as class attributes. An argument left out keeps the class-level
    value, so both styles work.

    Attributes:
        learner: The estimator class.
    """

    learner: type = estimators.TwoModelLearner
    family = "causal"

    def __init__(
        self,
        *,
        name: str | None = None,
        family: str | None = None,
        capabilities: models.Capabilities | None = None,
        learner: type | None = None,
    ) -> None:
        """Sets the declared data on this instance.

        Args:
            name: Unique registry name.
            family: Coarse family used for reporting.
            capabilities: Declared capabilities.
            learner: Estimator class the adapter builds.
        """
        given: dict[str, Any] = {
            "name": name,
            "family": family,
            "capabilities": capabilities,
            "learner": learner,
        }
        for attribute, value in given.items():
            if value is not None:
                setattr(self, attribute, value)

    def build(
        self,
        task: tasks.Task,  # noqa: ARG002 -- required by Model.build
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Builds the learner; the outcome kind is set by the wrapper.

        Args:
            task: Resolved task type.
            seed: Random seed of the outcome models.
            params: Only the `classify` entry is read (default True, a 0/1
                outcome); other entries are ignored.

        Returns:
            An unfitted uplift learner of the adapter's `learner` class.
        """
        settings = dict(params or {})
        return self.learner(
            classify=settings.pop("classify", True), random_state=seed
        )


T_LEARNER = Uplift(
    name="uplift_t_learner",
    capabilities=models.Capabilities(
        UPLIFT, handles_missing=True, cost=3, memory_factor=4.0
    ),
    learner=estimators.TwoModelLearner,
)

S_LEARNER = Uplift(
    name="uplift_s_learner",
    capabilities=models.Capabilities(
        UPLIFT, handles_missing=True, cost=3, memory_factor=3.0
    ),
    learner=estimators.SingleModelLearner,
)

X_LEARNER = Uplift(
    name="uplift_x_learner",
    capabilities=models.Capabilities(
        UPLIFT, handles_missing=True, cost=4, memory_factor=5.0
    ),
    learner=estimators.CrossLearner,
)

UPLIFT_ADAPTERS: tuple[models.Model, ...] = (
    T_LEARNER,
    S_LEARNER,
    X_LEARNER,
)
