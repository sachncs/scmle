"""The strategy interface, its thresholds and the default strategy."""

import abc
from collections.abc import Mapping
from typing import Any
from typing import ClassVar

from scmle.contracts import data
from scmle.contracts import experiments
from scmle.contracts import models
from scmle.contracts import tasks
from scmle.preparation import preprocessing

MIN_IMBALANCE_RATIO = 3.0


TARGET_SKEW_THRESHOLD = 1.0


SKEW_THRESHOLD = 1.0


MIN_INTERACTION_FEATURES = 2


MAX_INTERACTION_FEATURES = 12


MIN_SELECTION_FEATURES = 6


MIN_MISSING_RATE = 0.01


MAX_ITERATIVE_FEATURES = 50


CALIBRATION_FOLDS = 3


class Strategy(abc.ABC):
    """One way of preparing and wrapping a candidate pipeline.

    Attributes:
        name: Registry name recorded on trials.
        description: Why one would try it.
    """

    name: ClassVar[str]
    description: ClassVar[str]

    def applicability(
        self,
        profile: data.Portrait,  # noqa: ARG002 -- subclasses override with it
        task: tasks.Task,  # noqa: ARG002 -- subclasses override with it
        adapter: models.Model,
    ) -> str | None:
        """Explains why the strategy does not apply, if it does not.

        Args:
            profile: Dataset profile.
            task: Resolved task type.
            adapter: Model the strategy would be combined with.

        Returns:
            A reason, or None when the strategy is applicable.
        """
        if adapter.family == "baseline":
            return "the baseline model is never transformed"
        return None

    @abc.abstractmethod
    def build(
        self,
        plan: data.Blueprint,
        adapter: models.Model,
        task: tasks.Task,
        seed: int,
        params: Mapping[str, Any] | None = None,
        featurise: preprocessing.Featuriser | None = None,
    ) -> models.Estimator:
        """Builds the complete, unfitted candidate.

        Args:
            plan: Feature columns by kind.
            adapter: Model family.
            task: Resolved task type.
            seed: Random seed.
            params: Hyperparameter overrides.
            featurise: Builds the featuriser of the plan: the data kind's
                (`kinds.Kind.featurise`).

        Returns:
            The candidate estimator (preprocessing included).
        """


class Default(Strategy):
    """Standard preprocessing; the reference every other strategy must beat."""

    name = experiments.DEFAULT_STRATEGY
    description = "standard preprocessing"

    def applicability(
        self,
        profile: data.Portrait,  # noqa: ARG002 -- Strategy interface
        task: tasks.Task,  # noqa: ARG002 -- Strategy interface
        adapter: models.Model,  # noqa: ARG002 -- Strategy interface
    ) -> str | None:
        """Always applicable.

        Args:
            profile: Dataset profile.
            task: Resolved task type.
            adapter: Model the strategy would be combined with.

        Returns:
            A reason the strategy does not apply, or None when it applies.
        """
        return None

    def build(
        self,
        plan: data.Blueprint,
        adapter: models.Model,
        task: tasks.Task,
        seed: int,
        params: Mapping[str, Any] | None = None,
        featurise: preprocessing.Featuriser | None = None,
    ) -> models.Estimator:
        """Builds the standard pipeline.

        Args:
            plan: Feature columns by kind.
            adapter: Model family.
            task: Resolved task type.
            seed: Random seed.
            params: Hyperparameter overrides.
            featurise: Builds the featuriser of the plan: the data kind's
                (`kinds.Kind.featurise`).

        Returns:
            The unfitted candidate estimator, preprocessing included.
        """
        pipeline: models.Estimator = preprocessing.instantiate(
            plan, adapter, task, seed, params, featurise=featurise
        )
        return pipeline
