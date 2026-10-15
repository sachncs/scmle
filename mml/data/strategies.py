"""Data strategies: end-to-end variants of how a candidate is prepared.

A strategy decides *whether it applies* (with a reason when it does not) and
*how to build* the complete candidate pipeline. The search evaluates each
applicable strategy as its own independent, comparable candidate on the
original validation distribution, so a technique is adopted only on evidence.
"""

import abc
from collections.abc import Mapping
from typing import Any
from typing import ClassVar

from mml.contracts import data as data_contracts
from mml.contracts import errors
from mml.contracts import experiments
from mml.contracts import models
from mml.contracts import tasks
from mml.data import preprocessing
from mml.models import resampling

MIN_IMBALANCE_RATIO = 3.0
SKEW_THRESHOLD = 1.0
MIN_INTERACTION_FEATURES = 2
MAX_INTERACTION_FEATURES = 12
DEFAULT = experiments.DEFAULT_STRATEGY


class DataStrategy(abc.ABC):
    """One way of preparing and wrapping a candidate pipeline.

    Attributes:
        name: Registry name recorded on trials.
        description: Why one would try it.
    """

    name: ClassVar[str]
    description: ClassVar[str]

    def applicability(
        self,
        profile: data_contracts.DatasetProfile,
        task: tasks.TaskType,
        adapter: models.ModelAdapter,
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
        plan: data_contracts.FeaturePlan,
        adapter: models.ModelAdapter,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Builds the complete, unfitted candidate.

        Args:
            plan: Feature columns by kind.
            adapter: Model family.
            task: Resolved task type.
            seed: Random seed.
            params: Hyperparameter overrides.

        Returns:
            The candidate estimator (preprocessing included).
        """


class DefaultStrategy(DataStrategy):
    """Standard preprocessing; the reference every other strategy must beat."""

    name = DEFAULT
    description = "standard preprocessing"

    def applicability(
        self,
        profile: data_contracts.DatasetProfile,
        task: tasks.TaskType,
        adapter: models.ModelAdapter,
    ) -> str | None:
        """Always applicable."""
        return None

    def build(
        self,
        plan: data_contracts.FeaturePlan,
        adapter: models.ModelAdapter,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Builds the standard pipeline."""
        return preprocessing.build_pipeline(plan, adapter, task, seed, params)


class ImbalanceStrategy(DataStrategy):
    """Base for strategies that address class imbalance."""

    def applicability(
        self,
        profile: data_contracts.DatasetProfile,
        task: tasks.TaskType,
        adapter: models.ModelAdapter,
    ) -> str | None:
        """Requires a classification task with noticeably skewed classes."""
        reason = super().applicability(profile, task, adapter)
        if reason is not None:
            return reason
        if not task.is_classification:
            return "rebalancing is a classification technique"
        ratio = profile.target_summary.get("imbalance_ratio", 1.0)
        if ratio < MIN_IMBALANCE_RATIO:
            return (
                f"classes are balanced enough ({ratio:.1f}:1 < "
                f"{MIN_IMBALANCE_RATIO:.0f}:1)"
            )
        return None


class ClassWeightStrategy(ImbalanceStrategy):
    """Re-weights classes inversely to their frequency."""

    name = "class_weight"
    description = "balanced class weights"

    def applicability(
        self,
        profile: data_contracts.DatasetProfile,
        task: tasks.TaskType,
        adapter: models.ModelAdapter,
    ) -> str | None:
        """Also requires an estimator with a `class_weight` parameter."""
        reason = super().applicability(profile, task, adapter)
        if reason is not None:
            return reason
        if "class_weight" not in adapter.build(task, 0).get_params():
            return f"{adapter.name} has no class_weight parameter"
        return None

    def build(
        self,
        plan: data_contracts.FeaturePlan,
        adapter: models.ModelAdapter,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Builds the standard pipeline with `class_weight='balanced'`."""
        weighted = {**(params or {}), "class_weight": "balanced"}
        return preprocessing.build_pipeline(plan, adapter, task, seed, weighted)


class ResamplingStrategy(ImbalanceStrategy):
    """Base for strategies that resample each training fold.

    Attributes:
        resampler_class: Resampler applied inside `fit`.
    """

    resampler_class: ClassVar[type[resampling.Resampler]]

    def build(
        self,
        plan: data_contracts.FeaturePlan,
        adapter: models.ModelAdapter,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Wraps the standard pipeline so it trains on rebalanced data."""
        inner = preprocessing.build_pipeline(plan, adapter, task, seed, params)
        return resampling.ResamplingClassifier(
            inner, self.resampler_class(), seed
        )


class OversampleStrategy(ResamplingStrategy):
    """Random oversampling of minority classes inside each training fold."""

    name = "random_oversample"
    description = "random oversampling inside the training fold"
    resampler_class = resampling.RandomOverSampler


class UndersampleStrategy(ResamplingStrategy):
    """Random undersampling of majority classes inside each training fold."""

    name = "random_undersample"
    description = "random undersampling inside the training fold"
    resampler_class = resampling.RandomUnderSampler


class FeatureStrategy(DataStrategy):
    """Base for strategies that change the numeric preprocessing.

    Attributes:
        log_skewed: Log-transform right-skewed numeric features.
        interactions: Add pairwise numeric interactions.
    """

    log_skewed: ClassVar[bool] = False
    interactions: ClassVar[bool] = False

    def build(
        self,
        plan: data_contracts.FeaturePlan,
        adapter: models.ModelAdapter,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Builds the pipeline with the strategy's numeric steps."""
        options = data_contracts.PreprocessingOptions(
            scale=adapter.capabilities.needs_scaling,
            log_skewed=self.log_skewed,
            interactions=self.interactions,
        )
        return preprocessing.build_pipeline(
            plan, adapter, task, seed, params, options
        )


class LogSkewedStrategy(FeatureStrategy):
    """Log-transforms right-skewed, non-negative numeric features."""

    name = "log_skewed"
    description = "log1p of right-skewed non-negative numeric features"
    log_skewed = True

    def applicability(
        self,
        profile: data_contracts.DatasetProfile,
        task: tasks.TaskType,
        adapter: models.ModelAdapter,
    ) -> str | None:
        """Needs a skewed feature and a model sensitive to monotone shifts."""
        reason = super().applicability(profile, task, adapter)
        if reason is not None:
            return reason
        if adapter.family in ("tree", "boosting"):
            return "monotone transforms do not change tree splits"
        skewed = [
            name
            for name in profile.feature_plan.numeric
            if profile.column_statistics[name].get("skew", 0.0) > SKEW_THRESHOLD
            and profile.column_statistics[name].get("min", -1.0) >= 0
        ]
        if not skewed:
            return "no right-skewed non-negative numeric feature"
        return None


class InteractionStrategy(FeatureStrategy):
    """Adds pairwise numeric interactions for linear models."""

    name = "interactions"
    description = "pairwise numeric interaction features"
    interactions = True

    def applicability(
        self,
        profile: data_contracts.DatasetProfile,
        task: tasks.TaskType,
        adapter: models.ModelAdapter,
    ) -> str | None:
        """Linear models with a moderate number of numeric features."""
        reason = super().applicability(profile, task, adapter)
        if reason is not None:
            return reason
        if adapter.family != "linear":
            return "non-linear models learn interactions themselves"
        count = len(profile.feature_plan.numeric)
        if not MIN_INTERACTION_FEATURES <= count <= MAX_INTERACTION_FEATURES:
            return f"{count} numeric features is outside [2, 12]"
        return None


STRATEGIES: dict[str, DataStrategy] = {
    strategy.name: strategy
    for strategy in (
        DefaultStrategy(),
        ClassWeightStrategy(),
        OversampleStrategy(),
        UndersampleStrategy(),
        LogSkewedStrategy(),
        InteractionStrategy(),
    )
}
EXPERIMENTAL = tuple(name for name in STRATEGIES if name != DEFAULT)


def get_strategy(name: str) -> DataStrategy:
    """Returns the registered strategy.

    Raises:
        ConfigError: If the name is unknown.
    """
    if name not in STRATEGIES:
        raise errors.ConfigError(
            f"unknown strategy '{name}'; available: {sorted(STRATEGIES)}"
        )
    return STRATEGIES[name]
