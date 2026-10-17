"""Data strategies: end-to-end variants of how a candidate is prepared.

A strategy decides *whether it applies* (with a reason when it does not) and
*how to build* the complete candidate pipeline. The search evaluates each
applicable strategy as its own independent, comparable candidate on the
original validation distribution, so a technique is adopted only on evidence.
"""

import abc
from collections.abc import Mapping
import dataclasses
import importlib
import importlib.util
from typing import Any
from typing import ClassVar

from sklearn import calibration
from sklearn import compose
from sklearn.utils import validation

from mml.contracts import data as data_contracts
from mml.contracts import errors
from mml.contracts import experiments
from mml.contracts import models
from mml.contracts import tasks
from mml.data.curation import preprocessing
from mml.data.curation import transformers
from mml.data.splitting import inner_cv
from mml.sampling import resampling
from mml.sampling import synthesis

MIN_IMBALANCE_RATIO = 3.0
TARGET_SKEW_THRESHOLD = 1.0
SKEW_THRESHOLD = 1.0
MIN_INTERACTION_FEATURES = 2
MAX_INTERACTION_FEATURES = 12
MIN_SELECTION_FEATURES = 6
MIN_MISSING_RATE = 0.01
MAX_ITERATIVE_FEATURES = 50
DEFAULT = experiments.DEFAULT_STRATEGY
CALIBRATION_FOLDS = 3


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


class CalibrationStrategy(DataStrategy):
    """Recalibrates predicted probabilities (sigmoid, fitted inside folds)."""

    name = "calibrated"
    description = "sigmoid probability calibration inside the training fold"

    def applicability(
        self,
        profile: data_contracts.DatasetProfile,
        task: tasks.TaskType,
        adapter: models.ModelAdapter,
    ) -> str | None:
        """Needs a probabilistic classifier worth calibrating."""
        reason = super().applicability(profile, task, adapter)
        if reason is not None:
            return reason
        if not task.is_classification:
            return "calibration applies to classification"
        if not adapter.capabilities.probabilities:
            return f"{adapter.name} does not provide probabilities"
        if adapter.name == "logistic_regression":
            return "logistic regression is already well calibrated"
        return None

    def build(
        self,
        plan: data_contracts.FeaturePlan,
        adapter: models.ModelAdapter,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Wraps the standard pipeline in a cross-fitted calibrator."""
        inner = preprocessing.build_pipeline(plan, adapter, task, seed, params)
        return calibration.CalibratedClassifierCV(
            inner,
            method="sigmoid",
            cv=inner_cv.for_plan(plan, True, CALIBRATION_FOLDS, seed),
        )


class JitterStrategy(ResamplingStrategy):
    """Experimental augmentation: jittered minority copies in each fold.

    Not part of the default strategy list; request it explicitly with
    `StrategySpec(names=["jitter_augment"])`.
    """

    name = "jitter_augment"
    description = "EXPERIMENTAL jittered minority augmentation inside the fold"
    resampler_class = resampling.JitterOverSampler


class CopulaAugmentStrategy(ResamplingStrategy):
    """Experimental augmentation: Gaussian-copula synthetic minority rows.

    Opt-in only; adopted only when it beats the un-augmented baseline on the
    original validation folds.
    """

    name = "copula_augment"
    description = "EXPERIMENTAL Gaussian-copula synthetic minority rows"
    resampler_class = synthesis.CopulaOverSampler


class SmoteStrategy(ImbalanceStrategy):
    """SMOTE-family oversampling on the encoded training fold.

    Attributes:
        sampler_name: Class name in `imblearn.over_sampling`.
    """

    sampler_name: ClassVar[str] = "SMOTE"
    min_features: ClassVar[int] = 2

    def applicability(
        self,
        profile: data_contracts.DatasetProfile,
        task: tasks.TaskType,
        adapter: models.ModelAdapter,
    ) -> str | None:
        """Needs imbalanced-learn and a purely numeric feature space."""
        reason = super().applicability(profile, task, adapter)
        if reason is not None:
            return reason
        if importlib.util.find_spec("imblearn") is None:
            return "needs the 'sampling' extra (imbalanced-learn)"
        if profile.feature_plan.categorical:
            return "interpolating one-hot categories is not meaningful"
        if len(profile.feature_plan.numeric) < self.min_features:
            return "needs at least two numeric features"
        return None

    def build(
        self,
        plan: data_contracts.FeaturePlan,
        adapter: models.ModelAdapter,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Builds preprocess -> resample -> model with the SMOTE variant."""
        over_sampling = importlib.import_module("imblearn.over_sampling")
        sampler = getattr(over_sampling, self.sampler_name)(random_state=seed)
        options = data_contracts.PreprocessingOptions(
            scale=adapter.capabilities.needs_scaling
        )
        return resampling.PreprocessedResamplingClassifier(
            preprocessing.build_front_end(plan, options),
            sampler,
            adapter.build(task, seed, params),
        )


class BorderlineSmoteStrategy(SmoteStrategy):
    """Borderline-SMOTE: synthesizes only near the decision boundary."""

    name = "borderline_smote"
    description = "Borderline-SMOTE oversampling inside the training fold"
    sampler_name = "BorderlineSMOTE"


class PlainSmoteStrategy(SmoteStrategy):
    """Classic SMOTE."""

    name = "smote"
    description = "SMOTE oversampling inside the training fold"
    sampler_name = "SMOTE"


class AdasynStrategy(SmoteStrategy):
    """ADASYN: more synthetic rows where minority density is low."""

    name = "adasyn"
    description = "ADASYN oversampling inside the training fold"
    sampler_name = "ADASYN"


class FeatureStrategy(DataStrategy):
    """Base for strategies that change the numeric preprocessing.

    Attributes:
        log_skewed: Log-transform right-skewed numeric features.
        interactions: Add pairwise numeric interactions.
        construct: Append differences and ratios of the best numeric pairs.
        select: Keep only columns an L1-penalised linear model uses.
        missing: Missing-value strategy for numeric features.
    """

    log_skewed: ClassVar[bool] = False
    interactions: ClassVar[bool] = False
    construct: ClassVar[bool] = False
    select: ClassVar[bool] = False
    missing: ClassVar[str] = "median"

    def build(
        self,
        plan: data_contracts.FeaturePlan,
        adapter: models.ModelAdapter,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Builds the pipeline with the strategy's numeric steps."""
        options = dataclasses.replace(
            preprocessing.default_options(plan, adapter),
            log_skewed=self.log_skewed,
            interactions=self.interactions,
            construct=self.construct,
            select=self.select,
            missing=self.missing,
        )
        if self.select:
            options = dataclasses.replace(options, scale=True)
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
        stats = profile.column_statistics
        skewed = [
            name
            for name in profile.feature_plan.numeric
            if name in stats
            and stats[name].get("skew", 0.0) > SKEW_THRESHOLD
            and stats[name].get("min", -1.0) >= 0
        ]
        if not skewed:
            return "no right-skewed non-negative numeric feature"
        return None


class RareTargetStrategy(DataStrategy):
    """Regression policy for a long-tailed target: inverse-density weights.

    Classification resampling does not transfer to a continuous target, so
    rare target values are handled by weighting rows inversely to how densely
    populated their part of the target range is. It is adopted only if it
    helps on the original (unweighted) validation distribution.
    """

    name = "rare_target_weighting"
    description = "inverse-density sample weights for a long-tailed target"

    def applicability(
        self,
        profile: data_contracts.DatasetProfile,
        task: tasks.TaskType,
        adapter: models.ModelAdapter,
    ) -> str | None:
        """Needs a skewed regression target and a model taking weights."""
        reason = super().applicability(profile, task, adapter)
        if reason is not None:
            return reason
        if task.is_classification:
            return "regression only (classification uses class weighting)"
        skew = abs(profile.target_summary.get("skew", 0.0))
        if skew < TARGET_SKEW_THRESHOLD:
            return f"target is not long-tailed (|skew| {skew:.2f})"
        if not validation.has_fit_parameter(
            adapter.build(task, 0), "sample_weight"
        ):
            return f"{adapter.name} does not accept sample weights"
        return None

    def build(
        self,
        plan: data_contracts.FeaturePlan,
        adapter: models.ModelAdapter,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Wraps the standard pipeline in the density-weighted regressor."""
        inner = preprocessing.build_pipeline(plan, adapter, task, seed, params)
        return resampling.DensityWeightedRegressor(inner)


class TargetEncodingStrategy(DataStrategy):
    """Regularized, cross-fitted target encoding of high-cardinality columns.

    Replaces one-hot encoding of categorical features with many levels by a
    smoothed target encoding that is fitted inside each training fold, with
    internal cross-fitting so no row sees its own label.
    """

    name = "target_encoding"
    description = "regularized target encoding of high-cardinality categories"

    def applicability(
        self,
        profile: data_contracts.DatasetProfile,
        task: tasks.TaskType,
        adapter: models.ModelAdapter,
    ) -> str | None:
        """Needs at least one high-cardinality categorical feature."""
        reason = super().applicability(profile, task, adapter)
        if reason is not None:
            return reason
        if not profile.feature_plan.high_cardinality:
            return "no high-cardinality categorical feature"
        return None

    def build(
        self,
        plan: data_contracts.FeaturePlan,
        adapter: models.ModelAdapter,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Builds the pipeline with target-encoded high-cardinality columns."""
        options = dataclasses.replace(
            preprocessing.default_options(plan, adapter),
            high_cardinality="target",
            sparse=False,
        )
        return preprocessing.build_pipeline(
            plan, adapter, task, seed, params, options
        )


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


class ConstructFeaturesStrategy(FeatureStrategy):
    """Adds differences and ratios of the numeric features that matter most."""

    name = "construct_features"
    description = "differences and ratios of the most target-related features"
    construct = True

    def applicability(
        self,
        profile: data_contracts.DatasetProfile,
        task: tasks.TaskType,
        adapter: models.ModelAdapter,
    ) -> str | None:
        """Needs at least two numeric features to combine."""
        reason = super().applicability(profile, task, adapter)
        if reason is not None:
            return reason
        if len(profile.feature_plan.numeric) < MIN_INTERACTION_FEATURES:
            return "fewer than two numeric features to combine"
        return None


class SelectFeaturesStrategy(FeatureStrategy):
    """Drops columns an L1-penalised linear model gives no weight."""

    name = "select_features"
    description = "keep only the features a sparse linear model uses"
    select = True

    def applicability(
        self,
        profile: data_contracts.DatasetProfile,
        task: tasks.TaskType,
        adapter: models.ModelAdapter,
    ) -> str | None:
        """Needs enough features for removal to be meaningful."""
        reason = super().applicability(profile, task, adapter)
        if reason is not None:
            return reason
        plan = profile.feature_plan
        if plan.monotonic:
            return "feature selection would shift the monotonic constraints"
        if len(plan.numeric) + len(plan.categorical) < MIN_SELECTION_FEATURES:
            return f"fewer than {MIN_SELECTION_FEATURES} features"
        return None


class MissingValueStrategy(FeatureStrategy):
    """Base for strategies that change how missing numbers are filled."""

    def applicability(
        self,
        profile: data_contracts.DatasetProfile,
        task: tasks.TaskType,
        adapter: models.ModelAdapter,
    ) -> str | None:
        """Needs numeric gaps and a model that does not handle them itself."""
        reason = super().applicability(profile, task, adapter)
        if reason is not None:
            return reason
        if adapter.capabilities.handles_missing:
            return "the model handles missing values natively"
        stats = profile.column_statistics
        gaps = [
            name
            for name in profile.feature_plan.numeric
            if stats.get(name, {}).get("missing_rate", 0.0) >= MIN_MISSING_RATE
        ]
        if not gaps:
            return f"no numeric feature is missing in >= {MIN_MISSING_RATE:.0%}"
        return None


class MissingIndicatorStrategy(MissingValueStrategy):
    """Fills with the median and flags which values were missing."""

    name = "missing_indicators"
    description = (
        "median imputation plus a missing-value flag for features whose "
        "gaps relate to the target"
    )
    missing = "indicator"

    def applicability(
        self,
        profile: data_contracts.DatasetProfile,
        task: tasks.TaskType,
        adapter: models.ModelAdapter,
    ) -> str | None:
        """Needs numeric gaps that the missingness report calls informative."""
        reason = super().applicability(profile, task, adapter)
        if reason is not None:
            return reason
        plan = profile.feature_plan
        if plan.informative_missing is None:
            return None  # Not assessed: keep the earlier behaviour.
        flagged = [c for c in plan.informative_missing if c in plan.numeric]
        if not flagged:
            return "no numeric feature's gaps are associated with the target"
        return None


class IterativeImputationStrategy(MissingValueStrategy):
    """Predicts each missing value from the other features."""

    name = "iterative_imputation"
    description = "model-based imputation from the other numeric features"
    missing = "iterative"

    def applicability(
        self,
        profile: data_contracts.DatasetProfile,
        task: tasks.TaskType,
        adapter: models.ModelAdapter,
    ) -> str | None:
        """Also bounded in width, because it fits a model per feature."""
        reason = super().applicability(profile, task, adapter)
        if reason is not None:
            return reason
        width = len(profile.feature_plan.numeric)
        if width > MAX_ITERATIVE_FEATURES:
            return f"{width} numeric features is too many to impute iteratively"
        return None


class TargetTransformStrategy(DataStrategy):
    """Regression on a compressed target, predicted in original units.

    A long-tailed target makes squared error chase the extremes; modelling a
    more symmetric version often helps. Adopted only if the original-unit
    score on the validation folds improves.
    """

    name = "target_transform"
    description = "arcsinh transform of a skewed regression target"

    def applicability(
        self,
        profile: data_contracts.DatasetProfile,
        task: tasks.TaskType,
        adapter: models.ModelAdapter,
    ) -> str | None:
        """Needs a skewed continuous target."""
        reason = super().applicability(profile, task, adapter)
        if reason is not None:
            return reason
        if task is not tasks.TaskType.REGRESSION:
            return "target transforms are a regression technique"
        skew = abs(profile.target_summary.get("skew", 0.0))
        if skew < TARGET_SKEW_THRESHOLD:
            return f"target is not long-tailed (|skew| {skew:.2f})"
        return None

    def build(
        self,
        plan: data_contracts.FeaturePlan,
        adapter: models.ModelAdapter,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Wraps the standard pipeline so the target is transformed."""
        return compose.TransformedTargetRegressor(
            regressor=preprocessing.build_pipeline(
                plan, adapter, task, seed, params
            ),
            transformer=transformers.AsinhTarget(),
        )


STRATEGIES: dict[str, DataStrategy] = {
    strategy.name: strategy
    for strategy in (
        DefaultStrategy(),
        ClassWeightStrategy(),
        OversampleStrategy(),
        UndersampleStrategy(),
        PlainSmoteStrategy(),
        BorderlineSmoteStrategy(),
        AdasynStrategy(),
        JitterStrategy(),
        CopulaAugmentStrategy(),
        CalibrationStrategy(),
        LogSkewedStrategy(),
        InteractionStrategy(),
        ConstructFeaturesStrategy(),
        SelectFeaturesStrategy(),
        MissingIndicatorStrategy(),
        IterativeImputationStrategy(),
        TargetTransformStrategy(),
        TargetEncodingStrategy(),
        RareTargetStrategy(),
    )
}
EXPERIMENTAL = tuple(name for name in STRATEGIES if name != DEFAULT)
OPT_IN = frozenset({"jitter_augment", "copula_augment"})
DEFAULT_NAMES = tuple(n for n in EXPERIMENTAL if n not in OPT_IN)


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
