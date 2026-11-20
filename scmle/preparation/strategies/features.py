"""Strategies that change the features: transforms, encodings, imputation."""

from collections.abc import Mapping
import dataclasses
from typing import Any
from typing import ClassVar

from sklearn import compose
from sklearn.utils import validation

from scmle.contracts import data
from scmle.contracts import models
from scmle.contracts import tasks
from scmle.data import transformers
from scmle.preparation import preprocessing
from scmle.preparation import resampling
from scmle.preparation.strategies.interface import MAX_INTERACTION_FEATURES
from scmle.preparation.strategies.interface import MAX_ITERATIVE_FEATURES
from scmle.preparation.strategies.interface import MIN_INTERACTION_FEATURES
from scmle.preparation.strategies.interface import MIN_MISSING_RATE
from scmle.preparation.strategies.interface import MIN_SELECTION_FEATURES
from scmle.preparation.strategies.interface import SKEW_THRESHOLD
from scmle.preparation.strategies.interface import Strategy
from scmle.preparation.strategies.interface import TARGET_SKEW_THRESHOLD


class Features(Strategy):
    """Base for strategies that change the numeric preprocessing.

    Attributes:
        skewed: Log-transform right-skewed numeric features.
        interactions: Add pairwise numeric interactions.
        construct: Append differences and ratios of the best numeric pairs.
        select: Keep only columns an L1-penalised linear model uses.
        missing: Missing-value strategy for numeric features.
    """

    skewed: ClassVar[bool] = False
    interactions: ClassVar[bool] = False
    construct: ClassVar[bool] = False
    select: ClassVar[bool] = False
    missing: ClassVar[str] = "median"

    def build(
        self,
        plan: data.Blueprint,
        adapter: models.Model,
        task: tasks.Task,
        seed: int,
        params: Mapping[str, Any] | None = None,
        featurise: preprocessing.Featuriser | None = None,
    ) -> models.Estimator:
        """Builds the pipeline with the strategy's numeric steps.

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
        options = dataclasses.replace(
            preprocessing.default(plan, adapter),
            skewed=self.skewed,
            interactions=self.interactions,
            construct=self.construct,
            select=self.select,
            missing=self.missing,
        )
        if self.select:
            options = dataclasses.replace(options, scale=True)
        pipeline: models.Estimator = preprocessing.instantiate(
            plan, adapter, task, seed, params, options, featurise=featurise
        )
        return pipeline


class Logarithm(Features):
    """Log-transforms right-skewed, non-negative numeric features."""

    name = "log_skewed"
    description = "log1p of right-skewed non-negative numeric features"
    skewed = True

    def applicability(
        self,
        profile: data.Portrait,
        task: tasks.Task,
        adapter: models.Model,
    ) -> str | None:
        """Needs a skewed feature and a model sensitive to monotone shifts.

        Args:
            profile: Dataset profile.
            task: Resolved task type.
            adapter: Model the strategy would be combined with.

        Returns:
            A reason the strategy does not apply, or None when it applies.
        """
        reason = super().applicability(profile, task, adapter)
        if reason is not None:
            return reason
        if adapter.family in ("tree", "boosting"):
            return "monotone transforms do not change tree splits"
        stats = profile.statistics
        skewed = [
            name
            for name in profile.plan.numeric
            if name in stats
            and stats[name].get("skew", 0.0) > SKEW_THRESHOLD
            and stats[name].get("min", -1.0) >= 0
        ]
        if not skewed:
            return "no right-skewed non-negative numeric feature"
        return None


class Rarity(Strategy):
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
        profile: data.Portrait,
        task: tasks.Task,
        adapter: models.Model,
    ) -> str | None:
        """Needs a skewed regression target and a model taking weights.

        Checked in order: the baseline model is refused, then classification
        tasks, then a target whose absolute skew is below
        `TARGET_SKEW_THRESHOLD`, then a model whose estimator has no
        `sample_weight` fit parameter.

        Args:
            profile: Dataset profile.
            task: Resolved task type.
            adapter: Model the strategy would be combined with.

        Returns:
            The first reason the strategy does not apply, or None when it
            applies.
        """
        reason = super().applicability(profile, task, adapter)
        if reason is not None:
            return reason
        if task.classification:
            return "regression only (classification uses class weighting)"
        skew = abs(profile.summary.get("skew", 0.0))
        if skew < TARGET_SKEW_THRESHOLD:
            return f"target is not long-tailed (|skew| {skew:.2f})"
        if not validation.has_fit_parameter(
            adapter.build(task, 0), "sample_weight"
        ):
            return f"{adapter.name} does not accept sample weights"
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
        """Wraps the standard pipeline in the density-weighted regressor.

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
        inner = preprocessing.instantiate(
            plan, adapter, task, seed, params, featurise=featurise
        )
        return resampling.Density(inner)


class Encoding(Strategy):
    """Regularized, cross-fitted target encoding of high-cardinality columns.

    Replaces one-hot encoding of categorical features with many levels by a
    smoothed target encoding that is fitted inside each training fold, with
    internal cross-fitting so no row sees its own label.
    """

    name = "target_encoding"
    description = "regularized target encoding of high-cardinality categories"

    def applicability(
        self,
        profile: data.Portrait,
        task: tasks.Task,
        adapter: models.Model,
    ) -> str | None:
        """Needs at least one high-cardinality categorical feature.

        Args:
            profile: Dataset profile.
            task: Resolved task type.
            adapter: Model the strategy would be combined with.

        Returns:
            A reason the strategy does not apply, or None when it applies.
        """
        reason = super().applicability(profile, task, adapter)
        if reason is not None:
            return reason
        if not profile.plan.cardinal:
            return "no high-cardinality categorical feature"
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
        """Builds the pipeline with target-encoded high-cardinality columns.

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
        options = dataclasses.replace(
            preprocessing.default(plan, adapter),
            cardinality="target",
            sparse=False,
        )
        pipeline: models.Estimator = preprocessing.instantiate(
            plan, adapter, task, seed, params, options, featurise=featurise
        )
        return pipeline


class Interactions(Features):
    """Adds pairwise numeric interactions for linear models."""

    name = "interactions"
    description = "pairwise numeric interaction features"
    interactions = True

    def applicability(
        self,
        profile: data.Portrait,
        task: tasks.Task,
        adapter: models.Model,
    ) -> str | None:
        """Linear models with a moderate number of numeric features.

        Args:
            profile: Dataset profile.
            task: Resolved task type.
            adapter: Model the strategy would be combined with.

        Returns:
            A reason the strategy does not apply, or None when it applies.
        """
        reason = super().applicability(profile, task, adapter)
        if reason is not None:
            return reason
        if adapter.family != "linear":
            return "non-linear models learn interactions themselves"
        count = len(profile.plan.numeric)
        if not MIN_INTERACTION_FEATURES <= count <= MAX_INTERACTION_FEATURES:
            return f"{count} numeric features is outside [2, 12]"
        return None


class Construction(Features):
    """Adds differences and ratios of the numeric features that matter most."""

    name = "construct_features"
    description = "differences and ratios of the most target-related features"
    construct = True

    def applicability(
        self,
        profile: data.Portrait,
        task: tasks.Task,
        adapter: models.Model,
    ) -> str | None:
        """Needs at least two numeric features to combine.

        Args:
            profile: Dataset profile.
            task: Resolved task type.
            adapter: Model the strategy would be combined with.

        Returns:
            A reason the strategy does not apply, or None when it applies.
        """
        reason = super().applicability(profile, task, adapter)
        if reason is not None:
            return reason
        if len(profile.plan.numeric) < MIN_INTERACTION_FEATURES:
            return "fewer than two numeric features to combine"
        return None


class Selection(Features):
    """Drops columns an L1-penalised linear model gives no weight."""

    name = "select_features"
    description = "keep only the features a sparse linear model uses"
    select = True

    def applicability(
        self,
        profile: data.Portrait,
        task: tasks.Task,
        adapter: models.Model,
    ) -> str | None:
        """Needs enough features for removal to be meaningful.

        Args:
            profile: Dataset profile.
            task: Resolved task type.
            adapter: Model the strategy would be combined with.

        Returns:
            A reason the strategy does not apply, or None when it applies.
        """
        reason = super().applicability(profile, task, adapter)
        if reason is not None:
            return reason
        plan = profile.plan
        if plan.monotonic:
            return "feature selection would shift the monotonic constraints"
        if len(plan.numeric) + len(plan.categorical) < MIN_SELECTION_FEATURES:
            return f"fewer than {MIN_SELECTION_FEATURES} features"
        return None


class Imputation(Features):
    """Base for strategies that change how missing numbers are filled."""

    def applicability(
        self,
        profile: data.Portrait,
        task: tasks.Task,
        adapter: models.Model,
    ) -> str | None:
        """Needs numeric gaps and a model that does not handle them itself.

        Args:
            profile: Dataset profile.
            task: Resolved task type.
            adapter: Model the strategy would be combined with.

        Returns:
            A reason the strategy does not apply, or None when it applies.
        """
        reason = super().applicability(profile, task, adapter)
        if reason is not None:
            return reason
        if adapter.capabilities.missing:
            return "the model handles missing values natively"
        stats = profile.statistics
        gaps = [
            name
            for name in profile.plan.numeric
            if stats.get(name, {}).get("missing_rate", 0.0) >= MIN_MISSING_RATE
        ]
        if not gaps:
            return f"no numeric feature is missing in >= {MIN_MISSING_RATE:.0%}"
        return None


class Indicators(Imputation):
    """Fills with the median and flags which values were missing."""

    name = "missing_indicators"
    description = (
        "median imputation plus a missing-value flag for features whose "
        "gaps relate to the target"
    )
    missing = "indicator"

    def applicability(
        self,
        profile: data.Portrait,
        task: tasks.Task,
        adapter: models.Model,
    ) -> str | None:
        """Needs numeric gaps that the missingness report calls informative.

        Args:
            profile: Dataset profile.
            task: Resolved task type.
            adapter: Model the strategy would be combined with.

        Returns:
            A reason the strategy does not apply, or None when it applies.
        """
        reason = super().applicability(profile, task, adapter)
        if reason is not None:
            return reason
        plan = profile.plan
        problem = preprocessing.misalignment(
            plan, select=False, missing=self.missing
        )
        if problem:
            return problem
        if plan.informative_missing is None:
            return None  # Not assessed: keep the earlier behaviour.
        flagged = [c for c in plan.informative_missing if c in plan.numeric]
        if not flagged:
            return "no numeric feature's gaps are associated with the target"
        return None


class IterativeImputation(Imputation):
    """Predicts each missing value from the other features."""

    name = "iterative_imputation"
    description = "model-based imputation from the other numeric features"
    missing = "iterative"

    def applicability(
        self,
        profile: data.Portrait,
        task: tasks.Task,
        adapter: models.Model,
    ) -> str | None:
        """Also bounded in width, because it fits a model per feature.

        Args:
            profile: Dataset profile.
            task: Resolved task type.
            adapter: Model the strategy would be combined with.

        Returns:
            A reason the strategy does not apply, or None when it applies.
        """
        reason = super().applicability(profile, task, adapter)
        if reason is not None:
            return reason
        width = len(profile.plan.numeric)
        if width > MAX_ITERATIVE_FEATURES:
            return f"{width} numeric features is too many to impute iteratively"
        return None


class Transformation(Strategy):
    """Regression on a compressed target, predicted in original units.

    A long-tailed target makes squared error chase the extremes; modelling a
    more symmetric version often helps. Adopted only if the original-unit
    score on the validation folds improves.
    """

    name = "target_transform"
    description = "arcsinh transform of a skewed regression target"

    def applicability(
        self,
        profile: data.Portrait,
        task: tasks.Task,
        adapter: models.Model,
    ) -> str | None:
        """Needs a skewed regression target.

        Checked in order: the baseline model is refused, then any task other
        than plain regression, then a target whose absolute skew is below
        `TARGET_SKEW_THRESHOLD`.

        Args:
            profile: Dataset profile.
            task: Resolved task type.
            adapter: Model the strategy would be combined with.

        Returns:
            The first reason the strategy does not apply, or None when it
            applies.
        """
        reason = super().applicability(profile, task, adapter)
        if reason is not None:
            return reason
        if task is not tasks.Task.REGRESSION:
            return "target transforms are a regression technique"
        skew = abs(profile.summary.get("skew", 0.0))
        if skew < TARGET_SKEW_THRESHOLD:
            return f"target is not long-tailed (|skew| {skew:.2f})"
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
        """Wraps the standard pipeline so the target is transformed.

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
        transformed: models.Estimator = compose.TransformedTargetRegressor(
            regressor=preprocessing.instantiate(
                plan, adapter, task, seed, params, featurise=featurise
            ),
            transformer=transformers.Asinh(),
        )
        return transformed
