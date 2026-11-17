"""Strategies for imbalanced classes: weights, resampling, recalibration."""

from collections.abc import Mapping
import importlib
import importlib.util
from typing import Any
from typing import cast
from typing import ClassVar
from typing import TYPE_CHECKING

from scmle.contracts import data
from scmle.contracts import models
from scmle.contracts import tasks
from scmle.data.splitting import inner_cv
from scmle.preparation import preprocessing
from scmle.preparation import resampling
from scmle.preparation import synthesis
from scmle.preparation.strategies.interface import CALIBRATION_FOLDS
from scmle.preparation.strategies.interface import MIN_IMBALANCE_RATIO
from scmle.preparation.strategies.interface import Strategy

if TYPE_CHECKING:
    from sklearn import base


class Rebalancing(Strategy):
    """Base for strategies that address class imbalance."""

    def applicability(
        self,
        profile: data.Portrait,
        task: tasks.Task,
        adapter: models.Model,
    ) -> str | None:
        """Requires a classification task with noticeably skewed classes.

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
        if not task.is_classification:
            return "rebalancing is a classification technique"
        ratio = profile.target_summary.get("imbalance_ratio", 1.0)
        if ratio < MIN_IMBALANCE_RATIO:
            return (
                f"classes are balanced enough ({ratio:.1f}:1 < "
                f"{MIN_IMBALANCE_RATIO:.0f}:1)"
            )
        return None


class Weighting(Rebalancing):
    """Re-weights classes inversely to their frequency."""

    name = "class_weight"
    description = "balanced class weights"

    def applicability(
        self,
        profile: data.Portrait,
        task: tasks.Task,
        adapter: models.Model,
    ) -> str | None:
        """Also requires an estimator with a `class_weight` parameter.

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
        if "class_weight" not in adapter.build(task, 0).get_params():
            return f"{adapter.name} has no class_weight parameter"
        return None

    def build(
        self,
        plan: data.Blueprint,
        adapter: models.Model,
        task: tasks.Task,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Builds the standard pipeline with `class_weight='balanced'`.

        Args:
            plan: Feature columns by kind.
            adapter: Model family.
            task: Resolved task type.
            seed: Random seed.
            params: Hyperparameter overrides.

        Returns:
            The unfitted candidate estimator, preprocessing included.
        """
        weighted = {**(params or {}), "class_weight": "balanced"}
        pipeline: models.Estimator = preprocessing.instantiate(
            plan, adapter, task, seed, weighted
        )
        return pipeline


class Resampling(Rebalancing):
    """Base for strategies that resample each training fold.

    Attributes:
        resampler_class: Resampler applied inside `fit`.
    """

    resampler_class: ClassVar[type[resampling.Resampler]]

    def build(
        self,
        plan: data.Blueprint,
        adapter: models.Model,
        task: tasks.Task,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Wraps the standard pipeline so it trains on rebalanced data.

        Args:
            plan: Feature columns by kind.
            adapter: Model family.
            task: Resolved task type.
            seed: Random seed.
            params: Hyperparameter overrides.

        Returns:
            The unfitted candidate estimator, preprocessing included.
        """
        inner = preprocessing.instantiate(plan, adapter, task, seed, params)
        return resampling.ResamplingClassifier(
            inner, self.resampler_class(), seed
        )


class Oversampling(Resampling):
    """Random oversampling of minority classes inside each training fold."""

    name = "random_oversample"
    description = "random oversampling inside the training fold"
    resampler_class = resampling.RandomOverSampler


class Undersampling(Resampling):
    """Random undersampling of majority classes inside each training fold."""

    name = "random_undersample"
    description = "random undersampling inside the training fold"
    resampler_class = resampling.RandomUnderSampler


class Recalibration(Strategy):
    """Recalibrates predicted probabilities (sigmoid, fitted inside folds)."""

    name = "calibrated"
    description = "sigmoid probability calibration inside the training fold"

    def applicability(
        self,
        profile: data.Portrait,
        task: tasks.Task,
        adapter: models.Model,
    ) -> str | None:
        """Needs a probabilistic classifier worth calibrating.

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
        if not task.is_classification:
            return "calibration applies to classification"
        if not adapter.capabilities.probabilities:
            return f"{adapter.name} does not provide probabilities"
        if adapter.name == "logistic_regression":
            return "logistic regression is already well calibrated"
        return None

    def build(
        self,
        plan: data.Blueprint,
        adapter: models.Model,
        task: tasks.Task,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Wraps the standard pipeline in a cross-fitted calibrator.

        Args:
            plan: Feature columns by kind.
            adapter: Model family.
            task: Resolved task type.
            seed: Random seed.
            params: Hyperparameter overrides.

        Returns:
            The unfitted candidate estimator, preprocessing included.
        """
        inner = preprocessing.instantiate(plan, adapter, task, seed, params)
        calibrated: models.Estimator = inner_cv.Calibrated(
            inner,
            method="sigmoid",
            cv=inner_cv.for_plan(
                plan, classification=True, folds=CALIBRATION_FOLDS, seed=seed
            ),
        )
        return calibrated


class Jitter(Resampling):
    """Experimental augmentation: jittered minority copies in each fold.

    Not part of the default strategy list; request it explicitly with
    `Strategies(names=["jitter_augment"])`.
    """

    name = "jitter_augment"
    description = "EXPERIMENTAL jittered minority augmentation inside the fold"
    resampler_class = resampling.Jittered


class Copula(Resampling):
    """Experimental augmentation: Gaussian-copula synthetic minority rows.

    Opt-in only; adopted only when it beats the un-augmented baseline on the
    original validation folds.
    """

    name = "copula_augment"
    description = "EXPERIMENTAL Gaussian-copula synthetic minority rows"
    resampler_class = synthesis.CopulaOverSampler


class Smote(Rebalancing):
    """SMOTE-family oversampling on the encoded training fold.

    Attributes:
        sampler_name: Class name in `imblearn.over_sampling`.
    """

    sampler_name: ClassVar[str] = "SMOTE"
    min_features: ClassVar[int] = 2

    def applicability(
        self,
        profile: data.Portrait,
        task: tasks.Task,
        adapter: models.Model,
    ) -> str | None:
        """Needs imbalanced-learn and a purely numeric feature space.

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
        if importlib.util.find_spec("imblearn") is None:
            return "needs the 'sampling' extra (imbalanced-learn)"
        if profile.feature_plan.categorical:
            return "interpolating one-hot categories is not meaningful"
        if len(profile.feature_plan.numeric) < self.min_features:
            return "needs at least two numeric features"
        return None

    def build(
        self,
        plan: data.Blueprint,
        adapter: models.Model,
        task: tasks.Task,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Builds preprocess -> resample -> model with the SMOTE variant.

        Args:
            plan: Feature columns by kind.
            adapter: Model family.
            task: Resolved task type.
            seed: Random seed.
            params: Hyperparameter overrides.

        Returns:
            The unfitted candidate estimator, preprocessing included.
        """
        over_sampling = importlib.import_module("imblearn.over_sampling")
        sampler = getattr(over_sampling, self.sampler_name)(random_state=seed)
        options = data.Preprocessing(scale=adapter.capabilities.needs_scaling)
        return resampling.PreprocessedResamplingClassifier(
            preprocessing.frontend(plan, options),
            sampler,
            cast("base.BaseEstimator", adapter.build(task, seed, params)),
        )


class Borderline(Smote):
    """Borderline-SMOTE: synthesizes only near the decision boundary."""

    name = "borderline_smote"
    description = "Borderline-SMOTE oversampling inside the training fold"
    sampler_name = "BorderlineSMOTE"


class PlainSmote(Smote):
    """Classic SMOTE."""

    name = "smote"
    description = "SMOTE oversampling inside the training fold"
    sampler_name = "SMOTE"


class Adasyn(Smote):
    """ADASYN: more synthetic rows where minority density is low."""

    name = "adasyn"
    description = "ADASYN oversampling inside the training fold"
    sampler_name = "ADASYN"
