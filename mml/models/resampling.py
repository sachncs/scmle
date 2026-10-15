"""Training-fold resampling for class imbalance.

`ResamplingClassifier` resamples *inside* `fit`, so under cross-validation
only the training fold is ever rebalanced and validation folds keep the
original distribution; this is the only leakage-safe place to resample.
"""

import abc
from typing import Any
from typing import ClassVar
from typing import Self

import numpy as np
import pandas as pd
from sklearn import base


class Resampler(abc.ABC):
    """Strategy that rebalances a labelled training set.

    Attributes:
        name: Short identifier.
    """

    name: ClassVar[str]

    @abc.abstractmethod
    def indices(
        self, target: pd.Series, rng: np.random.Generator
    ) -> np.ndarray:
        """Chooses the row positions of the resampled training set.

        Args:
            target: Training labels.
            rng: Source of randomness.

        Returns:
            Positions (with repeats for oversampling) into the training rows.
        """

    def resample(
        self, features: pd.DataFrame, target: pd.Series, seed: int
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Applies the resampling.

        Args:
            features: Training features.
            target: Training labels.
            seed: Random seed.

        Returns:
            Resampled features and labels with a fresh index.
        """
        chosen = self.indices(target, np.random.default_rng(seed))
        return (
            features.iloc[chosen].reset_index(drop=True),
            target.iloc[chosen].reset_index(drop=True),
        )


class RandomOverSampler(Resampler):
    """Duplicates minority-class rows until every class matches the largest."""

    name = "random_oversample"

    def indices(
        self, target: pd.Series, rng: np.random.Generator
    ) -> np.ndarray:
        """Samples minority rows with replacement up to the majority count."""
        positions = np.arange(len(target))
        counts = target.value_counts()
        goal = int(counts.max())
        chosen = [positions]
        for label, count in counts.items():
            members = positions[(target == label).to_numpy()]
            if count < goal:
                chosen.append(
                    rng.choice(members, goal - int(count), replace=True)
                )
        return rng.permutation(np.concatenate(chosen))


class RandomUnderSampler(Resampler):
    """Drops majority-class rows until every class matches the smallest."""

    name = "random_undersample"

    def indices(
        self, target: pd.Series, rng: np.random.Generator
    ) -> np.ndarray:
        """Samples each class without replacement down to the smallest."""
        positions = np.arange(len(target))
        goal = int(target.value_counts().min())
        chosen = [
            rng.choice(
                positions[(target == label).to_numpy()], goal, replace=False
            )
            for label in target.unique()
        ]
        return rng.permutation(np.concatenate(chosen))


class JitterOverSampler(Resampler):
    """Experimental: oversamples with Gaussian-jittered copies of minority rows.

    Synthetic rows copy a real minority row and add noise (a fraction of the
    column's standard deviation) to its numeric columns; categorical values
    are copied unchanged. Treat it as an experiment: it is only adopted when
    it beats the un-augmented baseline on the original validation folds.

    Attributes:
        scale: Noise standard deviation as a fraction of each column's std.
    """

    name = "jitter_oversample"

    def __init__(self, scale: float = 0.05) -> None:
        """Initializes the sampler.

        Args:
            scale: Noise level relative to each numeric column's std.
        """
        self.scale = scale

    def indices(
        self, target: pd.Series, rng: np.random.Generator
    ) -> np.ndarray:
        """Chooses source rows exactly like plain oversampling."""
        return RandomOverSampler().indices(target, rng)

    def resample(
        self, features: pd.DataFrame, target: pd.Series, seed: int
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Oversamples, then jitters every repeated (synthetic) copy."""
        rng = np.random.default_rng(seed)
        chosen = self.indices(target, rng)
        sampled = features.iloc[chosen].reset_index(drop=True)
        repeated = pd.Series(chosen).duplicated().to_numpy()
        numeric = [
            column
            for column in features.columns
            if pd.api.types.is_numeric_dtype(features[column])
            and not pd.api.types.is_bool_dtype(features[column])
        ]
        for column in numeric:
            spread = float(pd.to_numeric(features[column]).std() or 0.0)
            noise = rng.normal(0.0, self.scale * spread, repeated.sum())
            values = (
                sampled[column].to_numpy(dtype=float, na_value=np.nan).copy()
            )
            values[repeated] = values[repeated] + noise
            sampled[column] = values
        return sampled, target.iloc[chosen].reset_index(drop=True)


class PreprocessedResamplingClassifier(
    base.ClassifierMixin, base.BaseEstimator
):
    """Resamples in *feature space*, after preprocessing, inside `fit`.

    Interpolating samplers such as SMOTE need numeric matrices, so the
    preprocessor is fitted on the training fold, the encoded training rows
    are resampled, and the model is fitted on the result. Prediction applies
    the fitted preprocessor only; validation rows are never resampled.

    Attributes:
        preprocessor: Unfitted column transformer.
        sampler: Unfitted imbalanced-learn style sampler (`fit_resample`).
        estimator: Unfitted classifier.
        preprocessor_: Fitted preprocessor.
        estimator_: Fitted classifier.
        classes_: Class labels.
    """

    def __init__(
        self,
        preprocessor: base.BaseEstimator,
        sampler: Any,
        estimator: base.BaseEstimator,
    ) -> None:
        """Initializes the wrapper.

        Args:
            preprocessor: Preprocessing transformer.
            sampler: Sampler with `fit_resample(features, target)`.
            estimator: Classifier trained on the resampled matrix.
        """
        self.preprocessor = preprocessor
        self.sampler = sampler
        self.estimator = estimator

    def fit(self, features: Any, target: Any) -> Self:
        """Fits the preprocessor, resamples the encoded rows, fits the model."""
        self.preprocessor_ = base.clone(self.preprocessor).fit(features, target)
        encoded = self.preprocessor_.transform(features)
        resampled, labels = base.clone(self.sampler).fit_resample(
            encoded, np.asarray(target)
        )
        self.estimator_ = base.clone(self.estimator).fit(resampled, labels)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, features: Any) -> np.ndarray:
        """Predicts from the preprocessed rows (no resampling)."""
        return self.estimator_.predict(self.preprocessor_.transform(features))

    def predict_proba(self, features: Any) -> np.ndarray:
        """Returns class probabilities (no resampling)."""
        return self.estimator_.predict_proba(
            self.preprocessor_.transform(features)
        )


class ResamplingClassifier(base.ClassifierMixin, base.BaseEstimator):
    """Fits a classifier on a resampled copy of its training data.

    Attributes:
        estimator: Unfitted classifier (typically a full pipeline).
        resampler: Resampling strategy.
        random_state: Seed for the resampling.
        estimator_: Fitted clone of `estimator`.
        classes_: Class labels of the fitted classifier.
    """

    def __init__(
        self,
        estimator: base.BaseEstimator,
        resampler: Resampler,
        random_state: int = 0,
    ) -> None:
        """Initializes the wrapper.

        Args:
            estimator: Classifier to fit on resampled data.
            resampler: How to rebalance the training data.
            random_state: Seed for the resampling.
        """
        self.estimator = estimator
        self.resampler = resampler
        self.random_state = random_state

    def fit(self, features: Any, target: Any) -> Self:
        """Rebalances the training data and fits the wrapped classifier."""
        labels = pd.Series(np.asarray(target))
        frame = pd.DataFrame(features).reset_index(drop=True)
        resampled, resampled_target = self.resampler.resample(
            frame, labels, self.random_state
        )
        self.estimator_ = base.clone(self.estimator).fit(
            resampled, resampled_target
        )
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, features: Any) -> np.ndarray:
        """Predicts with the fitted classifier (no resampling)."""
        return self.estimator_.predict(features)

    def predict_proba(self, features: Any) -> np.ndarray:
        """Returns class probabilities (no resampling)."""
        return self.estimator_.predict_proba(features)


class DensityWeightedRegressor(base.RegressorMixin, base.BaseEstimator):
    """Fits a regression pipeline with weights that favour rare target values.

    Rows whose target falls in a sparsely populated region of the target's
    range get proportionally larger weights (inverse bin frequency over
    equal-width bins, capped at `max_weight` times the lightest row), so a
    long tail is not ignored. This is the regression counterpart of class
    weighting; classification resampling does not apply to a continuous
    target. The weights are computed from the training fold only.

    Attributes:
        estimator: Unfitted pipeline whose final step is named `model`.
        bins: Number of equal-width target bins.
        max_weight: Largest weight relative to the smallest.
        estimator_: Fitted clone.
    """

    def __init__(
        self,
        estimator: base.BaseEstimator,
        bins: int = 10,
        max_weight: float = 10.0,
    ) -> None:
        """Initializes the wrapper.

        Args:
            estimator: Pipeline to fit with sample weights.
            bins: Number of equal-width target bins.
            max_weight: Largest weight relative to the smallest.
        """
        self.estimator = estimator
        self.bins = bins
        self.max_weight = max_weight

    def weights(self, target: np.ndarray) -> np.ndarray:
        """Inverse-density weights, normalized to mean 1."""
        edges = np.histogram_bin_edges(target, bins=self.bins)
        index = np.clip(np.digitize(target, edges[1:-1]), 0, len(edges) - 2)
        counts = np.bincount(index, minlength=len(edges) - 1).astype(float)
        raw = 1.0 / counts[index]
        raw = np.minimum(raw, raw.min() * self.max_weight)
        return raw * len(raw) / raw.sum()

    def fit(self, features: Any, target: Any) -> Self:
        """Fits the pipeline with density weights on the model step."""
        values = np.asarray(target, dtype=float)
        self.estimator_ = base.clone(self.estimator).fit(
            features, values, model__sample_weight=self.weights(values)
        )
        return self

    def predict(self, features: Any) -> np.ndarray:
        """Predicts with the fitted pipeline."""
        return self.estimator_.predict(features)
