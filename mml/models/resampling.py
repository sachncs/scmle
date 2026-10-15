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
