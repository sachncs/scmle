"""Predictions computed in batches, replayed to the metric functions.

When the locked rows are too many to hold at once, they are scored batch by
batch and only the predictions are kept. These estimators hand the stored
predictions back, so every metric works unchanged.
"""

from typing import Any

import numpy as np
from sklearn import base


class PredictionReplay(base.BaseEstimator):
    """Returns stored predictions whatever rows it is handed.

    Attributes:
        predictions: Predicted labels or values for the whole set.
        probabilities: Class probabilities, if the model has them.
        classes_: Class labels, if any.
    """

    def __init__(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray | None = None,
        classes: np.ndarray | None = None,
    ) -> None:
        """Stores the arrays."""
        self.predictions = predictions
        self.probabilities = probabilities
        self.classes_ = classes

    def predict(self, features: Any) -> np.ndarray:
        """The stored predictions."""
        return self.predictions

    def predict_proba(self, features: Any) -> np.ndarray:
        """The stored probabilities."""
        if self.probabilities is None:
            raise ValueError("no probabilities were stored")
        return self.probabilities


class ClassifierPredictions(base.ClassifierMixin, PredictionReplay):
    """Stored predictions that scikit-learn treats as a classifier."""


class RegressorPredictions(base.RegressorMixin, PredictionReplay):
    """Stored predictions that scikit-learn treats as a regressor."""
