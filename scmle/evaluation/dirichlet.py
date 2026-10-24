"""Dirichlet calibration of multiclass probabilities.

Sigmoid and isotonic calibration treat one class at a time. Dirichlet
calibration (Kull et al., 2019) learns a full map from all class
probabilities to calibrated ones: a multinomial logistic regression on the
log-probabilities, `q = softmax(W log p + b)`. It can fix confusion between
specific classes that per-class methods cannot. The L2 penalty on `W` keeps
the map near identity when calibration rows are few.
"""

from typing import Any
from typing import Self

import numpy as np
from sklearn import base
from sklearn import linear_model

FLOOR = 1e-6


class DirichletCalibrator(base.BaseEstimator, base.ClassifierMixin):
    """A frozen classifier whose probabilities are Dirichlet-calibrated.

    Attributes:
        estimator: A fitted classifier with `predict_proba`.
        regularization: Strength of the L2 penalty on the map.
    """

    def __init__(self, estimator: Any, regularization: float = 1.0) -> None:
        """Stores the settings (scikit-learn convention)."""
        self.estimator = estimator
        self.regularization = regularization

    def log_probabilities(self, features: Any) -> np.ndarray:
        """Log of the wrapped model's probabilities, floored."""
        probabilities = np.asarray(self.estimator.predict_proba(features))
        return np.log(np.clip(probabilities, FLOOR, 1.0))

    def fit(self, features: Any, labels: Any) -> Self:
        """Learns the map on labeled rows the model was not trained on."""
        self.map_ = linear_model.LogisticRegression(
            C=1.0 / max(self.regularization, 1e-9), max_iter=1000
        ).fit(self.log_probabilities(features), np.asarray(labels))
        self.classes_ = self.map_.classes_
        return self

    def predict_proba(self, features: Any) -> np.ndarray:
        """Calibrated class probabilities."""
        return self.map_.predict_proba(self.log_probabilities(features))

    def predict(self, features: Any) -> np.ndarray:
        """Most probable class after calibration."""
        return self.classes_[self.predict_proba(features).argmax(axis=1)]
