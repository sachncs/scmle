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

import numpy
from sklearn import base
from sklearn import linear_model

FLOOR = 1e-6


class DirichletCalibrator(base.BaseEstimator, base.ClassifierMixin):  # type: ignore[misc]  # scikit-learn is untyped
    """A frozen classifier whose probabilities are Dirichlet-calibrated.

    Attributes:
        estimator: A fitted classifier with `predict_proba`.
        regularization: Strength of the L2 penalty on the map.
    """

    def __init__(self, estimator: Any, regularization: float = 1.0) -> None:
        """Stores the settings (scikit-learn convention).

        Args:
            estimator: A fitted classifier with `predict_proba`.
            regularization: Strength of the L2 penalty on the map; the
                logistic regression uses `C = 1 / max(regularization, 1e-9)`.
        """
        self.estimator = estimator
        self.regularization = regularization

    def log_probabilities(self, features: Any) -> numpy.ndarray:
        """Log of the wrapped model's probabilities, floored.

        Args:
            features: Rows to predict probabilities for.

        Returns:
            The natural log of each class probability, with probabilities
            clipped to `[FLOOR, 1]` first; shape `(rows, classes)`.
        """
        probabilities = numpy.asarray(self.estimator.predict_proba(features))
        return numpy.asarray(numpy.log(numpy.clip(probabilities, FLOOR, 1.0)))

    def fit(self, features: Any, labels: Any) -> Self:
        """Fits a multinomial logistic map on the wrapped log-probabilities.

        The rows should be ones the wrapped model was not trained on; this is
        not checked. `classes_` is set to the classes present in `labels`.

        Args:
            features: Calibration rows.
            labels: Class labels of those rows.

        Returns:
            The fitted calibrator itself.
        """
        self.map_ = linear_model.LogisticRegression(
            C=1.0 / max(self.regularization, 1e-9), max_iter=1000
        ).fit(self.log_probabilities(features), numpy.asarray(labels))
        self.classes_ = self.map_.classes_
        return self

    def predict_proba(self, features: Any) -> numpy.ndarray:
        """Calibrated class probabilities.

        Args:
            features: Rows to predict for.

        Returns:
            Probabilities of shape `(rows, classes)`, columns in `classes_`
            order.
        """
        probabilities: numpy.ndarray = self.map_.predict_proba(
            self.log_probabilities(features)
        )
        return probabilities

    def predict(self, features: Any) -> numpy.ndarray:
        """Most probable class after calibration.

        Args:
            features: Rows to predict for.

        Returns:
            The class with the highest calibrated probability for each row.
        """
        labels: numpy.ndarray = self.classes_[
            self.predict_proba(features).argmax(axis=1)
        ]
        return labels
