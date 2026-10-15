"""Estimator wrappers adapting third-party quirks to the framework."""

from typing import Any
from typing import Self

import numpy as np
from sklearn import base
from sklearn import preprocessing


class LabelEncodedClassifier(base.ClassifierMixin, base.BaseEstimator):
    """Lets estimators that need integer labels accept arbitrary class labels.

    Fits the wrapped classifier on integer-encoded labels and decodes its
    predictions, so the rest of the pipeline (metrics, bundles, inference)
    always sees the user's original labels.

    Attributes:
        estimator: Unfitted classifier requiring labels 0..K-1.
        encoder_: Fitted label encoder.
        estimator_: Fitted clone of `estimator`.
        classes_: Original class labels, sorted.
    """

    def __init__(self, estimator: base.BaseEstimator) -> None:
        """Initializes the wrapper.

        Args:
            estimator: Classifier to wrap.
        """
        self.estimator = estimator

    def fit(self, features: Any, target: Any) -> Self:
        """Encodes the labels and fits the wrapped classifier."""
        self.encoder_ = preprocessing.LabelEncoder().fit(target)
        self.classes_ = self.encoder_.classes_
        self.estimator_ = base.clone(self.estimator).fit(
            features, self.encoder_.transform(target)
        )
        return self

    def predict(self, features: Any) -> np.ndarray:
        """Predicts and decodes back to the original labels."""
        encoded = np.asarray(self.estimator_.predict(features)).astype(int)
        return self.encoder_.inverse_transform(encoded)

    def predict_proba(self, features: Any) -> np.ndarray:
        """Returns class probabilities in `classes_` order."""
        return self.estimator_.predict_proba(features)


class FlatPredictionClassifier(base.ClassifierMixin, base.BaseEstimator):
    """Guarantees 1-D predictions from libraries that return column vectors.

    CatBoost's multiclass `predict` returns shape `(n, 1)`; the framework's
    contract is `(n,)`.

    Attributes:
        estimator: Unfitted classifier.
        estimator_: Fitted clone.
        classes_: Class labels.
    """

    def __init__(self, estimator: base.BaseEstimator) -> None:
        """Initializes the wrapper.

        Args:
            estimator: Classifier whose `predict` may be 2-D.
        """
        self.estimator = estimator

    def fit(self, features: Any, target: Any) -> Self:
        """Fits a clone of the wrapped classifier."""
        self.estimator_ = base.clone(self.estimator).fit(features, target)
        self.classes_ = np.asarray(self.estimator_.classes_).ravel()
        return self

    def predict(self, features: Any) -> np.ndarray:
        """Predicts, flattened to one dimension."""
        return np.asarray(self.estimator_.predict(features)).ravel()

    def predict_proba(self, features: Any) -> np.ndarray:
        """Returns class probabilities."""
        return self.estimator_.predict_proba(features)


class FitParamsClassifier(base.ClassifierMixin, base.BaseEstimator):
    """Passes fixed keyword arguments to the wrapped classifier's `fit`.

    Some libraries take dataset properties (for example which columns are
    categorical) only as `fit` arguments, which a scikit-learn pipeline
    cannot route; this wrapper carries them as an ordinary parameter.

    Attributes:
        estimator: Unfitted classifier.
        fit_params: Keyword arguments forwarded to `estimator.fit`.
        estimator_: Fitted clone.
        classes_: Class labels.
    """

    def __init__(
        self, estimator: base.BaseEstimator, fit_params: dict[str, Any]
    ) -> None:
        """Initializes the wrapper.

        Args:
            estimator: Classifier to wrap.
            fit_params: Arguments forwarded to `fit`.
        """
        self.estimator = estimator
        self.fit_params = fit_params

    def fit(self, features: Any, target: Any) -> Self:
        """Fits a clone with the fixed arguments."""
        self.estimator_ = base.clone(self.estimator).fit(
            features, target, **self.fit_params
        )
        self.classes_ = np.asarray(self.estimator_.classes_)
        return self

    def predict(self, features: Any) -> np.ndarray:
        """Predicts class labels."""
        return np.asarray(self.estimator_.predict(features))

    def predict_proba(self, features: Any) -> np.ndarray:
        """Returns class probabilities."""
        return self.estimator_.predict_proba(features)


class FitParamsRegressor(base.RegressorMixin, base.BaseEstimator):
    """Passes fixed keyword arguments to the wrapped regressor's `fit`.

    Attributes:
        estimator: Unfitted regressor.
        fit_params: Keyword arguments forwarded to `estimator.fit`.
        estimator_: Fitted clone.
    """

    def __init__(
        self, estimator: base.BaseEstimator, fit_params: dict[str, Any]
    ) -> None:
        """Initializes the wrapper.

        Args:
            estimator: Regressor to wrap.
            fit_params: Arguments forwarded to `fit`.
        """
        self.estimator = estimator
        self.fit_params = fit_params

    def fit(self, features: Any, target: Any) -> Self:
        """Fits a clone with the fixed arguments."""
        self.estimator_ = base.clone(self.estimator).fit(
            features, target, **self.fit_params
        )
        return self

    def predict(self, features: Any) -> np.ndarray:
        """Predicts values."""
        return np.asarray(self.estimator_.predict(features))
