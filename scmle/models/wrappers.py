"""Estimator wrappers adapting third-party quirks to the framework."""

from typing import Any
from typing import Self

import numpy
from sklearn import base
from sklearn import preprocessing


class LabelEncodedClassifier(base.ClassifierMixin, base.BaseEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
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
        """Encodes the labels and fits the wrapped classifier.

        Args:
            features: Training features.
            target: Training labels, of any sortable type.

        Returns:
            This wrapper, fitted.
        """
        self.encoder_ = preprocessing.LabelEncoder().fit(target)
        self.classes_ = self.encoder_.classes_
        self.estimator_ = base.clone(self.estimator).fit(
            features, self.encoder_.transform(target)
        )
        return self

    def predict(self, features: Any) -> numpy.ndarray:
        """Predicts and decodes back to the original labels.

        Args:
            features: Rows to predict.

        Returns:
            Array of original class labels, one per row.
        """
        encoded = numpy.asarray(self.estimator_.predict(features)).astype(int)
        labels: numpy.ndarray = self.encoder_.inverse_transform(encoded)
        return labels

    def predict_proba(self, features: Any) -> numpy.ndarray:
        """Predicts class probabilities.

        Args:
            features: Rows to predict.

        Returns:
            Array with one row per input row and one column per class, in
            `classes_` order.
        """
        probabilities: numpy.ndarray = self.estimator_.predict_proba(features)
        return probabilities


class FlatPredictionClassifier(base.ClassifierMixin, base.BaseEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
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
        """Fits a clone of the wrapped classifier.

        Args:
            features: Training features.
            target: Training labels.

        Returns:
            This wrapper, fitted.
        """
        self.estimator_ = base.clone(self.estimator).fit(features, target)
        self.classes_ = numpy.asarray(self.estimator_.classes_).ravel()
        return self

    def predict(self, features: Any) -> numpy.ndarray:
        """Predicts, flattened to one dimension.

        Args:
            features: Rows to predict.

        Returns:
            One-dimensional array of predicted labels, one per row.
        """
        return numpy.asarray(self.estimator_.predict(features)).ravel()

    def predict_proba(self, features: Any) -> numpy.ndarray:
        """Predicts class probabilities.

        Args:
            features: Rows to predict.

        Returns:
            Array with one row per input row and one column per class.
        """
        probabilities: numpy.ndarray = self.estimator_.predict_proba(features)
        return probabilities


class FitParamsClassifier(base.ClassifierMixin, base.BaseEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
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
        """Fits a clone with the fixed arguments.

        Args:
            features: Training features.
            target: Training labels.

        Returns:
            This wrapper, fitted.
        """
        self.estimator_ = base.clone(self.estimator).fit(
            features, target, **self.fit_params
        )
        self.classes_ = numpy.asarray(self.estimator_.classes_)
        return self

    def predict(self, features: Any) -> numpy.ndarray:
        """Predicts class labels.

        Args:
            features: Rows to predict.

        Returns:
            Array of predicted labels, one per row.
        """
        return numpy.asarray(self.estimator_.predict(features))

    def predict_proba(self, features: Any) -> numpy.ndarray:
        """Predicts class probabilities.

        Args:
            features: Rows to predict.

        Returns:
            Array with one row per input row and one column per class.
        """
        probabilities: numpy.ndarray = self.estimator_.predict_proba(features)
        return probabilities


class FitParamsRegressor(base.RegressorMixin, base.BaseEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
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
        """Fits a clone with the fixed arguments.

        Args:
            features: Training features.
            target: Training values.

        Returns:
            This wrapper, fitted.
        """
        self.estimator_ = base.clone(self.estimator).fit(
            features, target, **self.fit_params
        )
        return self

    def predict(self, features: Any) -> numpy.ndarray:
        """Predicts values.

        Args:
            features: Rows to predict.

        Returns:
            Array of predicted values, one per row.
        """
        return numpy.asarray(self.estimator_.predict(features))


class MonotoneEstimator(base.BaseEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
    """Sets per-column monotonic signs once the input width is known.

    The preprocessing decides how many columns the model receives (one-hot
    encoding changes it), so the constraint vector can only be completed at
    fit time: the given signs cover the leading columns and every remaining
    column is unconstrained.

    Attributes:
        estimator: Unfitted model whose `parameter` takes the sign vector.
        parameter: Name of that parameter (set on the innermost estimator,
            beneath any wrappers).
        signs: Signs (+1, -1, 0) for the leading columns.
        estimator_: Fitted clone.
    """

    def __init__(
        self,
        estimator: base.BaseEstimator,
        parameter: str,
        signs: tuple[int, ...],
    ) -> None:
        """Stores the model, the parameter name and the leading signs.

        Args:
            estimator: Unfitted model whose `parameter` takes the sign
                vector.
            parameter: Name of that parameter, set on the innermost
                estimator beneath any wrappers.
            signs: Signs (+1, -1, 0) for the leading columns.
        """
        self.estimator = estimator
        self.parameter = parameter
        self.signs = signs

    def fit(self, features: Any, target: Any) -> Self:
        """Completes the sign vector for this input width and fits.

        Args:
            features: Training features; the number of columns sets the
                vector length.
            target: Training targets.

        Returns:
            This wrapper, fitted.
        """
        width = features.shape[1]
        vector = list(self.signs)[:width]
        vector += [0] * (width - len(vector))
        model = base.clone(self.estimator)
        path, node = [], model
        while getattr(node, "estimator", None) is not None:
            path.append("estimator")
            node = node.estimator
        model.set_params(**{"__".join([*path, self.parameter]): vector})
        self.estimator_ = model.fit(features, target)
        return self

    def predict(self, features: Any) -> numpy.ndarray:
        """Predicts with the constrained model.

        Args:
            features: Rows to predict.

        Returns:
            Array of predictions, one per row.
        """
        return numpy.asarray(self.estimator_.predict(features))


class MonotoneClassifier(base.ClassifierMixin, MonotoneEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
    """Monotonically constrained classifier."""

    @property
    def classes_(self) -> numpy.ndarray:
        """The class labels of the fitted model."""
        return numpy.asarray(self.estimator_.classes_)

    def predict_proba(self, features: Any) -> numpy.ndarray:
        """Predicts class probabilities with the constrained model.

        Args:
            features: Rows to predict.

        Returns:
            Array with one row per input row and one column per class, in
            `classes_` order.
        """
        return numpy.asarray(self.estimator_.predict_proba(features))


class MonotoneRegressor(base.RegressorMixin, MonotoneEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
    """Monotonically constrained regressor."""


def monotone(
    estimator: base.BaseEstimator, parameter: str, signs: tuple[int, ...]
) -> MonotoneEstimator:
    """Wraps an estimator so its monotonic parameter is set at fit time.

    Args:
        estimator: Unfitted model to constrain.
        parameter: Name of the parameter that takes the sign vector.
        signs: Signs (+1, -1, 0) for the leading columns.

    Returns:
        A monotone classifier if `estimator` is a classifier, otherwise a
        monotone regressor.
    """
    kind = (
        MonotoneClassifier
        if base.is_classifier(estimator)
        else MonotoneRegressor
    )
    return kind(estimator, parameter, signs)
