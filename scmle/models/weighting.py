"""Per-row sample weights taken from a column of the training data."""

import inspect
from typing import Any
from typing import Self

import numpy
import pandas
from sklearn import base
from sklearn import pipeline


class WeightedEstimator(base.BaseEstimator):
    """Fits a pipeline so each training row counts by its weight.

    The weight column travels with the features, is read at fit time and
    ignored at prediction time (so inference never needs it). Models whose
    `fit` accepts `sample_weight` receive the weights directly. For the
    others (deep networks, wrapped estimators) rows are drawn with
    probability proportional to weight, which gives the same expected
    training distribution.

    Attributes:
        estimator: Unfitted pipeline whose final step is named `model`.
        column: Name of the weight column in the training features.
        random_state: Seed for the weighted draw.
        estimator_: Fitted clone of `estimator`.
    """

    def __init__(
        self,
        estimator: base.BaseEstimator,
        column: str,
        random_state: int = 0,
    ) -> None:
        """Initializes the wrapper.

        Args:
            estimator: Pipeline to fit with weights.
            column: Weight column name.
            random_state: Seed for the weighted draw.
        """
        self.estimator = estimator
        self.column = column
        self.random_state = random_state

    @property
    def event_column(self) -> str:
        """The wrapped survival pipeline's event column name."""
        return self.estimator.event_column

    @property
    def treatment_column(self) -> str:
        """The wrapped uplift pipeline's treatment column."""
        return self.estimator.treatment_column

    @property
    def value_column(self) -> str:
        """The wrapped forecasting pipeline's value column."""
        return self.estimator.value_column

    @property
    def target_columns(self) -> tuple[str, ...]:
        """The wrapped multi-target pipeline's target columns."""
        return self.estimator.target_columns

    @property
    def query_column(self) -> str:
        """The wrapped ranking pipeline's query column."""
        return self.estimator.query_column

    def accepts_weights(self, candidate: base.BaseEstimator) -> bool:
        """Checks whether a pipeline's model step takes `sample_weight`.

        Args:
            candidate: Estimator to inspect.

        Returns:
            True if `candidate` is a pipeline with a step named `model`
            whose `fit` accepts `sample_weight`; False otherwise.
        """
        if not isinstance(candidate, pipeline.Pipeline):
            return False
        model = candidate.named_steps.get("model")
        if model is None:
            return False
        return "sample_weight" in inspect.signature(model.fit).parameters

    def fit(self, features: Any, target: Any) -> Self:
        """Fits a clone of the pipeline weighted by the weight column.

        When the pipeline's `model` step takes `sample_weight`, the weights are
        passed to it. Otherwise rows are drawn with replacement, with
        probability proportional to weight, and the pipeline is fitted on the
        draw. The weight column stays in the features given to the pipeline.

        Args:
            features: Training features, convertible to a data frame, that
                include the weight column.
            target: Training labels, one per row.

        Returns:
            This estimator, fitted. The fitted pipeline is `estimator_`.
        """
        frame = pandas.DataFrame(features).reset_index(drop=True)
        labels = pandas.Series(numpy.asarray(target))
        weights = frame[self.column].to_numpy(dtype=float)
        candidate = base.clone(self.estimator)
        if self.accepts_weights(candidate):
            candidate.fit(frame, labels, model__sample_weight=weights)
        else:
            generator = numpy.random.default_rng(self.random_state)
            drawn = generator.choice(
                len(frame), size=len(frame), p=weights / weights.sum()
            )
            candidate.fit(frame.iloc[drawn], labels.iloc[drawn])
        self.estimator_ = candidate
        return self

    def predict(self, features: Any) -> numpy.ndarray:
        """Predicts with the fitted pipeline.

        Args:
            features: Rows to predict.

        Returns:
            The fitted pipeline's predictions, one per row.
        """
        return self.estimator_.predict(features)

    def predict_survival(self, features: Any, times: Any) -> numpy.ndarray:
        """Predicts survival curves with the fitted survival pipeline.

        Args:
            features: Rows to predict.
            times: Time points at which to evaluate each curve.

        Returns:
            The fitted pipeline's survival probabilities, one row per input
            row and one column per time point.
        """
        return self.estimator_.predict_survival(features, times)


class WeightedClassifier(base.ClassifierMixin, WeightedEstimator):
    """Weighted fitting for classifiers."""

    def fit(self, features: Any, target: Any) -> Self:
        """Fits the pipeline and exposes its class labels.

        Args:
            features: Training features that include the weight column.
            target: Training class labels, one per row.

        Returns:
            This classifier, fitted, with `classes_` set.
        """
        super().fit(features, target)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, features: Any) -> numpy.ndarray:
        """Predicts class probabilities.

        Args:
            features: Rows to predict.

        Returns:
            Array with one row per input row and one column per class, in
            `classes_` order.
        """
        return self.estimator_.predict_proba(features)


class WeightedRegressor(base.RegressorMixin, WeightedEstimator):
    """Weighted fitting for regressors."""


def weighted(
    estimator: base.BaseEstimator, column: str, seed: int
) -> WeightedEstimator:
    """Wraps a candidate so rows are weighted by `column`.

    Args:
        estimator: Pipeline to wrap.
        column: Name of the weight column in the training features.
        seed: Seed for the weighted draw used when the model has no
            `sample_weight` support.

    Returns:
        A weighted classifier if `estimator` is a classifier, otherwise a
        weighted regressor.
    """
    kind = (
        WeightedClassifier
        if base.is_classifier(estimator)
        else WeightedRegressor
    )
    return kind(estimator, column, seed)
