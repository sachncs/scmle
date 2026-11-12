"""Carries the event indicator beside the features into a survival model."""

from typing import Any
from typing import cast
from typing import Self

import numpy
import pandas
from sklearn import base
from sklearn import pipeline


class EventAwareEstimator(base.RegressorMixin, base.BaseEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
    """Fits a survival pipeline using the event column of the features frame.

    The preprocessing steps never see the event indicator (they select their
    feature columns by name), so it cannot leak into the features. At fit the
    wrapper reads it from the frame and hands it to the final model; at
    prediction time it is not needed.

    Attributes:
        estimator: Unfitted pipeline whose last step is a survival model.
        event_column: Name of the 0/1 event column.
        preprocessing_: Fitted steps before the model.
        model_: Fitted survival model.
    """

    def __init__(self, estimator: pipeline.Pipeline, event_column: str) -> None:
        """Initializes the wrapper.

        Args:
            estimator: Pipeline of preprocessing then a survival model.
            event_column: Name of the 0/1 event column.
        """
        self.estimator = estimator
        self.event_column = event_column

    def fit(self, features: Any, target: Any) -> Self:
        """Fits preprocessing, then the model on duration and events.

        Args:
            features: Frame holding the model inputs and the event column.
            target: Observed duration per row.

        Returns:
            This fitted estimator.
        """
        frame = pandas.DataFrame(features)
        events = frame[self.event_column].to_numpy(dtype=float)
        steps = cast("Any", base.clone(self.estimator))
        self.preprocessing_ = steps[:-1]
        transformed = self.preprocessing_.fit_transform(
            frame, numpy.asarray(target)
        )
        self.model_ = steps[-1].fit(
            transformed, numpy.asarray(target, dtype=float), event=events
        )
        return self

    def predict(self, features: Any) -> numpy.ndarray:
        """Risk score per row; larger means the event is expected sooner.

        Args:
            features: Frame with the same columns as at fit.

        Returns:
            Array of shape `(n_rows,)` of risk scores.
        """
        risk: numpy.ndarray = self.model_.predict(
            self.preprocessing_.transform(features)
        )
        return risk

    def predict_survival(self, features: Any, times: Any) -> numpy.ndarray:
        """Probability of no event yet at each time, `(rows, len(times))`.

        Args:
            features: Frame with the same columns as at fit.
            times: Times at which to evaluate survival.

        Returns:
            Array of shape `(n_rows, len(times))` of survival probabilities.
        """
        survival: numpy.ndarray = self.model_.predict_survival(
            self.preprocessing_.transform(features), times
        )
        return survival
