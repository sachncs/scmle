"""Carries the treatment column beside the features into an uplift model."""

from typing import Any
from typing import Self

import numpy
import pandas
from sklearn import base
from sklearn import pipeline


class TreatmentAwareEstimator(base.RegressorMixin, base.BaseEstimator):
    """Fits an uplift pipeline using the treatment column of the frame.

    The preprocessing never sees the treatment (it selects columns by name),
    so it cannot leak into the features. At fit the wrapper reads it from the
    frame and hands it to the final model; at prediction time it is not
    needed because the estimate is "what if this row were treated".

    Attributes:
        estimator: Unfitted pipeline whose last step is an uplift model.
        treatment_column: Name of the 0/1 treatment column.
        preprocessing_: Fitted steps before the model.
        model_: Fitted uplift model.
    """

    def __init__(
        self, estimator: pipeline.Pipeline, treatment_column: str
    ) -> None:
        """Initializes the wrapper.

        Args:
            estimator: Unfitted pipeline whose last step is an uplift model.
            treatment_column: Name of the 0/1 treatment column.
        """
        self.estimator = estimator
        self.treatment_column = treatment_column

    def fit(self, features: Any, target: Any) -> Self:
        """Fits preprocessing, then the model on outcome and treatment.

        Args:
            features: Frame holding the model inputs and the treatment column.
            target: Outcome per row.

        Returns:
            This fitted estimator.
        """
        frame = pandas.DataFrame(features)
        arm = frame[self.treatment_column].to_numpy(dtype=int)
        steps = base.clone(self.estimator)
        self.preprocessing_ = steps[:-1]
        transformed = self.preprocessing_.fit_transform(
            frame, numpy.asarray(target)
        )
        self.model_ = steps[-1].fit(transformed, numpy.asarray(target), arm)
        return self

    def predict(self, features: Any) -> numpy.ndarray:
        """Estimated effect of treating each row.

        Args:
            features: Frame with the same columns as at fit.

        Returns:
            Array of shape `(n_rows,)` of estimated effects.
        """
        return self.model_.predict(self.preprocessing_.transform(features))
