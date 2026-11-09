"""Carries the other target columns beside the features into a model."""

from typing import Any
from typing import Self

import numpy
import pandas
from sklearn import base
from sklearn import pipeline


class MultiTargetEstimator(base.BaseEstimator):
    """Fits a pipeline whose last step predicts a matrix of targets.

    The framework splits off the first (primary) target; the others stay in
    the features frame (the preprocessing selects columns by name, so they
    never become features). At fit the wrapper joins the primary target to
    them; at prediction time none is needed.

    Attributes:
        estimator: Unfitted pipeline whose last step takes a target matrix.
        target_columns: Names of all target columns, in output order.
        preprocessing_: Fitted steps before the model.
        model_: Fitted multi-target model.
    """

    def __init__(
        self, estimator: pipeline.Pipeline, target_columns: tuple[str, ...]
    ) -> None:
        """Initializes the wrapper.

        Args:
            estimator: Unfitted pipeline whose last step takes a target matrix.
            target_columns: Names of all target columns, primary target first.
        """
        self.estimator = estimator
        self.target_columns = target_columns

    def fit(self, features: Any, target: Any) -> Self:
        """Fits preprocessing, then the model on all target columns.

        Args:
            features: Frame holding the model inputs and the non-primary target
                columns.
            target: Primary target, the first of `target_columns`.

        Returns:
            This fitted estimator.
        """
        frame = pandas.DataFrame(features)
        matrix = numpy.column_stack(
            [
                numpy.asarray(target),
                frame[list(self.target_columns[1:])].to_numpy(),
            ]
        )
        steps = base.clone(self.estimator)
        self.preprocessing_ = steps[:-1]
        transformed = self.preprocessing_.fit_transform(frame, matrix[:, 0])
        self.model_ = steps[-1].fit(transformed, matrix)
        return self

    def predict(self, features: Any) -> numpy.ndarray:
        """Predictions, one column per target.

        Args:
            features: Frame with the same columns as at fit.

        Returns:
            Array of shape `(n_rows, len(target_columns))`.
        """
        return self.model_.predict(self.preprocessing_.transform(features))


class MultiLabelEstimator(base.ClassifierMixin, MultiTargetEstimator):
    """Multi-label flavour: also provides per-label probabilities."""

    def predict_proba(self, features: Any) -> numpy.ndarray:
        """Probability of each label being present, one column per label.

        Args:
            features: Frame with the same columns as at fit.

        Returns:
            Array of shape `(n_rows, n_labels)` as returned by the fitted model.
        """
        return self.model_.predict_proba(
            self.preprocessing_.transform(features)
        )


class MultiOutputEstimator(base.RegressorMixin, MultiTargetEstimator):
    """Multi-output regression flavour."""
