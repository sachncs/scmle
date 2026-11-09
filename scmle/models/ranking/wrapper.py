"""Carries the query column beside the features into a ranking model."""

from typing import Any
from typing import Self

import numpy
import pandas
from sklearn import base
from sklearn import pipeline


class QueryAwareEstimator(base.RegressorMixin, base.BaseEstimator):
    """Fits a ranking pipeline using the query column of the features frame.

    Rows are put in query order (a stable sort), the sizes of the consecutive
    query blocks are handed to the final model, and prediction needs no query
    column: scores come back in the rows' original order.

    Attributes:
        estimator: Unfitted pipeline whose last step is a ranker.
        query_column: Name of the query identifier column.
        preprocessing_: Fitted steps before the model.
        model_: Fitted ranker.
    """

    def __init__(self, estimator: pipeline.Pipeline, query_column: str) -> None:
        """Initializes the wrapper.

        Args:
            estimator: Unfitted pipeline whose last step is a ranker.
            query_column: Name of the query identifier column.
        """
        self.estimator = estimator
        self.query_column = query_column

    def fit(self, features: Any, target: Any) -> Self:
        """Fits preprocessing, then the ranker with query group sizes.

        Args:
            features: Frame holding the model inputs and the query column.
            target: Relevance grade per row.

        Returns:
            This fitted estimator.
        """
        frame = pandas.DataFrame(features).reset_index(drop=True)
        relevance = numpy.asarray(target)
        order = numpy.argsort(
            frame[self.query_column].astype(str).to_numpy(), kind="stable"
        )
        frame, relevance = frame.iloc[order], relevance[order]
        sizes = frame[self.query_column].astype(str).value_counts(sort=False)
        ordered = frame[self.query_column].astype(str).drop_duplicates()
        group = sizes.loc[ordered].to_numpy().tolist()
        steps = base.clone(self.estimator)
        self.preprocessing_ = steps[:-1]
        transformed = self.preprocessing_.fit_transform(frame, relevance)
        self.model_ = steps[-1].fit(transformed, relevance, group=group)
        return self

    def predict(self, features: Any) -> numpy.ndarray:
        """Ranking score per row, in the rows' order.

        Args:
            features: Frame with the same columns as at fit.

        Returns:
            Array of shape `(n_rows,)`; compare scores only within a query.
        """
        return self.model_.predict(self.preprocessing_.transform(features))
