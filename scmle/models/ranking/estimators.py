"""Ranking estimators: within a query, better rows get higher scores.

Every estimator follows one contract: `fit(features, relevance, group=None)`
where `group` lists how many consecutive rows belong to each query (a single
query when omitted), and `predict` returns one score per row. Scores are only
meaningful within a query.
"""

from typing import Any
from typing import cast
from typing import Self

import numpy
from sklearn import base
from sklearn import ensemble


class ConstantRanker(base.RegressorMixin, base.BaseEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
    """Gives every row the same score: the order carries no information."""

    def fit(self, features: Any, target: Any, group: Any = None) -> Self:  # noqa: ARG002 -- fit signature of the ranking estimators
        """Nothing to learn.

        Args:
            features: Training matrix.
            target: Relevance grade per row.
            group: Number of consecutive rows in each query; None means all rows
                form one query.

        Returns:
            This ranker, unchanged.
        """
        return self

    def predict(self, features: Any) -> numpy.ndarray:
        """Zero for every row.

        Args:
            features: Matrix to score.

        Returns:
            Array of zeros of shape `(n_rows,)`.
        """
        return numpy.zeros(features.shape[0])


class PointwiseRanker(base.RegressorMixin, base.BaseEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
    """Ranks by a regressor's prediction of the relevance grade.

    Attributes:
        regressor: Unfitted regressor.
        regressor_: Fitted clone.
    """

    def __init__(self, regressor: base.BaseEstimator | None = None) -> None:
        """Initializes the ranker (gradient boosting by default).

        Args:
            regressor: Unfitted regressor; None uses gradient boosting.
        """
        self.regressor = regressor

    def fit(self, features: Any, target: Any, group: Any = None) -> Self:  # noqa: ARG002 -- fit signature of the ranking estimators
        """Fits the regressor on relevance grades.

        The query groups are not used.

        Args:
            features: Training matrix.
            target: Relevance grade per row.
            group: Number of consecutive rows in each query; None means all rows
                form one query.

        Returns:
            This fitted ranker.
        """
        model = self.regressor or ensemble.HistGradientBoostingRegressor()
        self.regressor_ = cast("Any", base.clone(model)).fit(
            features, numpy.asarray(target, dtype=float)
        )
        return self

    def predict(self, features: Any) -> numpy.ndarray:
        """Predicted relevance.

        Args:
            features: Matrix to score.

        Returns:
            Array of shape `(n_rows,)` of predicted grades.
        """
        return numpy.asarray(self.regressor_.predict(features), dtype=float)


class LibraryRanker(base.RegressorMixin, base.BaseEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
    """A boosted ranker from LightGBM or XGBoost that learns the ordering.

    Attributes:
        library: `lightgbm` or `xgboost`.
        learning_rate: Boosting step.
        max_leaves: Tree size.
        random_state: Seed.
    """

    def __init__(
        self,
        library: str = "lightgbm",
        learning_rate: float = 0.1,
        max_leaves: int = 31,
        random_state: int = 0,
    ) -> None:
        """Stores the settings (scikit-learn convention).

        Args:
            library: `lightgbm` or `xgboost`.
            learning_rate: Boosting step.
            max_leaves: Tree size.
            random_state: Seed.
        """
        self.library = library
        self.learning_rate = learning_rate
        self.max_leaves = max_leaves
        self.random_state = random_state

    def build(self) -> Any:
        """Creates the underlying library ranker from the settings.

        Returns:
            An unfitted LightGBM LambdaRank model, or an unfitted XGBoost NDCG
            ranker for any other `library` value.
        """
        if self.library == "lightgbm":
            import lightgbm  # noqa: PLC0415 -- optional dependency

            return lightgbm.LGBMRanker(
                objective="lambdarank",
                learning_rate=self.learning_rate,
                num_leaves=self.max_leaves,
                random_state=self.random_state,
                n_jobs=1,
                verbose=-1,
            )
        import xgboost  # noqa: PLC0415 -- optional dependency

        return xgboost.XGBRanker(
            objective="rank:ndcg",
            learning_rate=self.learning_rate,
            max_leaves=self.max_leaves,
            grow_policy="lossguide",
            tree_method="hist",
            random_state=self.random_state,
            n_jobs=1,
        )

    def fit(self, features: Any, target: Any, group: Any = None) -> Self:
        """Trains on relevance grades with the query groups.

        Args:
            features: Training matrix.
            target: Relevance grade per row.
            group: Number of consecutive rows in each query; None means all rows
                form one query.

        Returns:
            This fitted ranker.
        """
        sizes = [features.shape[0]] if group is None else group
        self.model_ = self.build().fit(
            features, numpy.asarray(target).astype(int), group=list(sizes)
        )
        return self

    def predict(self, features: Any) -> numpy.ndarray:
        """Ranking scores.

        Args:
            features: Matrix to score.

        Returns:
            Array of shape `(n_rows,)`; compare scores only within a query.
        """
        return numpy.asarray(self.model_.predict(features), dtype=float)
