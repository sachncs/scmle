"""Ranking estimators: within a query, better rows get higher scores.

Every estimator follows one contract: `fit(features, relevance, group=None)`
where `group` lists how many consecutive rows belong to each query (a single
query when omitted), and `predict` returns one score per row. Scores are only
meaningful within a query.
"""

import math
from typing import Any
from typing import cast
from typing import Self

import numpy
from sklearn import base
from sklearn import ensemble
from sklearn import linear_model

from scmle.contracts import errors

PAIRS = 50_000
"""Most preference pairs the pairwise ranker trains on. The logistic fit's
coefficient error shrinks as `1 / sqrt(pairs)`, so 50,000 pairs leave about
0.5% of a standardised coefficient's scale, while the doubled difference
matrix stays under 100 MB at 120 features."""


class Constant(base.RegressorMixin, base.BaseEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
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


class Scorer(base.RegressorMixin, base.BaseEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
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


class Library(base.RegressorMixin, base.BaseEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
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
            import lightgbm  # noqa: PLC0415 -- optional dependency  # pyright: ignore[reportMissingImports]  # optional extra, imported lazily and guarded; not installed in the type-check environment

            return lightgbm.LGBMRanker(
                objective="lambdarank",
                learning_rate=self.learning_rate,
                num_leaves=self.max_leaves,
                random_state=self.random_state,
                n_jobs=1,
                verbose=-1,
            )
        import xgboost  # noqa: PLC0415 -- optional dependency  # pyright: ignore[reportMissingImports]  # optional extra, imported lazily and guarded; not installed in the type-check environment

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


def sizes(group: Any, rows: int) -> numpy.ndarray:
    """Query sizes that cover the rows.

    Args:
        group: Number of consecutive rows in each query; None means one query.
        rows: Rows in the training matrix.

    Returns:
        The sizes as integers.

    Raises:
        InputError: If the sizes do not add up to the rows.
    """
    found = numpy.asarray([rows] if group is None else group, dtype=int)
    if found.sum() != rows or (found < 0).any():
        raise errors.InputError(
            f"group sizes add up to {int(found.sum())}, not the {rows} rows"
        )
    return found


class Pairwise(base.RegressorMixin, base.BaseEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
    """A linear ranker learnt from preference pairs (RankSVM, logistic loss).

    Within each query, every pair of rows with different grades says which
    should rank higher. A logistic regression without intercept on the
    feature differences learns weights `w` so that `w @ (x_i - x_j) > 0`
    when row `i` is the more relevant; the score is `w @ x`. Pairs are drawn
    at random when a query has more than its share of `pairs`.

    Attributes:
        penalty: Ridge strength on the weights (the inverse of `C`).
        pairs: Most preference pairs to train on.
        random_state: Seed of the pair sampling.
        coefficients_: Fitted weights.
    """

    def __init__(
        self, penalty: float = 1.0, pairs: int = PAIRS, random_state: int = 0
    ) -> None:
        """Stores the settings (scikit-learn convention).

        Args:
            penalty: Ridge strength on the weights.
            pairs: Most preference pairs to train on.
            random_state: Seed of the pair sampling.
        """
        self.penalty = penalty
        self.pairs = pairs
        self.random_state = random_state

    def couple(
        self, grades: numpy.ndarray, group: Any
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Preference pairs within each query.

        A query gets the share of `pairs` its pair count holds; a query
        within its share gives every pair, a larger one a uniform draw.

        Args:
            grades: Relevance grade of each row.
            group: Size of each query, in row order.

        Returns:
            Row positions of the first and second row of each pair; the
            grades of the two always differ.
        """
        counts = numpy.asarray(group, dtype=int)
        starts = numpy.concatenate([[0], numpy.cumsum(counts)[:-1]])
        total = int((counts * (counts - 1) // 2).sum())
        share = min(1.0, self.pairs / total) if total else 1.0
        rng = numpy.random.default_rng(self.random_state)
        firsts, seconds = [numpy.empty(0, int)], [numpy.empty(0, int)]
        for start, size in zip(starts, counts, strict=True):
            available = size * (size - 1) // 2
            quota = math.ceil(available * share)
            if quota >= available:
                left, right = numpy.triu_indices(size, 1)
            else:
                left = rng.integers(0, size, quota)
                right = rng.integers(0, size, quota)
            keep = grades[start + left] != grades[start + right]
            firsts.append(start + left[keep])
            seconds.append(start + right[keep])
        return numpy.concatenate(firsts), numpy.concatenate(seconds)

    def fit(self, features: Any, target: Any, group: Any = None) -> Self:
        """Learns the weights from the preference pairs.

        Args:
            features: Training matrix (scaled; no missing values).
            target: Relevance grade per row.
            group: Number of consecutive rows in each query; None means all
                rows form one query.

        Returns:
            This fitted ranker; zero weights when no pair differs in grade.
        """
        matrix = numpy.asarray(features, dtype=float)
        grades = numpy.asarray(target, dtype=float)
        first, second = self.couple(grades, sizes(group, len(matrix)))
        if not len(first):
            self.coefficients_ = numpy.zeros(matrix.shape[1])
            return self
        difference = matrix[first] - matrix[second]
        better = grades[first] > grades[second]
        model = linear_model.LogisticRegression(
            C=1.0 / self.penalty, fit_intercept=False, max_iter=1000
        ).fit(
            numpy.vstack([difference, -difference]),
            numpy.concatenate([better, ~better]).astype(int),
        )
        self.coefficients_ = model.coef_[0]
        return self

    def predict(self, features: Any) -> numpy.ndarray:
        """Linear ranking score.

        Args:
            features: Matrix to score.

        Returns:
            Array of shape `(n_rows,)`; compare scores only within a query.
        """
        score: numpy.ndarray = (
            numpy.asarray(features, dtype=float) @ self.coefficients_
        )
        return score
