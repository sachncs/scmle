"""A regressor turned into an ordinal classifier with learned thresholds.

The target arrives as the integer rank of each level. A regressor learns a
latent score; cut points on that score separate the levels, and a logistic
link around the cut points turns scores into level probabilities. Levels are
ordered, so the model is judged by how far its guesses fall from the truth,
not only by whether they are exact.
"""

from typing import Any
from typing import cast
from typing import Self

import numpy
from scipy import special
from sklearn import base
from sklearn import model_selection

MIN_ROWS_FOR_HONEST_SCORES = 30
CROSS_FITS = 3
SCALE_GRID = numpy.geomspace(0.05, 5.0, 40)
EPSILON = 1e-12


class CumulativeThresholdRegressor(base.RegressorMixin, base.BaseEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
    """Ordinal classifier built from any regressor.

    Attributes:
        regressor: Unfitted scikit-learn regressor producing the latent score.
        levels: Number of ordered levels; inferred from the target if None
            (pass it when a training fold may miss the top level).
        random_state: Seed for the cross-fitting that places the thresholds.
        classes_: Ranks `0..levels-1`.
        thresholds_: Cut points between consecutive levels.
        scale_: Width of the logistic link around the cut points.
        regressor_: Fitted regressor.
    """

    def __init__(
        self,
        regressor: base.BaseEstimator,
        levels: int | None = None,
        random_state: int = 0,
    ) -> None:
        """Initializes the model.

        Args:
            regressor: Unfitted regressor producing the latent score.
            levels: Number of ordered levels; inferred from the target if None.
            random_state: Seed for the cross-fitting that places the thresholds.
        """
        self.regressor = regressor
        self.levels = levels
        self.random_state = random_state

    def fit(self, features: Any, target: Any) -> Self:
        """Fits the latent regressor, then places thresholds and the link.

        Thresholds are set on out-of-fold scores (the model has not seen the
        row it scores), so an overfit regressor cannot squeeze the levels
        together.

        Args:
            features: Training matrix.
            target: Integer rank of each row's level, `0..levels-1`.

        Returns:
            This fitted model.
        """
        ranks = numpy.asarray(target, dtype=float).round().astype(int)
        count = self.levels or int(ranks.max()) + 1
        self.classes_ = numpy.arange(count)
        self.regressor_ = cast("Any", base.clone(self.regressor)).fit(
            features, ranks
        )
        if len(ranks) >= MIN_ROWS_FOR_HONEST_SCORES:
            folds = model_selection.KFold(
                CROSS_FITS, shuffle=True, random_state=self.random_state
            )
            scores = model_selection.cross_val_predict(
                base.clone(self.regressor), features, ranks, cv=folds
            )
        else:
            scores = self.regressor_.predict(features)
        scores = numpy.asarray(scores, dtype=float)
        cumulative = numpy.cumsum(numpy.bincount(ranks, minlength=count))[:-1]
        shares = numpy.clip(cumulative / len(ranks), EPSILON, 1 - EPSILON)
        self.thresholds_ = numpy.maximum.accumulate(
            numpy.quantile(scores, shares)
        )
        self.scale_ = self.best_scale(scores, ranks)
        return self

    def level_probabilities(
        self, scores: numpy.ndarray, scale: float
    ) -> numpy.ndarray:
        """Probability of each level for latent `scores`.

        Args:
            scores: Latent scores of shape `(n_rows,)`.
            scale: Width of the logistic link around the thresholds.

        Returns:
            Array of shape `(n_rows, levels)`, floored at a tiny positive value;
            rows are not renormalised.
        """
        edges = numpy.concatenate([[-numpy.inf], self.thresholds_, [numpy.inf]])
        below = special.expit((edges[None, :] - scores[:, None]) / scale)
        clipped: numpy.ndarray = numpy.clip(
            numpy.diff(below, axis=1), EPSILON, None
        )
        return clipped

    def best_scale(self, scores: numpy.ndarray, ranks: numpy.ndarray) -> float:
        """The link width with the highest likelihood on the fitted scores.

        Args:
            scores: Latent scores of shape `(n_rows,)`.
            ranks: True integer level of each row.

        Returns:
            The scale from a fixed grid, in multiples of the score spread, with
            the highest mean log-likelihood of the true levels.
        """
        spread = float(numpy.std(scores)) or 1.0
        best, best_value = spread, -numpy.inf
        for factor in SCALE_GRID:
            scale = factor * spread
            probabilities = self.level_probabilities(scores, scale)
            value = float(
                numpy.log(probabilities[numpy.arange(len(ranks)), ranks]).mean()
            )
            if value > best_value:
                best, best_value = scale, value
        return best

    def predict_proba(self, features: Any) -> numpy.ndarray:
        """Level probabilities, columns in level order.

        Args:
            features: Matrix to predict for.

        Returns:
            Array of shape `(n_rows, levels)` whose rows sum to one.
        """
        scores = numpy.asarray(self.regressor_.predict(features), dtype=float)
        probabilities = self.level_probabilities(scores, self.scale_)
        normalized: numpy.ndarray = probabilities / probabilities.sum(
            axis=1, keepdims=True
        )
        return normalized

    def predict(self, features: Any) -> numpy.ndarray:
        """Level rank: the number of thresholds the latent score has passed.

        Args:
            features: Matrix to predict for.

        Returns:
            Integer array of shape `(n_rows,)` with values in `0..levels-1`.
        """
        scores = numpy.asarray(self.regressor_.predict(features), dtype=float)
        return numpy.searchsorted(self.thresholds_, scores, side="right")
