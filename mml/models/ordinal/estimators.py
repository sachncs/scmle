"""A regressor turned into an ordinal classifier with learned thresholds.

The target arrives as the integer rank of each level. A regressor learns a
latent score; cut points on that score separate the levels, and a logistic
link around the cut points turns scores into level probabilities. Levels are
ordered, so the model is judged by how far its guesses fall from the truth,
not only by whether they are exact.
"""

from typing import Any
from typing import Self

import numpy as np
from scipy import special
from sklearn import base
from sklearn import model_selection

MIN_ROWS_FOR_HONEST_SCORES = 30
CROSS_FITS = 3
SCALE_GRID = np.geomspace(0.05, 5.0, 40)
EPSILON = 1e-12


class CumulativeThresholdRegressor(base.RegressorMixin, base.BaseEstimator):
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
        """Initializes the model."""
        self.regressor = regressor
        self.levels = levels
        self.random_state = random_state

    def fit(self, features: Any, target: Any) -> Self:
        """Fits the latent regressor, then places thresholds and the link.

        Thresholds are set on out-of-fold scores (the model has not seen the
        row it scores), so an overfit regressor cannot squeeze the levels
        together.
        """
        ranks = np.asarray(target, dtype=float).round().astype(int)
        count = self.levels or int(ranks.max()) + 1
        self.classes_ = np.arange(count)
        self.regressor_ = base.clone(self.regressor).fit(features, ranks)
        if len(ranks) >= MIN_ROWS_FOR_HONEST_SCORES:
            folds = model_selection.KFold(
                CROSS_FITS, shuffle=True, random_state=self.random_state
            )
            scores = model_selection.cross_val_predict(
                base.clone(self.regressor), features, ranks, cv=folds
            )
        else:
            scores = self.regressor_.predict(features)
        scores = np.asarray(scores, dtype=float)
        cumulative = np.cumsum(np.bincount(ranks, minlength=count))[:-1]
        shares = np.clip(cumulative / len(ranks), EPSILON, 1 - EPSILON)
        self.thresholds_ = np.maximum.accumulate(np.quantile(scores, shares))
        self.scale_ = self.best_scale(scores, ranks)
        return self

    def level_probabilities(
        self, scores: np.ndarray, scale: float
    ) -> np.ndarray:
        """Probability of each level for latent `scores`."""
        edges = np.concatenate([[-np.inf], self.thresholds_, [np.inf]])
        below = special.expit((edges[None, :] - scores[:, None]) / scale)
        return np.clip(np.diff(below, axis=1), EPSILON, None)

    def best_scale(self, scores: np.ndarray, ranks: np.ndarray) -> float:
        """The link width with the highest likelihood on the fitted scores."""
        spread = float(np.std(scores)) or 1.0
        best, best_value = spread, -np.inf
        for factor in SCALE_GRID:
            scale = factor * spread
            probabilities = self.level_probabilities(scores, scale)
            value = float(
                np.log(probabilities[np.arange(len(ranks)), ranks]).mean()
            )
            if value > best_value:
                best, best_value = scale, value
        return best

    def predict_proba(self, features: Any) -> np.ndarray:
        """Level probabilities, columns in level order."""
        scores = np.asarray(self.regressor_.predict(features), dtype=float)
        probabilities = self.level_probabilities(scores, self.scale_)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, features: Any) -> np.ndarray:
        """Level rank: the number of thresholds the latent score has passed."""
        scores = np.asarray(self.regressor_.predict(features), dtype=float)
        return np.searchsorted(self.thresholds_, scores, side="right")
