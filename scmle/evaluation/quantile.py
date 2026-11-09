"""Conformalized quantile regression: intervals that widen where data is noisy.

Two boosted models learn a low and a high conditional quantile of the target.
Their gap already adapts to the row (wide where outcomes vary, narrow where
they do not). Out-of-fold rows then measure how far the true value falls
outside that band, and the finite-sample quantile of those misses widens (or
tightens) the band so a new exchangeable row lands inside it with at least the
requested probability. Compare `ConformalCalibrator`, whose margin is the same
width everywhere.
"""

from collections.abc import Sequence
import dataclasses
from typing import Any

import numpy as np
import pandas as pd

from scmle.evaluation import conformal

LOWER_LEVEL = 0.05
UPPER_LEVEL = 0.95


def predicted_band(
    lower_model: Any, upper_model: Any, features: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Uncalibrated band with the lower edge never above the upper.

    Separately fitted quantile models can cross; the edges are swapped back.

    Args:
        lower_model: Fitted model of the low quantile.
        upper_model: Fitted model of the high quantile.
        features: Rows to predict for.

    Returns:
        The lower and upper edges: the elementwise minimum and maximum of
        the two predictions.
    """
    low = np.asarray(lower_model.predict(features), dtype=float)
    high = np.asarray(upper_model.predict(features), dtype=float)
    return np.minimum(low, high), np.maximum(low, high)


@dataclasses.dataclass(frozen=True)
class QuantileBand:
    """Out-of-fold misses of the quantile band.

    Attributes:
        scores: For each held-out row, how far the true value lies outside
            the band (negative when inside, by the distance to the nearer
            edge).
    """

    scores: tuple[float, ...]

    @classmethod
    def calibrate(
        cls,
        observed: Sequence[float],
        lower: Sequence[float],
        upper: Sequence[float],
    ) -> "QuantileBand":
        """Scores held-out rows against their predicted band.

        Args:
            observed: True values of held-out rows.
            lower: Lower edge of the predicted band for those rows.
            upper: Upper edge of the predicted band for those rows.

        Returns:
            The band holding each row's miss: the larger of lower minus
            value and value minus upper.

        Raises:
            ValueError: If there are fewer rows than
                `ConformalCalibrator.checked` requires.
        """
        values = np.asarray(observed, dtype=float)
        low = np.asarray(lower, dtype=float)
        high = np.asarray(upper, dtype=float)
        misses = np.maximum(low - values, values - high)
        return cls(conformal.ConformalCalibrator.checked(misses))

    def margin(self, coverage: float) -> float:
        """Widening (or tightening, when negative) that gives `coverage`.

        Args:
            coverage: Desired coverage, strictly between 0 and 1.

        Returns:
            The finite-sample quantile of the misses; infinity when there are
            too few rows for `coverage`.

        Raises:
            ValueError: If `coverage` is not strictly between 0 and 1.
        """
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must lie strictly between 0 and 1")
        return conformal.finite_sample_quantile(
            np.asarray(self.scores), coverage
        )


@dataclasses.dataclass(frozen=True)
class QuantileIntervals:
    """Fitted low and high quantile models plus their calibration.

    Attributes:
        lower_model: Pipeline predicting the low conditional quantile.
        upper_model: Pipeline predicting the high conditional quantile.
        band: Calibration misses.
    """

    lower_model: Any
    upper_model: Any
    band: QuantileBand

    def interval(
        self, features: pd.DataFrame, coverage: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calibrated lower and upper bounds for each row.

        Args:
            features: Rows to predict for.
            coverage: Desired coverage, strictly between 0 and 1; an invalid
                value is rejected by `QuantileBand.margin` with a
                `ValueError`.

        Returns:
            The raw band widened by the margin on each side.
        """
        low, high = self.raw_band(features)
        margin = self.band.margin(coverage)
        return low - margin, high + margin

    def coverage_curve(
        self,
        features: pd.DataFrame,
        target: Any,
        levels: Sequence[float],
    ) -> list[dict[str, float]]:
        """Share of held-out rows inside the interval at each nominal level.

        Args:
            features: Rows the models did not train on.
            target: Their true values.
            levels: Nominal coverages in (0, 1).

        Returns:
            For each level: nominal and empirical coverage, mean width.
        """
        values = np.asarray(target, dtype=float)
        curve = []
        for level in levels:
            low, high = self.interval(features, level)
            curve.append(
                {
                    "nominal": float(level),
                    "empirical": float(
                        np.mean((values >= low) & (values <= high))
                    ),
                    "mean_width": float(np.mean(high - low)),
                }
            )
        return curve

    def raw_band(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """The model band before calibration.

        Args:
            features: Rows to predict for.

        Returns:
            The lower and upper edges from `predicted_band`.
        """
        return predicted_band(self.lower_model, self.upper_model, features)
