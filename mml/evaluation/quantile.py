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

from mml.evaluation import conformal

LOWER_LEVEL = 0.05
UPPER_LEVEL = 0.95


def predicted_band(
    lower_model: Any, upper_model: Any, features: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Uncalibrated band with the lower edge never above the upper.

    Separately fitted quantile models can cross; the edges are swapped back.
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

        Raises:
            ValueError: With too few rows.
        """
        values = np.asarray(observed, dtype=float)
        low = np.asarray(lower, dtype=float)
        high = np.asarray(upper, dtype=float)
        misses = np.maximum(low - values, values - high)
        return cls(conformal.ConformalCalibrator.checked(misses))

    def margin(self, coverage: float) -> float:
        """Widening (or tightening, when negative) that gives `coverage`."""
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
        """Calibrated lower and upper bounds for each row."""
        low, high = self.raw_band(features)
        margin = self.band.margin(coverage)
        return low - margin, high + margin

    def raw_band(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """The model band before calibration."""
        return predicted_band(self.lower_model, self.upper_model, features)
