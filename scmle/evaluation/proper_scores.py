"""Proper scores for probabilistic forecasts and the standard MASE.

The pinball loss at level `tau` is a proper score for the `tau` quantile. The
interval score of a central `1 - a` interval and the weighted interval score
(WIS) built from it (Bracher, Ray, Gneiting and Reich, arXiv:2005.12881)
reward narrow intervals that still contain the outcome. `crps_from_quantiles`
is twice the mean pinball loss over an equally spaced level grid, the usual
quantile-grid approximation of the continuous ranked probability score.
Propriety of the CRPS is quoted from Gneiting and Raftery (2007) from memory
and not re-verified here; the tests check the closed forms instead.

`mase` is the usual definition: the forecast's mean absolute error divided
by the in-sample mean absolute error of the seasonal naive forecast on the
training history, per series. `scmle.evaluation.forecast.mase` divides by the
no-change error on the same evaluation rows instead; the two differ, and only
this one is comparable with published values.
"""

from collections.abc import Sequence

import numpy as np

SCALE_FLOOR = 1e-12


def pinball_loss(
    truth: np.ndarray, forecast: np.ndarray, level: float
) -> float:
    """Mean pinball (quantile) loss of `forecast` as the `level` quantile."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between 0 and 1")
    gap = np.asarray(truth, dtype=float) - np.asarray(forecast, dtype=float)
    return float(np.mean(np.maximum(level * gap, (level - 1.0) * gap)))


def interval_score(
    truth: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> float:
    """Mean interval score of central `1 - alpha` intervals (lower is better).

    Width plus `2 / alpha` times the distance by which the truth falls
    outside.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    y = np.asarray(truth, dtype=float)
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    below = np.maximum(low - y, 0.0)
    above = np.maximum(y - high, 0.0)
    return float(np.mean((high - low) + 2.0 / alpha * (below + above)))


def weighted_interval_score(
    truth: np.ndarray,
    median: np.ndarray,
    intervals: dict[float, tuple[np.ndarray, np.ndarray]],
) -> float:
    """Weighted interval score over several central intervals.

    `WIS = (0.5 * |y - m| + sum_k (alpha_k / 2) * IS_k) / (K + 0.5)`.

    Args:
        truth: Observed values.
        median: Median forecast.
        intervals: `alpha -> (lower, upper)` for each central `1 - alpha`
            interval.

    Raises:
        ValueError: With no interval.
    """
    if not intervals:
        raise ValueError("need at least one interval")
    y = np.asarray(truth, dtype=float)
    total = 0.5 * float(np.mean(np.abs(y - np.asarray(median, dtype=float))))
    for alpha, (lower, upper) in intervals.items():
        total += alpha / 2.0 * interval_score(y, lower, upper, alpha)
    return total / (len(intervals) + 0.5)


def crps_from_quantiles(
    truth: np.ndarray, quantiles: np.ndarray, levels: Sequence[float]
) -> float:
    """CRPS approximated as twice the mean pinball loss over the grid.

    Args:
        truth: Observed values, length `rows`.
        quantiles: `(rows, len(levels))` forecast quantiles.
        levels: The probability levels of the columns, in (0, 1).
    """
    values = np.asarray(quantiles, dtype=float)
    grid = list(levels)
    if values.shape[1] != len(grid):
        raise ValueError("one column per level is required")
    return 2.0 * float(
        np.mean(
            [
                pinball_loss(truth, values[:, index], level)
                for index, level in enumerate(grid)
            ]
        )
    )


def insample_scale(history: np.ndarray, season: int = 1) -> float:
    """Mean absolute seasonal-naive error on the training history."""
    series = np.asarray(history, dtype=float)
    if season < 1 or len(series) <= season:
        raise ValueError("history must be longer than the season")
    return float(np.mean(np.abs(series[season:] - series[:-season])))


def mase(
    truth: np.ndarray,
    forecast: np.ndarray,
    history: np.ndarray,
    season: int = 1,
) -> float:
    """Mean absolute scaled error with the in-sample naive scale.

    Args:
        truth: Observed values in the forecast window.
        forecast: Forecasts for the same window.
        history: Training values of the same series.
        season: Seasonal period of the naive benchmark (1 is no-change).
    """
    scale = max(insample_scale(history, season), SCALE_FLOOR)
    gap = np.abs(np.asarray(truth, dtype=float) - np.asarray(forecast, float))
    return float(np.mean(gap) / scale)
