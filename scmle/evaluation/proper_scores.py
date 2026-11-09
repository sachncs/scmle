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

import numpy

SCALE_FLOOR = 1e-12


def pinball_loss(
    truth: numpy.ndarray, forecast: numpy.ndarray, level: float
) -> float:
    """Mean pinball (quantile) loss of `forecast` as the `level` quantile.

    Args:
        truth: Observed values.
        forecast: Forecast of the `level` quantile, one per observation.
        level: Quantile level, strictly between 0 and 1.

    Returns:
        The mean loss; lower is better.

    Raises:
        ValueError: If `level` is not strictly between 0 and 1.
    """
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between 0 and 1")
    gap = numpy.asarray(truth, dtype=float) - numpy.asarray(
        forecast, dtype=float
    )
    return float(numpy.mean(numpy.maximum(level * gap, (level - 1.0) * gap)))


def interval_score(
    truth: numpy.ndarray,
    lower: numpy.ndarray,
    upper: numpy.ndarray,
    alpha: float,
) -> float:
    """Mean interval score of central `1 - alpha` intervals (lower is better).

    Width plus `2 / alpha` times the distance by which the truth falls
    outside.

    Args:
        truth: Observed values.
        lower: Lower limits of the intervals.
        upper: Upper limits of the intervals.
        alpha: Miscoverage level, strictly between 0 and 1.

    Returns:
        The mean interval score.

    Raises:
        ValueError: If `alpha` is not strictly between 0 and 1.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    y = numpy.asarray(truth, dtype=float)
    low = numpy.asarray(lower, dtype=float)
    high = numpy.asarray(upper, dtype=float)
    below = numpy.maximum(low - y, 0.0)
    above = numpy.maximum(y - high, 0.0)
    return float(numpy.mean((high - low) + 2.0 / alpha * (below + above)))


def weighted_interval_score(
    truth: numpy.ndarray,
    median: numpy.ndarray,
    intervals: dict[float, tuple[numpy.ndarray, numpy.ndarray]],
) -> float:
    """Weighted interval score over several central intervals.

    `WIS = (0.5 * |y - m| + sum_k (alpha_k / 2) * IS_k) / (K + 0.5)`.

    Args:
        truth: Observed values.
        median: Median forecast.
        intervals: `alpha -> (lower, upper)` for each central `1 - alpha`
            interval.

    Returns:
        The weighted interval score; lower is better.

    Raises:
        ValueError: If `intervals` is empty, or if any `alpha` is not
            strictly between 0 and 1 (rejected by `interval_score`).
    """
    if not intervals:
        raise ValueError("need at least one interval")
    y = numpy.asarray(truth, dtype=float)
    total = 0.5 * float(
        numpy.mean(numpy.abs(y - numpy.asarray(median, dtype=float)))
    )
    for alpha, (lower, upper) in intervals.items():
        total += alpha / 2.0 * interval_score(y, lower, upper, alpha)
    return total / (len(intervals) + 0.5)


def crps_from_quantiles(
    truth: numpy.ndarray, quantiles: numpy.ndarray, levels: Sequence[float]
) -> float:
    """CRPS approximated as twice the mean pinball loss over the grid.

    Args:
        truth: Observed values, length `rows`.
        quantiles: `(rows, len(levels))` forecast quantiles.
        levels: The probability levels of the columns, in (0, 1).

    Returns:
        The approximate CRPS; lower is better.

    Raises:
        ValueError: If the number of columns differs from the number of
            levels. A level outside (0, 1) is rejected by `pinball_loss`
            with a `ValueError`.
    """
    values = numpy.asarray(quantiles, dtype=float)
    grid = list(levels)
    if values.shape[1] != len(grid):
        raise ValueError("one column per level is required")
    return 2.0 * float(
        numpy.mean(
            [
                pinball_loss(truth, values[:, index], level)
                for index, level in enumerate(grid)
            ]
        )
    )


def insample_scale(history: numpy.ndarray, season: int = 1) -> float:
    """Mean absolute seasonal-naive error on the training history.

    Args:
        history: Training values of the series.
        season: Seasonal period; 1 is the no-change forecast.

    Returns:
        The mean absolute difference between values `season` steps apart.

    Raises:
        ValueError: If `season` is below 1 or the history is not longer than
            the season.
    """
    series = numpy.asarray(history, dtype=float)
    if season < 1 or len(series) <= season:
        raise ValueError("history must be longer than the season")
    return float(numpy.mean(numpy.abs(series[season:] - series[:-season])))


def mase(
    truth: numpy.ndarray,
    forecast: numpy.ndarray,
    history: numpy.ndarray,
    season: int = 1,
) -> float:
    """Mean absolute scaled error with the in-sample naive scale.

    Args:
        truth: Observed values in the forecast window.
        forecast: Forecasts for the same window.
        history: Training values of the same series.
        season: Seasonal period of the naive benchmark (1 is no-change).

    Returns:
        The mean absolute error divided by the in-sample naive error, which
        is floored at `SCALE_FLOOR`. A history that is too short is rejected
        by `insample_scale` with a `ValueError`.
    """
    scale = max(insample_scale(history, season), SCALE_FLOOR)
    gap = numpy.abs(
        numpy.asarray(truth, dtype=float) - numpy.asarray(forecast, float)
    )
    return float(numpy.mean(gap) / scale)
