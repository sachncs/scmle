"""Hierarchical forecast reconciliation: bottom-up and MinT.

A hierarchy is a summing matrix `S` (all series by bottom series) so that
every aggregate equals the sum of its bottom series. Base forecasts of all
series rarely satisfy that. Bottom-up keeps the bottom forecasts and sums
them. MinT (Wickramasuriya, Athanasopoulos and Hyndman, 2019; recalled from
memory and not re-read here) picks the linear map
`G = (S' W^-1 S)^-1 S' W^-1` that minimises the trace of the reconciled
forecast-error covariance when `W` is the base-forecast error covariance, and
returns `S G y_hat`. `W` is estimated from in-sample one-step residuals with a
shrinkage of the off-diagonal terms toward zero, because the sample covariance
is singular or unstable when there are many series. The shrinkage intensity
is the Schaefer-Strimmer plug-in, also from memory; the measurement that
matters is gate G9 (`uncertainty_certify.py`), where MinT must not be worse
than bottom-up on a world with a non-diagonal covariance.

Not implemented: top-down, ensembles of reconcilers, non-negativity.
"""

import numpy as np

DIAGONAL_FLOOR = 1e-12


def bottom_up(bottom_forecasts: np.ndarray, summing: np.ndarray) -> np.ndarray:
    """Coherent forecasts by summing the bottom-level forecasts.

    Args:
        bottom_forecasts: `(steps, m)` forecasts of the bottom series.
        summing: `(n, m)` summing matrix.
    """
    return np.asarray(bottom_forecasts, dtype=float) @ np.asarray(summing).T


def shrunk_covariance(residuals: np.ndarray) -> np.ndarray:
    """Covariance of residuals with off-diagonals shrunk toward zero.

    Args:
        residuals: `(rows, n)` in-sample one-step errors of the base
            forecasts of all `n` series.

    Returns:
        A positive-definite matrix keeping the sample variances.
    """
    values = np.asarray(residuals, dtype=float)
    rows, count = values.shape
    centred = values - values.mean(axis=0)
    variance = (centred**2).sum(axis=0) / max(rows - 1, 1)
    scale = np.sqrt(np.maximum(variance, DIAGONAL_FLOOR))
    standard = centred / scale
    correlation = standard.T @ standard / max(rows - 1, 1)
    products = standard[:, :, None] * standard[:, None, :]
    spread = products.var(axis=0, ddof=1) * rows / max(rows - 1, 1) ** 2
    off = ~np.eye(count, dtype=bool)
    squared = float((correlation[off] ** 2).sum())
    intensity = (
        1.0
        if squared <= 0
        else min(1.0, max(0.0, float(spread[off].sum()) / squared))
    )
    shrunk = np.where(off, (1.0 - intensity) * correlation, 1.0)
    return shrunk * scale[:, None] * scale[None, :]


def mint(
    base_forecasts: np.ndarray, summing: np.ndarray, covariance: np.ndarray
) -> np.ndarray:
    """Coherent forecasts by trace minimisation.

    Args:
        base_forecasts: `(steps, n)` base forecasts of all series.
        summing: `(n, m)` summing matrix.
        covariance: `(n, n)` base-forecast error covariance `W`.

    Returns:
        `(steps, n)` reconciled forecasts `S G y_hat`.
    """
    s = np.asarray(summing, dtype=float)
    inverse = np.linalg.inv(np.asarray(covariance, dtype=float))
    gain = np.linalg.solve(s.T @ inverse @ s, s.T @ inverse)
    return np.asarray(base_forecasts, dtype=float) @ (s @ gain).T


def coherence_residual(forecasts: np.ndarray, summing: np.ndarray) -> float:
    """Largest gap between each series and the sum of its bottom series.

    The bottom series are the rows of `summing` that are unit vectors; they
    are taken as the last `m` columns of `forecasts`.
    """
    s = np.asarray(summing, dtype=float)
    values = np.asarray(forecasts, dtype=float)
    bottom = values[:, -s.shape[1] :]
    return float(np.abs(values - bottom @ s.T).max())
