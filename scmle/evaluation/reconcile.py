"""Hierarchical forecast reconciliation: bottom-up and MinT.

A hierarchy is a summing matrix `S` (all series by bottom series) so that
every aggregate equals the sum of its bottom series. Base forecasts of all
series rarely satisfy that. Bottom-up keeps the bottom forecasts and sums
them. MinT (Wickramasuriya, Athanasopoulos and Hyndman, JASA 114(526), 2019;
the working-paper version of 23 January 2018 was read on 2026-11-05: Theorem
1, eq. 9, and the covariance estimators of Section 3) picks the linear map
`G = (S' W^-1 S)^-1 S' W^-1` (the paper writes `P`) that minimises the trace
of the reconciled forecast-error covariance among the unbiased maps (those
with `S G S = S`), where `W` is the base-forecast error covariance, and
returns `S G y_hat`. `W` is estimated from in-sample one-step residuals with a
shrinkage of the off-diagonal terms toward zero, because the sample covariance
is singular or unstable when there are many series. The shrinkage intensity
is the Schaefer-Strimmer plug-in as the paper describes it (intensity
`sum Var(r_ij) / sum r_ij^2` over the off-diagonal sample correlations); the
paper states it under a constant-variance assumption, and its source
(Schaefer and Strimmer 2005) was not read. The property that matters is that
MinT is not worse than bottom-up on simulated data with a non-diagonal
covariance.

Not implemented: top-down, ensembles of reconcilers, non-negativity.
"""

import numpy

DIAGONAL_FLOOR = 1e-12


def bottom_up(
    bottom_forecasts: numpy.ndarray, summing: numpy.ndarray
) -> numpy.ndarray:
    """Coherent forecasts by summing the bottom-level forecasts.

    Args:
        bottom_forecasts: `(steps, m)` forecasts of the bottom series.
        summing: `(n, m)` summing matrix.

    Returns:
        `(steps, n)` coherent forecasts of all series.
    """
    coherent: numpy.ndarray = (
        numpy.asarray(bottom_forecasts, dtype=float) @ numpy.asarray(summing).T
    )
    return coherent


def shrunk_covariance(residuals: numpy.ndarray) -> numpy.ndarray:
    """Covariance of residuals with off-diagonals shrunk toward zero.

    Args:
        residuals: `(rows, n)` in-sample one-step errors of the base
            forecasts of all `n` series.

    Returns:
        An `(n, n)` matrix with the sample variances (floored at
        `DIAGONAL_FLOOR`) on the diagonal and the sample correlations
        shrunk toward zero off the diagonal by the plug-in intensity
        (clipped to [0, 1]). It is diagonal when the intensity is 1; it is
        positive definite only if the shrunk correlation matrix is, which
        is not guaranteed for little shrinkage with fewer rows than series.
    """
    values = numpy.asarray(residuals, dtype=float)
    rows, count = values.shape
    centred = values - values.mean(axis=0)
    variance = (centred**2).sum(axis=0) / max(rows - 1, 1)
    scale = numpy.sqrt(numpy.maximum(variance, DIAGONAL_FLOOR))
    standard = centred / scale
    correlation = standard.T @ standard / max(rows - 1, 1)
    products = standard[:, :, None] * standard[:, None, :]
    spread = products.var(axis=0, ddof=1) * rows / max(rows - 1, 1) ** 2
    off = ~numpy.eye(count, dtype=bool)
    squared = float((correlation[off] ** 2).sum())
    intensity = (
        1.0
        if squared <= 0
        else min(1.0, max(0.0, float(spread[off].sum()) / squared))
    )
    shrunk = numpy.where(off, (1.0 - intensity) * correlation, 1.0)
    covariance: numpy.ndarray = shrunk * scale[:, None] * scale[None, :]
    return covariance


def mint(
    base_forecasts: numpy.ndarray,
    summing: numpy.ndarray,
    covariance: numpy.ndarray,
) -> numpy.ndarray:
    """Coherent forecasts by trace minimisation.

    Args:
        base_forecasts: `(steps, n)` base forecasts of all series.
        summing: `(n, m)` summing matrix.
        covariance: `(n, n)` base-forecast error covariance `W`.

    Returns:
        `(steps, n)` reconciled forecasts `S G y_hat`.

    Raises:
        numpy.linalg.LinAlgError: If `covariance` or `S' W^-1 S` is singular.
    """
    s = numpy.asarray(summing, dtype=float)
    inverse = numpy.linalg.inv(numpy.asarray(covariance, dtype=float))
    gain = numpy.linalg.solve(s.T @ inverse @ s, s.T @ inverse)
    reconciled: numpy.ndarray = (
        numpy.asarray(base_forecasts, dtype=float) @ (s @ gain).T
    )
    return reconciled


def coherence_residual(
    forecasts: numpy.ndarray, summing: numpy.ndarray
) -> float:
    """Largest gap between each series and the sum of its bottom series.

    The bottom series are taken to be the last `m` columns of `forecasts`
    (`m` being the number of columns of `summing`); the row order of
    `summing` is not checked against that assumption.

    Args:
        forecasts: `(steps, n)` forecasts of all series.
        summing: `(n, m)` summing matrix.

    Returns:
        The largest absolute difference between a forecast and the sum of
        the bottom forecasts that make it up, over all steps and series.
    """
    s = numpy.asarray(summing, dtype=float)
    values = numpy.asarray(forecasts, dtype=float)
    bottom = values[:, -s.shape[1] :]
    return float(numpy.abs(values - bottom @ s.T).max())
