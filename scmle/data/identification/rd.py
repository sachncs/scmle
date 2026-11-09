"""Sharp regression discontinuity at a cutoff.

Estimator. Triangular-kernel local polynomial fits on each side of the
cutoff. The conventional estimate is the difference of the two local-linear
intercepts. The reported interval is bias-corrected: it uses the difference
of the local-quadratic intercepts at the same bandwidth and the
heteroskedasticity-robust variance of that fit. This is the Calonico,
Cattaneo and Titiunik (2014) construction with the bias bandwidth equal to
the estimation bandwidth (their `rho = 1`), where the bias-corrected
estimator is exactly the local quadratic intercept; the reference package
`rdrobust` was not available here, so agreement with it is not claimed.

Bandwidth. Chosen by one-sided leave-out cross-validation (in the manner of
Imbens and Lemieux 2008, from memory): each of up to 40 evaluation points
among the nearest half of a side is predicted by a local-linear fit that uses
only the points on the same side that lie farther from the cutoff than the
evaluation point, which mimics estimation at a boundary. The grid is fixed
(`GRID`, quantiles of the distance to the cutoff) and the larger of the two
sides' choices is used. It is not the CCT mean-squared-error-optimal
bandwidth.

Diagnostics. A binomial test of the counts just left and right of the cutoff
within the bandwidth (a crude density check, valid when the density is flat
across the window; it is not the McCrary or Cattaneo-Jansson-Ma test), placebo
cutoffs at the median of each side (estimated with that side's data only),
covariate discontinuities, and the estimate at multiples of the bandwidth.

The estimand is the effect at the cutoff for units at the cutoff; it is never
extended further. Continuity and absence of manipulation are not checked
beyond the diagnostics above.
"""

import dataclasses
from typing import Any

import numpy
import pandas
from scipy import stats

GRID = (0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)
CV_POINTS = 40
MIN_SIDE = 15
SENSITIVITY_MULTIPLES = (0.5, 0.75, 1.25, 1.5)
MIN_FAR_ROWS = 3
MIN_NEAR_ROWS = 3
MIN_CV_ERRORS = 5


@dataclasses.dataclass(frozen=True)
class Fit:
    """Intercept of a weighted polynomial fit and its robust variance.

    Attributes:
        intercept: The fitted value at the evaluation point.
        variance: Heteroskedasticity-robust variance of the intercept.
        n: Number of points with positive weight.
    """

    intercept: float
    variance: float
    n: int


def local_fit(
    x: numpy.ndarray, y: numpy.ndarray, h: float, order: int
) -> Fit | None:
    """Fits a triangular-kernel polynomial and reads it at 0.

    Args:
        x: Running variable, already centred at the evaluation point.
        y: Outcome.
        h: Bandwidth; points at distance `h` or more get zero weight.
        order: Polynomial order (1 for linear, 2 for quadratic).

    Returns:
        The intercept, its robust variance and the number of points used,
        or None when fewer than ``order + 3`` points have positive weight
        or the weighted Gram matrix is singular.
    """
    weight = numpy.clip(1.0 - numpy.abs(x) / h, 0.0, None)
    keep = weight > 0
    if keep.sum() < order + 3:
        return None
    xs, ys, ws = x[keep], y[keep], weight[keep]
    design = numpy.vander(xs, order + 1, increasing=True)
    gram = design.T @ (design * ws[:, None])
    try:
        bread = numpy.linalg.inv(gram)
    except numpy.linalg.LinAlgError:
        return None
    coef = bread @ (design.T @ (ws * ys))
    resid = ys - design @ coef
    meat = (design * (ws * resid)[:, None]).T @ (design * (ws * resid)[:, None])
    n, k = len(xs), order + 1
    variance = float((bread @ meat @ bread)[0, 0]) * n / max(n - k, 1)
    return Fit(float(coef[0]), variance, n)


def cv_bandwidth(x: numpy.ndarray, y: numpy.ndarray) -> float:
    """Leave-out cross-validated bandwidth for one signed side of the cutoff.

    `x` holds distances from the cutoff, positive, nearest first after
    sorting; evaluation points are at most `CV_POINTS` distinct positions
    evenly spread among the nearest half (at least five positions).

    Args:
        x: Distances from the cutoff, positive.
        y: Outcomes, aligned with `x`.

    Returns:
        The bandwidth, one of the grid quantiles of `x`, that has the
        smallest mean squared leave-out prediction error among those with at
        least five evaluation errors; the median of `x` if none has.
    """
    order = numpy.argsort(x)
    x, y = x[order], y[order]
    pool = max(len(x) // 2, 5)
    picks = numpy.unique(numpy.linspace(2, pool - 1, CV_POINTS).astype(int))
    best, best_error = float(numpy.quantile(x, 0.5)), numpy.inf
    for fraction in GRID:
        h = float(numpy.quantile(x, fraction))
        errors = []
        for i in picks:
            far = numpy.arange(len(x)) > i
            if h <= 0 or far.sum() < MIN_FAR_ROWS:
                continue
            near = far & (x - x[i] < h)
            if near.sum() < MIN_NEAR_ROWS:
                continue
            fit = local_fit(x[near] - x[i], y[near], h, 1)
            if fit is not None:
                errors.append((y[i] - fit.intercept) ** 2)
        if len(errors) >= MIN_CV_ERRORS and numpy.mean(errors) < best_error:
            best, best_error = h, float(numpy.mean(errors))
    return best


@dataclasses.dataclass(frozen=True)
class Estimate:
    """A regression discontinuity fit.

    Attributes:
        estimate: Conventional (local linear) estimate.
        bias_corrected: Local quadratic estimate (the interval's centre).
        se: Robust standard error of the bias-corrected estimate.
        low: Lower interval limit.
        high: Upper interval limit.
        level: Confidence level.
        bandwidth: The bandwidth used on both sides.
        n_left: Rows inside the bandwidth to the left.
        n_right: Rows inside the bandwidth to the right.
        density_p: Binomial p-value of the left/right counts.
        placebos: Placebo name (`left_median` or `right_median`) to its
            `cutoff` and the `discontinuity` record at that cutoff.
        balance: Covariate to its `discontinuity` record.
        sensitivity: Bandwidth multiple (as text) to the `discontinuity`
            record at that bandwidth.
    """

    estimate: float
    bias_corrected: float
    se: float
    low: float
    high: float
    level: float
    bandwidth: float
    n_left: int
    n_right: int
    density_p: float
    placebos: dict[str, dict[str, float]]
    balance: dict[str, dict[str, float]]
    sensitivity: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        """Builds the JSON-ready record.

        Returns:
            A mapping of every field.
        """
        return dataclasses.asdict(self)


def discontinuity(
    x: numpy.ndarray,
    y: numpy.ndarray,
    cutoff: float,
    h: float,
    level: float,
) -> dict[str, float] | None:
    """Computes the conventional and bias-corrected jump at a cutoff.

    Args:
        x: Running variable.
        y: Outcome (or a covariate whose jump is tested).
        cutoff: Where the jump is measured.
        h: Bandwidth, used on both sides.
        level: Confidence level.

    Returns:
        None when a side has too few points for a fit or a singular one.
        Otherwise a mapping
        with the conventional `estimate`, the `bias_corrected` jump, its
        robust `se`, the interval limits `low` and `high` around the
        bias-corrected jump, and the two-sided normal p-value `p` of the
        bias-corrected jump (1 when `se` is not positive).
    """
    left = x < cutoff
    pieces = {}
    for name, mask in (("left", left), ("right", ~left)):
        linear = local_fit(x[mask] - cutoff, y[mask], h, 1)
        quadratic = local_fit(x[mask] - cutoff, y[mask], h, 2)
        if linear is None or quadratic is None:
            return None
        pieces[name] = (linear, quadratic)
    conventional = pieces["right"][0].intercept - pieces["left"][0].intercept
    corrected = pieces["right"][1].intercept - pieces["left"][1].intercept
    se = float(
        numpy.sqrt(pieces["right"][1].variance + pieces["left"][1].variance)
    )
    z = float(stats.norm.ppf(0.5 + level / 2))
    return {
        "estimate": float(conventional),
        "bias_corrected": float(corrected),
        "se": se,
        "low": float(corrected - z * se),
        "high": float(corrected + z * se),
        "p": float(2 * stats.norm.sf(abs(corrected) / se)) if se > 0 else 1.0,
    }


def chosen_bandwidth(
    x: numpy.ndarray,
    y: numpy.ndarray,
    left: numpy.ndarray,
    cutoff: float,
    bandwidth: float | None,
) -> float:
    """Takes the fixed bandwidth, or cross-validates one on each side.

    Args:
        x: Running variable.
        y: Outcomes, aligned with `x`.
        left: Mask of the rows below the cutoff.
        cutoff: The threshold.
        bandwidth: A fixed bandwidth, or None to cross-validate.

    Returns:
        `bandwidth` as a float, or the larger of the two cross-validated
        bandwidths.
    """
    if bandwidth is None:
        return max(
            cv_bandwidth(cutoff - x[left], y[left]),
            cv_bandwidth(x[~left] - cutoff, y[~left]),
        )
    return float(bandwidth)


def placebo_fits(
    x: numpy.ndarray,
    y: numpy.ndarray,
    left: numpy.ndarray,
    h: float,
    level: float,
) -> dict[str, dict[str, float]]:
    """Fits the discontinuity at the median of each side.

    Args:
        x: Running variable.
        y: Outcomes, aligned with `x`.
        left: Mask of the rows below the cutoff.
        h: The bandwidth.
        level: Confidence level.

    Returns:
        The placebo fits keyed `left_median` and `right_median`, each with
        its `cutoff`; a side whose fit was degenerate is left out.
    """
    placebos = {}
    for name, mask in (("left_median", left), ("right_median", ~left)):
        point = float(numpy.median(x[mask]))
        found = discontinuity(x[mask], y[mask], point, h, level)
        if found is not None:
            placebos[name] = {"cutoff": point, **found}
    return placebos


def estimate(
    frame: pandas.DataFrame,
    running: str,
    outcome: str,
    cutoff: float,
    level: float = 0.95,
    bandwidth: float | None = None,
    covariates: tuple[str, ...] = (),
) -> Estimate:
    """Fits the discontinuity, its diagnostics and bandwidth sensitivity.

    Args:
        frame: Rows with the running variable and outcome.
        running: Running variable column.
        outcome: Outcome column.
        cutoff: The threshold; treatment is assigned at `running >= cutoff`.
        level: Confidence level.
        bandwidth: A fixed bandwidth, or None to cross-validate.
        covariates: Predetermined covariates to test for a jump.

    Returns:
        The fit: the estimates and interval at the bandwidth, the binomial
        density p-value, and the placebo, covariate and bandwidth-multiple
        results (entries whose fit was degenerate are left out).

    Raises:
        ValueError: If a running or outcome value is missing or non-finite,
            a side has fewer than `MIN_SIDE` rows, or the main fit has too
            few rows inside the bandwidth.
    """
    x = frame[running].to_numpy(dtype=float)
    y = frame[outcome].to_numpy(dtype=float)
    if not (numpy.isfinite(x).all() and numpy.isfinite(y).all()):
        raise ValueError("RD columns must not hold missing values")
    left = x < cutoff
    if min(left.sum(), (~left).sum()) < MIN_SIDE:
        raise ValueError(f"each side needs at least {MIN_SIDE} rows")
    h = chosen_bandwidth(x, y, left, cutoff, bandwidth)
    main = discontinuity(x, y, cutoff, h, level)
    if main is None:
        raise ValueError("too few rows inside the bandwidth to fit")
    inside_left = int(((x < cutoff) & (x > cutoff - h)).sum())
    inside_right = int(((x >= cutoff) & (x < cutoff + h)).sum())
    total = inside_left + inside_right
    density_p = float(stats.binomtest(inside_right, total, 0.5).pvalue)
    placebos = placebo_fits(x, y, left, h, level)
    balance = {}
    for name in covariates:
        found = discontinuity(
            x, frame[name].to_numpy(dtype=float), cutoff, h, level
        )
        if found is not None:
            balance[name] = found
    sensitivity = {}
    for multiple in SENSITIVITY_MULTIPLES:
        found = discontinuity(x, y, cutoff, h * multiple, level)
        if found is not None:
            sensitivity[str(multiple)] = found
    return Estimate(
        estimate=main["estimate"],
        bias_corrected=main["bias_corrected"],
        se=main["se"],
        low=main["low"],
        high=main["high"],
        level=level,
        bandwidth=h,
        n_left=inside_left,
        n_right=inside_right,
        density_p=density_p,
        placebos=placebos,
        balance=balance,
        sensitivity=sensitivity,
    )
