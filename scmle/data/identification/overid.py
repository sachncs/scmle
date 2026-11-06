"""Overidentification tests for several instruments (F10, descriptive).

`iv.py` handles one instrument, where the model is just identified and the
exclusion restriction cannot be tested. With `k >= 2` instruments for one
endogenous regressor there are `k - 1` testable restrictions: if all
instruments are valid, the 2SLS residual is uncorrelated with every one of
them. Two tests of that, both after partialling out the exogenous controls
(and an intercept):

* `sargan`: `n * R^2` of the 2SLS residual regressed on the instruments, chi-
  square with `k - 1` degrees of freedom; valid under homoskedastic errors;
* `hansen_j`: the two-step efficient GMM criterion with the heteroskedasticity
  robust weight matrix, chi-square with `k - 1` degrees of freedom.

What a result shows. A rejection says that at least one instrument is not
valid or that the effect is not the same for every instrument (several local
average treatment effects); it does not say which instrument. A pass does not
show validity: all instruments can fail in the same direction, and with weak
instruments the tests have little power. The tests are descriptive: they do
not enter the tier of an IV claim, which stays `attested` because exclusion
is never verified (`iv.py`). The false-alarm rate with valid instruments and
the detection of one invalid instrument are checked by simulation in
`tests/unit/test_identification_overid.py` against floors written before the
first run; there is no certificate in `f10_certificates.json`.
"""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from scipy import stats

NOTE = (
    "a rejection means some instrument is invalid or the instruments "
    "identify different effects, not which one; a pass does not show "
    "validity, and weak instruments give the tests little power"
)


def prepare(
    y: ArrayLike,
    d: ArrayLike,
    instruments: ArrayLike,
    controls: ArrayLike | None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Outcome, regressor and instruments with the controls partialled out."""
    outcome = np.asarray(y, dtype=float)
    regressor = np.asarray(d, dtype=float)
    z = np.asarray(instruments, dtype=float)
    if z.ndim == 1:
        z = z[:, None]
    if not len(outcome) == len(regressor) == len(z):
        raise ValueError("y, d and the instruments must have the same length")
    rows = len(outcome)
    base = np.ones((rows, 1))
    if controls is not None:
        extra = np.asarray(controls, dtype=float)
        if extra.ndim == 1:
            extra = extra[:, None]
        if len(extra) != rows:
            raise ValueError("controls must have the length of y")
        base = np.column_stack([base, extra])

    def residual(values: NDArray[np.float64]) -> NDArray[np.float64]:
        fit = np.linalg.lstsq(base, values, rcond=None)[0]
        return np.asarray(values - base @ fit)

    return (
        residual(outcome),
        residual(regressor),
        np.column_stack([residual(z[:, k]) for k in range(z.shape[1])]),
    )


def not_testable(name: str, count: int) -> dict[str, Any]:
    """The result when there is nothing to test."""
    return {
        "test": name,
        "status": "not_testable",
        "statistic": None,
        "degrees_of_freedom": 0,
        "p_value": None,
        "instruments": count,
        "note": "one instrument is just identified: exclusion is not testable",
    }


def result(
    name: str, statistic: float, count: int, rows: int
) -> dict[str, Any]:
    """The described result of a test with `count - 1` degrees of freedom."""
    df = count - 1
    return {
        "test": name,
        "status": "described",
        "statistic": float(statistic),
        "degrees_of_freedom": df,
        "p_value": float(stats.chi2.sf(statistic, df)),
        "instruments": count,
        "rows": rows,
        "note": NOTE,
    }


def two_stage(
    y: NDArray[np.float64], d: NDArray[np.float64], z: NDArray[np.float64]
) -> float:
    """The 2SLS coefficient of `d` (all variables already residualised)."""
    projected = z @ np.linalg.lstsq(z, d, rcond=None)[0]
    return float(projected @ y / (projected @ d))


def sargan(
    y: ArrayLike,
    d: ArrayLike,
    instruments: ArrayLike,
    controls: ArrayLike | None = None,
) -> dict[str, Any]:
    """Sargan's test: `n * R^2` of the 2SLS residual on the instruments.

    Raises:
        ValueError: On inputs of unequal length.
    """
    outcome, regressor, z = prepare(y, d, instruments, controls)
    count = z.shape[1]
    if count < 2:
        return not_testable("sargan", count)
    rows = len(outcome)
    residual = outcome - two_stage(outcome, regressor, z) * regressor
    fit = z @ np.linalg.lstsq(z, residual, rcond=None)[0]
    total = float(residual @ residual)
    r_squared = float(fit @ fit) / total if total > 0 else 0.0
    return result("sargan", rows * r_squared, count, rows)


def hansen_j(
    y: ArrayLike,
    d: ArrayLike,
    instruments: ArrayLike,
    controls: ArrayLike | None = None,
) -> dict[str, Any]:
    """Hansen's J: the two-step efficient GMM criterion, robust weights.

    Raises:
        ValueError: On inputs of unequal length.
    """
    outcome, regressor, z = prepare(y, d, instruments, controls)
    count = z.shape[1]
    if count < 2:
        return not_testable("hansen_j", count)
    rows = len(outcome)
    first = outcome - two_stage(outcome, regressor, z) * regressor
    moments = z * first[:, None]
    weight = np.linalg.pinv(moments.T @ moments / rows)
    left = z.T @ regressor / rows
    right = z.T @ outcome / rows
    beta = float(left @ weight @ right / (left @ weight @ left))
    gap = z.T @ (outcome - beta * regressor) / rows
    return result("hansen_j", rows * float(gap @ weight @ gap), count, rows)
