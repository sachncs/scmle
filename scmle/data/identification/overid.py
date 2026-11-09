"""Overidentification tests for several instruments (descriptive).

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

import numpy
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
    NDArray[numpy.float64],
    NDArray[numpy.float64],
    NDArray[numpy.float64],
]:
    """Partials the controls (and an intercept) out of every variable.

    Args:
        y: Outcome.
        d: Endogenous regressor.
        instruments: One column per instrument, or one vector.
        controls: Exogenous controls, or None for an intercept only.

    Returns:
        The residualised outcome, regressor and instrument matrix (one
        column per instrument).

    Raises:
        ValueError: If the outcome, regressor and instruments differ in
            length, or the controls differ in length from the outcome.
    """
    outcome = numpy.asarray(y, dtype=float)
    regressor = numpy.asarray(d, dtype=float)
    z = numpy.asarray(instruments, dtype=float)
    if z.ndim == 1:
        z = z[:, None]
    if not len(outcome) == len(regressor) == len(z):
        raise ValueError("y, d and the instruments must have the same length")
    rows = len(outcome)
    base = numpy.ones((rows, 1))
    if controls is not None:
        extra = numpy.asarray(controls, dtype=float)
        if extra.ndim == 1:
            extra = extra[:, None]
        if len(extra) != rows:
            raise ValueError("controls must have the length of y")
        base = numpy.column_stack([base, extra])

    def residual(values: NDArray[numpy.float64]) -> NDArray[numpy.float64]:
        fit = numpy.linalg.lstsq(base, values, rcond=None)[0]
        return numpy.asarray(values - base @ fit)

    return (
        residual(outcome),
        residual(regressor),
        numpy.column_stack([residual(z[:, k]) for k in range(z.shape[1])]),
    )


def not_testable(name: str, count: int) -> dict[str, Any]:
    """Builds the result when there is nothing to test.

    Args:
        name: Name of the test.
        count: Number of instruments.

    Returns:
        A record with status `not_testable`, no statistic or p-value, zero
        degrees of freedom, the instrument count and a note that a single
        instrument is just identified.
    """
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
    """Builds the described result of a test on `count - 1` degrees of freedom.

    Args:
        name: Name of the test.
        statistic: The test statistic.
        count: Number of instruments.
        rows: Number of rows used.

    Returns:
        A record with status `described`, the statistic, the degrees of
        freedom, the chi-square upper-tail p-value, `count`, `rows` and a
        note on how to read the result.
    """
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
    y: NDArray[numpy.float64],
    d: NDArray[numpy.float64],
    z: NDArray[numpy.float64],
) -> float:
    """Computes the 2SLS coefficient of `d`.

    Args:
        y: Residualised outcome.
        d: Residualised regressor.
        z: Residualised instruments, one column each.

    Returns:
        The coefficient of `d` in the 2SLS fit.
    """
    projected = z @ numpy.linalg.lstsq(z, d, rcond=None)[0]
    return float(projected @ y / (projected @ d))


def sargan(
    y: ArrayLike,
    d: ArrayLike,
    instruments: ArrayLike,
    controls: ArrayLike | None = None,
) -> dict[str, Any]:
    """Computes Sargan's test: `n * R^2` of the 2SLS residual on instruments.

    Inputs of unequal length are refused with a `ValueError` by `prepare`.

    Args:
        y: Outcome.
        d: Endogenous regressor.
        instruments: One column per instrument, or one vector.
        controls: Exogenous controls, or None.

    Returns:
        A result record; with fewer than two instruments the status is
        `not_testable`, otherwise `described` with the statistic and its
        p-value on `k - 1` degrees of freedom.
    """
    outcome, regressor, z = prepare(y, d, instruments, controls)
    count = z.shape[1]
    if count < 2:
        return not_testable("sargan", count)
    rows = len(outcome)
    residual = outcome - two_stage(outcome, regressor, z) * regressor
    fit = z @ numpy.linalg.lstsq(z, residual, rcond=None)[0]
    total = float(residual @ residual)
    r_squared = float(fit @ fit) / total if total > 0 else 0.0
    return result("sargan", rows * r_squared, count, rows)


def hansen_j(
    y: ArrayLike,
    d: ArrayLike,
    instruments: ArrayLike,
    controls: ArrayLike | None = None,
) -> dict[str, Any]:
    """Computes Hansen's J: the two-step efficient GMM criterion.

    The weight matrix is heteroskedasticity robust. Inputs of unequal length
    are refused with a `ValueError` by `prepare`.

    Args:
        y: Outcome.
        d: Endogenous regressor.
        instruments: One column per instrument, or one vector.
        controls: Exogenous controls, or None.

    Returns:
        A result record; with fewer than two instruments the status is
        `not_testable`, otherwise `described` with the statistic and its
        p-value on one fewer degrees of freedom than instruments.
    """
    outcome, regressor, z = prepare(y, d, instruments, controls)
    count = z.shape[1]
    if count < 2:
        return not_testable("hansen_j", count)
    rows = len(outcome)
    first = outcome - two_stage(outcome, regressor, z) * regressor
    moments = z * first[:, None]
    weight = numpy.linalg.pinv(moments.T @ moments / rows)
    left = z.T @ regressor / rows
    right = z.T @ outcome / rows
    beta = float(left @ weight @ right / (left @ weight @ left))
    gap = z.T @ (outcome - beta * regressor) / rows
    return result("hansen_j", rows * float(gap @ weight @ gap), count, rows)
