"""Description of whether released rows look exchangeable with calibration.

A conformal guarantee holds for rows exchangeable with the calibration rows.
This module compares the nonconformity scores of the calibration rows with
the scores of the released rows by two tests, the Kolmogorov-Smirnov
two-sample test (any difference of shape) and the Mann-Whitney test (a
shift of location), and flags a difference when the smaller p-value, doubled
for the two tests, is at most `ALPHA`.

This is not a gate. The flag is a description beside the coverage
claim, never a condition on it. Its size follows from the tests (for rows
that are exchangeable the doubled minimum p-value is at most `ALPHA` with
probability at most `ALPHA`, which `tests/unit/test_exchangeability.py`
checks by simulation), but no recall certificate exists for it, so it
carries no decision. The calibration scores come from out-of-fold models and
the released scores from the final model, so a mild difference of scale is
expected and is not evidence of a broken run.
"""

from collections.abc import Sequence
from typing import Any

import numpy
from scipy import stats

from scmle.evaluation import conformal

ALPHA = 0.05
MIN_ROWS = 20
NOTE = (
    "description only: a flag says the released scores differ from the "
    "calibration scores; it does not change the coverage status. The "
    "calibration scores are out of fold and the released scores come from the "
    "final model."
)


def compare(
    calibration: Sequence[float] | numpy.ndarray,
    released: Sequence[float] | numpy.ndarray,
) -> dict[str, Any]:
    """Compares two score samples.

    Args:
        calibration: Nonconformity scores of the calibration rows.
        released: Nonconformity scores of the released rows.

    Returns:
        The two p-values, the doubled minimum compared with `ALPHA`, and
        `status` `described` (or `not_tested` when either sample has fewer
        than `MIN_ROWS` rows).
    """
    first = numpy.asarray(calibration, dtype=float)
    second = numpy.asarray(released, dtype=float)
    counts = {
        "calibration_rows": int(first.size),
        "released_rows": int(second.size),
        "alpha": ALPHA,
        "note": NOTE,
    }
    if first.size < MIN_ROWS or second.size < MIN_ROWS:
        return {
            **counts,
            "ks_p_value": None,
            "mann_whitney_p_value": None,
            "flag": None,
            "status": "not_tested",
        }
    ks_p = float(stats.ks_2samp(first, second).pvalue)
    mw_p = float(stats.mannwhitneyu(first, second).pvalue)
    return {
        **counts,
        "ks_p_value": ks_p,
        "mann_whitney_p_value": mw_p,
        "flag": bool(2.0 * min(ks_p, mw_p) <= ALPHA),
        "status": "described",
    }


def released_scores(bundle: Any, features: Any, target: Any) -> numpy.ndarray:
    """Nonconformity scores of the released rows under the bundle's model.

    Absolute error for regression and one minus the probability of the true
    class for classification, the scores the calibrator stores.

    Args:
        bundle: A model bundle with a conformal calibrator.
        features: Released rows.
        target: True targets of the released rows.

    Returns:
        One score per released row.
    """
    calibrator = bundle.conformal
    if calibrator.kind == "regression":
        band = bundle.predict_interval(
            features, coverage=conformal.DEFAULT_COVERAGE, method="residual"
        )
        residuals: numpy.ndarray = numpy.abs(
            numpy.asarray(target, dtype=float)
            - band["prediction"].to_numpy(dtype=float)
        )
        return residuals
    probabilities = numpy.asarray(bundle.predict_proba(features), dtype=float)
    position = {label: i for i, label in enumerate(calibrator.classes)}
    columns = [position[label] for label in list(target)]
    return 1.0 - probabilities[numpy.arange(len(columns)), columns]


def describe(bundle: Any, features: Any, target: Any) -> dict[str, Any]:
    """The comparison for a bundle's calibration and the released rows.

    Args:
        bundle: A model bundle with a conformal calibrator.
        features: Released rows.
        target: True targets of the released rows.

    Returns:
        The result of `compare` for the bundle's calibration scores and the
        released scores.
    """
    return compare(
        bundle.conformal.scores, released_scores(bundle, features, target)
    )
