"""Measured coverage: exact intervals, slice flags and the coverage tier.

A conformal theorem says what coverage to expect under exchangeability. What
this module reports is what was measured on rows the calibration never saw:
the Clopper-Pearson interval of the covered share, the same by slice with a
Holm correction over the slice family fixed beforehand, and the tier the
measurement earns (canon P4). Conditional coverage is measured, never
promised; only class-conditional sets carry a within-class construction.

The slice flag changes a decision, so its false-flag rate and its recall by
slice size are certificates (`uncertainty_certify.py`). Below the smallest
certified size a flag is descriptive.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import math
from typing import Any

import numpy as np
from scipy import stats

DEFAULT_TOLERANCE = 0.02
DEFAULT_FAMILYWISE = 0.05
MIN_SLICE_ROWS = 30
CONFIRMED = "confirmed"
ATTESTED = "attested"
UNVERIFIED = "unverified"


def clopper_pearson(
    covered: int, rows: int, level: float = 0.95
) -> tuple[float, float]:
    """Exact two-sided binomial interval for a covered share.

    Args:
        covered: Number of rows inside their interval or set.
        rows: Number of rows scored.
        level: Confidence level in (0, 1).

    Returns:
        Lower and upper limits; `(0, 1)` when `rows` is zero.

    Raises:
        ValueError: For counts that are not `0 <= covered <= rows`, or a
            level outside (0, 1).
    """
    if not 0 <= covered <= rows:
        raise ValueError("need 0 <= covered <= rows")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between 0 and 1")
    if rows == 0:
        return 0.0, 1.0
    tail = (1.0 - level) / 2.0
    lower = (
        0.0
        if covered == 0
        else float(stats.beta.ppf(tail, covered, rows - covered + 1))
    )
    upper = (
        1.0
        if covered == rows
        else float(stats.beta.ppf(1.0 - tail, covered + 1, rows - covered))
    )
    return lower, upper


def holm_rejections(p_values: Sequence[float], alpha: float) -> list[bool]:
    """Holm step-down: which hypotheses are rejected at familywise `alpha`."""
    count = len(p_values)
    order = np.argsort(np.asarray(p_values, dtype=float), kind="stable")
    rejected = [False] * count
    for step, index in enumerate(order):
        if p_values[int(index)] <= alpha / (count - step):
            rejected[int(index)] = True
        else:
            break
    return rejected


def undercoverage_p_value(covered: int, rows: int, nominal: float) -> float:
    """One-sided exact binomial p-value for "coverage is below nominal".

    The null is that each row is covered with probability at least
    `nominal`; a small value says the interval or set under-covers.
    """
    if rows == 0:
        return 1.0
    return float(stats.binom.cdf(covered, rows, nominal))


def coverage_summary(
    covered: np.ndarray,
    widths: np.ndarray | None = None,
    nominal: float = 0.9,
    tolerance: float = DEFAULT_TOLERANCE,
    level: float = 0.95,
) -> dict[str, Any]:
    """Marginal coverage with its exact interval.

    Args:
        covered: Boolean array, one entry per scored row.
        widths: Interval widths (or set sizes), for the mean.
        nominal: The requested coverage.
        tolerance: Slack below nominal that still counts as met.
        level: Confidence level of the interval.

    Returns:
        Counts, the observed share, the Clopper-Pearson limits, the mean
        width and whether the lower limit reaches `nominal - tolerance`.
    """
    flags = np.asarray(covered, dtype=bool)
    rows = int(flags.size)
    hits = int(flags.sum())
    lower, upper = clopper_pearson(hits, rows, level)
    mean_width = None
    if widths is not None and len(widths):
        finite = np.asarray(widths, dtype=float)
        mean_width = None if np.isinf(finite).any() else float(finite.mean())
    return {
        "nominal": nominal,
        "rows": rows,
        "covered": hits,
        "coverage": hits / rows if rows else float("nan"),
        "lower": lower,
        "upper": upper,
        "level": level,
        "tolerance": tolerance,
        "mean_width": mean_width,
        "meets_nominal": bool(rows and lower >= nominal - tolerance),
    }


def slice_coverage(
    covered: np.ndarray,
    labels: Sequence[Any],
    nominal: float = 0.9,
    familywise: float = DEFAULT_FAMILYWISE,
    min_rows: int = MIN_SLICE_ROWS,
    level: float = 0.95,
) -> dict[str, Any]:
    """Coverage by slice with a Holm-corrected under-coverage flag.

    The slice family is every level of `labels` with at least `min_rows`
    rows; fix it before the test rows are read. Slices below `min_rows` are
    counted in `skipped` and never tested.

    Args:
        covered: Boolean array, one entry per scored row.
        labels: Slice label of each row.
        nominal: The requested coverage.
        familywise: Familywise error rate of the Holm step-down.
        min_rows: Smallest slice that is tested.
        level: Confidence level of each reported interval.

    Returns:
        `slices` (n, coverage, interval, p-value, `flagged`), `skipped` and
        the number of tests in the family.
    """
    flags = np.asarray(covered, dtype=bool)
    names = np.asarray([str(label) for label in labels])
    found: list[dict[str, Any]] = []
    skipped = 0
    for name in sorted(set(names)):
        mask = names == name
        rows = int(mask.sum())
        if rows < min_rows:
            skipped += 1
            continue
        hits = int(flags[mask].sum())
        lower, upper = clopper_pearson(hits, rows, level)
        found.append(
            {
                "slice": name,
                "rows": rows,
                "covered": hits,
                "coverage": hits / rows,
                "lower": lower,
                "upper": upper,
                "p_value": undercoverage_p_value(hits, rows, nominal),
            }
        )
    rejected = holm_rejections([item["p_value"] for item in found], familywise)
    for item, flag in zip(found, rejected, strict=True):
        item["flagged"] = bool(flag)
    return {
        "nominal": nominal,
        "familywise": familywise,
        "tests": len(found),
        "skipped": skipped,
        "slices": found,
        "flagged": [item["slice"] for item in found if item["flagged"]],
    }


def class_coverage(
    covered: np.ndarray,
    true_labels: Sequence[Any],
    nominal: float = 0.9,
    level: float = 0.95,
) -> dict[str, dict[str, Any]]:
    """Coverage within each true class (for class-conditional sets)."""
    flags = np.asarray(covered, dtype=bool)
    names = np.asarray([str(label) for label in true_labels])
    return {
        name: coverage_summary(flags[names == name], None, nominal, 0.0, level)
        for name in sorted(set(names))
    }


def coverage_tier(
    summary: Mapping[str, Any],
    attested_exchangeable: bool = False,
    clean_holdout: bool = True,
) -> str:
    """Tier of a coverage claim (canon P4).

    `confirmed` needs a clean held-out measurement whose lower limit reaches
    `nominal - tolerance`; the theorem's assumption is then not what is
    trusted. Without such a measurement, a named person's attestation that
    the data are exchangeable with deployment gives `attested`; a measured
    shortfall is never overridden by an attestation. Anything else is
    `unverified`.
    """
    measured = bool(clean_holdout and summary.get("rows", 0) > 0)
    if measured and summary.get("meets_nominal"):
        return CONFIRMED
    if measured and not math.isnan(summary.get("coverage", math.nan)):
        return UNVERIFIED
    return ATTESTED if attested_exchangeable else UNVERIFIED


def coverage_report(
    covered: np.ndarray,
    widths: np.ndarray | None,
    nominal: float,
    labels: Sequence[Any] | None = None,
    true_classes: Sequence[Any] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    attested_exchangeable: bool = False,
    clean_holdout: bool = True,
) -> dict[str, Any]:
    """The `coverage_report.json` body for one nominal level.

    Args:
        covered: Boolean array over the held-out rows (read once).
        widths: Interval widths or set sizes.
        nominal: The requested coverage.
        labels: Slice label of each row, for the slice family.
        true_classes: True class of each row, for sets.
        tolerance: Charter tolerance below nominal (default 0.02).
        attested_exchangeable: A named person attests exchangeability.
        clean_holdout: Whether `covered` comes from untouched data.
    """
    summary = coverage_summary(covered, widths, nominal, tolerance)
    report: dict[str, Any] = {
        "marginal": summary,
        "tier": coverage_tier(summary, attested_exchangeable, clean_holdout),
    }
    if labels is not None:
        report["by_slice"] = slice_coverage(covered, labels, nominal)
    if true_classes is not None:
        report["by_class"] = class_coverage(covered, true_classes, nominal)
    return report
