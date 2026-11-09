"""Uncertainty of a cross-validated score that respects fold dependence.

Fold scores are not independent observations: every pair of training sets
overlaps heavily. Treating them as independent makes the usual standard error
too small. The Nadeau-Bengio correction inflates the variance by the ratio of
validation to training size, which is the standard remedy for resampled
estimates; for temporal validation the windows are even more dependent, so
the interval is flagged as optimistic.
"""

import math
from typing import Any

import numpy as np
from scipy import stats

LEVEL = 0.95


def cv_interval(
    fold_scores: list[float],
    training_rows: float,
    validation_rows: float,
    strategy: str,
) -> dict[str, Any]:
    """Confidence interval for the mean cross-validation score.

    Args:
        fold_scores: Per-fold scores of the selected candidate.
        training_rows: Mean training rows per fold.
        validation_rows: Mean validation rows per fold.
        strategy: Validation strategy name (stratified, random, group,
            temporal); `temporal` and `group` add their own remark to the
            note.

    Returns:
        With at least two folds, a mapping with `available` (True), `level`
        (0.95), the `mean`, the `naive_standard_error`, the
        `corrected_standard_error`, the t-based `interval` as a two-item
        list, `folds` and a `note` on how far to trust it. With fewer
        folds, only `available` (False) and a `reason`.
    """
    scores = np.asarray(fold_scores, dtype=float)
    folds = len(scores)
    if folds < 2:
        return {"available": False, "reason": "needs at least two folds"}
    mean = float(scores.mean())
    variance = float(scores.var(ddof=1))
    naive = math.sqrt(variance / folds)
    corrected = math.sqrt(
        (1.0 / folds + validation_rows / max(training_rows, 1.0)) * variance
    )
    margin = float(stats.t.ppf(0.5 + LEVEL / 2, folds - 1)) * corrected
    note = (
        "overlapping training sets make fold scores dependent; the "
        "Nadeau-Bengio corrected error is reported"
    )
    if strategy == "temporal":
        note += (
            "; expanding temporal windows are more dependent still, so the "
            "interval is optimistic"
        )
    elif strategy == "group":
        note += "; folds are group-disjoint, so entities are independent units"
    return {
        "available": True,
        "level": LEVEL,
        "mean": mean,
        "naive_standard_error": naive,
        "corrected_standard_error": corrected,
        "interval": [mean - margin, mean + margin],
        "folds": folds,
        "note": note,
    }
