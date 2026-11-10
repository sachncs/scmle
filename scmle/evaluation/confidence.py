"""Uncertainty of a cross-validated score that respects fold dependence.

Fold scores are not independent observations: every pair of training sets
overlaps heavily. Treating them as independent makes the usual standard error
too small. The Nadeau-Bengio correction inflates the variance by the ratio of
validation to training size, which is the standard remedy for resampled
estimates; for temporal validation the windows are even more dependent, so
the interval is flagged as optimistic.

The winner of a search was picked because its mean score was the best of
several, so its mean flatters it, and the more candidates were tried, the
more it does. `selection_optimism` estimates that flattery from the fold
scores already at hand, and `cv_interval` moves the interval by it.
"""

import math
from typing import Any

import numpy
from scipy import stats

LEVEL = 0.95
MIN_FOLDS = 2
MIN_CANDIDATES = 2


def selection_optimism(
    selected_scores: list[float],
    candidate_scores: list[list[float]],
    greater_is_better: bool,
) -> float:
    """Estimates how much choosing the best candidate flatters its mean score.

    For each fold the candidate with the best mean over the other folds is
    picked, as the search would have picked it without that fold, and scored
    on the fold. The mean of those scores is what choosing a winner delivers
    on rows it did not choose on. The optimism is how far the selected
    candidate's own mean lies above it (below it for a loss), and never
    negative. The pick uses the plain best mean, not the run's tie-breaking
    rules, and the folds are not independent, so this is an estimate.

    Args:
        selected_scores: Per-fold scores of the selected candidate.
        candidate_scores: Per-fold scores of every finished candidate,
            including the selected one, over the same folds.
        greater_is_better: Whether a higher score is better.

    Returns:
        The optimism in score units; 0 for a single candidate.
    """
    folds = len(selected_scores)
    comparable = numpy.asarray(
        [scores for scores in candidate_scores if len(scores) == folds],
        dtype=float,
    )
    if len(comparable) < MIN_CANDIDATES:
        return 0.0
    sign = 1.0 if greater_is_better else -1.0
    comparable = sign * comparable
    honest = [
        comparable[
            numpy.argmax(numpy.delete(comparable, fold, axis=1).mean(axis=1)),
            fold,
        ]
        for fold in range(folds)
    ]
    apparent = sign * float(numpy.mean(selected_scores))
    return max(apparent - float(numpy.mean(honest)), 0.0)


def cv_interval(
    fold_scores: list[float],
    training_rows: float,
    validation_rows: float,
    strategy: str,
    candidate_scores: list[list[float]] | None = None,
    greater_is_better: bool = True,
) -> dict[str, Any]:
    """Confidence interval for the mean cross-validation score.

    Args:
        fold_scores: Per-fold scores of the selected candidate.
        training_rows: Mean training rows per fold.
        validation_rows: Mean validation rows per fold.
        strategy: Validation strategy name (stratified, random, group,
            temporal); `temporal` and `group` add their own remark to the
            note.
        candidate_scores: Per-fold scores of every finished candidate the
            winner was chosen among, including the winner; None or a single
            candidate leaves the interval centred on the mean.
        greater_is_better: Whether a higher score is better; it sets the
            direction the selection optimism is removed in.

    Returns:
        With at least two folds, a mapping with `available` (True), `level`
        (0.95), the `mean`, the `candidates` compared, the
        `selection_optimism` removed, the `naive_standard_error`, the
        `corrected_standard_error`, the t-based `interval` (centred on the mean
        less the optimism) as a two-item list, `folds` and a `note` on how far
        to trust it. With fewer folds, only `available` (False) and a
        `reason`.
    """
    scores = numpy.asarray(fold_scores, dtype=float)
    folds = len(scores)
    if folds < MIN_FOLDS:
        return {"available": False, "reason": "needs at least two folds"}
    mean = float(scores.mean())
    variance = float(scores.var(ddof=1))
    naive = math.sqrt(variance / folds)
    corrected = math.sqrt(
        (1.0 / folds + validation_rows / max(training_rows, 1.0)) * variance
    )
    margin = float(stats.t.ppf(0.5 + LEVEL / 2, folds - 1)) * corrected
    optimism = selection_optimism(
        fold_scores, candidate_scores or [], greater_is_better
    )
    centre = mean - optimism if greater_is_better else mean + optimism
    note = (
        "overlapping training sets make fold scores dependent; the "
        "Nadeau-Bengio corrected error is reported"
    )
    if optimism > 0:
        note += (
            f"; the winner was chosen among {len(candidate_scores or [])} "
            f"candidates and the interval is moved by {optimism:.4f} for "
            "that, an estimate that assumes the winner is the best mean"
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
        "candidates": len(candidate_scores or []),
        "selection_optimism": optimism,
        "naive_standard_error": naive,
        "corrected_standard_error": corrected,
        "interval": [centre - margin, centre + margin],
        "folds": folds,
        "note": note,
    }
