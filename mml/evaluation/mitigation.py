"""Reducing outcome gaps between groups by moving decision thresholds.

After a model is chosen, each group of a sensitive column can get its own
probability threshold. Two criteria are supported: equal selection rates
(demographic parity) or equal true-positive rates (equal opportunity). The
change is judged on out-of-fold development predictions and kept only if it
shrinks the gap enough without costing too much accuracy; the evidence is
recorded either way.
"""

import dataclasses
from typing import Any

import numpy as np
import pandas as pd

from mml.models import thresholds

MIN_POSITIVES = 10
CRITERIA = ("demographic_parity", "equal_opportunity")


@dataclasses.dataclass(frozen=True)
class Mitigation:
    """A judged set of group thresholds.

    Attributes:
        column: Sensitive column.
        criterion: Gap the thresholds aim to close.
        thresholds: Group name to threshold.
        gap_before: Largest between-group difference with one threshold.
        gap_after: The same with group thresholds.
        accuracy_before: Accuracy with one threshold.
        accuracy_after: Accuracy with group thresholds.
        applied: Whether the thresholds are kept.
        reason: Why they were kept or dropped.
    """

    column: str
    criterion: str
    thresholds: dict[str, float]
    gap_before: float
    gap_after: float
    accuracy_before: float
    accuracy_after: float
    applied: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)


def rates(
    groups: pd.Series, flagged: np.ndarray, actual: np.ndarray, criterion: str
) -> dict[str, float]:
    """Per-group selection rate or true-positive rate."""
    found = {}
    for level in groups.dropna().unique():
        rows = (groups == level).to_numpy()
        if rows.sum() < thresholds.MIN_GROUP_ROWS:
            continue
        if criterion == "demographic_parity":
            found[str(level)] = float(flagged[rows].mean())
        elif (actual[rows]).sum() >= MIN_POSITIVES:
            found[str(level)] = float(
                (flagged[rows] & actual[rows]).sum() / actual[rows].sum()
            )
    return found


def gap(
    groups: pd.Series, flagged: np.ndarray, actual: np.ndarray, criterion: str
) -> float:
    """Largest between-group difference in the criterion's rate."""
    found = list(rates(groups, flagged, actual, criterion).values())
    return float(max(found) - min(found)) if len(found) > 1 else 0.0


def group_cuts(
    probabilities: np.ndarray,
    groups: pd.Series,
    actual: np.ndarray,
    criterion: str,
    base: float,
) -> dict[str, float]:
    """Threshold per group that matches the overall rate at `base`."""
    flagged = probabilities >= base
    if criterion == "demographic_parity":
        target = float(flagged.mean())
    else:
        target = float(flagged[actual].mean()) if actual.any() else 0.0
    cuts = {}
    for level in groups.dropna().unique():
        rows = (groups == level).to_numpy()
        pool = rows if criterion == "demographic_parity" else rows & actual
        enough = (
            rows.sum() >= thresholds.MIN_GROUP_ROWS
            if criterion == "demographic_parity"
            else pool.sum() >= MIN_POSITIVES
        )
        if not enough:
            continue
        cuts[str(level)] = float(
            np.clip(np.quantile(probabilities[pool], 1.0 - target), 0.0, 1.0)
        )
    return cuts


def judge(
    probabilities: np.ndarray,
    groups: pd.Series,
    actual: np.ndarray,
    column: str,
    criterion: str,
    base: float,
    trigger_gap: float,
    min_reduction: float,
    max_accuracy_cost: float,
) -> Mitigation:
    """Finds group thresholds and decides whether they earn their place.

    Args:
        probabilities: Out-of-fold positive-class probabilities.
        groups: Group name per row.
        actual: True where the label is the positive class.
        column: Sensitive column name.
        criterion: `demographic_parity` or `equal_opportunity`.
        base: The single threshold in use.
        trigger_gap: Smallest gap worth acting on.
        min_reduction: Share of the gap the thresholds must remove.
        max_accuracy_cost: Most accuracy that may be given up.

    Returns:
        The judged mitigation with its evidence.
    """
    cuts = group_cuts(probabilities, groups, actual, criterion, base)
    assigned = groups.map(cuts).fillna(base).to_numpy(float)
    before_flags = probabilities >= base
    after_flags = probabilities >= assigned
    before = gap(groups, before_flags, actual, criterion)
    after = gap(groups, after_flags, actual, criterion)
    accuracy_before = float((before_flags == actual).mean())
    accuracy_after = float((after_flags == actual).mean())
    if before <= trigger_gap:
        applied, reason = False, f"gap {before:.3f} is within {trigger_gap}"
    elif after > before * (1.0 - min_reduction):
        applied, reason = (
            False,
            (f"thresholds cut the gap only from {before:.3f} to {after:.3f}"),
        )
    elif accuracy_before - accuracy_after > max_accuracy_cost:
        applied, reason = (
            False,
            (
                f"accuracy cost {accuracy_before - accuracy_after:.3f} exceeds "
                f"{max_accuracy_cost}"
            ),
        )
    else:
        applied, reason = (
            True,
            (
                f"gap {before:.3f} -> {after:.3f} for an accuracy cost of "
                f"{accuracy_before - accuracy_after:.3f}"
            ),
        )
    return Mitigation(
        column,
        criterion,
        cuts,
        before,
        after,
        accuracy_before,
        accuracy_after,
        applied,
        reason,
    )
