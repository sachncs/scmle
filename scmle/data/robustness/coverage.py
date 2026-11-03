"""Subgroup coverage: sample share against deployment share (F6).

For each level of a named subgroup column the sample share carries an exact
(Clopper-Pearson) binomial interval. The deployment share comes from a
deployment sample or from stated marginals. Without either, every level is
`unverifiable`: the check never reports `pass` without a reference (canon P5).

A `pass` means the deployment share lies inside the sample-share interval,
not that the sample represents the population. The deployment share is
treated as exact, so with a small deployment sample the check is
anti-conservative. A coverage gap is where the validation design should
stratify.
"""

from collections.abc import Mapping
import dataclasses
from typing import Any

import pandas as pd
from scipy import stats

from scmle.data.robustness import shift

ALPHA = 0.05
MISSING_LEVEL = "<missing>"


@dataclasses.dataclass(frozen=True, slots=True)
class LevelCoverage:
    """Coverage of one level of one subgroup column.

    Attributes:
        column: Subgroup column.
        level: The level.
        count: Sample rows at the level.
        rows: Sample rows in total.
        share: ``count / rows``.
        interval: Exact interval for the sample share.
        deployment_share: Share in the deployment population, or None.
        status: `pass`, `fail` or `unverifiable`.
    """

    column: str
    level: str
    count: int
    rows: int
    share: float
    interval: tuple[float, float]
    deployment_share: float | None
    status: shift.Status

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        record = dataclasses.asdict(self)
        record["status"] = self.status.value
        record["interval"] = list(self.interval)
        return record


def shares(column: pd.Series) -> dict[str, float]:
    """Level shares of a column, missing values as their own level."""
    labels = column.astype("object").where(column.notna(), MISSING_LEVEL)
    return {
        str(k): float(v)
        for k, v in labels.astype(str).value_counts(normalize=True).items()
    }


def subgroup_coverage(
    sample: pd.Series,
    deployment: Mapping[str, float] | None,
    alpha: float = ALPHA,
) -> tuple[LevelCoverage, ...]:
    """Compares a column's level shares with the deployment shares.

    Args:
        sample: The column in the exploration set.
        deployment: Level to deployment share, or None when there is no
            reference (every level is then `unverifiable`).
        alpha: One minus the confidence of the interval, in (0, 1).

    Returns:
        One record per level seen in the sample or the deployment, sorted by
        level. A level absent from the sample has count 0.

    Raises:
        ValueError: On an alpha outside (0, 1) or an empty sample.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be within (0, 1)")
    rows = len(sample)
    if rows == 0:
        raise ValueError("the sample is empty")
    labels = sample.astype("object").where(sample.notna(), MISSING_LEVEL)
    counts = labels.astype(str).value_counts()
    levels = sorted(set(counts.index) | set(deployment or {}))
    out = []
    for level in levels:
        count = int(counts.get(level, 0))
        interval = stats.binomtest(count, rows).proportion_ci(
            confidence_level=1.0 - alpha, method="exact"
        )
        target = None if deployment is None else deployment.get(level, 0.0)
        if target is None:
            status = shift.Status.UNVERIFIABLE
        elif interval.low <= target <= interval.high:
            status = shift.Status.PASS
        else:
            status = shift.Status.FAIL
        out.append(
            LevelCoverage(
                str(sample.name),
                str(level),
                count,
                rows,
                count / rows,
                (float(interval.low), float(interval.high)),
                target,
                status,
            )
        )
    return tuple(out)
