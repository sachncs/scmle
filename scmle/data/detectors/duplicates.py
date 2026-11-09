"""Exact and near-duplicate rows.

Exact duplicates reuse the grouping of `scmle.data.splitting.duplicates`
(floats rounded to six decimals, text exact), so detection and the
group-aware split agree. Near-duplicates are found by blocking and a
nearest-neighbour distance: non-numeric columns form a blocking key (rows
differing there are never compared); numeric columns are scaled by their
standard deviation and the distance is the root-mean-square difference per
column, in units of column standard deviations. The default cut is a policy
value selected by the certification harness from a grid on clean data (see
`benchmarks/qualification/f5.py`), not chosen by hand.

MinHash on tokenised strings is not implemented: text columns only block
exact matches.

The duplicate rate carries an exact (Clopper-Pearson) binomial interval.
"""

from collections.abc import Sequence
import dataclasses
import math
from typing import Any

import numpy
import pandas
from scipy import stats
from sklearn import neighbors

import scmle.data.splitting.duplicates

DEFAULT_CUT = 0.01
MIN_BLOCK = 2
CONFIDENCE = 0.95


@dataclasses.dataclass(frozen=True)
class Duplicates:
    """Duplicate evidence for a table.

    Attributes:
        rows: Rows examined.
        exact_extra: Rows that repeat an earlier row exactly.
        exact_rate: `exact_extra / rows`.
        exact_interval: Clopper-Pearson interval for the rate.
        near_extra: Rows within the cut of an earlier, non-identical row.
        cut: Distance cut used (column standard deviations, RMS).
        label_conflicts: Exact groups whose rows disagree on the target
            (0 when no target was given).
        groups: Number of exact duplicate groups.
    """

    rows: int
    exact_extra: int
    exact_rate: float
    exact_interval: tuple[float, float]
    near_extra: int
    cut: float
    label_conflicts: int
    groups: int

    def to_dict(self) -> dict[str, Any]:
        """Returns the report as a JSON-ready record.

        Returns:
            A mapping of every field to its value.
        """
        return dataclasses.asdict(self)


def exact_flags(
    frame: pandas.DataFrame, columns: Sequence[str]
) -> numpy.ndarray:
    """Flags every row after the first of its exact-duplicate group.

    Args:
        frame: Table.
        columns: Columns that define a record.

    Returns:
        A boolean per row, True for a row that repeats an earlier row.
    """
    ids = scmle.data.splitting.duplicates.duplicate_groups(frame, list(columns))
    flags = numpy.zeros(len(frame), dtype=bool)
    seen: set[int] = set()
    for position, group in enumerate(ids):
        if group == scmle.data.splitting.duplicates.NOT_DUPLICATED:
            continue
        if int(group) in seen:
            flags[position] = True
        seen.add(int(group))
    return flags


def standardized_block(
    values: pandas.DataFrame, spread: pandas.Series
) -> numpy.ndarray:
    """Scales a numeric block by column spread, filling gaps with the mean.

    Args:
        values: Numeric columns. Non-finite entries count as missing.
        spread: Spread (standard deviation) of each column, nonzero.

    Returns:
        A float array with missing entries replaced by the column mean, then
        divided by `spread`.
    """
    block = values.to_numpy(dtype=float, copy=True)
    block[~numpy.isfinite(block)] = numpy.nan
    centre = numpy.nanmean(block, axis=0)
    block = numpy.where(numpy.isnan(block), centre, block)
    return numpy.asarray(block / spread.to_numpy(dtype=float))


def near_flags(
    frame: pandas.DataFrame, columns: Sequence[str], cut: float = DEFAULT_CUT
) -> numpy.ndarray:
    """True for rows within `cut` of an earlier row in the same block.

    Args:
        frame: Table.
        columns: Columns that define a record.
        cut: Maximum RMS standardized distance, positive.

    Returns:
        A boolean per row. Rows that are exact copies are also flagged here
        when within the cut; combine with `exact_flags` as needed.

    Raises:
        ValueError: If `cut` is not positive and finite.
    """
    if not math.isfinite(cut) or cut <= 0.0:
        raise ValueError("cut must be positive and finite")
    flags = numpy.zeros(len(frame), dtype=bool)
    numeric = [
        c
        for c in columns
        if pandas.api.types.is_numeric_dtype(frame[c])
        and not pandas.api.types.is_bool_dtype(frame[c])
    ]
    other = [c for c in columns if c not in numeric]
    if not numeric:
        return flags
    spread = frame[numeric].std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    block = standardized_block(frame[numeric], spread)
    if other:
        text = frame[other].astype(str).fillna("<missing>")
        keys = text.agg("\x1f".join, axis=1).to_numpy()
        labels = pandas.factorize(keys)[0]
    else:
        labels = numpy.zeros(len(frame), dtype=int)
    radius = cut * math.sqrt(len(numeric))
    for label in numpy.unique(labels):
        members = numpy.flatnonzero(labels == label)
        if len(members) < MIN_BLOCK:
            continue
        tree = neighbors.NearestNeighbors(radius=radius).fit(block[members])
        near = tree.radius_neighbors(block[members], return_distance=False)
        for local, row_neighbours in zip(members, near, strict=True):
            earlier = members[row_neighbours]
            if numpy.any(earlier < local):
                flags[local] = True
    return flags


def duplicate_flags(
    frame: pandas.DataFrame, cut: float = DEFAULT_CUT
) -> pandas.Series:
    """Flags rows that repeat an earlier row exactly or within the near cut.

    A `cut` that is not positive and finite raises `ValueError`.

    Args:
        frame: Table; every column defines a record.
        cut: Largest root-mean-square standardized distance over the numeric
            columns at which two rows are near duplicates; rows must also
            match exactly on the non-numeric columns.

    Returns:
        A boolean Series on the index of `frame`.
    """
    columns = list(frame.columns)
    flags = exact_flags(frame, columns) | near_flags(frame, columns, cut)
    return pandas.Series(flags, index=frame.index)


def clopper_pearson(
    successes: int, n: int, confidence: float = CONFIDENCE
) -> tuple[float, float]:
    """Computes an exact two-sided binomial interval, via scipy.

    Args:
        successes: Number of successes.
        n: Number of trials.
        confidence: Confidence level of the interval.

    Returns:
        The lower and upper limits of the interval for the proportion.
    """
    result = stats.binomtest(successes, n).proportion_ci(confidence)
    return float(result.low), float(result.high)


def assess_duplicates(
    frame: pandas.DataFrame,
    columns: Sequence[str] | None = None,
    target: str | None = None,
    cut: float = DEFAULT_CUT,
) -> Duplicates:
    """Counts exact and near duplicates and label conflicts.

    Args:
        frame: Exploration rows.
        columns: Columns that define a record; defaults to all.
        target: Optional label column checked for conflicts among exact
            duplicates.
        cut: Near-duplicate distance cut.

    Returns:
        The report.
    """
    names = list(frame.columns) if columns is None else list(columns)
    exact = exact_flags(frame, names)
    near = near_flags(frame, names, cut) & ~exact
    ids = scmle.data.splitting.duplicates.duplicate_groups(frame, names)
    members = ids[ids != scmle.data.splitting.duplicates.NOT_DUPLICATED]
    conflicts = 0
    if target is not None:
        for group in numpy.unique(members):
            labels = frame[target].to_numpy()[ids == group]
            if len(pandas.unique(pandas.Series(labels))) > 1:
                conflicts += 1
    extra = int(exact.sum())
    return Duplicates(
        rows=len(frame),
        exact_extra=extra,
        exact_rate=extra / len(frame),
        exact_interval=clopper_pearson(extra, len(frame)),
        near_extra=int(near.sum()),
        cut=cut,
        label_conflicts=conflicts,
        groups=len(numpy.unique(members)),
    )
