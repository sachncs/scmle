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

Text near-copies (`resemble`): the non-numeric columns of a row are
read as one text and compared by MinHash with locality-sensitive hashing
(`minhash`), so two rows whose texts are near copies (Jaccard similarity of
their shingles at least `DEFAULT_SIMILARITY`, derived in
`minhash.similarity` from the edits a near copy may make) are compared on
their numeric columns too, and are near duplicates when those are within
the cut. Rows with no numeric column are near duplicates on the text alone.

The duplicate rate carries an exact (Clopper-Pearson) binomial interval.
"""

from collections.abc import Sequence
import dataclasses
import math
from typing import Any
from typing import ClassVar

import numpy
import pandas
from scipy import stats
from sklearn import neighbors

from scmle.assurance.detectors import minhash
from scmle.contracts import errors
from scmle.contracts import records
import scmle.data.splitting.duplicates

DEFAULT_CUT = 0.01
DEFAULT_SIMILARITY = minhash.similarity()
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
        near: Rows within the cut of an earlier, non-identical row.
        cut: Distance cut used (column standard deviations, RMS).
        conflicts: Exact groups whose rows disagree on the target
            (0 when no target was given).
        groups: Number of exact duplicate groups.
        similar: Rows whose text nearly copies an earlier row's (MinHash),
            with numeric columns within the cut, that are neither exact
            nor numeric near duplicates.
        similarity: Jaccard threshold of the text comparison.
        migrations: The written layout's version steps, oldest first;
            version 2 writes `exact.{extra, rate, interval}`, `near` and
            `conflicts`; version 3 adds `text.{near, threshold}`.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            {"near_extra": "near", "label_conflicts": "conflicts"},
            groups={
                "exact": {
                    "exact_extra": "extra",
                    "exact_rate": "rate",
                    "exact_interval": "interval",
                }
            },
        ),
        records.nest(
            {}, groups={"text": {"similar": "near", "similarity": "threshold"}}
        ),
    )

    rows: int
    exact_extra: int
    exact_rate: float
    exact_interval: tuple[float, float]
    near: int
    cut: float
    conflicts: int
    groups: int
    similar: int = 0
    similarity: float = DEFAULT_SIMILARITY

    def to_dict(self) -> dict[str, Any]:
        """Returns the report as a JSON-ready record.

        Returns:
            A mapping of every field to its value.
        """
        return records.upgrade(
            self.migrations, dataclasses.asdict(self), records.ORIGIN
        )


def verbatim(frame: pandas.DataFrame, columns: Sequence[str]) -> numpy.ndarray:
    """Flags every row after the first of its exact-duplicate group.

    Args:
        frame: Table.
        columns: Columns that define a record.

    Returns:
        A boolean per row, True for a row that repeats an earlier row.
    """
    ids = scmle.data.splitting.duplicates.group(frame, list(columns))
    flags = numpy.zeros(len(frame), dtype=bool)
    seen: set[int] = set()
    for position, group in enumerate(ids):
        if group == scmle.data.splitting.duplicates.NOT_DUPLICATED:
            continue
        if int(group) in seen:
            flags[position] = True
        seen.add(int(group))
    return flags


def approximate(
    frame: pandas.DataFrame, columns: Sequence[str], cut: float = DEFAULT_CUT
) -> numpy.ndarray:
    """True for rows within `cut` of an earlier row in the same block.

    Args:
        frame: Table.
        columns: Columns that define a record.
        cut: Maximum RMS standardized distance, positive.

    Returns:
        A boolean per row. Rows that are exact copies are also flagged here
        when within the cut; combine with `verbatim` as needed.

    Raises:
        ValueError: If `cut` is not positive and finite.
    """
    if not math.isfinite(cut) or cut <= 0.0:
        raise errors.InputError("cut must be positive and finite")
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
    block = standardize(frame, numeric)
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


def resemble(
    frame: pandas.DataFrame,
    columns: Sequence[str],
    cut: float = DEFAULT_CUT,
    similarity: float = DEFAULT_SIMILARITY,
) -> numpy.ndarray:
    """True for rows whose text nearly copies an earlier row's.

    The non-numeric columns of a row, joined, are its text; texts at or
    above `similarity` (exact Jaccard, after MinHash proposes the pair) are
    compared on the numeric columns as in `approximate`.

    Args:
        frame: Table.
        columns: Columns that define a record.
        cut: Maximum RMS standardized distance, positive.
        similarity: Jaccard threshold of the texts, within (0, 1].

    Returns:
        A boolean per row; rows whose texts are identical are left to
        `approximate` and `verbatim`.

    Raises:
        InputError: If `cut` is not positive and finite, or `similarity`
            is outside (0, 1].
    """
    if not math.isfinite(cut) or cut <= 0.0:
        raise errors.InputError("cut must be positive and finite")
    bands = minhash.Bands(similarity)
    flags = numpy.zeros(len(frame), dtype=bool)
    numeric = [
        c
        for c in columns
        if pandas.api.types.is_numeric_dtype(frame[c])
        and not pandas.api.types.is_bool_dtype(frame[c])
    ]
    other = [c for c in columns if c not in numeric]
    if not other or len(frame) < MIN_BLOCK:
        return flags
    text = (
        frame[other].astype(str).fillna("nan").agg(" ".join, axis=1).to_numpy()
    )
    keys, labels = minhash.group([minhash.shingles(t) for t in text])
    block = standardize(frame, numeric)
    radius = cut * math.sqrt(max(len(numeric), 1))
    for link in minhash.links(keys, bands):
        mine = numpy.flatnonzero(labels == link[0])
        theirs = numpy.flatnonzero(labels == link[1])
        if not numeric:
            flags[mine[mine > theirs[0]]] = True
            flags[theirs[theirs > mine[0]]] = True
            continue
        tree = neighbors.NearestNeighbors(radius=radius).fit(block[mine])
        found = tree.radius_neighbors(block[theirs], return_distance=False)
        for row, near in zip(theirs, found, strict=True):
            for peer in mine[near]:
                flags[max(row, peer)] = True
    return flags


def standardize(
    frame: pandas.DataFrame, numeric: Sequence[str]
) -> numpy.ndarray:
    """Scales numeric columns by their spread, filling non-finite values.

    Args:
        frame: Table.
        numeric: Numeric columns.

    Returns:
        The scaled block, one row per row of `frame` (no columns when
        `numeric` is empty).
    """
    if not numeric:
        return numpy.zeros((len(frame), 0))
    # Non-finite entries count as missing: fill them with the column mean,
    # then scale each column by its spread.
    spread = frame[list(numeric)].std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    block = frame[list(numeric)].to_numpy(dtype=float, copy=True)
    block[~numpy.isfinite(block)] = numpy.nan
    block = numpy.where(numpy.isnan(block), numpy.nanmean(block, axis=0), block)
    return numpy.asarray(block / spread.to_numpy(dtype=float))


def flag(
    frame: pandas.DataFrame,
    cut: float = DEFAULT_CUT,
    similarity: float = DEFAULT_SIMILARITY,
) -> pandas.Series:
    """Flags rows that repeat an earlier row exactly or nearly.

    A `cut` that is not positive and finite raises `InputError`.

    Args:
        frame: Table; every column defines a record.
        cut: Largest root-mean-square standardized distance over the numeric
            columns at which two rows are near duplicates; rows must also
            match exactly on the non-numeric columns, or be text near
            copies (`resemble`).
        similarity: Jaccard threshold of the text comparison.

    Returns:
        A boolean Series on the index of `frame`.
    """
    columns = list(frame.columns)
    flags = (
        verbatim(frame, columns)
        | approximate(frame, columns, cut)
        | resemble(frame, columns, cut, similarity)
    )
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


def assess(
    frame: pandas.DataFrame,
    columns: Sequence[str] | None = None,
    target: str | None = None,
    cut: float = DEFAULT_CUT,
    similarity: float = DEFAULT_SIMILARITY,
) -> Duplicates:
    """Counts exact, near and text-near duplicates and label conflicts.

    Args:
        frame: Exploration rows.
        columns: Columns that define a record; defaults to all.
        target: Optional label column checked for conflicts among exact
            duplicates.
        cut: Near-duplicate distance cut.
        similarity: Jaccard threshold of the text comparison.

    Returns:
        The report.
    """
    names = list(frame.columns) if columns is None else list(columns)
    exact = verbatim(frame, names)
    near = approximate(frame, names, cut) & ~exact
    similar = resemble(frame, names, cut, similarity) & ~exact & ~near
    ids = scmle.data.splitting.duplicates.group(frame, names)
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
        near=int(near.sum()),
        cut=cut,
        conflicts=conflicts,
        groups=len(numpy.unique(members)),
        similar=int(similar.sum()),
        similarity=similarity,
    )
