"""One pass that locks test rows by position hash and samples the rest."""

import dataclasses

import numpy
import pandas

from scmle.contracts import errors
from scmle.data.readers.streaming import feed

SPLIT_COLUMN = "split_side"
DEVELOPMENT = 0
LOCKED = 1
SAMPLE_SHARE = 0.5
MIN_SAMPLE_ROWS = 1_000
MIX = numpy.uint64(0x9E3779B97F4A7C15)
MIX_A = numpy.uint64(0xBF58476D1CE4E5B9)
MIX_B = numpy.uint64(0x94D049BB133111EB)


def row_hash(positions: numpy.ndarray, seed: int) -> numpy.ndarray:
    """A deterministic number in [0, 1) for each row position (SplitMix64).

    Args:
        positions: Integer row positions (zero-based) to hash.
        seed: Integer that selects the hash stream; different seeds give
            different, independent-looking values for the same position.

    Returns:
        A float64 array with one value in [0, 1) per entry of `positions`,
        in the same order.
    """
    with numpy.errstate(over="ignore"):
        z = positions.astype(numpy.uint64) + MIX * numpy.uint64(seed + 1)
        z = (z ^ (z >> numpy.uint64(30))) * MIX_A
        z = (z ^ (z >> numpy.uint64(27))) * MIX_B
        z = z ^ (z >> numpy.uint64(31))
    unit: numpy.ndarray = (z >> numpy.uint64(11)).astype(numpy.float64) / float(
        1 << 53
    )
    return unit


def allot(
    positions: numpy.ndarray, fraction: float, seed: int
) -> numpy.ndarray:
    """Assigns each row position to development (0) or locked (1).

    Args:
        positions: Integer row positions (zero-based).
        fraction: Share of rows to lock, between 0 and 1.
        seed: Seed of the position hash.

    Returns:
        An integer array, one entry per position: 1 (locked) where the
        position's hash is below `fraction`, otherwise 0 (development).
    """
    return (row_hash(positions, seed) < fraction).astype(int)


@dataclasses.dataclass(frozen=True)
class Totals:
    """What one pass over the file saw.

    Attributes:
        rows: All rows in the file.
        locked: Rows assigned to the locked test partition.
        development_sample: Development rows kept in memory.
        locked_sample: Locked rows kept in memory.
    """

    rows: int
    locked: int
    development_sample: int
    locked_sample: int


def budgets(
    columns: int, budget_gb: float, test_fraction: float
) -> tuple[int, int]:
    """Rows to keep in memory for development and for the sampled test.

    Args:
        columns: Number of columns in the table.
        budget_gb: Memory budget in gigabytes.
        test_fraction: Share of rows assigned to the locked side.

    Returns:
        The development row cap and the locked row cap. Half the budget
        is used (8 bytes per cell) and split by `test_fraction`; each cap
        has a floor of 1,000 and 250 rows respectively.
    """
    cells = max(columns, 1) * feed.BYTES_PER_CELL
    rows = int(budget_gb * 1e9 * SAMPLE_SHARE / cells)
    development = max(int(rows * (1 - test_fraction)), MIN_SAMPLE_ROWS)
    locked = max(int(rows * test_fraction), MIN_SAMPLE_ROWS // 4)
    return development, locked


def retain(
    buffer: pandas.DataFrame | None, batch: pandas.DataFrame, cap: int
) -> pandas.DataFrame:
    """Merges a batch into a buffer, keeping the `cap` lowest-priority rows.

    Args:
        buffer: Rows kept so far, or None if there are none.
        batch: New rows to merge in.
        cap: Maximum number of rows to keep.

    Returns:
        The merged frame; when it exceeds `cap` rows, only the `cap` rows
        with the smallest `priority` value remain.
    """
    merged = batch if buffer is None else pandas.concat([buffer, batch])
    if len(merged) > cap:
        merged = merged.nsmallest(cap, "priority")
    return merged


def sweep(
    source: feed.Feed,
    development_cap: int,
    locked_cap: int,
    test_fraction: float,
    seed: int,
) -> tuple[pandas.DataFrame, Totals]:
    """One pass: hash-assign every row, keep a priority sample of each side.

    Args:
        source: The table to read.
        development_cap: Maximum development rows to keep.
        locked_cap: Maximum locked rows to keep.
        test_fraction: Share of rows assigned to the locked side.
        seed: Seed of the position hash.

    Returns:
        The kept rows with a `split_side` column, ordered by sampling
        priority, and the totals of the pass.

    Raises:
        DataError: If either side ends up with no rows.
    """
    dev: pandas.DataFrame | None = None
    locked: pandas.DataFrame | None = None
    offset = 0
    locked_total = 0
    for batch in source.batches():
        positions = numpy.arange(offset, offset + len(batch))
        offset += len(batch)
        side = allot(positions, test_fraction, seed)
        locked_total += int(side.sum())
        priority = row_hash(positions, seed + 7919)
        tagged = batch.assign(**{SPLIT_COLUMN: side, "priority": priority})
        dev = retain(dev, tagged[side == DEVELOPMENT], development_cap)
        locked = retain(locked, tagged[side == LOCKED], locked_cap)
    if dev is None or locked is None or dev.empty or locked.empty:
        raise errors.DataError("the file is too small to split")
    kept = pandas.concat([dev, locked]).sort_values("priority")
    kept = kept.drop(columns="priority").reset_index(drop=True)
    return kept, Totals(offset, locked_total, len(dev), len(locked))
