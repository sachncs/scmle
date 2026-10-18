"""Reading tables larger than memory, in batches.

A big file is never loaded whole. One pass over its batches picks the locked
test rows by hashing each row's position (so the choice needs no memory and
is the same on every run), and keeps a bounded random sample of the other
rows for development. Each row's sampling priority is also a hash of its
position, so the sample does not depend on the batch size.
"""

import abc
from collections.abc import Iterator
import dataclasses
import pathlib
from typing import ClassVar

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scml.contracts import errors

SPLIT_COLUMN = "split_side"
DEVELOPMENT = 0
LOCKED = 1
BATCH_ROWS = 50_000
CSV_EXPANSION = 4
BYTES_PER_CELL = 8
SAMPLE_SHARE = 0.5
MIN_SAMPLE_ROWS = 1_000
MIX = np.uint64(0x9E3779B97F4A7C15)
MIX_A = np.uint64(0xBF58476D1CE4E5B9)
MIX_B = np.uint64(0x94D049BB133111EB)


def row_hash(positions: np.ndarray, seed: int) -> np.ndarray:
    """A deterministic number in [0, 1) for each row position (SplitMix64)."""
    with np.errstate(over="ignore"):
        z = positions.astype(np.uint64) + MIX * np.uint64(seed + 1)
        z = (z ^ (z >> np.uint64(30))) * MIX_A
        z = (z ^ (z >> np.uint64(27))) * MIX_B
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


class ChunkSource(abc.ABC):
    """A table that can be read batch by batch.

    Attributes:
        suffixes: File suffixes the source handles.
        path: The file.
    """

    suffixes: ClassVar[tuple[str, ...]]

    def __init__(self, path: pathlib.Path) -> None:
        """Binds the source to a file."""
        self.path = path

    @abc.abstractmethod
    def batches(self, rows: int = BATCH_ROWS) -> Iterator[pd.DataFrame]:
        """Yields the table in consecutive frames of about `rows` rows."""

    @abc.abstractmethod
    def shape(self) -> tuple[int | None, int]:
        """Row count (None if unknown without a pass) and column count."""

    def estimated_bytes(self) -> int:
        """In-memory size of the whole table (an estimate for CSV)."""
        rows, columns = self.shape()
        if rows is None:
            return self.path.stat().st_size * CSV_EXPANSION
        return rows * columns * BYTES_PER_CELL


class ParquetChunks(ChunkSource):
    """Parquet read by row group."""

    suffixes = (".parquet", ".pq")

    def batches(self, rows: int = BATCH_ROWS) -> Iterator[pd.DataFrame]:
        """Yields Arrow record batches as frames.

        Pre-buffering and threads are off: with them Arrow keeps decoded
        pages of many row groups alive and the reader's memory grows with
        the file instead of staying at one row group.
        """
        reader = pq.ParquetFile(self.path, pre_buffer=False)
        for batch in reader.iter_batches(batch_size=rows, use_threads=False):
            yield batch.to_pandas()

    def shape(self) -> tuple[int | None, int]:
        """Both numbers come from the file's metadata."""
        meta = pq.ParquetFile(self.path).metadata
        return meta.num_rows, meta.num_columns


class CsvChunks(ChunkSource):
    """CSV read in chunks (types are inferred per chunk)."""

    suffixes = (".csv",)

    def batches(self, rows: int = BATCH_ROWS) -> Iterator[pd.DataFrame]:
        """Yields consecutive chunks."""
        yield from pd.read_csv(self.path, chunksize=rows)

    def shape(self) -> tuple[int | None, int]:
        """Only the column count is known without a pass."""
        return None, len(pd.read_csv(self.path, nrows=1).columns)


SOURCES: tuple[type[ChunkSource], ...] = (ParquetChunks, CsvChunks)


def source_for(uri: str | pathlib.Path) -> ChunkSource:
    """The chunked reader for a file.

    Raises:
        DataError: For a missing file or a format that cannot be streamed.
    """
    path = pathlib.Path(uri)
    if not path.exists():
        raise errors.DataError(f"dataset not found: {path}")
    for kind in SOURCES:
        if path.suffix.lower() in kind.suffixes:
            return kind(path)
    raise errors.DataError(
        f"cannot stream '{path.suffix}' files; use Parquet or CSV"
    )


def side_of(positions: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Development (0) or locked (1) for each row position."""
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


def sample_caps(
    columns: int, budget_gb: float, test_fraction: float
) -> tuple[int, int]:
    """Rows to keep in memory for development and for the sampled test."""
    cells = max(columns, 1) * BYTES_PER_CELL
    rows = int(budget_gb * 1e9 * SAMPLE_SHARE / cells)
    development = max(int(rows * (1 - test_fraction)), MIN_SAMPLE_ROWS)
    locked = max(int(rows * test_fraction), MIN_SAMPLE_ROWS // 4)
    return development, locked


def keep_smallest(
    buffer: pd.DataFrame | None, batch: pd.DataFrame, cap: int
) -> pd.DataFrame:
    """Merges a batch into a buffer, keeping the `cap` lowest-priority rows."""
    merged = batch if buffer is None else pd.concat([buffer, batch])
    if len(merged) > cap:
        merged = merged.nsmallest(cap, "priority")
    return merged


def sample_pass(
    source: ChunkSource,
    development_cap: int,
    locked_cap: int,
    test_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, Totals]:
    """One pass: hash-assign every row, keep a priority sample of each side.

    Returns:
        The kept rows with a `split_side` column, and the pass totals.
    """
    dev: pd.DataFrame | None = None
    locked: pd.DataFrame | None = None
    offset = 0
    locked_total = 0
    for batch in source.batches():
        positions = np.arange(offset, offset + len(batch))
        offset += len(batch)
        side = side_of(positions, test_fraction, seed)
        locked_total += int(side.sum())
        priority = row_hash(positions, seed + 7919)
        tagged = batch.assign(**{SPLIT_COLUMN: side, "priority": priority})
        dev = keep_smallest(dev, tagged[side == DEVELOPMENT], development_cap)
        locked = keep_smallest(locked, tagged[side == LOCKED], locked_cap)
    if dev is None or locked is None or dev.empty or locked.empty:
        raise errors.DataError("the file is too small to split")
    kept = pd.concat([dev, locked]).sort_values("priority")
    kept = kept.drop(columns="priority").reset_index(drop=True)
    return kept, Totals(offset, locked_total, len(dev), len(locked))
