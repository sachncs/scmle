"""Tables read batch by batch, one reader per file format."""

import abc
from collections.abc import Iterable
from collections.abc import Iterator
import pathlib
from typing import ClassVar

import pandas
import pyarrow.parquet

from scmle.contracts import errors
from scmle.data.readers import connectors
from scmle.data.readers.streaming import failure
from scmle.data.readers.streaming import progress

BATCH_ROWS = 50_000
CSV_EXPANSION = 4
BYTES_PER_CELL = 8


def align(
    checkpoint: progress.Checkpoint | None,
    current: progress.Checkpoint,
    watermark_column: str | None,
    identity: str,
) -> str | None:
    """Chooses the watermark column a read tracks.

    Args:
        checkpoint: The checkpoint the caller passed, or None.
        current: That checkpoint bound to the source, or a new one.
        watermark_column: The column the caller asked to track.
        identity: Identity of the source, used in error evidence.

    Returns:
        The checkpoint's own column when resuming past row zero, otherwise
        `watermark_column`.

    Raises:
        StreamingError: If a resumed read asks for a different column than
            the checkpoint's.
    """
    if (
        checkpoint is not None
        and current.offset > 0
        and current.column != watermark_column
    ):
        raise failure.StreamingError(
            "watermark_mismatch",
            "watermark column does not match the checkpoint",
            identity,
            {
                "checkpoint_watermark_column": current.column,
                "requested_watermark_column": watermark_column,
            },
        )
    return (
        current.column
        if checkpoint is not None and current.offset > 0
        else watermark_column
    )


def pick(
    frames: Iterable[pandas.DataFrame],
    position: int,
    rows: int,
    identity: str,
) -> tuple[pandas.DataFrame, bool]:
    """Takes the rows after a position from consecutive frames.

    Args:
        frames: The table's frames in file order.
        position: Number of rows already consumed.
        rows: Maximum number of rows to take; positive.
        identity: Identity of the source, used in error evidence.

    Returns:
        At most `rows` rows starting at `position`, taken from as many
        consecutive frames as needed, and whether rows remain after them.

    Raises:
        StreamingError: If a frame is not a DataFrame, or no rows remain
            after `position`.
    """
    parts: list[pandas.DataFrame] = []
    taken = 0
    offset = 0
    has_more = False
    for batch in frames:
        if not isinstance(batch, pandas.DataFrame):
            raise failure.StreamingError(
                "invalid_batch",
                "chunk source yielded a non-DataFrame batch",
                identity,
                {"type": type(batch).__name__},
            )
        batch_start = offset
        offset += len(batch)
        if offset <= position or batch.empty:
            continue
        if taken == rows:
            has_more = True
            break
        candidate = batch.iloc[max(0, position - batch_start) :]
        take = min(rows - taken, len(candidate))
        parts.append(candidate.iloc[:take])
        taken += take
        if len(candidate) > take:
            has_more = True
            break
    selected = pandas.concat(parts).copy() if parts else None
    if selected is None or selected.empty:
        raise failure.StreamingError(
            "end_of_source",
            "streaming source has no rows after the checkpoint",
            identity,
            {"offset": position},
        )
    return selected, has_more


class Feed(abc.ABC):
    """A table that can be read batch by batch.

    Attributes:
        suffixes: File suffixes the source handles.
        path: The file.
    """

    suffixes: ClassVar[tuple[str, ...]]

    def __init__(self, path: pathlib.Path) -> None:
        """Binds the source to a file.

        Args:
            path: The file to read.
        """
        self.path = path

    @abc.abstractmethod
    def batches(self, rows: int = BATCH_ROWS) -> Iterator[pandas.DataFrame]:
        """Yields the table in consecutive frames of about `rows` rows.

        Args:
            rows: Target number of rows per frame.

        Yields:
            Consecutive frames in file order that together cover the table.
        """

    @abc.abstractmethod
    def shape(self) -> tuple[int | None, int]:
        """Reports the row count and column count of the table.

        Returns:
            A pair of the row count and the column count. The row count is None
            when it cannot be known without reading the whole file.
        """

    def footprint(self) -> int:
        """Estimates the in-memory size of the whole table.

        The estimate is rows x columns x 8 bytes when the row count is known.
        Otherwise (CSV, JSON lines) it is the file size on disk times 4.

        Returns:
            The estimated size in bytes.
        """
        rows, columns = self.shape()
        if rows is None:
            return self.path.stat().st_size * CSV_EXPANSION
        return rows * columns * BYTES_PER_CELL

    def source_identity(self) -> str:
        """Returns the stable identity of this source URI.

        Returns:
            The string "src_" followed by the SHA-256 hex digest of the
            canonical URI of the file.
        """
        return connectors.source_identity(self.path)

    def snapshot_sha256(self) -> str:
        """Returns the digest of the raw source snapshot.

        The digest is intentionally over the source bytes, not the decoded
        frame. A resumed stream must therefore observe the same exact file,
        including formatting and metadata that a decoder might otherwise
        discard.

        Returns:
            The SHA-256 hex digest of the file's bytes.

        Raises:
            StreamingError: With code "source_unreadable" if the file cannot be
                read.
        """
        try:
            return connectors.checksum(self.path)
        except OSError as error:
            raise failure.StreamingError(
                "source_unreadable",
                f"cannot fingerprint streaming source: {self.path}",
                self.source_identity(),
                {"path": str(self.path)},
            ) from error

    def read_batch(
        self,
        checkpoint: progress.Checkpoint | None = None,
        rows: int = BATCH_ROWS,
        watermark_column: str | None = None,
    ) -> progress.Batch:
        """Reads one bounded batch bound to an immutable source snapshot.

        The source is re-opened from row zero for each call because file
        readers do not share a transport cursor. The durable checkpoint makes
        that replay safe; the raw digest prevents a changed file from being
        silently appended to an old run.

        Args:
            checkpoint: Position to resume from. None starts at row zero.
            rows: Maximum number of rows in the batch; must be positive.
            watermark_column: Column whose greatest timestamp is tracked as
                the watermark. When resuming from a checkpoint that is past
                row zero it must equal the checkpoint's own watermark column;
                a differing value is rejected.

        Returns:
            The next batch of at most `rows` rows after the checkpoint's
            offset, the checkpoint that follows it, and whether more rows
            remain.

        Raises:
            ValueError: If `rows` is not positive.
            StreamingError: If the checkpoint belongs to another source or
                snapshot, the watermark column differs from the checkpoint's,
                the source yields something other than a DataFrame, no rows
                remain after the checkpoint, the file changes while it is
                read, or the watermark column is missing or holds invalid
                timestamps.
        """
        if rows <= 0:
            raise errors.InputError("rows must be positive")
        identity = self.source_identity()
        observed_snapshot = self.snapshot_sha256()
        current = self.rebind(checkpoint, observed_snapshot)
        effective_watermark_column = align(
            checkpoint, current, watermark_column, identity
        )
        selected, has_more = pick(
            self.batches(rows), current.offset, rows, identity
        )
        final_snapshot = self.snapshot_sha256()
        if final_snapshot != observed_snapshot:
            raise failure.StreamingError(
                "source_snapshot_changed",
                "streaming source changed while it was being read",
                identity,
                {
                    "initial_snapshot_sha256": observed_snapshot,
                    "final_snapshot_sha256": final_snapshot,
                },
            )
        next_watermark = progress.advance(
            selected,
            effective_watermark_column,
            current.watermark,
            identity,
        )
        next_checkpoint = progress.Checkpoint(
            identity,
            {"snapshot": observed_snapshot},
            current.offset + len(selected),
            next_watermark,
            effective_watermark_column,
        )
        return progress.Batch(
            identity,
            observed_snapshot,
            current.offset,
            selected,
            next_checkpoint,
            has_more,
        )

    def rebind(
        self, checkpoint: progress.Checkpoint | None, snapshot: str
    ) -> progress.Checkpoint:
        """Binds a checkpoint to this source and current raw snapshot.

        Args:
            checkpoint: The checkpoint to check, or None for a new run.
            snapshot: The SHA-256 hex digest of the file as currently observed.

        Returns:
            `checkpoint` itself, or a new checkpoint at offset zero for this
            source and snapshot when `checkpoint` is None.

        Raises:
            StreamingError: If the checkpoint belongs to another source or was
                taken on a different snapshot of the file.
        """
        identity = self.source_identity()
        if checkpoint is None:
            return progress.Checkpoint(identity, {"snapshot": snapshot})
        if checkpoint.source != identity:
            raise failure.StreamingError(
                "checkpoint_source_mismatch",
                "checkpoint belongs to another streaming source",
                identity,
                {"checkpoint_source_identity": checkpoint.source},
            )
        if checkpoint.digests["snapshot"] != snapshot:
            raise failure.StreamingError(
                "source_snapshot_changed",
                "source snapshot does not match the checkpoint",
                identity,
                {
                    "checkpoint_snapshot_sha256": checkpoint.digests[
                        "snapshot"
                    ],
                    "observed_snapshot_sha256": snapshot,
                },
            )
        return checkpoint


class Chunked(Feed):
    """Parquet read by row group."""

    suffixes = (".parquet", ".pq")

    def batches(self, rows: int = BATCH_ROWS) -> Iterator[pandas.DataFrame]:
        """Yields Arrow record batches as frames.

        Pre-buffering and threads are off: with them Arrow keeps decoded
        pages of many row groups alive and the reader's memory grows with
        the file instead of staying at one row group.

        Args:
            rows: Maximum rows per Arrow record batch.

        Yields:
            One frame per Arrow record batch, in file order.
        """
        reader = pyarrow.parquet.ParquetFile(self.path, pre_buffer=False)
        for batch in reader.iter_batches(batch_size=rows, use_threads=False):
            yield batch.to_pandas()

    def shape(self) -> tuple[int | None, int]:
        """Reports both counts from the file's metadata.

        Returns:
            The row count and the column count recorded in the Parquet
            metadata.
        """
        meta = pyarrow.parquet.ParquetFile(self.path).metadata
        return meta.num_rows, meta.num_columns


class Csv(Feed):
    """CSV read in chunks (types are inferred per chunk)."""

    suffixes = (".csv", ".csv.gz", ".csv.bz2", ".csv.xz", ".csv.zip")

    def batches(self, rows: int = BATCH_ROWS) -> Iterator[pandas.DataFrame]:
        """Yields consecutive chunks.

        Args:
            rows: Rows per chunk.

        Yields:
            Frames of `rows` rows each, except possibly the last.
        """
        yield from pandas.read_csv(self.path, chunksize=rows)

    def shape(self) -> tuple[int | None, int]:
        """Reports the column count; the row count is unknown without a pass.

        Returns:
            None for the row count and the number of columns in the header.
        """
        return None, len(pandas.read_csv(self.path, nrows=1).columns)


class Ndjson(Feed):
    """Newline-delimited JSON read in bounded chunks."""

    suffixes = (".jsonl", ".ndjson", ".jsonl.gz", ".ndjson.gz")

    def batches(self, rows: int = BATCH_ROWS) -> Iterator[pandas.DataFrame]:
        """Yields JSON records in bounded batches.

        Args:
            rows: Records per chunk.

        Yields:
            Frames of `rows` records each, except possibly the last.
        """
        yield from pandas.read_json(self.path, lines=True, chunksize=rows)

    def shape(self) -> tuple[int | None, int]:
        """Reports the column count; the row count is unknown without a pass.

        Returns:
            None for the row count, and the number of columns of the first
            record (0 if the file has no records).
        """
        first = next(iter(self.batches(rows=1)), None)
        return None, 0 if first is None else len(first.columns)


SOURCES: tuple[type[Feed], ...] = (
    Chunked,
    Csv,
    Ndjson,
)


def locate(uri: str | pathlib.Path) -> Feed:
    """Returns the chunked reader for a file.

    Args:
        uri: Path of the file to stream.

    Returns:
        A reader of the kind whose suffixes match the end of the file
        name (case-insensitive), bound to the file.

    Raises:
        DataError: For a missing file or a format that cannot be streamed.
    """
    path = pathlib.Path(uri)
    if not path.exists():
        raise errors.DataError(f"dataset not found: {path}")
    name = path.name.lower()
    for kind in SOURCES:
        if any(name.endswith(suffix) for suffix in kind.suffixes):
            return kind(path)
    raise errors.DataError(
        f"cannot stream '{path.suffix}' files; use Parquet, CSV or JSONL"
    )
