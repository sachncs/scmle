"""Reading tables larger than memory, in batches.

A big file is never loaded whole. One pass over its batches picks the locked
test rows by hashing each row's position (so the choice needs no memory and
is the same on every run), and keeps a bounded random sample of the other
rows for development. Each row's sampling priority is also a hash of its
position, so the sample does not depend on the batch size.
"""

import abc
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
import contextlib
import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
from typing import Any
from typing import ClassVar
from typing import Self

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scmle import durability
from scmle.contracts import errors
from scmle.data.readers import connectors

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
STREAM_STATE_SCHEMA = 2
STREAM_ACKNOWLEDGEMENT_SCHEMA = 1
STREAM_TRANSACTION_SCHEMA = 1


def row_hash(positions: np.ndarray, seed: int) -> np.ndarray:
    """A deterministic number in [0, 1) for each row position (SplitMix64).

    Args:
        positions: Integer row positions (zero-based) to hash.
        seed: Integer that selects the hash stream; different seeds give
            different, independent-looking values for the same position.

    Returns:
        A float64 array with one value in [0, 1) per entry of `positions`,
        in the same order.
    """
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
        """Binds the source to a file.

        Args:
            path: The file to read.
        """
        self.path = path

    @abc.abstractmethod
    def batches(self, rows: int = BATCH_ROWS) -> Iterator[pd.DataFrame]:
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

    def estimated_bytes(self) -> int:
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
            return connectors.file_content_sha256(self.path)
        except OSError as error:
            raise StreamingError(
                "source_unreadable",
                f"cannot fingerprint streaming source: {self.path}",
                self.source_identity(),
                {"path": str(self.path)},
            ) from error

    def read_batch(
        self,
        checkpoint: "StreamCheckpoint | None" = None,
        rows: int = BATCH_ROWS,
        watermark_column: str | None = None,
    ) -> "StreamBatch":
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
            raise ValueError("rows must be positive")
        identity = self.source_identity()
        observed_snapshot = self.snapshot_sha256()
        current = self.validate_checkpoint(checkpoint, observed_snapshot)
        if (
            checkpoint is not None
            and current.offset > 0
            and current.watermark_column != watermark_column
        ):
            raise StreamingError(
                "watermark_mismatch",
                "watermark column does not match the checkpoint",
                identity,
                {
                    "checkpoint_watermark_column": current.watermark_column,
                    "requested_watermark_column": watermark_column,
                },
            )
        effective_watermark_column = (
            current.watermark_column
            if checkpoint is not None and current.offset > 0
            else watermark_column
        )
        selected: pd.DataFrame | None = None
        offset = 0
        has_more = False
        batches = iter(self.batches(rows))
        for batch in batches:
            if not isinstance(batch, pd.DataFrame):
                raise StreamingError(
                    "invalid_batch",
                    "chunk source yielded a non-DataFrame batch",
                    identity,
                    {"type": type(batch).__name__},
                )
            batch_start = offset
            offset += len(batch)
            if batch_start + len(batch) <= current.offset:
                continue
            start = max(0, current.offset - batch_start)
            candidate = batch.iloc[start:]
            take = min(rows, len(candidate))
            selected = candidate.iloc[:take].copy()
            has_more = len(candidate) > take
            if take == rows:
                if not has_more:
                    next_batch = next(batches, None)
                    has_more = next_batch is not None and len(next_batch) > 0
                break
            break
        if selected is None or selected.empty:
            raise StreamingError(
                "end_of_source",
                "streaming source has no rows after the checkpoint",
                identity,
                {"offset": current.offset},
            )
        final_snapshot = self.snapshot_sha256()
        if final_snapshot != observed_snapshot:
            raise StreamingError(
                "source_snapshot_changed",
                "streaming source changed while it was being read",
                identity,
                {
                    "initial_snapshot_sha256": observed_snapshot,
                    "final_snapshot_sha256": final_snapshot,
                },
            )
        next_watermark = watermark_value(
            selected,
            effective_watermark_column,
            current.watermark,
            identity,
        )
        next_checkpoint = StreamCheckpoint(
            identity,
            observed_snapshot,
            current.offset + len(selected),
            next_watermark,
            effective_watermark_column,
        )
        return StreamBatch(
            identity,
            observed_snapshot,
            current.offset,
            selected,
            next_checkpoint,
            has_more,
        )

    def validate_checkpoint(
        self, checkpoint: "StreamCheckpoint | None", snapshot: str
    ) -> "StreamCheckpoint":
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
            return StreamCheckpoint(identity, snapshot)
        if checkpoint.source_identity != identity:
            raise StreamingError(
                "checkpoint_source_mismatch",
                "checkpoint belongs to another streaming source",
                identity,
                {"checkpoint_source_identity": checkpoint.source_identity},
            )
        if checkpoint.snapshot_sha256 != snapshot:
            raise StreamingError(
                "source_snapshot_changed",
                "source snapshot does not match the checkpoint",
                identity,
                {
                    "checkpoint_snapshot_sha256": checkpoint.snapshot_sha256,
                    "observed_snapshot_sha256": snapshot,
                },
            )
        return checkpoint


class StreamingError(errors.DataError):
    """A bounded streaming operation cannot safely advance its state.

    Attributes:
        code: Short machine-readable failure code.
        source_identity: Identity of the source involved.
        details: Evidence mapping holding `code`, `source_identity` and
            any extra detail supplied.
    """

    def __init__(
        self,
        code: str,
        message: str,
        source_identity: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Initializes a structured, transport-neutral streaming failure.

        Args:
            code: Short machine-readable failure code.
            message: Human-readable description of the failure.
            source_identity: Identity of the source involved.
            details: Extra evidence merged into `details` after the code and
                source identity.
        """
        self.code = code
        self.source_identity = source_identity
        self.details = {
            "code": code,
            "source_identity": source_identity,
            **dict(details or {}),
        }
        super().__init__(message, self.details)


def validate_digest(value: str, field_name: str) -> None:
    """Validates a SHA-256 digest used as durable source evidence.

    Args:
        value: The candidate digest.
        field_name: Name used in the error message.

    Raises:
        ValueError: If `value` is not 64 lower-case hexadecimal characters.
    """
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} must be a SHA-256 hex digest")


def validate_timestamp(value: str, field_name: str) -> datetime.datetime:
    """Parses a timezone-aware watermark timestamp.

    Args:
        value: ISO-8601 timestamp; a trailing "Z" means UTC.
        field_name: Name used in the error message.

    Returns:
        The timestamp converted to UTC.

    Raises:
        ValueError: If `value` is not ISO-8601 or carries no timezone.
    """
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field_name} must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed.astimezone(datetime.UTC)


@dataclasses.dataclass(frozen=True, slots=True)
class StreamCheckpoint:
    """A durable row position bound to a source and exact raw snapshot.

    Attributes:
        source_identity: Identity of the source the position belongs to.
        snapshot_sha256: SHA-256 hex digest of the file the position was
            taken on.
        offset: Number of rows already consumed, so the next row to read
            is the one at this zero-based position.
        watermark: Greatest timestamp seen so far as an ISO-8601 UTC
            string; empty when none has been seen.
        watermark_column: Column the watermark is taken from, or None.
    """

    source_identity: str
    snapshot_sha256: str
    offset: int = 0
    watermark: str = ""
    watermark_column: str | None = None

    def __post_init__(self) -> None:
        """Validates source, snapshot, offset, and optional watermark.

        Raises:
            ValueError: If the source identity is blank, the snapshot is not a
                SHA-256 hex digest, the offset is negative, the watermark
                column is an empty string, or the watermark is not a
                timezone-aware ISO-8601 timestamp.
            TypeError: If the offset is not an integer.
        """
        if not self.source_identity.strip():
            raise ValueError("source_identity must not be empty")
        validate_digest(self.snapshot_sha256, "snapshot_sha256")
        if isinstance(self.offset, bool) or not isinstance(self.offset, int):
            raise TypeError("offset must be an integer")
        if self.offset < 0:
            raise ValueError("offset must be non-negative")
        if self.watermark_column is not None and not self.watermark_column:
            raise ValueError("watermark_column must not be empty")
        if self.watermark:
            validate_timestamp(self.watermark, "watermark")

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible checkpoint evidence.

        Returns:
            A dictionary with the five checkpoint fields.
        """
        return {
            "source_identity": self.source_identity,
            "snapshot_sha256": self.snapshot_sha256,
            "offset": self.offset,
            "watermark": self.watermark,
            "watermark_column": self.watermark_column,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a checkpoint from durable JSON.

        Args:
            record: Mapping in the form produced by `to_dict`.

        Returns:
            The checkpoint described by `record`.
        """
        return cls(
            source_identity=str(record["source_identity"]),
            snapshot_sha256=str(record["snapshot_sha256"]),
            offset=int(record["offset"]),
            watermark=str(record.get("watermark", "")),
            watermark_column=(
                str(record["watermark_column"])
                if record.get("watermark_column") is not None
                else None
            ),
        )


def batch_identity(
    source_identity: str,
    snapshot_sha256: str,
    start_offset: int,
    end_offset: int,
) -> str:
    """Returns the stable idempotency key for one source row range.

    Args:
        source_identity: Identity of the source.
        snapshot_sha256: Digest of the file snapshot.
        start_offset: Offset of the first row of the batch.
        end_offset: Offset just past the last row of the batch.

    Returns:
        The string "batch_" followed by a SHA-256 hex digest of the four
        arguments.
    """
    payload = (
        f"{source_identity}:{snapshot_sha256}:{start_offset}:{end_offset}"
    ).encode()
    return f"batch_{hashlib.sha256(payload).hexdigest()}"


def watermark_value(
    frame: pd.DataFrame,
    column: str | None,
    current: str,
    source_identity: str,
) -> str:
    """Returns the greatest timezone-aware timestamp in a batch.

    A malformed `current` value is rejected with ValueError.

    Args:
        frame: The batch of rows.
        column: Column holding timestamps, or None to disable tracking.
        current: The watermark so far (ISO-8601 UTC string, possibly
            empty).
        source_identity: Identity of the source, used in error evidence.

    Returns:
        The later of `current` and the greatest timestamp in `column`, as
        an ISO-8601 UTC string. `current` is returned unchanged when
        `column` is None or every value in the column is null.

    Raises:
        StreamingError: If `column` is not in the frame or its values
            cannot be parsed as timestamps.
    """
    if column is None:
        return current
    if column not in frame.columns:
        raise StreamingError(
            "watermark_column_missing",
            f"watermark column '{column}' is not present in the batch",
            source_identity,
            {"column": column},
        )
    try:
        values = pd.to_datetime(frame[column], utc=True, errors="raise")
    except (TypeError, ValueError) as error:
        raise StreamingError(
            "invalid_watermark",
            f"watermark column '{column}' contains invalid timestamps",
            source_identity,
            {"column": column},
        ) from error
    observed = values.dropna()
    if observed.empty:
        return current
    candidate = observed.max().to_pydatetime().astimezone(datetime.UTC)
    if not current:
        return candidate.isoformat()
    return max(validate_timestamp(current, "watermark"), candidate).isoformat()


@dataclasses.dataclass(frozen=True, slots=True)
class StreamBatch:
    """A bounded frame plus the checkpoint it may advance.

    Attributes:
        source_identity: Identity of the source the rows came from.
        snapshot_sha256: Digest of the file snapshot the rows came from.
        start_offset: Zero-based position of the frame's first row in the
            source.
        frame: The rows; never empty.
        next_checkpoint: The checkpoint to persist once the batch is
            committed.
        has_more: Whether rows remain after this batch.
        batch_id: Idempotency key derived from the source, snapshot and
            row range; set on construction.
    """

    source_identity: str
    snapshot_sha256: str
    start_offset: int
    frame: pd.DataFrame = dataclasses.field(repr=False)
    next_checkpoint: StreamCheckpoint
    has_more: bool = False
    batch_id: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Validates the batch and derives its deterministic idempotency key.

        Raises:
            ValueError: If the checkpoint's source or snapshot differs from the
                batch's, the start offset is negative, the frame is empty or
                not a DataFrame, or the checkpoint does not advance past the
                start offset.
        """
        if self.next_checkpoint.source_identity != self.source_identity:
            raise ValueError("batch and checkpoint sources differ")
        if self.next_checkpoint.snapshot_sha256 != self.snapshot_sha256:
            raise ValueError("batch and checkpoint snapshots differ")
        if self.start_offset < 0:
            raise ValueError("start_offset must be non-negative")
        if not isinstance(self.frame, pd.DataFrame) or self.frame.empty:
            raise ValueError("frame must contain at least one row")
        if self.next_checkpoint.offset <= self.start_offset:
            raise ValueError("checkpoint must advance past the batch start")
        object.__setattr__(
            self,
            "batch_id",
            batch_identity(
                self.source_identity,
                self.snapshot_sha256,
                self.start_offset,
                self.next_checkpoint.offset,
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class StreamAcknowledgement:
    """Durable evidence awaiting delivery to an external transport.

    Attributes:
        source_identity: Identity of the source.
        consumer_group: Consumer group the acknowledgement is for.
        batch_id: Identity of the acknowledged batch.
        checkpoint: Checkpoint reached once the batch is committed.
    """

    source_identity: str
    consumer_group: str
    batch_id: str
    checkpoint: StreamCheckpoint

    def __post_init__(self) -> None:
        """Validates acknowledgement identity and checkpoint binding.

        Raises:
            ValueError: If the source identity or consumer group is blank, the
                batch identity does not start with "batch_", or the checkpoint
                belongs to another source.
        """
        if not self.source_identity.strip():
            raise ValueError("source_identity must not be empty")
        if not self.consumer_group.strip():
            raise ValueError("consumer_group must not be empty")
        if not self.batch_id.startswith("batch_"):
            raise ValueError("batch_id must be a streaming batch identity")
        if self.checkpoint.source_identity != self.source_identity:
            raise ValueError("acknowledgement and checkpoint sources differ")

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible acknowledgement evidence.

        Returns:
            A dictionary holding the schema version and the four fields.
        """
        return {
            "schema_version": STREAM_ACKNOWLEDGEMENT_SCHEMA,
            "source_identity": self.source_identity,
            "consumer_group": self.consumer_group,
            "batch_id": self.batch_id,
            "checkpoint": self.checkpoint.to_dict(),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a pending acknowledgement from durable JSON.

        Args:
            record: Mapping in the form produced by `to_dict`.

        Returns:
            The acknowledgement described by `record`.

        Raises:
            ValueError: If the record's schema version is not supported.
        """
        if record.get("schema_version") != STREAM_ACKNOWLEDGEMENT_SCHEMA:
            raise ValueError("unsupported streaming acknowledgement schema")
        return cls(
            source_identity=str(record["source_identity"]),
            consumer_group=str(record["consumer_group"]),
            batch_id=str(record["batch_id"]),
            checkpoint=StreamCheckpoint.from_dict(record["checkpoint"]),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class SinkCommitReceipt:
    """A transport-neutral receipt bound to one streaming batch.

    Attributes:
        source_identity: Identity of the source.
        batch_id: Identity of the committed batch.
        checkpoint: Checkpoint reached by the commit.
        receipt: Opaque, non-empty receipt text issued by the sink.
    """

    source_identity: str
    batch_id: str
    checkpoint: StreamCheckpoint
    receipt: str

    def __post_init__(self) -> None:
        """Validates the receipt's basic identities and checkpoint source.

        Raises:
            ValueError: If the source identity or receipt text is blank, the
                batch identity does not start with "batch_", or the checkpoint
                belongs to another source.
        """
        if not self.source_identity.strip():
            raise ValueError("source_identity must not be empty")
        if not self.batch_id.startswith("batch_"):
            raise ValueError("batch_id must be a streaming batch identity")
        if not self.receipt.strip():
            raise ValueError("receipt must not be empty")
        if self.checkpoint.source_identity != self.source_identity:
            raise ValueError("receipt and checkpoint sources differ")

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible sink receipt evidence.

        Returns:
            A dictionary with the four receipt fields.
        """
        return {
            "source_identity": self.source_identity,
            "batch_id": self.batch_id,
            "checkpoint": self.checkpoint.to_dict(),
            "receipt": self.receipt,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a sink receipt from durable JSON.

        Args:
            record: Mapping in the form produced by `to_dict`.

        Returns:
            The receipt described by `record`.
        """
        return cls(
            source_identity=str(record["source_identity"]),
            batch_id=str(record["batch_id"]),
            checkpoint=StreamCheckpoint.from_dict(record["checkpoint"]),
            receipt=str(record["receipt"]),
        )


class TransactionalSink(abc.ABC):
    """Caller-owned sink contract for an idempotent streaming commit.

    A sink must commit or deduplicate the supplied ``batch_id`` and return a
    receipt containing the exact source, batch, and checkpoint identities. The
    ledger atomically records that receipt with checkpoint acknowledgement in
    its local JSON state, but this contract does not provide a broker
    transaction or distributed exactly-once delivery.
    """

    @abc.abstractmethod
    def commit(self, batch: StreamBatch) -> SinkCommitReceipt:
        """Commits one batch and returns its transport-owned receipt.

        Args:
            batch: The batch to commit; implementations deduplicate on
                `batch.batch_id`.

        Returns:
            A receipt carrying the batch's source, batch and checkpoint
            identities.
        """


@dataclasses.dataclass(frozen=True, slots=True)
class StreamCommitEvidence:
    """Durable binding of a sink receipt and checkpoint acknowledgement.

    Attributes:
        source_identity: Identity of the source.
        batch_id: Identity of the committed batch.
        sink_receipt: Receipt issued by the sink for the batch.
        acknowledgement: Acknowledgement bound to the batch.
    """

    source_identity: str
    batch_id: str
    sink_receipt: SinkCommitReceipt
    acknowledgement: StreamAcknowledgement

    def __post_init__(self) -> None:
        """Validates all identities in one durable transaction record.

        Raises:
            ValueError: If the receipt or acknowledgement differs from the
                evidence in source or batch, or their checkpoints differ.
        """
        if self.sink_receipt.source_identity != self.source_identity:
            raise ValueError("evidence and sink receipt sources differ")
        if self.acknowledgement.source_identity != self.source_identity:
            raise ValueError("evidence and acknowledgement sources differ")
        if self.sink_receipt.batch_id != self.batch_id:
            raise ValueError("evidence and sink receipt batches differ")
        if self.acknowledgement.batch_id != self.batch_id:
            raise ValueError("evidence and acknowledgement batches differ")
        if self.sink_receipt.checkpoint != self.acknowledgement.checkpoint:
            raise ValueError(
                "sink receipt and acknowledgement checkpoints differ"
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible transaction evidence.

        Returns:
            A dictionary holding the schema version and the nested records.
        """
        return {
            "schema_version": STREAM_TRANSACTION_SCHEMA,
            "source_identity": self.source_identity,
            "batch_id": self.batch_id,
            "sink_receipt": self.sink_receipt.to_dict(),
            "acknowledgement": self.acknowledgement.to_dict(),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds transaction evidence from durable JSON.

        Args:
            record: Mapping in the form produced by `to_dict`.

        Returns:
            The evidence described by `record`.

        Raises:
            ValueError: If the record's schema version is not supported.
        """
        if record.get("schema_version") != STREAM_TRANSACTION_SCHEMA:
            raise ValueError("unsupported streaming transaction schema")
        return cls(
            source_identity=str(record["source_identity"]),
            batch_id=str(record["batch_id"]),
            sink_receipt=SinkCommitReceipt.from_dict(record["sink_receipt"]),
            acknowledgement=StreamAcknowledgement.from_dict(
                record["acknowledgement"]
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class StreamState:
    """Durable idempotency and checkpoint state for one source snapshot.

    Attributes:
        checkpoint: Position after the last committed batch.
        committed_batch_ids: Identities of all committed batches, sorted
            and unique.
        transaction_evidence: Evidence for committed batches, sorted by
            batch identity; may cover only some committed batches.
    """

    checkpoint: StreamCheckpoint
    committed_batch_ids: tuple[str, ...] = ()
    transaction_evidence: tuple[StreamCommitEvidence, ...] = ()

    def __post_init__(self) -> None:
        """Validates deterministic committed batch identities.

        Raises:
            ValueError: If the committed identities or the evidence are not
                sorted and unique, or evidence names a batch that is not
                committed.
        """
        if tuple(sorted(set(self.committed_batch_ids))) != (
            self.committed_batch_ids
        ):
            raise ValueError("committed_batch_ids must be sorted and unique")
        evidence_ids = tuple(
            evidence.batch_id for evidence in self.transaction_evidence
        )
        if tuple(sorted(set(evidence_ids))) != evidence_ids:
            raise ValueError("transaction_evidence must be sorted by batch_id")
        if not set(evidence_ids).issubset(self.committed_batch_ids):
            raise ValueError(
                "transaction evidence must identify committed batches"
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible durable state.

        Returns:
            A dictionary holding the schema version, checkpoint, committed
            batch identities and transaction evidence.
        """
        return {
            "schema_version": STREAM_STATE_SCHEMA,
            "checkpoint": self.checkpoint.to_dict(),
            "committed_batch_ids": list(self.committed_batch_ids),
            "transaction_evidence": [
                evidence.to_dict() for evidence in self.transaction_evidence
            ],
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds state from durable JSON.

        Args:
            record: Mapping in the form produced by `to_dict`. Schema
                versions 1 and 2 are accepted.

        Returns:
            The state described by `record`.

        Raises:
            ValueError: If the record's schema version is not supported.
        """
        if record.get("schema_version") not in (1, STREAM_STATE_SCHEMA):
            raise ValueError("unsupported streaming state schema")
        return cls(
            checkpoint=StreamCheckpoint.from_dict(record["checkpoint"]),
            committed_batch_ids=tuple(
                str(value) for value in record["committed_batch_ids"]
            ),
            transaction_evidence=tuple(
                StreamCommitEvidence.from_dict(value)
                for value in record.get("transaction_evidence", [])
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class StreamCommitReceipt:
    """Evidence from one idempotent streaming commit attempt.

    Attributes:
        source_identity: Identity of the source.
        batch_id: Identity of the batch.
        accepted: True if this call committed the batch.
        duplicate: True if the batch had already been committed.
        checkpoint: Checkpoint after the batch.
        sink_receipt: Receipt issued by the sink, when one is involved.
        acknowledgement: Acknowledgement bound to the batch, when one is
            involved.
    """

    source_identity: str
    batch_id: str
    accepted: bool
    duplicate: bool
    checkpoint: StreamCheckpoint
    sink_receipt: SinkCommitReceipt | None = None
    acknowledgement: StreamAcknowledgement | None = None

    def __post_init__(self) -> None:
        """Validates mutually exclusive commit classification.

        Raises:
            ValueError: If `accepted` and `duplicate` are equal, or the
                checkpoint, sink receipt or acknowledgement does not match the
                commit.
        """
        if self.accepted == self.duplicate:
            raise ValueError("a commit must be accepted or duplicate")
        if self.checkpoint.source_identity != self.source_identity:
            raise ValueError("receipt and checkpoint sources differ")
        if self.sink_receipt is not None and (
            self.sink_receipt.source_identity != self.source_identity
            or self.sink_receipt.batch_id != self.batch_id
            or self.sink_receipt.checkpoint != self.checkpoint
        ):
            raise ValueError("receipt does not match the commit")
        if self.acknowledgement is not None and (
            self.acknowledgement.source_identity != self.source_identity
            or self.acknowledgement.batch_id != self.batch_id
            or self.acknowledgement.checkpoint != self.checkpoint
        ):
            raise ValueError("acknowledgement does not match the commit")


@dataclasses.dataclass(frozen=True, slots=True)
class StreamingLedger:
    """Atomic local ledger for replay-safe streaming evidence.

    This ledger provides a durable idempotency boundary for a caller-owned
    writer and a durable acknowledgement retry record for a caller-owned
    transport callback. It does not provide broker transactions or distributed
    exactly-once delivery; external sinks and transports must honor the batch
    idempotency key themselves.

    Attributes:
        path: JSON file holding the durable state.
        source: The chunked source being consumed.
    """

    path: pathlib.Path
    source: ChunkSource

    @property
    def lock_path(self) -> pathlib.Path:
        """The sidecar file used to serialize local ledger writers."""
        return self.path.with_name(f"{self.path.name}.lock")

    @property
    def pending_acknowledgement_path(self) -> pathlib.Path:
        """The sidecar file holding an undelivered acknowledgement."""
        return self.path.with_name(f"{self.path.name}.ack")

    @contextlib.contextmanager
    def lock(self) -> Iterator[None]:
        """Serializes state transitions with an advisory file lock.

        Yields:
            None, while the exclusive lock on the sidecar file is held.
        """
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as handle:
            if os.name == "nt":
                import msvcrt

                windows_lock: Any = msvcrt
                handle.seek(0)
                windows_lock.locking(handle.fileno(), windows_lock.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":
                    handle.seek(0)
                    windows_lock.locking(
                        handle.fileno(), windows_lock.LK_UNLCK, 1
                    )
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def load(self) -> StreamState:
        """Loads state, failing closed if the source snapshot changed.

        Returns:
            The durable state, or a fresh state at offset zero when the state
            file does not exist.

        Raises:
            StreamingError: If the state file is unreadable or malformed, or
                the state (including its transaction evidence) belongs to
                another source or another snapshot of the file.
        """
        snapshot = self.source.snapshot_sha256()
        identity = self.source.source_identity()
        if not self.path.exists():
            return StreamState(StreamCheckpoint(identity, snapshot))
        try:
            state = StreamState.from_dict(
                json.loads(self.path.read_text(encoding="utf-8"))
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise StreamingError(
                "invalid_state",
                "durable streaming state is unreadable",
                identity,
                {"path": str(self.path)},
            ) from error
        if state.checkpoint.source_identity != identity:
            raise StreamingError(
                "checkpoint_source_mismatch",
                "durable streaming state belongs to another source",
                identity,
                {"state_source_identity": state.checkpoint.source_identity},
            )
        if state.checkpoint.snapshot_sha256 != snapshot:
            raise StreamingError(
                "source_snapshot_changed",
                "durable streaming state belongs to another source snapshot",
                identity,
                {
                    "checkpoint_snapshot_sha256": (
                        state.checkpoint.snapshot_sha256
                    ),
                    "observed_snapshot_sha256": snapshot,
                },
            )
        for evidence in state.transaction_evidence:
            if (
                evidence.sink_receipt.checkpoint.snapshot_sha256 != snapshot
                or evidence.acknowledgement.checkpoint.snapshot_sha256
                != snapshot
            ):
                raise StreamingError(
                    "source_snapshot_changed",
                    "transaction evidence belongs to another source snapshot",
                    identity,
                    {"batch_id": evidence.batch_id},
                )
        return state

    def read_batch(
        self,
        rows: int = BATCH_ROWS,
        watermark_column: str | None = None,
    ) -> StreamBatch:
        """Reads the next bounded batch from the durable checkpoint.

        Failures from `load` and from the source's `read_batch` propagate
        unchanged.

        Args:
            rows: Maximum number of rows in the batch.
            watermark_column: Column whose greatest timestamp is tracked as the
                watermark.

        Returns:
            The next batch after the durable checkpoint.
        """
        state = self.load()
        return self.source.read_batch(
            state.checkpoint, rows=rows, watermark_column=watermark_column
        )

    def write_atomically(self, state: StreamState) -> None:
        """Publishes state with durable temporary-file replacement.

        Args:
            state: The state to persist.

        Raises:
            StreamingError: If the state's checkpoint belongs to another source
                or another snapshot of the file.
        """
        identity = self.source.source_identity()
        if state.checkpoint.source_identity != identity:
            raise StreamingError(
                "checkpoint_source_mismatch",
                "state does not belong to this source",
                identity,
                {},
            )
        if state.checkpoint.snapshot_sha256 != self.source.snapshot_sha256():
            raise StreamingError(
                "source_snapshot_changed",
                "state does not belong to the current source snapshot",
                identity,
                {},
            )
        durability.write_atomically(
            self.path,
            (json.dumps(state.to_dict(), sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )

    def write_pending_acknowledgement(
        self, acknowledgement: StreamAcknowledgement
    ) -> None:
        """Persists an acknowledgement while the ledger lock is held.

        Args:
            acknowledgement: The acknowledgement to stage for delivery.

        Raises:
            StreamingError: If the acknowledgement belongs to another source or
                another snapshot of the file.
        """
        identity = self.source.source_identity()
        if acknowledgement.source_identity != identity:
            raise StreamingError(
                "acknowledgement_source_mismatch",
                "acknowledgement belongs to another source",
                identity,
                {},
            )
        if acknowledgement.checkpoint.snapshot_sha256 != (
            self.source.snapshot_sha256()
        ):
            raise StreamingError(
                "source_snapshot_changed",
                "acknowledgement belongs to another source snapshot",
                identity,
                {},
            )
        durability.write_atomically(
            self.pending_acknowledgement_path,
            (
                json.dumps(acknowledgement.to_dict(), sort_keys=True) + "\n"
            ).encode("utf-8"),
        )

    def load_pending_acknowledgement(self) -> StreamAcknowledgement | None:
        """Loads the pending acknowledgement, if one exists.

        Returns:
            The staged acknowledgement, or None if there is none.

        Raises:
            StreamingError: If the staged file is unreadable or malformed, or
                belongs to another source or another snapshot of the file.
        """
        if not self.pending_acknowledgement_path.exists():
            return None
        try:
            acknowledgement = StreamAcknowledgement.from_dict(
                json.loads(
                    self.pending_acknowledgement_path.read_text(
                        encoding="utf-8"
                    )
                )
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise StreamingError(
                "invalid_acknowledgement",
                "pending streaming acknowledgement is unreadable",
                self.source.source_identity(),
                {"path": str(self.pending_acknowledgement_path)},
            ) from error
        if acknowledgement.source_identity != self.source.source_identity():
            raise StreamingError(
                "acknowledgement_source_mismatch",
                "pending acknowledgement belongs to another source",
                self.source.source_identity(),
                {},
            )
        if acknowledgement.checkpoint.snapshot_sha256 != (
            self.source.snapshot_sha256()
        ):
            raise StreamingError(
                "source_snapshot_changed",
                "pending acknowledgement belongs to another source snapshot",
                self.source.source_identity(),
                {},
            )
        return acknowledgement

    def clear_pending_acknowledgement(self) -> None:
        """Removes a delivered acknowledgement and flushes the directory.

        Nothing happens if no acknowledgement is staged.
        """
        try:
            self.pending_acknowledgement_path.unlink()
            durability.sync_directory(self.pending_acknowledgement_path.parent)
        except FileNotFoundError:
            return

    def validate_batch(self, batch: StreamBatch) -> None:
        """Validates a batch against this ledger's source snapshot.

        Args:
            batch: The batch to check.

        Raises:
            StreamingError: If the batch belongs to another source or another
                snapshot of the file.
        """
        identity = self.source.source_identity()
        if batch.source_identity != identity:
            raise StreamingError(
                "batch_source_mismatch",
                "batch belongs to another source",
                identity,
                {},
            )
        if batch.snapshot_sha256 != self.source.snapshot_sha256():
            raise StreamingError(
                "source_snapshot_changed",
                "batch belongs to another source snapshot",
                identity,
                {},
            )

    def validate_acknowledgement(
        self, batch: StreamBatch, acknowledgement: StreamAcknowledgement
    ) -> None:
        """Validates the acknowledgement that will be bound to a batch.

        Args:
            batch: The batch being committed.
            acknowledgement: The acknowledgement to bind to it.

        Raises:
            StreamingError: If the acknowledgement's source, batch identity or
                checkpoint differs from the batch's.
        """
        identity = self.source.source_identity()
        if acknowledgement.source_identity != batch.source_identity:
            raise StreamingError(
                "acknowledgement_source_mismatch",
                "acknowledgement does not match the batch source",
                identity,
                {},
            )
        if acknowledgement.batch_id != batch.batch_id:
            raise StreamingError(
                "acknowledgement_batch_mismatch",
                "acknowledgement does not match the batch",
                identity,
                {},
            )
        if acknowledgement.checkpoint != batch.next_checkpoint:
            raise StreamingError(
                "acknowledgement_checkpoint_mismatch",
                "acknowledgement does not match the batch checkpoint",
                identity,
                {},
            )

    def validate_sink_receipt(
        self, batch: StreamBatch, receipt: SinkCommitReceipt
    ) -> None:
        """Validates a sink receipt before it enters durable state.

        Args:
            batch: The batch that was committed.
            receipt: The receipt returned by the sink.

        Raises:
            StreamingError: If the receipt's source, batch identity or
                checkpoint differs from the batch's.
        """
        identity = self.source.source_identity()
        if receipt.source_identity != batch.source_identity:
            raise StreamingError(
                "sink_receipt_source_mismatch",
                "sink receipt belongs to another source",
                identity,
                {"receipt_source_identity": receipt.source_identity},
            )
        if receipt.batch_id != batch.batch_id:
            raise StreamingError(
                "sink_receipt_batch_mismatch",
                "sink receipt belongs to another batch",
                identity,
                {"receipt_batch_id": receipt.batch_id},
            )
        if receipt.checkpoint != batch.next_checkpoint:
            raise StreamingError(
                "sink_receipt_checkpoint_mismatch",
                "sink receipt belongs to another checkpoint",
                identity,
                {},
            )

    def commit_transaction(
        self,
        batch: StreamBatch,
        sink: TransactionalSink,
        acknowledgement: StreamAcknowledgement,
    ) -> StreamCommitReceipt:
        """Commits a batch and atomically persists its receipt evidence.

        The sink is caller-owned and must deduplicate ``batch.batch_id`` when
        a process fails after the sink commit but before this local JSON state
        is replaced. A matching retry is handled without calling the sink a
        second time. This local transaction binds evidence atomically within
        the ledger only; it is not a broker transaction or a distributed
        exactly-once guarantee. Exceptions raised by the sink propagate and
        leave the state unchanged.

        Args:
            batch: The batch to commit; it must start at the durable
                checkpoint.
            sink: Caller-owned sink that commits the batch.
            acknowledgement: Acknowledgement to bind to the batch; it must
                match the batch's source, identity and next checkpoint.

        Returns:
            A receipt with `accepted` set after a new commit, or with
            `duplicate` set and the stored evidence when the batch was already
            committed with an equal acknowledgement.

        Raises:
            TypeError: If `sink` is not a TransactionalSink or
                `acknowledgement` is not a StreamAcknowledgement.
            StreamingError: If the batch, acknowledgement or sink receipt does
                not match the source and snapshot, the batch is bound to a
                different acknowledgement, the batch is recorded as committed
                without evidence, the batch does not start at the durable
                checkpoint, or the sink returns something other than a
                SinkCommitReceipt.
        """
        if not isinstance(sink, TransactionalSink):
            raise TypeError("sink must implement TransactionalSink")
        if not isinstance(acknowledgement, StreamAcknowledgement):
            raise TypeError("acknowledgement must be StreamAcknowledgement")
        self.validate_batch(batch)
        self.validate_acknowledgement(batch, acknowledgement)
        with self.lock():
            current = self.load()
            evidence = next(
                (
                    item
                    for item in current.transaction_evidence
                    if item.batch_id == batch.batch_id
                ),
                None,
            )
            if evidence is not None:
                if evidence.acknowledgement != acknowledgement:
                    raise StreamingError(
                        "acknowledgement_mismatch",
                        "batch is already bound to another acknowledgement",
                        self.source.source_identity(),
                        {},
                    )
                return StreamCommitReceipt(
                    batch.source_identity,
                    batch.batch_id,
                    accepted=False,
                    duplicate=True,
                    checkpoint=evidence.acknowledgement.checkpoint,
                    sink_receipt=evidence.sink_receipt,
                    acknowledgement=evidence.acknowledgement,
                )
            if batch.batch_id in current.committed_batch_ids:
                raise StreamingError(
                    "transaction_evidence_missing",
                    "batch was committed without transactional evidence",
                    self.source.source_identity(),
                    {},
                )
            if batch.start_offset != current.checkpoint.offset:
                raise StreamingError(
                    "checkpoint_mismatch",
                    "batch does not start at the durable checkpoint",
                    self.source.source_identity(),
                    {
                        "batch_start_offset": batch.start_offset,
                        "checkpoint_offset": current.checkpoint.offset,
                    },
                )
            sink_receipt = sink.commit(batch)
            if not isinstance(sink_receipt, SinkCommitReceipt):
                raise StreamingError(
                    "invalid_sink_receipt",
                    "transactional sink returned an invalid receipt",
                    self.source.source_identity(),
                    {"type": type(sink_receipt).__name__},
                )
            self.validate_sink_receipt(batch, sink_receipt)
            evidence = StreamCommitEvidence(
                batch.source_identity,
                batch.batch_id,
                sink_receipt,
                acknowledgement,
            )
            state = StreamState(
                batch.next_checkpoint,
                tuple(sorted((*current.committed_batch_ids, batch.batch_id))),
                tuple(
                    sorted(
                        (*current.transaction_evidence, evidence),
                        key=lambda item: item.batch_id,
                    )
                ),
            )
            self.write_atomically(state)
            return StreamCommitReceipt(
                batch.source_identity,
                batch.batch_id,
                accepted=True,
                duplicate=False,
                checkpoint=batch.next_checkpoint,
                sink_receipt=sink_receipt,
                acknowledgement=acknowledgement,
            )

    def commit_with_writer(
        self,
        batch: StreamBatch,
        writer: Callable[[pd.DataFrame, str], None],
        acknowledgement: StreamAcknowledgement | None = None,
    ) -> StreamCommitReceipt:
        """Commits a batch once and stages its external acknowledgement.

        The writer receives the stable batch idempotency key. If it succeeds
        and the process fails before the local state replacement, a retry can
        replay the writer; the sink must deduplicate that key. This is an
        explicit at-least-once seam, not a broker SDK exactly-once claim.
        Exceptions raised by the writer propagate and leave the state
        unchanged.

        Args:
            batch: The batch to commit; it must start at the durable
                checkpoint unless it is already committed.
            writer: Callable receiving the batch's frame and its batch
                identity.
            acknowledgement: Acknowledgement to stage for delivery, or None.
                Its source and batch identity must match the batch's. It is
                staged with the batch's next checkpoint.

        Returns:
            A receipt with `accepted` set after a new commit. When the batch
            was already committed, a receipt with `duplicate` set and the
            current checkpoint; the writer is not called again.

        Raises:
            TypeError: If `writer` is not callable.
            StreamingError: If the batch or acknowledgement does not match the
                source, snapshot or batch, or a new batch does not start at
                the durable checkpoint.
        """
        if not callable(writer):
            raise TypeError("writer must be callable")
        if batch.source_identity != self.source.source_identity():
            raise StreamingError(
                "batch_source_mismatch",
                "batch belongs to another source",
                self.source.source_identity(),
                {},
            )
        if batch.snapshot_sha256 != self.source.snapshot_sha256():
            raise StreamingError(
                "source_snapshot_changed",
                "batch belongs to another source snapshot",
                self.source.source_identity(),
                {},
            )
        if acknowledgement is not None and (
            acknowledgement.source_identity != batch.source_identity
            or acknowledgement.batch_id != batch.batch_id
        ):
            raise StreamingError(
                "acknowledgement_batch_mismatch",
                "acknowledgement does not match the batch",
                self.source.source_identity(),
                {},
            )
        with self.lock():
            current = self.load()
            if batch.batch_id in current.committed_batch_ids:
                if acknowledgement is not None:
                    self.write_pending_acknowledgement(acknowledgement)
                return StreamCommitReceipt(
                    batch.source_identity,
                    batch.batch_id,
                    accepted=False,
                    duplicate=True,
                    checkpoint=current.checkpoint,
                )
            if batch.start_offset != current.checkpoint.offset:
                raise StreamingError(
                    "checkpoint_mismatch",
                    "batch does not start at the durable checkpoint",
                    self.source.source_identity(),
                    {
                        "batch_start_offset": batch.start_offset,
                        "checkpoint_offset": current.checkpoint.offset,
                    },
                )
            writer(batch.frame, batch.batch_id)
            state = StreamState(
                batch.next_checkpoint,
                tuple(sorted((*current.committed_batch_ids, batch.batch_id))),
                current.transaction_evidence,
            )
            self.write_atomically(state)
            if acknowledgement is not None:
                self.write_pending_acknowledgement(
                    dataclasses.replace(
                        acknowledgement, checkpoint=batch.next_checkpoint
                    )
                )
            return StreamCommitReceipt(
                batch.source_identity,
                batch.batch_id,
                accepted=True,
                duplicate=False,
                checkpoint=batch.next_checkpoint,
            )


class StreamingConsumer:
    """Coordinate a durable writer with an explicit external ack callback.

    Attributes:
        ledger: The ledger holding the durable state.
        consumer_group: Non-empty name of the consumer group.
        writer: Callable receiving a batch's frame and batch identity.
        acknowledge: Callable that delivers an acknowledgement to the
            transport.
    """

    def __init__(
        self,
        ledger: StreamingLedger,
        consumer_group: str,
        writer: Callable[[pd.DataFrame, str], None],
        acknowledge: Callable[[StreamAcknowledgement], None],
    ) -> None:
        """Initializes a transport-neutral consumer boundary.

        Args:
            ledger: The ledger holding the durable state.
            consumer_group: Non-empty name of the consumer group.
            writer: Callable receiving a batch's frame and batch identity.
            acknowledge: Callable that delivers an acknowledgement to the
                transport.

        Raises:
            ValueError: If `consumer_group` is blank.
            TypeError: If `writer` or `acknowledge` is not callable.
        """
        if not consumer_group.strip():
            raise ValueError("consumer_group must not be empty")
        if not callable(writer) or not callable(acknowledge):
            raise TypeError("writer and acknowledge must be callable")
        self.ledger = ledger
        self.consumer_group = consumer_group
        self.writer = writer
        self.acknowledge = acknowledge

    def deliver_pending_acknowledgement(self) -> str | None:
        """Delivers and clears one staged acknowledgement after restart.

        Returns:
            The batch identity of the acknowledgement delivered, or None if
            none was staged.

        Raises:
            StreamingError: If the staged acknowledgement belongs to another
                consumer group, or the acknowledge callback raises (the
                staged record is kept for retry).
        """
        pending = self.ledger.load_pending_acknowledgement()
        if pending is None:
            return None
        if pending.consumer_group != self.consumer_group:
            raise StreamingError(
                "consumer_group_mismatch",
                "pending acknowledgement belongs to another consumer group",
                self.ledger.source.source_identity(),
                {"pending_consumer_group": pending.consumer_group},
            )
        try:
            self.acknowledge(pending)
        except Exception as error:
            raise StreamingError(
                "acknowledgement_failed",
                "external acknowledgement failed; retry is required",
                self.ledger.source.source_identity(),
                {"batch_id": pending.batch_id},
            ) from error
        self.ledger.clear_pending_acknowledgement()
        return pending.batch_id

    def consume(
        self,
        rows: int = BATCH_ROWS,
        watermark_column: str | None = None,
    ) -> StreamCommitReceipt:
        """Commits one batch and acknowledges it only after durable state.

        A previously staged acknowledgement is delivered first. Failures from
        reading, committing and delivering propagate unchanged.

        Args:
            rows: Maximum number of rows in the batch.
            watermark_column: Column whose greatest timestamp is tracked as the
                watermark.

        Returns:
            The receipt of the ledger commit.
        """
        self.deliver_pending_acknowledgement()
        batch = self.ledger.read_batch(rows, watermark_column)
        acknowledgement = StreamAcknowledgement(
            batch.source_identity,
            self.consumer_group,
            batch.batch_id,
            batch.next_checkpoint,
        )
        receipt = self.ledger.commit_with_writer(
            batch, self.writer, acknowledgement
        )
        self.deliver_pending_acknowledgement()
        return receipt


class ParquetChunks(ChunkSource):
    """Parquet read by row group."""

    suffixes = (".parquet", ".pq")

    def batches(self, rows: int = BATCH_ROWS) -> Iterator[pd.DataFrame]:
        """Yields Arrow record batches as frames.

        Pre-buffering and threads are off: with them Arrow keeps decoded
        pages of many row groups alive and the reader's memory grows with
        the file instead of staying at one row group.

        Args:
            rows: Maximum rows per Arrow record batch.

        Yields:
            One frame per Arrow record batch, in file order.
        """
        reader = pq.ParquetFile(self.path, pre_buffer=False)
        for batch in reader.iter_batches(batch_size=rows, use_threads=False):
            yield batch.to_pandas()

    def shape(self) -> tuple[int | None, int]:
        """Reports both counts from the file's metadata.

        Returns:
            The row count and the column count recorded in the Parquet
            metadata.
        """
        meta = pq.ParquetFile(self.path).metadata
        return meta.num_rows, meta.num_columns


class CsvChunks(ChunkSource):
    """CSV read in chunks (types are inferred per chunk)."""

    suffixes = (".csv", ".csv.gz", ".csv.bz2", ".csv.xz", ".csv.zip")

    def batches(self, rows: int = BATCH_ROWS) -> Iterator[pd.DataFrame]:
        """Yields consecutive chunks.

        Args:
            rows: Rows per chunk.

        Yields:
            Frames of `rows` rows each, except possibly the last.
        """
        yield from pd.read_csv(self.path, chunksize=rows)

    def shape(self) -> tuple[int | None, int]:
        """Reports the column count; the row count is unknown without a pass.

        Returns:
            None for the row count and the number of columns in the header.
        """
        return None, len(pd.read_csv(self.path, nrows=1).columns)


class JsonLinesChunks(ChunkSource):
    """Newline-delimited JSON read in bounded chunks."""

    suffixes = (".jsonl", ".ndjson", ".jsonl.gz", ".ndjson.gz")

    def batches(self, rows: int = BATCH_ROWS) -> Iterator[pd.DataFrame]:
        """Yields JSON records in bounded batches.

        Args:
            rows: Records per chunk.

        Yields:
            Frames of `rows` records each, except possibly the last.
        """
        yield from pd.read_json(self.path, lines=True, chunksize=rows)

    def shape(self) -> tuple[int | None, int]:
        """Reports the column count; the row count is unknown without a pass.

        Returns:
            None for the row count, and the number of columns of the first
            record (0 if the file has no records).
        """
        first = next(iter(self.batches(rows=1)), None)
        return None, 0 if first is None else len(first.columns)


SOURCES: tuple[type[ChunkSource], ...] = (
    ParquetChunks,
    CsvChunks,
    JsonLinesChunks,
)


def source_for(uri: str | pathlib.Path) -> ChunkSource:
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


def side_of(positions: np.ndarray, fraction: float, seed: int) -> np.ndarray:
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


def sample_caps(
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
    cells = max(columns, 1) * BYTES_PER_CELL
    rows = int(budget_gb * 1e9 * SAMPLE_SHARE / cells)
    development = max(int(rows * (1 - test_fraction)), MIN_SAMPLE_ROWS)
    locked = max(int(rows * test_fraction), MIN_SAMPLE_ROWS // 4)
    return development, locked


def keep_smallest(
    buffer: pd.DataFrame | None, batch: pd.DataFrame, cap: int
) -> pd.DataFrame:
    """Merges a batch into a buffer, keeping the `cap` lowest-priority rows.

    Args:
        buffer: Rows kept so far, or None if there are none.
        batch: New rows to merge in.
        cap: Maximum number of rows to keep.

    Returns:
        The merged frame; when it exceeds `cap` rows, only the `cap` rows
        with the smallest `priority` value remain.
    """
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
