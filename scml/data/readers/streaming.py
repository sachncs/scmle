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

from scml import durability
from scml.contracts import errors
from scml.data.readers import connectors

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

    def source_identity(self) -> str:
        """Return the stable identity of this source URI."""
        return connectors.source_identity(self.path)

    def snapshot_sha256(self) -> str:
        """Return the digest of the raw source snapshot.

        The digest is intentionally over the source bytes, not the decoded
        frame. A resumed stream must therefore observe the same exact file,
        including formatting and metadata that a decoder might otherwise
        discard.
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
        """Read one bounded batch bound to an immutable source snapshot.

        The source is re-opened from row zero for each call because file
        readers do not share a transport cursor. The durable checkpoint makes
        that replay safe; the raw digest prevents a changed file from being
        silently appended to an old run.
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
        """Bind a checkpoint to this source and current raw snapshot."""
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
    """A bounded streaming operation cannot safely advance its state."""

    def __init__(
        self,
        code: str,
        message: str,
        source_identity: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize a structured, transport-neutral streaming failure."""
        self.code = code
        self.source_identity = source_identity
        self.details = {
            "code": code,
            "source_identity": source_identity,
            **dict(details or {}),
        }
        super().__init__(message, self.details)


def validate_digest(value: str, field_name: str) -> None:
    """Validate a SHA-256 digest used as durable source evidence."""
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} must be a SHA-256 hex digest")


def validate_timestamp(value: str, field_name: str) -> datetime.datetime:
    """Parse a timezone-aware watermark timestamp."""
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field_name} must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed.astimezone(datetime.UTC)


@dataclasses.dataclass(frozen=True, slots=True)
class StreamCheckpoint:
    """A durable row position bound to a source and exact raw snapshot."""

    source_identity: str
    snapshot_sha256: str
    offset: int = 0
    watermark: str = ""
    watermark_column: str | None = None

    def __post_init__(self) -> None:
        """Validate source, snapshot, offset, and optional watermark."""
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
        """Return JSON-compatible checkpoint evidence."""
        return {
            "source_identity": self.source_identity,
            "snapshot_sha256": self.snapshot_sha256,
            "offset": self.offset,
            "watermark": self.watermark,
            "watermark_column": self.watermark_column,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild a checkpoint from durable JSON."""
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
    """Return the stable idempotency key for one source row range."""
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
    """Return the greatest timezone-aware timestamp in a batch."""
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
    """A bounded frame plus the checkpoint it may advance."""

    source_identity: str
    snapshot_sha256: str
    start_offset: int
    frame: pd.DataFrame = dataclasses.field(repr=False)
    next_checkpoint: StreamCheckpoint
    has_more: bool = False
    batch_id: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Validate the batch and derive its deterministic idempotency key."""
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
    """Durable evidence awaiting delivery to an external transport."""

    source_identity: str
    consumer_group: str
    batch_id: str
    checkpoint: StreamCheckpoint

    def __post_init__(self) -> None:
        """Validate acknowledgement identity and checkpoint binding."""
        if not self.source_identity.strip():
            raise ValueError("source_identity must not be empty")
        if not self.consumer_group.strip():
            raise ValueError("consumer_group must not be empty")
        if not self.batch_id.startswith("batch_"):
            raise ValueError("batch_id must be a streaming batch identity")
        if self.checkpoint.source_identity != self.source_identity:
            raise ValueError("acknowledgement and checkpoint sources differ")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible acknowledgement evidence."""
        return {
            "schema_version": STREAM_ACKNOWLEDGEMENT_SCHEMA,
            "source_identity": self.source_identity,
            "consumer_group": self.consumer_group,
            "batch_id": self.batch_id,
            "checkpoint": self.checkpoint.to_dict(),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild a pending acknowledgement from durable JSON."""
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
    """A transport-neutral receipt bound to one streaming batch."""

    source_identity: str
    batch_id: str
    checkpoint: StreamCheckpoint
    receipt: str

    def __post_init__(self) -> None:
        """Validate the receipt's basic identities and checkpoint source."""
        if not self.source_identity.strip():
            raise ValueError("source_identity must not be empty")
        if not self.batch_id.startswith("batch_"):
            raise ValueError("batch_id must be a streaming batch identity")
        if not self.receipt.strip():
            raise ValueError("receipt must not be empty")
        if self.checkpoint.source_identity != self.source_identity:
            raise ValueError("receipt and checkpoint sources differ")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible sink receipt evidence."""
        return {
            "source_identity": self.source_identity,
            "batch_id": self.batch_id,
            "checkpoint": self.checkpoint.to_dict(),
            "receipt": self.receipt,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild a sink receipt from durable JSON."""
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
        """Commit one batch and return its transport-owned receipt."""


@dataclasses.dataclass(frozen=True, slots=True)
class StreamCommitEvidence:
    """Durable binding of a sink receipt and checkpoint acknowledgement."""

    source_identity: str
    batch_id: str
    sink_receipt: SinkCommitReceipt
    acknowledgement: StreamAcknowledgement

    def __post_init__(self) -> None:
        """Validate all identities in one durable transaction record."""
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
        """Return JSON-compatible transaction evidence."""
        return {
            "schema_version": STREAM_TRANSACTION_SCHEMA,
            "source_identity": self.source_identity,
            "batch_id": self.batch_id,
            "sink_receipt": self.sink_receipt.to_dict(),
            "acknowledgement": self.acknowledgement.to_dict(),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild transaction evidence from durable JSON."""
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
    """Durable idempotency and checkpoint state for one source snapshot."""

    checkpoint: StreamCheckpoint
    committed_batch_ids: tuple[str, ...] = ()
    transaction_evidence: tuple[StreamCommitEvidence, ...] = ()

    def __post_init__(self) -> None:
        """Validate deterministic committed batch identities."""
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
        """Return JSON-compatible durable state."""
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
        """Rebuild state from durable JSON."""
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
    """Evidence from one idempotent streaming commit attempt."""

    source_identity: str
    batch_id: str
    accepted: bool
    duplicate: bool
    checkpoint: StreamCheckpoint
    sink_receipt: SinkCommitReceipt | None = None
    acknowledgement: StreamAcknowledgement | None = None

    def __post_init__(self) -> None:
        """Validate mutually exclusive commit classification."""
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
    """

    path: pathlib.Path
    source: ChunkSource

    @property
    def lock_path(self) -> pathlib.Path:
        """Return the sidecar used to serialize local ledger writers."""
        return self.path.with_name(f"{self.path.name}.lock")

    @property
    def pending_acknowledgement_path(self) -> pathlib.Path:
        """Return the sidecar containing an undelivered acknowledgement."""
        return self.path.with_name(f"{self.path.name}.ack")

    @contextlib.contextmanager
    def lock(self) -> Iterator[None]:
        """Serialize state transitions with an advisory file lock."""
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
        """Load state and fail closed if the source snapshot changed."""
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
        """Read the next bounded batch from the durable checkpoint."""
        state = self.load()
        return self.source.read_batch(
            state.checkpoint, rows=rows, watermark_column=watermark_column
        )

    def write_atomically(self, state: StreamState) -> None:
        """Publish state with durable temporary-file replacement."""
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
        """Persist an acknowledgement while the ledger lock is held."""
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
        """Load the pending acknowledgement, if one exists."""
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
        """Remove a delivered acknowledgement and flush the directory."""
        try:
            self.pending_acknowledgement_path.unlink()
            durability.sync_directory(self.pending_acknowledgement_path.parent)
        except FileNotFoundError:
            return

    def validate_batch(self, batch: StreamBatch) -> None:
        """Validate a batch against this ledger's source snapshot."""
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
        """Validate the acknowledgement that will be bound to a batch."""
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
        """Validate a sink receipt before it enters durable state."""
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
        """Commit a batch and atomically persist receipt/checkpoint evidence.

        The sink is caller-owned and must deduplicate ``batch.batch_id`` when
        a process fails after the sink commit but before this local JSON state
        is replaced. A matching retry is handled without calling the sink a
        second time. This local transaction binds evidence atomically within
        the ledger only; it is not a broker transaction or a distributed
        exactly-once guarantee.
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
        """Commit a batch once and stage its external acknowledgement.

        The writer receives the stable batch idempotency key. If it succeeds
        and the process fails before the local state replacement, a retry can
        replay the writer; the sink must deduplicate that key. This is an
        explicit at-least-once seam, not a broker SDK exactly-once claim.
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
    """Coordinate a durable writer with an explicit external ack callback."""

    def __init__(
        self,
        ledger: StreamingLedger,
        consumer_group: str,
        writer: Callable[[pd.DataFrame, str], None],
        acknowledge: Callable[[StreamAcknowledgement], None],
    ) -> None:
        """Initialize a transport-neutral consumer boundary."""
        if not consumer_group.strip():
            raise ValueError("consumer_group must not be empty")
        if not callable(writer) or not callable(acknowledge):
            raise TypeError("writer and acknowledge must be callable")
        self.ledger = ledger
        self.consumer_group = consumer_group
        self.writer = writer
        self.acknowledge = acknowledge

    def deliver_pending_acknowledgement(self) -> str | None:
        """Deliver and clear one staged acknowledgement after restart."""
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
        """Commit one batch and acknowledge it only after durable state."""
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

    suffixes = (".csv", ".csv.gz", ".csv.bz2", ".csv.xz", ".csv.zip")

    def batches(self, rows: int = BATCH_ROWS) -> Iterator[pd.DataFrame]:
        """Yields consecutive chunks."""
        yield from pd.read_csv(self.path, chunksize=rows)

    def shape(self) -> tuple[int | None, int]:
        """Only the column count is known without a pass."""
        return None, len(pd.read_csv(self.path, nrows=1).columns)


class JsonLinesChunks(ChunkSource):
    """Newline-delimited JSON read in bounded chunks."""

    suffixes = (".jsonl", ".ndjson", ".jsonl.gz", ".ndjson.gz")

    def batches(self, rows: int = BATCH_ROWS) -> Iterator[pd.DataFrame]:
        """Yields JSON records in bounded batches."""
        yield from pd.read_json(self.path, lines=True, chunksize=rows)

    def shape(self) -> tuple[int | None, int]:
        """Only the column count is known without a pass."""
        first = next(iter(self.batches(rows=1)), None)
        return None, 0 if first is None else len(first.columns)


SOURCES: tuple[type[ChunkSource], ...] = (
    ParquetChunks,
    CsvChunks,
    JsonLinesChunks,
)


def source_for(uri: str | pathlib.Path) -> ChunkSource:
    """The chunked reader for a file.

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
