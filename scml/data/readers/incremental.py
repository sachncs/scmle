"""Transport-neutral incremental event ingestion contracts.

This module owns the semantics that every event source must provide: stable
event identity, ordered checkpoints, watermarks, late-data policy, duplicate
classification, and an atomic durable commit. It deliberately does not
import a broker or warehouse SDK. A transport adapter supplies available
events through ``IncrementalEventSource``.
"""

import abc
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
import contextlib
import dataclasses
import datetime
import enum
import json
import os
import pathlib
import tempfile
from typing import Any
from typing import Self

from scml.contracts import errors
from scml.data.readers import connectors


class IncrementalError(errors.DataError):
    """A source cannot safely advance its incremental state."""

    def __init__(
        self,
        code: str,
        message: str,
        source_uri: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize a structured incremental-ingestion failure."""
        self.code = code
        self.source_uri = source_uri
        self.details = {
            "code": code,
            "source_uri": source_uri,
            **dict(details or {}),
        }
        super().__init__(message, self.details)


class LateDataPolicy(enum.StrEnum):
    """Action for an event older than the committed sequence checkpoint."""

    ACCEPT = "accept"
    DROP = "drop"
    REJECT = "reject"


def parse_timestamp(value: str, field_name: str) -> datetime.datetime:
    """Parse a timezone-aware ISO-8601 timestamp."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field_name} must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed.astimezone(datetime.UTC)


@dataclasses.dataclass(frozen=True, slots=True)
class IncrementalEvent:
    """One immutable event from an incremental source."""

    event_id: str
    sequence: int
    event_time: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate event identity, ordering, time, and payload shape."""
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError("event_id must not be empty")
        if isinstance(self.sequence, bool) or not isinstance(
            self.sequence, int
        ):
            raise TypeError("sequence must be an integer")
        if self.sequence < 1:
            raise ValueError("sequence must be a positive integer")
        parse_timestamp(self.event_time, "event_time")
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping")
        object.__setattr__(self, "payload", dict(self.payload))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible event record."""
        return {
            "event_id": self.event_id,
            "sequence": self.sequence,
            "event_time": self.event_time,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild an event from a persisted record."""
        return cls(
            event_id=str(record["event_id"]),
            sequence=int(record["sequence"]),
            event_time=str(record["event_time"]),
            payload=dict(record["payload"]),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class EventCheckpoint:
    """Committed position and watermark for one incremental source."""

    source_identity: str
    sequence: int = 0
    watermark: str = ""

    def __post_init__(self) -> None:
        """Validate source identity, sequence, and optional watermark."""
        if not self.source_identity.strip():
            raise ValueError("source_identity must not be empty")
        if isinstance(self.sequence, bool) or not isinstance(
            self.sequence, int
        ):
            raise TypeError("sequence must be an integer")
        if self.sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        if self.watermark:
            parse_timestamp(self.watermark, "watermark")

    def bind(self, source_identity: str) -> "EventCheckpoint":
        """Bind an uninitialized checkpoint to exactly one source."""
        if self.source_identity != source_identity:
            raise ValueError("checkpoint belongs to a different source")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible checkpoint."""
        return {
            "source_identity": self.source_identity,
            "sequence": self.sequence,
            "watermark": self.watermark,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild a checkpoint from durable state."""
        return cls(
            source_identity=str(record["source_identity"]),
            sequence=int(record["sequence"]),
            watermark=str(record.get("watermark", "")),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class EventBatch:
    """Events selected for processing and the checkpoint they can advance."""

    source_identity: str
    events: tuple[IncrementalEvent, ...]
    next_checkpoint: EventCheckpoint
    duplicate_event_ids: tuple[str, ...] = ()
    late_event_ids: tuple[str, ...] = ()
    has_more: bool = False

    def __post_init__(self) -> None:
        """Validate batch identity and deterministic event collections."""
        if self.next_checkpoint.source_identity != self.source_identity:
            raise ValueError("batch and checkpoint source identities differ")
        if any(
            not isinstance(event, IncrementalEvent) for event in self.events
        ):
            raise TypeError("events must contain IncrementalEvent values")
        if tuple(sorted(set(self.duplicate_event_ids))) != (
            self.duplicate_event_ids
        ):
            raise ValueError("duplicate_event_ids must be sorted and unique")
        if tuple(sorted(set(self.late_event_ids))) != self.late_event_ids:
            raise ValueError("late_event_ids must be sorted and unique")

    @property
    def event_ids(self) -> tuple[str, ...]:
        """Return selected event identities in source order."""
        return tuple(event.event_id for event in self.events)


class IncrementalEventSource(abc.ABC):
    """Polymorphic contract for a source that yields ordered events."""

    def __init__(
        self,
        source_uri: str,
        late_data_policy: LateDataPolicy = LateDataPolicy.REJECT,
    ) -> None:
        """Initialize source identity and late-data policy."""
        if not isinstance(late_data_policy, LateDataPolicy):
            late_data_policy = LateDataPolicy(late_data_policy)
        self.source_uri = connectors.canonical_source_uri(source_uri)
        self.source_identity = connectors.source_identity(self.source_uri)
        self.late_data_policy = late_data_policy

    @abc.abstractmethod
    def available_events(self) -> Iterable[IncrementalEvent]:
        """Return available events without changing committed state."""

    def read_events(
        self,
        checkpoint: EventCheckpoint | None = None,
        seen_event_ids: Iterable[str] = (),
        limit: int = 1_000,
    ) -> EventBatch:
        """Select a bounded batch without committing its side effects.

        Args:
            checkpoint: Last committed sequence and watermark.
            seen_event_ids: IDs already durably committed for deduplication.
            limit: Maximum number of new events to return.

        Raises:
            IncrementalError: If a late event violates the policy.
        """
        if limit <= 0:
            raise ValueError("limit must be positive")
        current = checkpoint or EventCheckpoint(self.source_identity)
        try:
            current.bind(self.source_identity)
        except ValueError as error:
            raise IncrementalError(
                "checkpoint_mismatch",
                str(error),
                self.source_uri,
                {"checkpoint": current.to_dict()},
            ) from error
        committed = frozenset(seen_event_ids)
        records = tuple(self.available_events())
        if any(not isinstance(event, IncrementalEvent) for event in records):
            raise TypeError(
                "available_events must yield IncrementalEvent values"
            )
        ordered = tuple(
            sorted(records, key=lambda event: (event.sequence, event.event_id))
        )
        duplicates: set[str] = set()
        late: set[str] = set()
        eligible: list[IncrementalEvent] = []
        selected_ids: set[str] = set()
        for event in ordered:
            if event.event_id in committed or event.event_id in selected_ids:
                duplicates.add(event.event_id)
                continue
            if event.sequence <= current.sequence:
                late.add(event.event_id)
                if self.late_data_policy is LateDataPolicy.DROP:
                    continue
                if self.late_data_policy is LateDataPolicy.REJECT:
                    raise IncrementalError(
                        "late_event",
                        f"event '{event.event_id}' is older than checkpoint",
                        self.source_uri,
                        {
                            "event_id": event.event_id,
                            "event_sequence": event.sequence,
                            "checkpoint_sequence": current.sequence,
                        },
                    )
            eligible.append(event)
            selected_ids.add(event.event_id)
        selected = tuple(eligible[:limit])
        next_sequence = max(
            (event.sequence for event in selected), default=current.sequence
        )
        timestamps = [
            parse_timestamp(event.event_time, "event_time")
            for event in selected
        ]
        current_watermark = (
            parse_timestamp(current.watermark, "watermark")
            if current.watermark
            else None
        )
        watermark_values = [*timestamps]
        if current_watermark is not None:
            watermark_values.append(current_watermark)
        next_watermark = max(watermark_values, default=None)
        checkpoint_value = EventCheckpoint(
            self.source_identity,
            next_sequence,
            next_watermark.isoformat() if next_watermark else "",
        )
        return EventBatch(
            self.source_identity,
            selected,
            checkpoint_value,
            tuple(sorted(duplicates)),
            tuple(sorted(late)),
            has_more=len(eligible) > len(selected),
        )


class SequenceEventSource(IncrementalEventSource):
    """Deterministic source adapter for tests and local replay fixtures."""

    def __init__(
        self,
        source_uri: str,
        events: Iterable[IncrementalEvent]
        | Callable[[], Iterable[IncrementalEvent]],
        late_data_policy: LateDataPolicy = LateDataPolicy.REJECT,
    ) -> None:
        """Initialize a source from a fixed sequence or event factory."""
        super().__init__(source_uri, late_data_policy)
        if callable(events):
            self.event_factory = events
        else:
            values = tuple(events)
            self.event_factory = lambda: values

    def available_events(self) -> Iterable[IncrementalEvent]:
        """Return the current event view supplied by the transport seam."""
        return self.event_factory()


@dataclasses.dataclass(frozen=True, slots=True)
class CommittedIncrementalState:
    """Durable event IDs and checkpoint after an atomic ingestion commit."""

    checkpoint: EventCheckpoint
    seen_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate deterministic deduplication state."""
        if tuple(sorted(set(self.seen_event_ids))) != self.seen_event_ids:
            raise ValueError("seen_event_ids must be sorted and unique")

    def to_dict(self) -> dict[str, Any]:
        """Return durable state as JSON-compatible data."""
        return {
            "schema_version": 1,
            "checkpoint": self.checkpoint.to_dict(),
            "seen_event_ids": list(self.seen_event_ids),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild committed state from durable JSON."""
        if record.get("schema_version") != 1:
            raise ValueError("unsupported incremental state schema version")
        return cls(
            checkpoint=EventCheckpoint.from_dict(record["checkpoint"]),
            seen_event_ids=tuple(
                str(item) for item in record["seen_event_ids"]
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class IncrementalIngestionLedger:
    """Atomic durable commit boundary for incremental event processing."""

    path: pathlib.Path
    source_identity: str

    @property
    def lock_path(self) -> pathlib.Path:
        """Return the sidecar path used to serialize ledger writers."""
        return self.path.with_name(f"{self.path.name}.lock")

    @contextlib.contextmanager
    def lock(self) -> Iterator[None]:
        """Serialize ledger writers with the platform's advisory file lock."""
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

    def load(self) -> CommittedIncrementalState:
        """Load committed state or initialize an empty source state."""
        if not self.path.exists():
            return CommittedIncrementalState(
                EventCheckpoint(self.source_identity)
            )
        record = json.loads(self.path.read_text(encoding="utf-8"))
        state = CommittedIncrementalState.from_dict(record)
        if state.checkpoint.source_identity != self.source_identity:
            raise IncrementalError(
                "state_source_mismatch",
                "durable state belongs to another source",
                self.source_identity,
                {"state_source_identity": state.checkpoint.source_identity},
            )
        return state

    def read_batch(
        self, source: IncrementalEventSource, limit: int = 1_000
    ) -> EventBatch:
        """Read an uncommitted batch using the durable deduplication state."""
        if source.source_identity != self.source_identity:
            raise IncrementalError(
                "source_mismatch",
                "source does not match ingestion ledger",
                self.source_identity,
                {"source_identity": source.source_identity},
            )
        state = self.load()
        return source.read_events(state.checkpoint, state.seen_event_ids, limit)

    def write_atomically(self, state: CommittedIncrementalState) -> None:
        """Publish state with a durable temporary-file replacement.

        Callers that combine a read and write must hold ``lock``. This method
        remains public so transports can persist a prepared state explicitly.
        """
        if state.checkpoint.source_identity != self.source_identity:
            raise IncrementalError(
                "state_source_mismatch",
                "state does not match ingestion ledger",
                self.source_identity,
                {"state_source_identity": state.checkpoint.source_identity},
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=self.path.parent
        )
        try:
            with os.fdopen(
                descriptor, "w", encoding="utf-8", newline="\n"
            ) as handle:
                json.dump(state.to_dict(), handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
        except BaseException:
            pathlib.Path(temporary_name).unlink(missing_ok=True)
            raise

    def commit(self, batch: EventBatch) -> CommittedIncrementalState:
        """Atomically publish a processed batch's IDs and checkpoint."""
        if batch.source_identity != self.source_identity:
            raise IncrementalError(
                "batch_source_mismatch",
                "batch does not match ingestion ledger",
                self.source_identity,
                {"batch_source_identity": batch.source_identity},
            )
        with self.lock():
            current = self.load()
            candidate_ids = set(batch.event_ids)
            current_ids = set(current.seen_event_ids)
            if (
                candidate_ids <= current_ids
                and batch.next_checkpoint.sequence
                <= current.checkpoint.sequence
            ):
                return current
            if batch.next_checkpoint.sequence < current.checkpoint.sequence:
                raise IncrementalError(
                    "checkpoint_regression",
                    "batch checkpoint moves backwards",
                    self.source_identity,
                    {
                        "current": current.checkpoint.to_dict(),
                        "candidate": batch.next_checkpoint.to_dict(),
                    },
                )
            state = CommittedIncrementalState(
                batch.next_checkpoint,
                tuple(
                    sorted(set(current.seen_event_ids) | set(batch.event_ids))
                ),
            )
            self.write_atomically(state)
            return state
