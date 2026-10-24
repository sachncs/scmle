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
from urllib.parse import urlsplit

from scmle import durability
from scmle.contracts import errors
from scmle.data.readers import connectors


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


@dataclasses.dataclass(frozen=True, slots=True)
class EventCommitReceipt:
    """Result of an idempotent sink commit attempt."""

    source_identity: str
    accepted_event_ids: tuple[str, ...]
    duplicate_event_ids: tuple[str, ...]
    checkpoint: EventCheckpoint

    def __post_init__(self) -> None:
        """Validate deterministic accepted and duplicate identities."""
        if not self.source_identity.strip():
            raise ValueError("source_identity must not be empty")
        if self.checkpoint.source_identity != self.source_identity:
            raise ValueError("receipt and checkpoint source identities differ")
        for name in ("accepted_event_ids", "duplicate_event_ids"):
            values = getattr(self, name)
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be sorted and unique")
        if set(self.accepted_event_ids) & set(self.duplicate_event_ids):
            raise ValueError(
                "accepted and duplicate event IDs must be disjoint"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible commit receipt."""
        return {
            "schema_version": 1,
            "source_identity": self.source_identity,
            "accepted_event_ids": list(self.accepted_event_ids),
            "duplicate_event_ids": list(self.duplicate_event_ids),
            "checkpoint": self.checkpoint.to_dict(),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class EventAcknowledgement:
    """Broker acknowledgement prepared after a durable sink commit.

    The transport adapter owns the actual acknowledgement call. This record
    carries the source, consumer group, event identities, and committed
    checkpoint that the adapter must bind to that call. Persisting it before
    delivery lets a restarted consumer retry an acknowledgement without
    re-running the sink writer.
    """

    source_identity: str
    consumer_group: str
    event_ids: tuple[str, ...]
    checkpoint: EventCheckpoint

    def __post_init__(self) -> None:
        """Validate the acknowledgement's source and deterministic IDs."""
        if not self.source_identity.strip():
            raise ValueError("source_identity must not be empty")
        if not self.consumer_group.strip():
            raise ValueError("consumer_group must not be empty")
        if tuple(sorted(set(self.event_ids))) != self.event_ids:
            raise ValueError("event_ids must be sorted and unique")
        if self.checkpoint.source_identity != self.source_identity:
            raise ValueError(
                "acknowledgement and checkpoint source identities differ"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible acknowledgement record."""
        return {
            "schema_version": 1,
            "source_identity": self.source_identity,
            "consumer_group": self.consumer_group,
            "event_ids": list(self.event_ids),
            "checkpoint": self.checkpoint.to_dict(),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild a staged acknowledgement from durable JSON."""
        if record.get("schema_version") != 1:
            raise ValueError("unsupported acknowledgement schema version")
        return cls(
            source_identity=str(record["source_identity"]),
            consumer_group=str(record["consumer_group"]),
            event_ids=tuple(str(value) for value in record["event_ids"]),
            checkpoint=EventCheckpoint.from_dict(record["checkpoint"]),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class EventConsumerResult:
    """Evidence from one durable consume-and-acknowledge cycle."""

    source_identity: str
    consumer_group: str
    receipt: EventCommitReceipt
    acknowledged_event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the result's source, group, and acknowledgement IDs."""
        if self.receipt.source_identity != self.source_identity:
            raise ValueError("result and receipt source identities differ")
        if not self.consumer_group.strip():
            raise ValueError("consumer_group must not be empty")
        if tuple(sorted(set(self.acknowledged_event_ids))) != (
            self.acknowledged_event_ids
        ):
            raise ValueError("acknowledged_event_ids must be sorted and unique")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible consume evidence."""
        return {
            "schema_version": 1,
            "source_identity": self.source_identity,
            "consumer_group": self.consumer_group,
            "receipt": self.receipt.to_dict(),
            "acknowledged_event_ids": list(self.acknowledged_event_ids),
        }


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

    def validate_checkpoint(
        self, checkpoint: EventCheckpoint | None
    ) -> EventCheckpoint:
        """Bind a checkpoint to this source or fail before polling."""
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
        return current

    def select_events(
        self,
        records: Iterable[IncrementalEvent],
        checkpoint: EventCheckpoint | None = None,
        seen_event_ids: Iterable[str] = (),
        limit: int = 1_000,
    ) -> EventBatch:
        """Select a bounded, deduplicated batch from transport records.

        Args:
            records: Events returned by a transport boundary.
            checkpoint: Last committed sequence and watermark.
            seen_event_ids: IDs already durably committed for deduplication.
            limit: Maximum number of new events to return.

        Raises:
            IncrementalError: If a late event violates the policy.
        """
        if limit <= 0:
            raise ValueError("limit must be positive")
        current = self.validate_checkpoint(checkpoint)
        committed = frozenset(seen_event_ids)
        values = tuple(records)
        if any(not isinstance(event, IncrementalEvent) for event in values):
            raise TypeError(
                "transport records must yield IncrementalEvent values"
            )
        ordered = tuple(
            sorted(values, key=lambda event: (event.sequence, event.event_id))
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
        return self.select_events(
            self.available_events(), checkpoint, seen_event_ids, limit
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


class EventStreamSource(IncrementalEventSource):
    """Bounded event-stream adapter around a caller-owned poll boundary.

    The callback is the only transport-specific surface. It receives the
    canonical source URI, the committed checkpoint, and the requested limit;
    it must return at most that many events without acknowledging them. The
    durable ledger remains responsible for commit and downstream idempotency.
    """

    supported_schemes = (
        "eventhubs",
        "kafka",
        "kinesis",
        "nats",
        "pulsar",
        "pubsub",
        "redpanda",
    )

    def __init__(
        self,
        source_uri: str,
        event_reader: Callable[
            [str, EventCheckpoint, int], Iterable[IncrementalEvent]
        ],
        late_data_policy: LateDataPolicy = LateDataPolicy.REJECT,
        retry_budget: connectors.RetryBudget | None = None,
    ) -> None:
        """Initialize a source around an injected broker poll callback.

        Args:
            source_uri: Broker URI identifying topic, stream, or subscription.
            event_reader: Callback receiving URI, checkpoint, and batch limit.
            late_data_policy: Policy applied to events behind the checkpoint.
            retry_budget: Retry policy for transport polling.
        """
        value = source_uri.strip()
        scheme = urlsplit(value).scheme.lower()
        if scheme not in self.supported_schemes:
            raise connectors.UnsupportedUriSchemeError(
                value, scheme, self.supported_schemes
            )
        if not callable(event_reader):
            raise TypeError("event_reader must be callable")
        super().__init__(value, late_data_policy)
        self.event_reader = event_reader
        self.retry_budget = retry_budget or connectors.RetryBudget()

    def available_events(self) -> Iterable[IncrementalEvent]:
        """Reject unbounded polling without a checkpoint and limit."""
        raise IncrementalError(
            "bounded_poll_required",
            "event streams require read_events with a checkpoint and limit",
            self.source_uri,
            {},
        )

    def poll_events(
        self, checkpoint: EventCheckpoint, limit: int
    ) -> tuple[IncrementalEvent, ...]:
        """Poll a bounded transport batch with retry evidence."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        records = connectors.execute_with_retries(
            lambda: tuple(
                self.event_reader(self.source_uri, checkpoint, limit)
            ),
            "poll_events",
            self.source_uri,
            self.retry_budget,
        )
        if len(records) > limit:
            raise IncrementalError(
                "batch_limit_exceeded",
                "event reader returned more events than requested",
                self.source_uri,
                {"limit": limit, "received": len(records)},
            )
        return records

    def read_events(
        self,
        checkpoint: EventCheckpoint | None = None,
        seen_event_ids: Iterable[str] = (),
        limit: int = 1_000,
    ) -> EventBatch:
        """Poll and select one bounded event batch without acknowledging it."""
        current = self.validate_checkpoint(checkpoint)
        return self.select_events(
            self.poll_events(current, limit), current, seen_event_ids, limit
        )


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

    @property
    def pending_acknowledgement_path(self) -> pathlib.Path:
        """Return the durable sidecar awaiting acknowledgement delivery."""
        return self.path.with_name(f"{self.path.name}.ack")

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
            durability.sync_directory(self.path.parent)
        except BaseException:
            pathlib.Path(temporary_name).unlink(missing_ok=True)
            raise

    def write_pending_acknowledgement(
        self, acknowledgement: EventAcknowledgement
    ) -> None:
        """Stage an acknowledgement with durable atomic replacement."""
        if acknowledgement.source_identity != self.source_identity:
            raise IncrementalError(
                "acknowledgement_source_mismatch",
                "acknowledgement does not match ingestion ledger",
                self.source_identity,
                {"ack_source_identity": acknowledgement.source_identity},
            )
        self.pending_acknowledgement_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.pending_acknowledgement_path.name}.",
            dir=self.pending_acknowledgement_path.parent,
        )
        try:
            with os.fdopen(
                descriptor, "w", encoding="utf-8", newline="\n"
            ) as handle:
                json.dump(acknowledgement.to_dict(), handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.pending_acknowledgement_path)
            durability.sync_directory(self.pending_acknowledgement_path.parent)
        except BaseException:
            pathlib.Path(temporary_name).unlink(missing_ok=True)
            raise

    def load_pending_acknowledgement(self) -> EventAcknowledgement | None:
        """Load the acknowledgement that still needs transport delivery."""
        if not self.pending_acknowledgement_path.exists():
            return None
        try:
            record = json.loads(
                self.pending_acknowledgement_path.read_text(encoding="utf-8")
            )
            acknowledgement = EventAcknowledgement.from_dict(record)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise IncrementalError(
                "invalid_pending_acknowledgement",
                "pending acknowledgement is not valid JSON evidence",
                self.source_identity,
                {"path": str(self.pending_acknowledgement_path)},
            ) from error
        if acknowledgement.source_identity != self.source_identity:
            raise IncrementalError(
                "acknowledgement_source_mismatch",
                "pending acknowledgement belongs to another source",
                self.source_identity,
                {"ack_source_identity": acknowledgement.source_identity},
            )
        return acknowledgement

    def clear_pending_acknowledgement(self) -> None:
        """Remove a delivered acknowledgement and flush the directory entry."""
        self.pending_acknowledgement_path.unlink(missing_ok=True)
        durability.sync_directory(self.pending_acknowledgement_path.parent)

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

    def commit_with_writer(
        self,
        events: Iterable[IncrementalEvent],
        writer: Callable[[tuple[IncrementalEvent, ...]], None],
        acknowledgement: EventAcknowledgement | None = None,
    ) -> EventCommitReceipt:
        """Write unseen events and commit their IDs under one ledger lock.

        The writer receives only events not already present in the durable
        ledger. If it raises, the ledger is unchanged and the caller may
        retry. A process crash after the writer succeeds but before the
        ledger replacement can cause a replay; external writers must accept
        ``event_id`` as an idempotency key. This is an at-least-once boundary,
        not a claim of distributed exactly-once delivery.
        """
        records = tuple(events)
        if not all(isinstance(event, IncrementalEvent) for event in records):
            raise TypeError("events must contain IncrementalEvent values")
        if not callable(writer):
            raise TypeError("writer must be callable")
        if (
            acknowledgement is not None
            and acknowledgement.source_identity != self.source_identity
        ):
            raise IncrementalError(
                "acknowledgement_source_mismatch",
                "acknowledgement does not match ingestion ledger",
                self.source_identity,
                {"ack_source_identity": acknowledgement.source_identity},
            )
        with self.lock():
            current = self.load()
            seen = set(current.seen_event_ids)
            duplicates: set[str] = set()
            accepted: list[IncrementalEvent] = []
            for event in sorted(
                records, key=lambda value: (value.sequence, value.event_id)
            ):
                if event.event_id in seen:
                    duplicates.add(event.event_id)
                else:
                    seen.add(event.event_id)
                    accepted.append(event)
            accepted_ids = tuple(event.event_id for event in accepted)
            if accepted:
                writer(tuple(accepted))
                timestamps = [
                    parse_timestamp(event.event_time, "event_time")
                    for event in accepted
                ]
                current_watermark = (
                    parse_timestamp(current.checkpoint.watermark, "watermark")
                    if current.checkpoint.watermark
                    else None
                )
                if current_watermark is not None:
                    timestamps.append(current_watermark)
                checkpoint = EventCheckpoint(
                    self.source_identity,
                    max(
                        current.checkpoint.sequence,
                        *(event.sequence for event in accepted),
                    ),
                    max(timestamps).isoformat(),
                )
                self.write_atomically(
                    CommittedIncrementalState(checkpoint, tuple(sorted(seen)))
                )
            else:
                checkpoint = current.checkpoint
            if acknowledgement is not None and acknowledgement.event_ids:
                self.write_pending_acknowledgement(
                    dataclasses.replace(acknowledgement, checkpoint=checkpoint)
                )
            return EventCommitReceipt(
                self.source_identity,
                tuple(sorted(accepted_ids)),
                tuple(sorted(duplicates)),
                checkpoint,
            )


class EventConsumer:
    """Consume events with a durable sink commit and explicit acknowledgements.

    This class is the transport-neutral consumer-group boundary. The source
    supplies bounded events, the writer owns the downstream sink, and the
    acknowledgement callback owns the broker SDK. The ledger stages the
    acknowledgement after the sink commit and retries it on the next call if
    delivery fails. A callback must therefore make acknowledgement delivery
    idempotent; distributed broker transactions remain outside this package.
    """

    def __init__(
        self,
        source: IncrementalEventSource,
        ledger: IncrementalIngestionLedger,
        consumer_group: str,
        writer: Callable[[tuple[IncrementalEvent, ...]], None],
        acknowledge: Callable[[EventAcknowledgement], None],
    ) -> None:
        """Initialize a consumer around explicit transport and sink seams."""
        group = consumer_group.strip()
        if not group:
            raise ValueError("consumer_group must not be empty")
        if source.source_identity != ledger.source_identity:
            raise IncrementalError(
                "source_mismatch",
                "source does not match consumer ledger",
                ledger.source_identity,
                {"source_identity": source.source_identity},
            )
        if not callable(writer):
            raise TypeError("writer must be callable")
        if not callable(acknowledge):
            raise TypeError("acknowledge must be callable")
        self.source = source
        self.ledger = ledger
        self.consumer_group = group
        self.writer = writer
        self.acknowledge = acknowledge

    def deliver_pending_acknowledgement(self) -> tuple[str, ...]:
        """Deliver and clear a previously staged acknowledgement."""
        pending = self.ledger.load_pending_acknowledgement()
        if pending is None:
            return ()
        if pending.consumer_group != self.consumer_group:
            raise IncrementalError(
                "consumer_group_mismatch",
                "pending acknowledgement belongs to another consumer group",
                self.source.source_uri,
                {"pending_group": pending.consumer_group},
            )
        try:
            self.acknowledge(pending)
        except Exception as error:
            raise IncrementalError(
                "acknowledgement_failed",
                "transport acknowledgement failed; retry is required",
                self.source.source_uri,
                {
                    "consumer_group": self.consumer_group,
                    "event_ids": list(pending.event_ids),
                },
            ) from error
        self.ledger.clear_pending_acknowledgement()
        return pending.event_ids

    def consume(self, limit: int = 1_000) -> EventConsumerResult:
        """Consume, durably commit, acknowledge, and return evidence."""
        self.deliver_pending_acknowledgement()
        batch = self.ledger.read_batch(self.source, limit)
        event_ids = tuple(
            sorted(
                set(batch.event_ids)
                | set(batch.duplicate_event_ids)
                | set(batch.late_event_ids)
            )
        )
        acknowledgement = EventAcknowledgement(
            source_identity=self.source.source_identity,
            consumer_group=self.consumer_group,
            event_ids=event_ids,
            checkpoint=batch.next_checkpoint,
        )
        receipt = self.ledger.commit_with_writer(
            batch.events, self.writer, acknowledgement
        )
        acknowledged = self.deliver_pending_acknowledgement()
        return EventConsumerResult(
            source_identity=self.source.source_identity,
            consumer_group=self.consumer_group,
            receipt=receipt,
            acknowledged_event_ids=acknowledged,
        )
