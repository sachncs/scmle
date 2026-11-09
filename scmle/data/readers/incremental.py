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
    """A source cannot safely advance its incremental state.

    Attributes:
        code: Short machine-readable failure code.
        source_uri: URI (or source identity) of the source involved.
        details: Evidence mapping holding `code`, `source_uri` and any
            extra detail supplied.
    """

    def __init__(
        self,
        code: str,
        message: str,
        source_uri: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize a structured incremental-ingestion failure.

        Args:
            code: Short machine-readable failure code.
            message: Human-readable description of the failure.
            source_uri: URI (or source identity) of the source involved.
            details: Extra evidence merged into `details` after the code and
                source URI.
        """
        self.code = code
        self.source_uri = source_uri
        self.details = {
            "code": code,
            "source_uri": source_uri,
            **dict(details or {}),
        }
        super().__init__(message, self.details)


class LateDataPolicy(enum.StrEnum):
    """Action for an event older than the committed sequence checkpoint.

    Attributes:
        ACCEPT: Keep the event and select it like any other.
        DROP: Discard the event, reporting its identity as late.
        REJECT: Raise an error when such an event is found.
    """

    ACCEPT = "accept"
    DROP = "drop"
    REJECT = "reject"


def parse_timestamp(value: str, field_name: str) -> datetime.datetime:
    """Parse a timezone-aware ISO-8601 timestamp.

    Args:
        value: ISO-8601 timestamp; a trailing "Z" means UTC.
        field_name: Name used in the error message.

    Returns:
        The timestamp converted to UTC.

    Raises:
        ValueError: If `value` is not a non-blank string, is not ISO-8601,
            or carries no timezone.
    """
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
    """One immutable event from an incremental source.

    Attributes:
        event_id: Non-empty, stable identity of the event.
        sequence: Positive integer giving the event's position in the
            source's order.
        event_time: Time the event occurred, as a timezone-aware ISO-8601
            string.
        payload: The event's content; copied into a plain dict on
            construction.
        ingest_time: Time the system recorded the event, as a
            timezone-aware ISO-8601 string; empty when the source does not
            report it.
    """

    event_id: str
    sequence: int
    event_time: str
    payload: Mapping[str, Any]
    ingest_time: str = ""

    def __post_init__(self) -> None:
        """Validate event identity, ordering, time, and payload shape.

        ``ingest_time`` is when the system recorded the event (transaction
        time). It is empty when the source does not report it; a point-in-
        time reader must then fall back to ``event_time`` plus a declared
        delay and say so.

        Raises:
            ValueError: If the event identity is blank, the sequence is below
                one, or a timestamp is not timezone-aware ISO-8601.
            TypeError: If the sequence is not an integer or the payload is not
                a mapping.
        """
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError("event_id must not be empty")
        if isinstance(self.sequence, bool) or not isinstance(
            self.sequence, int
        ):
            raise TypeError("sequence must be an integer")
        if self.sequence < 1:
            raise ValueError("sequence must be a positive integer")
        parse_timestamp(self.event_time, "event_time")
        if self.ingest_time:
            parse_timestamp(self.ingest_time, "ingest_time")
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping")
        object.__setattr__(self, "payload", dict(self.payload))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible event record.

        Returns:
            A dictionary of the event's fields; `ingest_time` is included only
            when it is non-empty.
        """
        record: dict[str, Any] = {
            "event_id": self.event_id,
            "sequence": self.sequence,
            "event_time": self.event_time,
            "payload": dict(self.payload),
        }
        if self.ingest_time:
            record["ingest_time"] = self.ingest_time
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild an event from a persisted record.

        Args:
            record: Mapping in the form produced by `to_dict`.

        Returns:
            The event described by `record`.
        """
        return cls(
            event_id=str(record["event_id"]),
            sequence=int(record["sequence"]),
            event_time=str(record["event_time"]),
            payload=dict(record["payload"]),
            ingest_time=str(record.get("ingest_time", "")),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class EventCheckpoint:
    """Committed position and watermark for one incremental source.

    Attributes:
        source_identity: Identity of the source.
        sequence: Highest committed event sequence; 0 when nothing has been
            committed.
        watermark: Greatest committed event time as an ISO-8601 string;
            empty when none.
    """

    source_identity: str
    sequence: int = 0
    watermark: str = ""

    def __post_init__(self) -> None:
        """Validate source identity, sequence, and optional watermark.

        Raises:
            ValueError: If the source identity is blank, the sequence is
                negative, or the watermark is not timezone-aware ISO-8601.
            TypeError: If the sequence is not an integer.
        """
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
        """Bind an uninitialized checkpoint to exactly one source.

        Args:
            source_identity: Identity of the source the checkpoint must belong
                to.

        Returns:
            This checkpoint, unchanged.

        Raises:
            ValueError: If the checkpoint belongs to a different source.
        """
        if self.source_identity != source_identity:
            raise ValueError("checkpoint belongs to a different source")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible checkpoint.

        Returns:
            A dictionary with the three checkpoint fields.
        """
        return {
            "source_identity": self.source_identity,
            "sequence": self.sequence,
            "watermark": self.watermark,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild a checkpoint from durable state.

        Args:
            record: Mapping in the form produced by `to_dict`.

        Returns:
            The checkpoint described by `record`.
        """
        return cls(
            source_identity=str(record["source_identity"]),
            sequence=int(record["sequence"]),
            watermark=str(record.get("watermark", "")),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class EventBatch:
    """Events selected for processing and the checkpoint they can advance.

    Attributes:
        source_identity: Identity of the source.
        events: The selected events, in the order they were selected.
        next_checkpoint: The checkpoint to persist once the batch is
            committed.
        duplicate_event_ids: Sorted, unique identities of events skipped as
            already seen or repeated.
        late_event_ids: Sorted, unique identities of events at or behind
            the checkpoint's sequence.
        has_more: Whether eligible events remain beyond the selected ones.
    """

    source_identity: str
    events: tuple[IncrementalEvent, ...]
    next_checkpoint: EventCheckpoint
    duplicate_event_ids: tuple[str, ...] = ()
    late_event_ids: tuple[str, ...] = ()
    has_more: bool = False

    def __post_init__(self) -> None:
        """Validate batch identity and deterministic event collections.

        Raises:
            ValueError: If the checkpoint belongs to another source, or the
                duplicate or late identities are not sorted and unique.
            TypeError: If `events` holds something other than IncrementalEvent
                values.
        """
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
        """The identities of the selected events, in the order of `events`."""
        return tuple(event.event_id for event in self.events)


@dataclasses.dataclass(frozen=True, slots=True)
class EventCommitReceipt:
    """Result of an idempotent sink commit attempt.

    Attributes:
        source_identity: Identity of the source.
        accepted_event_ids: Sorted, unique identities of events written by
            this attempt.
        duplicate_event_ids: Sorted, unique identities of events skipped
            because they were already committed.
        checkpoint: The checkpoint after the attempt.
    """

    source_identity: str
    accepted_event_ids: tuple[str, ...]
    duplicate_event_ids: tuple[str, ...]
    checkpoint: EventCheckpoint

    def __post_init__(self) -> None:
        """Validate deterministic accepted and duplicate identities.

        Raises:
            ValueError: If the source identity is blank, the checkpoint
                belongs to another source, either identity collection is not
                sorted and unique, or the two collections overlap.
        """
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
        """Return a JSON-compatible commit receipt.

        Returns:
            A dictionary holding the schema version and the receipt fields.
        """
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

    Attributes:
        source_identity: Identity of the source.
        consumer_group: Consumer group the acknowledgement is for.
        event_ids: Sorted, unique identities of the events to acknowledge.
        checkpoint: The committed checkpoint the acknowledgement refers to.
    """

    source_identity: str
    consumer_group: str
    event_ids: tuple[str, ...]
    checkpoint: EventCheckpoint

    def __post_init__(self) -> None:
        """Validate the acknowledgement's source and deterministic IDs.

        Raises:
            ValueError: If the source identity or consumer group is blank, the
                event identities are not sorted and unique, or the checkpoint
                belongs to another source.
        """
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
        """Return a JSON-compatible acknowledgement record.

        Returns:
            A dictionary holding the schema version and the four fields.
        """
        return {
            "schema_version": 1,
            "source_identity": self.source_identity,
            "consumer_group": self.consumer_group,
            "event_ids": list(self.event_ids),
            "checkpoint": self.checkpoint.to_dict(),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild a staged acknowledgement from durable JSON.

        Args:
            record: Mapping in the form produced by `to_dict`.

        Returns:
            The acknowledgement described by `record`.

        Raises:
            ValueError: If the record's schema version is not 1.
        """
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
    """Evidence from one durable consume-and-acknowledge cycle.

    Attributes:
        source_identity: Identity of the source.
        consumer_group: Consumer group that ran the cycle.
        receipt: Receipt of the sink commit.
        acknowledged_event_ids: Sorted, unique identities of the events
            whose acknowledgement was delivered in this cycle.
    """

    source_identity: str
    consumer_group: str
    receipt: EventCommitReceipt
    acknowledged_event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the result's source, group, and acknowledgement IDs.

        Raises:
            ValueError: If the receipt belongs to another source, the consumer
                group is blank, or the acknowledged identities are not sorted
                and unique.
        """
        if self.receipt.source_identity != self.source_identity:
            raise ValueError("result and receipt source identities differ")
        if not self.consumer_group.strip():
            raise ValueError("consumer_group must not be empty")
        if tuple(sorted(set(self.acknowledged_event_ids))) != (
            self.acknowledged_event_ids
        ):
            raise ValueError("acknowledged_event_ids must be sorted and unique")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible consume evidence.

        Returns:
            A dictionary holding the schema version and the result fields.
        """
        return {
            "schema_version": 1,
            "source_identity": self.source_identity,
            "consumer_group": self.consumer_group,
            "receipt": self.receipt.to_dict(),
            "acknowledged_event_ids": list(self.acknowledged_event_ids),
        }


class IncrementalEventSource(abc.ABC):
    """Polymorphic contract for a source that yields ordered events.

    Attributes:
        source_uri: Canonical URI of the source.
        source_identity: Stable identity derived from the canonical URI.
        late_data_policy: Action taken for events at or behind the
            checkpoint's sequence.
    """

    def __init__(
        self,
        source_uri: str,
        late_data_policy: LateDataPolicy = LateDataPolicy.REJECT,
    ) -> None:
        """Initialize source identity and late-data policy.

        Args:
            source_uri: URI of the source; canonicalized on binding.
            late_data_policy: Policy for late events; a plain string value of
                LateDataPolicy is accepted.
        """
        if not isinstance(late_data_policy, LateDataPolicy):
            late_data_policy = LateDataPolicy(late_data_policy)
        self.source_uri = connectors.canonical_source_uri(source_uri)
        self.source_identity = connectors.source_identity(self.source_uri)
        self.late_data_policy = late_data_policy

    @abc.abstractmethod
    def available_events(self) -> Iterable[IncrementalEvent]:
        """Return available events without changing committed state.

        Returns:
            The events the transport currently offers.
        """

    def validate_checkpoint(
        self, checkpoint: EventCheckpoint | None
    ) -> EventCheckpoint:
        """Bind a checkpoint to this source or fail before polling.

        Args:
            checkpoint: The checkpoint to check, or None for a new run.

        Returns:
            `checkpoint`, or an empty checkpoint (sequence 0) for this source
            when `checkpoint` is None.

        Raises:
            IncrementalError: With code "checkpoint_mismatch" if the
                checkpoint belongs to another source.
        """
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

        Records are ordered by sequence then event identity. A record whose
        identity is already committed or already selected is a duplicate. A
        record whose sequence is at or behind the checkpoint is late and is
        handled by the late-data policy: dropped, rejected, or (for ACCEPT)
        selected like any other.

        Args:
            records: Events returned by a transport boundary.
            checkpoint: Last committed sequence and watermark.
            seen_event_ids: IDs already durably committed for deduplication.
            limit: Maximum number of new events to return.

        Returns:
            A batch of at most `limit` events in sequence order, the sorted
            duplicate and late identities, a next checkpoint at the highest
            selected sequence and latest event time (or the unchanged
            checkpoint values when nothing was selected), and `has_more` set
            when more eligible events remained than were selected.

        Raises:
            ValueError: If `limit` is not positive.
            TypeError: If `records` holds something other than IncrementalEvent
                values.
            IncrementalError: With code "late_event" if a late event is found
                and the policy is REJECT. An invalid checkpoint also raises
                IncrementalError, from `validate_checkpoint`.
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

        Failures from `available_events` and from `select_events` (such as an
        IncrementalError for a late event under the REJECT policy) propagate
        unchanged.

        Args:
            checkpoint: Last committed sequence and watermark.
            seen_event_ids: IDs already durably committed for deduplication.
            limit: Maximum number of new events to return.

        Returns:
            The selected batch, as described for `select_events`.
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
        """Initialize a source from a fixed sequence or event factory.

        Args:
            source_uri: URI identifying the source.
            events: Either an iterable of events, captured once, or a
                zero-argument callable returning the events each time they are
                requested.
            late_data_policy: Policy for events behind the checkpoint.
        """
        super().__init__(source_uri, late_data_policy)
        if callable(events):
            self.event_factory = events
        else:
            values = tuple(events)
            self.event_factory = lambda: values

    def available_events(self) -> Iterable[IncrementalEvent]:
        """Return the current event view supplied by the transport seam.

        Returns:
            The events held or produced by the factory.
        """
        return self.event_factory()


class EventStreamSource(IncrementalEventSource):
    """Bounded event-stream adapter around a caller-owned poll boundary.

    The callback is the only transport-specific surface. It receives the
    canonical source URI, the committed checkpoint, and the requested limit;
    it must return at most that many events without acknowledging them. The
    durable ledger remains responsible for commit and downstream idempotency.

    Attributes:
        supported_schemes: URI schemes the adapter accepts.
        event_reader: The injected poll callback.
        retry_budget: Retry policy applied to polling.
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
            retry_budget: Retry policy for transport polling; the default
                budget is used when None.

        Raises:
            UnsupportedUriSchemeError: If the URI scheme is not one of
                `supported_schemes`.
            TypeError: If `event_reader` is not callable.
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
        """Reject unbounded polling without a checkpoint and limit.

        Raises:
            IncrementalError: Always, with code "bounded_poll_required";
                callers must use `read_events`.
        """
        raise IncrementalError(
            "bounded_poll_required",
            "event streams require read_events with a checkpoint and limit",
            self.source_uri,
            {},
        )

    def poll_events(
        self, checkpoint: EventCheckpoint, limit: int
    ) -> tuple[IncrementalEvent, ...]:
        """Poll a bounded transport batch with retry evidence.

        The reader is called under the retry budget; if every attempt fails
        the retry helper's exhaustion error propagates.

        Args:
            checkpoint: Committed position passed to the reader.
            limit: Maximum number of events to request.

        Returns:
            The events the reader returned, at most `limit`.

        Raises:
            ValueError: If `limit` is not positive.
            IncrementalError: With code "batch_limit_exceeded" if the reader
                returns more than `limit` events.
        """
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
        """Poll and select one bounded event batch without acknowledging it.

        Args:
            checkpoint: Last committed sequence and watermark.
            seen_event_ids: IDs already durably committed for deduplication.
            limit: Maximum number of events to poll and return.

        Returns:
            The selected batch, as described for `select_events`.
        """
        current = self.validate_checkpoint(checkpoint)
        return self.select_events(
            self.poll_events(current, limit), current, seen_event_ids, limit
        )


@dataclasses.dataclass(frozen=True, slots=True)
class CommittedIncrementalState:
    """Durable event IDs and checkpoint after an atomic ingestion commit.

    Attributes:
        checkpoint: Position and watermark after the last commit.
        seen_event_ids: Sorted, unique identities of all committed events.
    """

    checkpoint: EventCheckpoint
    seen_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate deterministic deduplication state.

        Raises:
            ValueError: If `seen_event_ids` is not sorted and unique.
        """
        if tuple(sorted(set(self.seen_event_ids))) != self.seen_event_ids:
            raise ValueError("seen_event_ids must be sorted and unique")

    def to_dict(self) -> dict[str, Any]:
        """Return durable state as JSON-compatible data.

        Returns:
            A dictionary holding the schema version, checkpoint and seen event
            identities.
        """
        return {
            "schema_version": 1,
            "checkpoint": self.checkpoint.to_dict(),
            "seen_event_ids": list(self.seen_event_ids),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild committed state from durable JSON.

        Args:
            record: Mapping in the form produced by `to_dict`.

        Returns:
            The state described by `record`.

        Raises:
            ValueError: If the record's schema version is not 1.
        """
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
    """Atomic durable commit boundary for incremental event processing.

    Attributes:
        path: JSON file holding the committed state.
        source_identity: Identity of the one source the ledger serves.
    """

    path: pathlib.Path
    source_identity: str

    @property
    def lock_path(self) -> pathlib.Path:
        """The sidecar file used to serialize ledger writers."""
        return self.path.with_name(f"{self.path.name}.lock")

    @property
    def pending_acknowledgement_path(self) -> pathlib.Path:
        """The durable sidecar file awaiting acknowledgement delivery."""
        return self.path.with_name(f"{self.path.name}.ack")

    @contextlib.contextmanager
    def lock(self) -> Iterator[None]:
        """Serialize ledger writers with the platform's advisory file lock.

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

    def load(self) -> CommittedIncrementalState:
        """Load committed state or initialize an empty source state.

        An unreadable or malformed state file raises the parsing error
        (for example JSONDecodeError or ValueError) unchanged.

        Returns:
            The committed state, or an empty state for this source when the
            state file does not exist.

        Raises:
            IncrementalError: With code "state_source_mismatch" if the stored
                state belongs to another source.
        """
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
        """Read an uncommitted batch using the durable deduplication state.

        Args:
            source: The event source to read from.
            limit: Maximum number of new events to return.

        Returns:
            The batch selected by `source.read_events` from the durable
            checkpoint, skipping already committed event identities.

        Raises:
            IncrementalError: With code "source_mismatch" if the source's
                identity differs from the ledger's.
        """
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
        If writing fails, the temporary file is removed and the error
        propagates.

        Args:
            state: The state to persist.

        Raises:
            IncrementalError: With code "state_source_mismatch" if the state's
                checkpoint belongs to another source.
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
        """Stage an acknowledgement with durable atomic replacement.

        If writing fails, the temporary file is removed and the error
        propagates.

        Args:
            acknowledgement: The acknowledgement to stage for delivery.

        Raises:
            IncrementalError: With code "acknowledgement_source_mismatch" if
                the acknowledgement belongs to another source.
        """
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
        """Load the acknowledgement that still needs transport delivery.

        Returns:
            The staged acknowledgement, or None if there is none.

        Raises:
            IncrementalError: If the staged file is unreadable or malformed
                (code "invalid_pending_acknowledgement"), or it belongs to
                another source (code "acknowledgement_source_mismatch").
        """
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
        """Remove a delivered acknowledgement and flush the directory entry.

        Nothing is removed if no acknowledgement is staged.
        """
        self.pending_acknowledgement_path.unlink(missing_ok=True)
        durability.sync_directory(self.pending_acknowledgement_path.parent)

    def commit(self, batch: EventBatch) -> CommittedIncrementalState:
        """Atomically publish a processed batch's IDs and checkpoint.

        Args:
            batch: The processed batch.

        Returns:
            The state now committed. When every event of the batch is already
            committed and its checkpoint is not ahead of the current one, the
            current state is returned and nothing is written.

        Raises:
            IncrementalError: If the batch belongs to another source (code
                "batch_source_mismatch"), or its checkpoint sequence is behind
                the committed one (code "checkpoint_regression"), or the
                stored state belongs to another source.
        """
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

        Args:
            events: Events to commit; processed in sequence then identity
                order.
            writer: Callable receiving the tuple of unseen events. It is not
                called when every event is already committed.
            acknowledgement: Acknowledgement to stage, or None. It is staged,
                rebound to the resulting checkpoint, only if it lists at
                least one event identity.

        Returns:
            A receipt listing the sorted identities accepted and the sorted
            identities skipped as duplicates (including repeats within
            `events`), and the resulting checkpoint: advanced to the highest
            sequence and latest event time when anything was accepted,
            otherwise the unchanged current checkpoint.

        Raises:
            TypeError: If `events` holds something other than IncrementalEvent
                values or `writer` is not callable.
            IncrementalError: If the acknowledgement belongs to another source
                (code "acknowledgement_source_mismatch") or the stored state
                belongs to another source.
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

    Attributes:
        source: The event source.
        ledger: The ledger holding the durable state.
        consumer_group: Name of the consumer group, stripped of
            surrounding whitespace.
        writer: Callable receiving the tuple of unseen events.
        acknowledge: Callable that delivers an acknowledgement to the
            transport.
    """

    def __init__(
        self,
        source: IncrementalEventSource,
        ledger: IncrementalIngestionLedger,
        consumer_group: str,
        writer: Callable[[tuple[IncrementalEvent, ...]], None],
        acknowledge: Callable[[EventAcknowledgement], None],
    ) -> None:
        """Initialize a consumer around explicit transport and sink seams.

        Args:
            source: The event source.
            ledger: The ledger holding the durable state; it must serve the
                same source identity.
            consumer_group: Non-blank name of the consumer group.
            writer: Callable receiving the tuple of unseen events.
            acknowledge: Callable that delivers an acknowledgement to the
                transport.

        Raises:
            ValueError: If `consumer_group` is blank.
            IncrementalError: With code "source_mismatch" if the source and
                ledger identities differ.
            TypeError: If `writer` or `acknowledge` is not callable.
        """
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
        """Deliver and clear a previously staged acknowledgement.

        Returns:
            The event identities acknowledged, or an empty tuple if none was
            staged.

        Raises:
            IncrementalError: If the staged acknowledgement belongs to another
                consumer group, or the acknowledge callback raises (the
                staged record is kept for retry).
        """
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
        """Consume, durably commit, acknowledge, and return evidence.

        A previously staged acknowledgement is delivered first. Failures from
        reading, committing and delivering propagate unchanged.

        Args:
            limit: Maximum number of new events to read.

        Returns:
            The commit receipt and the identities whose acknowledgement was
            delivered in this cycle.
        """
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
