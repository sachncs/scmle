"""Events, their committed checkpoint and the batches that advance it."""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import datetime
import enum
from typing import Any
from typing import Self

from scmle.contracts import errors


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
        """Initializes a structured incremental-ingestion failure.

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


class Lateness(enum.StrEnum):
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
    """Parses a timezone-aware ISO-8601 timestamp.

    Args:
        value: ISO-8601 timestamp; a trailing "Z" means UTC.
        field_name: Name used in the error message.

    Returns:
        The timestamp converted to UTC.

    Raises:
        ValueError: If `value` is not a non-blank string, is not ISO-8601,
            or carries no timezone.
    """
    candidate: object = value
    if not isinstance(candidate, str) or not candidate.strip():
        raise errors.InputError(f"{field_name} must not be empty")
    try:
        parsed = datetime.datetime.fromisoformat(value)
    except ValueError as error:
        raise errors.InputError(f"{field_name} must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise errors.InputError(f"{field_name} must include a timezone")
    return parsed.astimezone(datetime.UTC)


@dataclasses.dataclass(frozen=True, slots=True)
class Event:
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
        """Validates event identity, ordering, time, and payload shape.

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
        event_id: object = self.event_id
        if not isinstance(event_id, str) or not event_id.strip():
            raise errors.InputError("event_id must not be empty")
        if isinstance(self.sequence, bool) or not isinstance(
            self.sequence, int
        ):
            raise errors.KindError("sequence must be an integer")
        if self.sequence < 1:
            raise errors.InputError("sequence must be a positive integer")
        parse_timestamp(self.event_time, "event_time")
        if self.ingest_time:
            parse_timestamp(self.ingest_time, "ingest_time")
        if not isinstance(self.payload, Mapping):
            raise errors.KindError("payload must be a mapping")
        object.__setattr__(self, "payload", dict(self.payload))

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible event record.

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
        """Rebuilds an event from a persisted record.

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
class Checkpoint:
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
        """Validates source identity, sequence, and optional watermark.

        Raises:
            ValueError: If the source identity is blank, the sequence is
                negative, or the watermark is not timezone-aware ISO-8601.
            TypeError: If the sequence is not an integer.
        """
        if not self.source_identity.strip():
            raise errors.InputError("source_identity must not be empty")
        if isinstance(self.sequence, bool) or not isinstance(
            self.sequence, int
        ):
            raise errors.KindError("sequence must be an integer")
        if self.sequence < 0:
            raise errors.InputError("sequence must be a non-negative integer")
        if self.watermark:
            parse_timestamp(self.watermark, "watermark")

    def bind(self, source_identity: str) -> "Checkpoint":
        """Binds an uninitialized checkpoint to exactly one source.

        Args:
            source_identity: Identity of the source the checkpoint must belong
                to.

        Returns:
            This checkpoint, unchanged.

        Raises:
            ValueError: If the checkpoint belongs to a different source.
        """
        if self.source_identity != source_identity:
            raise errors.InputError("checkpoint belongs to a different source")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible checkpoint.

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
        """Rebuilds a checkpoint from durable state.

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
class Batch:
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
    events: tuple[Event, ...]
    next_checkpoint: Checkpoint
    duplicate_event_ids: tuple[str, ...] = ()
    late_event_ids: tuple[str, ...] = ()
    has_more: bool = False

    def __post_init__(self) -> None:
        """Validates batch identity and deterministic event collections.

        Raises:
            ValueError: If the checkpoint belongs to another source, or the
                duplicate or late identities are not sorted and unique.
            TypeError: If `events` holds something other than Event
                values.
        """
        if self.next_checkpoint.source_identity != self.source_identity:
            raise errors.InputError(
                "batch and checkpoint source identities differ"
            )
        if any(not isinstance(event, Event) for event in self.events):
            raise errors.KindError("events must contain Event values")
        if tuple(sorted(set(self.duplicate_event_ids))) != (
            self.duplicate_event_ids
        ):
            raise errors.InputError(
                "duplicate_event_ids must be sorted and unique"
            )
        if tuple(sorted(set(self.late_event_ids))) != self.late_event_ids:
            raise errors.InputError("late_event_ids must be sorted and unique")

    @property
    def event_ids(self) -> tuple[str, ...]:
        """The identities of the selected events, in the order of `events`."""
        return tuple(event.event_id for event in self.events)


def horizon(events: Sequence[Event], watermark: str) -> str:
    """Returns the latest of the events' times and a watermark.

    The event times are parsed before the watermark.

    Args:
        events: Events whose times are compared.
        watermark: The current watermark as an ISO-8601 string, or empty.

    Returns:
        The latest time as an ISO-8601 UTC string, or empty when there are
        no events and no watermark.
    """
    timestamps = [
        parse_timestamp(event.event_time, "event_time") for event in events
    ]
    if watermark:
        timestamps.append(parse_timestamp(watermark, "watermark"))
    latest = max(timestamps, default=None)
    return latest.isoformat() if latest else ""


def forward(checkpoint: Checkpoint, events: Sequence[Event]) -> Checkpoint:
    """Returns the checkpoint after committing events.

    Args:
        checkpoint: The committed checkpoint.
        events: The newly committed events; not empty.

    Returns:
        A checkpoint for the same source at the highest of its sequence and
        the events' sequences, with the latest of its watermark and the
        events' times.
    """
    return Checkpoint(
        checkpoint.source_identity,
        max(checkpoint.sequence, *(event.sequence for event in events)),
        horizon(events, checkpoint.watermark),
    )
