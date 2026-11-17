"""Commit receipts, acknowledgements and the durable ledger state."""

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
from typing import Any
from typing import ClassVar
from typing import Self

from scmle.contracts import errors
from scmle.contracts import records
from scmle.data.readers.incremental import timeline

VERSION = 2


def lift(layout: records.Layout) -> records.Layout:
    """Moves the checkpoint of a version-1 record to version 2.

    Args:
        layout: The record after its own keys were renamed.

    Returns:
        The record with its checkpoint's keys renamed.
    """
    return shift(layout, timeline.Checkpoint.migrations[0].upgrade)


def lower(layout: records.Layout) -> records.Layout:
    """Undoes `lift`.

    Args:
        layout: A version-2 record.

    Returns:
        The record with its checkpoint in version-1 form.
    """
    return shift(layout, timeline.Checkpoint.migrations[0].downgrade)


def shift(
    layout: records.Layout,
    step: Callable[[records.Layout], records.Layout],
) -> records.Layout:
    """Applies a step to a stored record's checkpoint.

    Args:
        layout: The stored record.
        step: Moves one stored checkpoint.

    Returns:
        The record with its checkpoint moved, or unchanged when it holds no
        checkpoint mapping (loading then rejects it).
    """
    checkpoint = layout.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        return layout
    return {**layout, "checkpoint": step(dict(checkpoint))}


@dataclasses.dataclass(frozen=True, slots=True)
class Ack:
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
    checkpoint: timeline.Checkpoint

    def __post_init__(self) -> None:
        """Validates deterministic accepted and duplicate identities.

        Raises:
            ValueError: If the source identity is blank, the checkpoint
                belongs to another source, either identity collection is not
                sorted and unique, or the two collections overlap.
        """
        if not self.source_identity.strip():
            raise errors.InputError("source_identity must not be empty")
        if self.checkpoint.source != self.source_identity:
            raise errors.InputError(
                "receipt and checkpoint source identities differ"
            )
        for name in ("accepted_event_ids", "duplicate_event_ids"):
            values = getattr(self, name)
            if tuple(sorted(set(values))) != values:
                raise errors.InputError(f"{name} must be sorted and unique")
        if set(self.accepted_event_ids) & set(self.duplicate_event_ids):
            raise errors.InputError(
                "accepted and duplicate event IDs must be disjoint"
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible commit receipt.

        Returns:
            A dictionary holding the schema version and the receipt fields.
        """
        return {
            "schema_version": VERSION,
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
        source: Identity of the source.
        consumer: Consumer group the acknowledgement is for.
        events: Sorted, unique identities of the events to acknowledge.
        checkpoint: The committed checkpoint the acknowledgement refers to.
        migrations: The record's version steps, oldest first.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={
                "source_identity": "source",
                "consumer_group": "consumer",
                "event_ids": "events",
            },
            change=lift,
            revert=lower,
        ),
    )

    source: str
    consumer: str
    events: tuple[str, ...]
    checkpoint: timeline.Checkpoint

    def __post_init__(self) -> None:
        """Validates the acknowledgement's source and deterministic IDs.

        Raises:
            ValueError: If the source identity or consumer group is blank, the
                event identities are not sorted and unique, or the checkpoint
                belongs to another source.
        """
        if not self.source.strip():
            raise errors.InputError("source must not be empty")
        if not self.consumer.strip():
            raise errors.InputError("consumer must not be empty")
        if tuple(sorted(set(self.events))) != self.events:
            raise errors.InputError("events must be sorted and unique")
        if self.checkpoint.source != self.source:
            raise errors.InputError(
                "acknowledgement and checkpoint source identities differ"
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible acknowledgement record.

        Returns:
            A dictionary holding the schema version and the four fields.
        """
        return {
            "schema_version": VERSION,
            "source": self.source,
            "consumer": self.consumer,
            "events": list(self.events),
            "checkpoint": self.checkpoint.to_dict(),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a staged acknowledgement from durable JSON.

        A version-1 record is migrated first.

        Args:
            record: Mapping in the form produced by `to_dict`.

        Returns:
            The acknowledgement described by `record`.

        Raises:
            ValueError: If the record's schema version is missing or newer
                than this build.
        """
        if not isinstance(record.get("schema_version"), int):
            raise errors.InputError(
                "unsupported acknowledgement schema version"
            )
        layout = records.load("EventAcknowledgement", cls.migrations, record)
        return cls(
            source=str(layout["source"]),
            consumer=str(layout["consumer"]),
            events=tuple(str(value) for value in layout["events"]),
            checkpoint=timeline.Checkpoint.from_dict(layout["checkpoint"]),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Consumption:
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
    receipt: Ack
    acknowledged_event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validates the result's source, group, and acknowledgement IDs.

        Raises:
            ValueError: If the receipt belongs to another source, the consumer
                group is blank, or the acknowledged identities are not sorted
                and unique.
        """
        if self.receipt.source_identity != self.source_identity:
            raise errors.InputError(
                "result and receipt source identities differ"
            )
        if not self.consumer_group.strip():
            raise errors.InputError("consumer_group must not be empty")
        if tuple(sorted(set(self.acknowledged_event_ids))) != (
            self.acknowledged_event_ids
        ):
            raise errors.InputError(
                "acknowledged_event_ids must be sorted and unique"
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible consume evidence.

        Returns:
            A dictionary holding the schema version and the result fields.
        """
        return {
            "schema_version": VERSION,
            "source_identity": self.source_identity,
            "consumer_group": self.consumer_group,
            "receipt": self.receipt.to_dict(),
            "acknowledged_event_ids": list(self.acknowledged_event_ids),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Settlement:
    """Durable event IDs and checkpoint after an atomic ingestion commit.

    Attributes:
        checkpoint: Position and watermark after the last commit.
        seen: Sorted, unique identities of all committed events.
        migrations: The record's version steps, oldest first.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={"seen_event_ids": "seen"}, change=lift, revert=lower
        ),
    )

    checkpoint: timeline.Checkpoint
    seen: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validates deterministic deduplication state.

        Raises:
            ValueError: If `seen` is not sorted and unique.
        """
        if tuple(sorted(set(self.seen))) != self.seen:
            raise errors.InputError("seen must be sorted and unique")

    def to_dict(self) -> dict[str, Any]:
        """Returns durable state as JSON-compatible data.

        Returns:
            A dictionary holding the schema version, checkpoint and seen event
            identities.
        """
        return {
            "schema_version": VERSION,
            "checkpoint": self.checkpoint.to_dict(),
            "seen": list(self.seen),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds committed state from durable JSON.

        A version-1 record is migrated first.

        Args:
            record: Mapping in the form produced by `to_dict`.

        Returns:
            The state described by `record`.

        Raises:
            ValueError: If the record's schema version is missing or newer
                than this build.
        """
        if not isinstance(record.get("schema_version"), int):
            raise errors.InputError(
                "unsupported incremental state schema version"
            )
        layout = records.load("Settlement", cls.migrations, record)
        return cls(
            checkpoint=timeline.Checkpoint.from_dict(layout["checkpoint"]),
            seen=tuple(str(item) for item in layout["seen"]),
        )
