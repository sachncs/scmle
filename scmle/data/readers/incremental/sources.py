"""Event sources: the selection contract and its transport adapters."""

import abc
from collections.abc import Callable
from collections.abc import Iterable
from urllib.parse import urlsplit

from scmle.contracts import errors
from scmle.data.readers import connectors
from scmle.data.readers.incremental import timeline


class Source(abc.ABC):
    """Polymorphic contract for a source that yields ordered events.

    Attributes:
        uri: Canonical URI of the source.
        identity: Stable identity derived from the canonical URI.
        lateness: Action taken for events at or behind the
            checkpoint's sequence.
    """

    def __init__(
        self,
        source_uri: str,
        late_data_policy: timeline.Lateness | str = timeline.Lateness.REJECT,
    ) -> None:
        """Initializes source identity and late-data policy.

        Args:
            source_uri: URI of the source; canonicalized on binding.
            late_data_policy: Policy for late events; a plain string value of
                Lateness is accepted.
        """
        if not isinstance(late_data_policy, timeline.Lateness):
            late_data_policy = timeline.Lateness(late_data_policy)
        self.uri = connectors.canonicalize(source_uri)
        self.identity = connectors.identify(self.uri)
        self.lateness = late_data_policy

    @abc.abstractmethod
    def unconsumed(self) -> Iterable[timeline.Event]:
        """Returns available events without changing committed state.

        Returns:
            The events the transport currently offers.
        """

    def rebind(
        self, checkpoint: timeline.Checkpoint | None
    ) -> timeline.Checkpoint:
        """Binds a checkpoint to this source, failing before any polling.

        Args:
            checkpoint: The checkpoint to check, or None for a new run.

        Returns:
            `checkpoint`, or an empty checkpoint (sequence 0) for this source
            when `checkpoint` is None.

        Raises:
            IncrementalError: With code "checkpoint_mismatch" if the
                checkpoint belongs to another source.
        """
        current = checkpoint or timeline.Checkpoint(self.identity)
        try:
            current.bind(self.identity)
        except ValueError as error:
            raise timeline.IncrementalError(
                "checkpoint_mismatch",
                str(error),
                self.uri,
                {"checkpoint": current.to_dict()},
            ) from error
        return current

    def select(
        self,
        records: Iterable[timeline.Event],
        checkpoint: timeline.Checkpoint | None = None,
        seen_event_ids: Iterable[str] = (),
        limit: int = 1_000,
    ) -> timeline.Batch:
        """Selects a bounded, deduplicated batch from transport records.

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
            selected sequence and the latest of the selected event times and
            the checkpoint's watermark (the unchanged checkpoint values when
            nothing was selected), and `has_more` set
            when more eligible events remained than were selected.

        Raises:
            ValueError: If `limit` is not positive, or a selected event's
                time is not a timezone-aware ISO-8601 string.
            TypeError: If `records` holds something other than Event
                values.
            IncrementalError: With code "late_event" if a late event is found
                and the policy is REJECT. An invalid checkpoint also raises
                IncrementalError, from `rebind`.
        """
        if limit <= 0:
            raise errors.InputError("limit must be positive")
        current = self.rebind(checkpoint)
        committed = frozenset(seen_event_ids)
        values = tuple(records)
        if any(not isinstance(event, timeline.Event) for event in values):
            raise errors.KindError("transport records must yield Event values")
        ordered = tuple(
            sorted(values, key=lambda event: (event.sequence, event.identifier))
        )
        eligible, duplicates, late = self.sift(ordered, committed, current)
        selected = tuple(eligible[:limit])
        next_sequence = max(
            (event.sequence for event in selected), default=current.sequence
        )
        next_sequence = max(next_sequence, current.sequence)
        checkpoint_value = timeline.Checkpoint(
            self.identity,
            next_sequence,
            timeline.horizon(selected, current.watermark),
        )
        return timeline.Batch(
            self.identity,
            selected,
            checkpoint_value,
            tuple(sorted(duplicates)),
            tuple(sorted(late)),
            more=len(eligible) > len(selected),
        )

    def sift(
        self,
        ordered: tuple[timeline.Event, ...],
        committed: frozenset[str],
        current: timeline.Checkpoint,
    ) -> tuple[list[timeline.Event], set[str], set[str]]:
        """Separates eligible events from duplicate and late ones.

        Args:
            ordered: Events in sequence then identity order.
            committed: Identities already durably committed.
            current: The bound checkpoint.

        Returns:
            The eligible events in order, the duplicate identities and the
            late identities. A late event stays eligible under ACCEPT.

        Raises:
            IncrementalError: With code "late_event" if a late event is found
                and the policy is REJECT.
        """
        duplicates: set[str] = set()
        late: set[str] = set()
        eligible: list[timeline.Event] = []
        selected_ids: set[str] = set()
        for event in ordered:
            if (
                event.identifier in committed
                or event.identifier in selected_ids
            ):
                duplicates.add(event.identifier)
                continue
            if event.sequence <= current.sequence:
                late.add(event.identifier)
                if self.lateness is timeline.Lateness.DROP:
                    continue
                if self.lateness is timeline.Lateness.REJECT:
                    raise timeline.IncrementalError(
                        "late_event",
                        f"event '{event.identifier}' is older than checkpoint",
                        self.uri,
                        {
                            "event_id": event.identifier,
                            "event_sequence": event.sequence,
                            "checkpoint_sequence": current.sequence,
                        },
                    )
            eligible.append(event)
            selected_ids.add(event.identifier)
        return eligible, duplicates, late

    def read(
        self,
        checkpoint: timeline.Checkpoint | None = None,
        seen_event_ids: Iterable[str] = (),
        limit: int = 1_000,
    ) -> timeline.Batch:
        """Selects a bounded batch without committing its side effects.

        Failures from `unconsumed` and from `select` (such as an
        IncrementalError for a late event under the REJECT policy) propagate
        unchanged.

        Args:
            checkpoint: Last committed sequence and watermark.
            seen_event_ids: IDs already durably committed for deduplication.
            limit: Maximum number of new events to return.

        Returns:
            The selected batch, as described for `select`.
        """
        return self.select(self.unconsumed(), checkpoint, seen_event_ids, limit)


class Replay(Source):
    """Deterministic source adapter for tests and local replay fixtures."""

    def __init__(
        self,
        source_uri: str,
        events: Iterable[timeline.Event]
        | Callable[[], Iterable[timeline.Event]],
        late_data_policy: timeline.Lateness = timeline.Lateness.REJECT,
    ) -> None:
        """Initializes a source from a fixed sequence or event factory.

        Args:
            source_uri: URI identifying the source.
            events: Either an iterable of events, captured once, or a
                zero-argument callable returning the events each time they are
                requested.
            late_data_policy: Policy for events behind the checkpoint.
        """
        super().__init__(source_uri, late_data_policy)
        if callable(events):
            self.factory = events
        else:
            values = tuple(events)
            self.factory = lambda: values

    def unconsumed(self) -> Iterable[timeline.Event]:
        """Returns the current event view supplied by the transport seam.

        Returns:
            The events held or produced by the factory.
        """
        return self.factory()


class Stream(Source):
    """Bounded event-stream adapter around a caller-owned poll boundary.

    The callback is the only transport-specific surface. It receives the
    canonical source URI, the committed checkpoint, and the requested limit;
    it must return at most that many events without acknowledging them. The
    durable ledger remains responsible for commit and downstream idempotency.

    Attributes:
        schemes: URI schemes the adapter accepts.
        reader: The injected poll callback.
        budget: Retry policy applied to polling.
    """

    schemes = (
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
            [str, timeline.Checkpoint, int], Iterable[timeline.Event]
        ],
        late_data_policy: timeline.Lateness = timeline.Lateness.REJECT,
        budget: connectors.Retries | None = None,
    ) -> None:
        """Initializes a source around an injected broker poll callback.

        Args:
            source_uri: Broker URI identifying topic, stream, or subscription.
            event_reader: Callback receiving URI, checkpoint, and batch limit.
            late_data_policy: Policy applied to events behind the checkpoint.
            budget: Retry policy for transport polling; the default
                budget is used when None.

        Raises:
            SchemeError: If the URI scheme is not one of
                `schemes`.
            TypeError: If `reader` is not callable.
        """
        value = source_uri.strip()
        scheme = urlsplit(value).scheme.lower()
        if scheme not in self.schemes:
            raise connectors.SchemeError(value, scheme, self.schemes)
        if not callable(event_reader):
            raise errors.KindError("event_reader must be callable")
        super().__init__(value, late_data_policy)
        self.reader = event_reader
        self.budget = budget or connectors.Retries()

    def unconsumed(self) -> Iterable[timeline.Event]:
        """Rejects unbounded polling without a checkpoint and limit.

        Raises:
            IncrementalError: Always, with code "bounded_poll_required";
                callers must use `read`.
        """
        raise timeline.IncrementalError(
            "bounded_poll_required",
            "event streams require read_events with a checkpoint and limit",
            self.uri,
            {},
        )

    def poll(
        self, checkpoint: timeline.Checkpoint, limit: int
    ) -> tuple[timeline.Event, ...]:
        """Polls a bounded transport batch with retry evidence.

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
            raise errors.InputError("limit must be positive")
        records = connectors.persevere(
            lambda: tuple(self.reader(self.uri, checkpoint, limit)),
            "poll_events",
            self.uri,
            self.budget,
        )
        if len(records) > limit:
            raise timeline.IncrementalError(
                "batch_limit_exceeded",
                "event reader returned more events than requested",
                self.uri,
                {"limit": limit, "received": len(records)},
            )
        return records

    def read(
        self,
        checkpoint: timeline.Checkpoint | None = None,
        seen_event_ids: Iterable[str] = (),
        limit: int = 1_000,
    ) -> timeline.Batch:
        """Polls and selects one bounded event batch without acknowledging it.

        Args:
            checkpoint: Last committed sequence and watermark.
            seen_event_ids: IDs already durably committed for deduplication.
            limit: Maximum number of events to poll and return.

        Returns:
            The selected batch, as described for `select_events`.
        """
        current = self.rebind(checkpoint)
        return self.select(
            self.poll(current, limit), current, seen_event_ids, limit
        )
