"""The consumer-group boundary with durable commits and acknowledgements."""

from collections.abc import Callable

from scmle.contracts import errors
from scmle.data.readers.incremental import evidence
from scmle.data.readers.incremental import sources
from scmle.data.readers.incremental import store
from scmle.data.readers.incremental import timeline


class Ingestor:
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
        group: Name of the consumer group, stripped of
            surrounding whitespace.
        writer: Callable receiving the tuple of unseen events.
        acknowledge: Callable that delivers an acknowledgement to the
            transport.
    """

    def __init__(
        self,
        source: sources.Source,
        ledger: store.Ledger,
        consumer_group: str,
        writer: Callable[[tuple[timeline.Event, ...]], None],
        acknowledge: Callable[[evidence.Acknowledgement], None],
    ) -> None:
        """Initializes a consumer around explicit transport and sink seams.

        Args:
            source: The event source.
            ledger: The ledger holding the durable state; it must serve the
                same source identity.
            consumer_group: Non-blank name of the consumer group.
            writer: Callable receiving the tuple of unseen events.
            acknowledge: Callable that delivers an acknowledgement to the
                transport.

        Raises:
            ValueError: If `group` is blank.
            IncrementalError: With code "source_mismatch" if the source and
                ledger identities differ.
            TypeError: If `writer` or `acknowledge` is not callable.
        """
        group = consumer_group.strip()
        if not group:
            raise errors.InputError("consumer_group must not be empty")
        if source.identity != ledger.identity:
            raise timeline.IncrementalError(
                "source_mismatch",
                "source does not match consumer ledger",
                ledger.identity,
                {"source_identity": source.identity},
            )
        if not callable(writer):
            raise errors.KindError("writer must be callable")
        if not callable(acknowledge):
            raise errors.KindError("acknowledge must be callable")
        self.source = source
        self.ledger = ledger
        self.group = group
        self.writer = writer
        self.acknowledge = acknowledge

    def handoff(self) -> tuple[str, ...]:
        """Delivers and clears a previously staged acknowledgement.

        Returns:
            The event identities acknowledged, or an empty tuple if none was
            staged.

        Raises:
            IncrementalError: If the staged acknowledgement belongs to another
                consumer group, or the acknowledge callback raises (the
                staged record is kept for retry).
        """
        pending = self.ledger.reclaim()
        if pending is None:
            return ()
        if pending.consumer != self.group:
            raise timeline.IncrementalError(
                "consumer_group_mismatch",
                "pending acknowledgement belongs to another consumer group",
                self.source.uri,
                {"pending_group": pending.consumer},
            )
        try:
            self.acknowledge(pending)
        except Exception as error:
            raise timeline.IncrementalError(
                "acknowledgement_failed",
                "transport acknowledgement failed; retry is required",
                self.source.uri,
                {
                    "consumer_group": self.group,
                    "event_ids": list(pending.events),
                },
            ) from error
        self.ledger.retract()
        return pending.events

    def consume(self, limit: int = 1_000) -> evidence.Consumption:
        """Consume, durably commit, acknowledge, and return evidence.

        A previously staged acknowledgement is delivered first. Failures from
        reading, committing and delivering propagate unchanged.

        Args:
            limit: Maximum number of new events to read.

        Returns:
            The commit receipt and the identities whose acknowledgement was
            delivered in this cycle.
        """
        self.handoff()
        batch = self.ledger.read(self.source, limit)
        event_ids = tuple(
            sorted(
                set(batch.identities) | set(batch.duplicates) | set(batch.late)
            )
        )
        acknowledgement = evidence.Acknowledgement(
            source=self.source.identity,
            consumer=self.group,
            events=event_ids,
            checkpoint=batch.checkpoint,
        )
        receipt = self.ledger.finalize(
            batch.events, self.writer, acknowledgement
        )
        acknowledged = self.handoff()
        return evidence.Consumption(
            identity=self.source.identity,
            group=self.group,
            receipt=receipt,
            acknowledged=acknowledged,
        )
