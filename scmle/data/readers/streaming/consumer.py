"""A durable writer coordinated with an external acknowledgement."""

from collections.abc import Callable

import pandas

from scmle.contracts import errors
from scmle.data.readers.streaming import failure
from scmle.data.readers.streaming import feed
from scmle.data.readers.streaming import journal
from scmle.data.readers.streaming import transaction


class Consumer:
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
        ledger: journal.Journal,
        consumer_group: str,
        writer: Callable[[pandas.DataFrame, str], None],
        acknowledge: Callable[[transaction.Acknowledgement], None],
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
            raise errors.InputError("consumer_group must not be empty")
        writer_candidate: object = writer
        acknowledge_candidate: object = acknowledge
        if not callable(writer_candidate) or not callable(
            acknowledge_candidate
        ):
            raise errors.KindError("writer and acknowledge must be callable")
        self.ledger = ledger
        self.consumer_group = consumer_group
        self.writer = writer
        self.acknowledge = acknowledge

    def handoff(self) -> str | None:
        """Delivers and clears one staged acknowledgement after restart.

        Returns:
            The batch identity of the acknowledgement delivered, or None if
            none was staged.

        Raises:
            StreamingError: If the staged acknowledgement belongs to another
                consumer group, or the acknowledge callback raises (the
                staged record is kept for retry).
        """
        pending = self.ledger.reclaim()
        if pending is None:
            return None
        if pending.consumer != self.consumer_group:
            raise failure.StreamingError(
                "consumer_group_mismatch",
                "pending acknowledgement belongs to another consumer group",
                self.ledger.source.source_identity(),
                {"pending_consumer_group": pending.consumer},
            )
        try:
            self.acknowledge(pending)
        except Exception as error:
            raise failure.StreamingError(
                "acknowledgement_failed",
                "external acknowledgement failed; retry is required",
                self.ledger.source.source_identity(),
                {"batch_id": pending.batch},
            ) from error
        self.ledger.retract()
        return pending.batch

    def consume(
        self,
        rows: int = feed.BATCH_ROWS,
        watermark_column: str | None = None,
    ) -> transaction.Receipt:
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
        self.handoff()
        batch = self.ledger.read_batch(rows, watermark_column)
        acknowledgement = transaction.Acknowledgement(
            batch.source_identity,
            self.consumer_group,
            batch.batch_id,
            batch.next_checkpoint,
        )
        receipt = self.ledger.finalize(batch, self.writer, acknowledgement)
        self.handoff()
        return receipt
