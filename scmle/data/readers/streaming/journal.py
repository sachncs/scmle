"""The atomic local ledger that commits streaming batches once."""

from collections.abc import Callable
import dataclasses

import pandas

from scmle.contracts import errors
from scmle.data.readers.streaming import archive
from scmle.data.readers.streaming import failure
from scmle.data.readers.streaming import feed
from scmle.data.readers.streaming import progress
from scmle.data.readers.streaming import transaction


def adjoin(
    current: transaction.Settlement, batch: progress.Batch, source: feed.Feed
) -> None:
    """Rejects a new batch that does not start at the durable checkpoint.

    Args:
        current: The durable state.
        batch: The batch about to be committed.
        source: The source the ledger consumes, named in the failure.

    Raises:
        StreamingError: With code "checkpoint_mismatch" if the batch starts
            elsewhere.
    """
    if batch.start_offset != current.checkpoint.offset:
        raise failure.StreamingError(
            "checkpoint_mismatch",
            "batch does not start at the durable checkpoint",
            source.source_identity(),
            {
                "batch_start_offset": batch.start_offset,
                "checkpoint_offset": current.checkpoint.offset,
            },
        )


def replay(
    current: transaction.Settlement,
    batch: progress.Batch,
    acknowledgement: transaction.Acknowledgement,
    source: feed.Feed,
) -> transaction.Receipt | None:
    """Answers a retried transactional commit from the stored evidence.

    Args:
        current: The durable state.
        batch: The batch being committed.
        acknowledgement: The acknowledgement the caller binds to it.
        source: The source the ledger consumes, named in a failure.

    Returns:
        A receipt with `duplicate` set and the stored evidence when the batch
        was committed with an equal acknowledgement, or None when the batch
        has not been committed.

    Raises:
        StreamingError: If the batch is bound to another acknowledgement, or
            it is recorded as committed without transaction evidence.
    """
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
            raise failure.StreamingError(
                "acknowledgement_mismatch",
                "batch is already bound to another acknowledgement",
                source.source_identity(),
                {},
            )
        return transaction.Receipt(
            batch.source_identity,
            batch.batch_id,
            accepted=False,
            duplicate=True,
            checkpoint=evidence.acknowledgement.checkpoint,
            sink_receipt=evidence.sink_receipt,
            acknowledgement=evidence.acknowledgement,
        )
    if batch.batch_id in current.committed_batch_ids:
        raise failure.StreamingError(
            "transaction_evidence_missing",
            "batch was committed without transactional evidence",
            source.source_identity(),
            {},
        )
    return None


def settle(
    current: transaction.Settlement,
    batch: progress.Batch,
    evidence: transaction.Commitment | None = None,
) -> transaction.Settlement:
    """Returns the durable state once a batch is committed.

    Args:
        current: The durable state before the commit.
        batch: The committed batch.
        evidence: Transaction evidence for the batch, or None to keep the
            current evidence unchanged.

    Returns:
        The state at the batch's next checkpoint with the batch identity
        added, and the evidence added in batch identity order.
    """
    recorded = current.transaction_evidence
    if evidence is not None:
        recorded = tuple(
            sorted((*recorded, evidence), key=lambda item: item.batch_id)
        )
    return transaction.Settlement(
        batch.next_checkpoint,
        tuple(sorted((*current.committed_batch_ids, batch.batch_id))),
        recorded,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class Journal(archive.Archive):
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

    def commit_transaction(
        self,
        batch: progress.Batch,
        sink: transaction.Sink,
        acknowledgement: transaction.Acknowledgement,
    ) -> transaction.Receipt:
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
            TypeError: If `sink` is not a Sink or
                `acknowledgement` is not a Acknowledgement.
            StreamingError: If the batch, acknowledgement or sink receipt does
                not match the source and snapshot, the batch is bound to a
                different acknowledgement, the batch is recorded as committed
                without evidence, the batch does not start at the durable
                checkpoint, or the sink returns something other than a
                Voucher.
        """
        if not isinstance(sink, transaction.Sink):
            raise errors.KindError("sink must implement Sink")
        if not isinstance(acknowledgement, transaction.Acknowledgement):
            raise errors.KindError("acknowledgement must be Acknowledgement")
        self.validate(batch)
        self.vouch(batch, acknowledgement)
        with self.lock():
            current = self.load()
            duplicate = replay(current, batch, acknowledgement, self.source)
            if duplicate is not None:
                return duplicate
            adjoin(current, batch, self.source)
            return self.deliver(current, batch, sink, acknowledgement)

    def deliver(
        self,
        current: transaction.Settlement,
        batch: progress.Batch,
        sink: transaction.Sink,
        acknowledgement: transaction.Acknowledgement,
    ) -> transaction.Receipt:
        """Commits a new batch to the sink and records the evidence.

        The caller holds the ledger lock and has checked that the batch is new
        and starts at the durable checkpoint.

        Args:
            current: The durable state read under the lock.
            batch: The batch to commit.
            sink: Caller-owned sink that commits the batch.
            acknowledgement: Acknowledgement bound to the batch.

        Returns:
            A receipt with `accepted` set, the sink receipt and the
            acknowledgement.

        Raises:
            StreamingError: If the sink returns something other than a
                Voucher, or a receipt for another source, batch or
                checkpoint.
        """
        sink_receipt = sink.commit(batch)
        if not isinstance(sink_receipt, transaction.Voucher):
            raise failure.StreamingError(
                "invalid_sink_receipt",
                "transactional sink returned an invalid receipt",
                self.source.source_identity(),
                {"type": type(sink_receipt).__name__},
            )
        self.attest(batch, sink_receipt)
        evidence = transaction.Commitment(
            batch.source_identity,
            batch.batch_id,
            sink_receipt,
            acknowledgement,
        )
        self.write_atomically(settle(current, batch, evidence))
        return transaction.Receipt(
            batch.source_identity,
            batch.batch_id,
            accepted=True,
            duplicate=False,
            checkpoint=batch.next_checkpoint,
            sink_receipt=sink_receipt,
            acknowledgement=acknowledgement,
        )

    def finalize(
        self,
        batch: progress.Batch,
        writer: Callable[[pandas.DataFrame, str], None],
        acknowledgement: transaction.Acknowledgement | None = None,
    ) -> transaction.Receipt:
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
            raise errors.KindError("writer must be callable")
        self.validate(batch)
        if acknowledgement is not None and (
            acknowledgement.source_identity != batch.source_identity
            or acknowledgement.batch_id != batch.batch_id
        ):
            raise failure.StreamingError(
                "acknowledgement_batch_mismatch",
                "acknowledgement does not match the batch",
                self.source.source_identity(),
                {},
            )
        with self.lock():
            current = self.load()
            if batch.batch_id in current.committed_batch_ids:
                if acknowledgement is not None:
                    self.earmark(acknowledgement)
                return transaction.Receipt(
                    batch.source_identity,
                    batch.batch_id,
                    accepted=False,
                    duplicate=True,
                    checkpoint=current.checkpoint,
                )
            adjoin(current, batch, self.source)
            writer(batch.frame, batch.batch_id)
            self.write_atomically(settle(current, batch))
            if acknowledgement is not None:
                self.earmark(
                    dataclasses.replace(
                        acknowledgement, checkpoint=batch.next_checkpoint
                    )
                )
            return transaction.Receipt(
                batch.source_identity,
                batch.batch_id,
                accepted=True,
                duplicate=False,
                checkpoint=batch.next_checkpoint,
            )
