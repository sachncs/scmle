"""Durable streaming state and the checks that bind evidence to its source."""

import dataclasses
import json

from scmle import durability
from scmle.data.readers import sidecar
from scmle.data.readers.streaming import failure
from scmle.data.readers.streaming import feed
from scmle.data.readers.streaming import progress
from scmle.data.readers.streaming import transaction


def confine(
    source: feed.Feed,
    identity: str,
    snapshot: str,
    code: str,
    messages: tuple[str, str],
) -> None:
    """Rejects evidence taken from another source or source snapshot.

    The snapshot is read only when the identity matches.

    Args:
        source: The source the ledger consumes.
        identity: Source identity the evidence names.
        snapshot: Snapshot digest the evidence names.
        code: Failure code for evidence from another source.
        messages: Failure messages for another source and for another
            snapshot of the file.

    Raises:
        StreamingError: With `code` if `identity` is not the source's, or
            with code "source_snapshot_changed" if `snapshot` is not the
            digest of the file as it is now.
    """
    expected = source.identify()
    if identity != expected:
        raise failure.StreamingError(code, messages[0], expected, {})
    if snapshot != source.digest():
        raise failure.StreamingError(
            "source_snapshot_changed", messages[1], expected, {}
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Archive(sidecar.Sidecar):
    """Durable streaming state bound to one source and its raw snapshot.

    The archive reads and replaces the state and the staged acknowledgement,
    and checks that every batch, acknowledgement and receipt belongs to the
    source and snapshot it serves. `journal.Journal` adds the commits.

    Attributes:
        path: JSON file holding the durable state.
        source: The chunked source being consumed.
    """

    source: feed.Feed

    def load(self) -> transaction.Settlement:
        """Loads state, failing closed if the source snapshot changed.

        Returns:
            The durable state, or a fresh state at offset zero when the state
            file does not exist.

        Raises:
            StreamingError: If the state file is unreadable or malformed, or
                the state (including its transaction evidence) belongs to
                another source or another snapshot of the file.
        """
        snapshot = self.source.digest()
        identity = self.source.identify()
        if not self.path.exists():
            return transaction.Settlement(
                progress.Checkpoint(identity, {"snapshot": snapshot})
            )
        try:
            state = transaction.Settlement.from_dict(
                json.loads(self.path.read_text(encoding="utf-8"))
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise failure.StreamingError(
                "invalid_state",
                "durable streaming state is unreadable",
                identity,
                {"path": str(self.path)},
            ) from error
        if state.checkpoint.source != identity:
            raise failure.StreamingError(
                "checkpoint_source_mismatch",
                "durable streaming state belongs to another source",
                identity,
                {"state_source_identity": state.checkpoint.source},
            )
        if state.checkpoint.digests["snapshot"] != snapshot:
            raise failure.StreamingError(
                "source_snapshot_changed",
                "durable streaming state belongs to another source snapshot",
                identity,
                {
                    "checkpoint_snapshot_sha256": (
                        state.checkpoint.digests["snapshot"]
                    ),
                    "observed_snapshot_sha256": snapshot,
                },
            )
        for evidence in state.commitments:
            if (
                evidence.receipt.checkpoint.digests["snapshot"] != snapshot
                or evidence.acknowledgement.checkpoint.digests["snapshot"]
                != snapshot
            ):
                raise failure.StreamingError(
                    "source_snapshot_changed",
                    "transaction evidence belongs to another source snapshot",
                    identity,
                    {"batch_id": evidence.batch},
                )
        return state

    def read(
        self,
        rows: int = feed.BATCH_ROWS,
        watermark_column: str | None = None,
    ) -> progress.Batch:
        """Reads the next bounded batch from the durable checkpoint.

        Failures from `load` and from the source's `read` propagate
        unchanged.

        Args:
            rows: Maximum number of rows in the batch.
            watermark_column: Column whose greatest timestamp is tracked as the
                watermark.

        Returns:
            The next batch after the durable checkpoint.
        """
        state = self.load()
        return self.source.read(
            state.checkpoint, rows=rows, watermark_column=watermark_column
        )

    def write(self, state: transaction.Settlement) -> None:
        """Publishes state with durable temporary-file replacement.

        Args:
            state: The state to persist.

        Raises:
            StreamingError: If the state's checkpoint belongs to another source
                or another snapshot of the file.
        """
        confine(
            self.source,
            state.checkpoint.source,
            state.checkpoint.digests["snapshot"],
            "checkpoint_source_mismatch",
            (
                "state does not belong to this source",
                "state does not belong to the current source snapshot",
            ),
        )
        durability.write(
            self.path,
            (json.dumps(state.to_dict(), sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )

    def earmark(self, acknowledgement: transaction.Acknowledgement) -> None:
        """Persists an acknowledgement while the ledger lock is held.

        Args:
            acknowledgement: The acknowledgement to stage for delivery.

        Raises:
            StreamingError: If the acknowledgement belongs to another source or
                another snapshot of the file.
        """
        confine(
            self.source,
            acknowledgement.source,
            acknowledgement.checkpoint.digests["snapshot"],
            "acknowledgement_source_mismatch",
            (
                "acknowledgement belongs to another source",
                "acknowledgement belongs to another source snapshot",
            ),
        )
        durability.write(
            self.pending,
            (
                json.dumps(acknowledgement.to_dict(), sort_keys=True) + "\n"
            ).encode("utf-8"),
        )

    def reclaim(self) -> transaction.Acknowledgement | None:
        """Loads the pending acknowledgement, if one exists.

        Returns:
            The staged acknowledgement, or None if there is none.

        Raises:
            StreamingError: If the staged file is unreadable or malformed, or
                belongs to another source or another snapshot of the file.
        """
        if not self.pending.exists():
            return None
        try:
            acknowledgement = transaction.Acknowledgement.from_dict(
                json.loads(self.pending.read_text(encoding="utf-8"))
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise failure.StreamingError(
                "invalid_acknowledgement",
                "pending streaming acknowledgement is unreadable",
                self.source.identify(),
                {"path": str(self.pending)},
            ) from error
        confine(
            self.source,
            acknowledgement.source,
            acknowledgement.checkpoint.digests["snapshot"],
            "acknowledgement_source_mismatch",
            (
                "pending acknowledgement belongs to another source",
                "pending acknowledgement belongs to another source snapshot",
            ),
        )
        return acknowledgement

    def retract(self) -> None:
        """Removes a delivered acknowledgement and flushes the directory.

        Nothing happens if no acknowledgement is staged.
        """
        try:
            self.pending.unlink()
            durability.flush(self.pending.parent)
        except FileNotFoundError:
            # handled: documented no-op when no acknowledgement staged
            return

    def validate(self, batch: progress.Batch) -> None:
        """Validates a batch against this ledger's source snapshot.

        Args:
            batch: The batch to check.

        Raises:
            StreamingError: If the batch belongs to another source or another
                snapshot of the file.
        """
        confine(
            self.source,
            batch.identity,
            batch.snapshot,
            "batch_source_mismatch",
            (
                "batch belongs to another source",
                "batch belongs to another source snapshot",
            ),
        )

    def vouch(
        self,
        batch: progress.Batch,
        acknowledgement: transaction.Acknowledgement,
    ) -> None:
        """Validates the acknowledgement that will be bound to a batch.

        Args:
            batch: The batch being committed.
            acknowledgement: The acknowledgement to bind to it.

        Raises:
            StreamingError: If the acknowledgement's source, batch identity or
                checkpoint differs from the batch's.
        """
        identity = self.source.identify()
        if acknowledgement.source != batch.identity:
            raise failure.StreamingError(
                "acknowledgement_source_mismatch",
                "acknowledgement does not match the batch source",
                identity,
                {},
            )
        if acknowledgement.batch != batch.key:
            raise failure.StreamingError(
                "acknowledgement_batch_mismatch",
                "acknowledgement does not match the batch",
                identity,
                {},
            )
        if acknowledgement.checkpoint != batch.checkpoint:
            raise failure.StreamingError(
                "acknowledgement_checkpoint_mismatch",
                "acknowledgement does not match the batch checkpoint",
                identity,
                {},
            )

    def attest(
        self, batch: progress.Batch, receipt: transaction.Voucher
    ) -> None:
        """Validates a sink receipt before it enters durable state.

        Args:
            batch: The batch that was committed.
            receipt: The receipt returned by the sink.

        Raises:
            StreamingError: If the receipt's source, batch identity or
                checkpoint differs from the batch's.
        """
        identity = self.source.identify()
        if receipt.source != batch.identity:
            raise failure.StreamingError(
                "sink_receipt_source_mismatch",
                "sink receipt belongs to another source",
                identity,
                {"receipt_source_identity": receipt.source},
            )
        if receipt.batch != batch.key:
            raise failure.StreamingError(
                "sink_receipt_batch_mismatch",
                "sink receipt belongs to another batch",
                identity,
                {"receipt_batch_id": receipt.batch},
            )
        if receipt.checkpoint != batch.checkpoint:
            raise failure.StreamingError(
                "sink_receipt_checkpoint_mismatch",
                "sink receipt belongs to another checkpoint",
                identity,
                {},
            )
