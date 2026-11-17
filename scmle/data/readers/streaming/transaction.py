"""Acknowledgements, sink receipts and the durable commit state."""

import abc
from collections.abc import Mapping
import dataclasses
from typing import Any
from typing import ClassVar
from typing import Self

from scmle.contracts import errors
from scmle.contracts import records
from scmle.data.readers.streaming import progress

STREAM_STATE_SCHEMA = 3
STREAM_ACKNOWLEDGEMENT_SCHEMA = 2
STREAM_TRANSACTION_SCHEMA = 2


def lift(layout: records.Layout) -> records.Layout:
    """Moves the checkpoint of a renamed version-1 record to version 2.

    Args:
        layout: The record after its own keys were renamed.

    Returns:
        The record with its checkpoint in version-2 form.
    """
    step = progress.Checkpoint.migrations[0].upgrade
    return progress.shift(layout, "checkpoint", step)


def lower(layout: records.Layout) -> records.Layout:
    """Undoes `lift`.

    Args:
        layout: A version-2 record.

    Returns:
        The record with its checkpoint in version-1 form.
    """
    step = progress.Checkpoint.migrations[0].downgrade
    return progress.shift(layout, "checkpoint", step)


@dataclasses.dataclass(frozen=True, slots=True)
class Acknowledgement:
    """Durable evidence awaiting delivery to an external transport.

    Attributes:
        source: Identity of the source.
        consumer: Consumer group the acknowledgement is for.
        batch: Identity of the acknowledged batch.
        checkpoint: Checkpoint reached once the batch is committed.
        migrations: The record's version steps, oldest first.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={
                "source_identity": "source",
                "consumer_group": "consumer",
                "batch_id": "batch",
            },
            change=lift,
            revert=lower,
        ),
    )

    source: str
    consumer: str
    batch: str
    checkpoint: progress.Checkpoint

    def __post_init__(self) -> None:
        """Validates acknowledgement identity and checkpoint binding.

        Raises:
            ValueError: If the source identity or consumer group is blank, the
                batch identity does not start with "batch_", or the checkpoint
                belongs to another source.
        """
        if not self.source.strip():
            raise errors.InputError("source must not be empty")
        if not self.consumer.strip():
            raise errors.InputError("consumer must not be empty")
        if not self.batch.startswith("batch_"):
            raise errors.InputError("batch must be a streaming batch identity")
        if self.checkpoint.source != self.source:
            raise errors.InputError(
                "acknowledgement and checkpoint sources differ"
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible acknowledgement evidence.

        Returns:
            A dictionary holding the schema version and the four fields.
        """
        return {
            "schema_version": STREAM_ACKNOWLEDGEMENT_SCHEMA,
            "source": self.source,
            "consumer": self.consumer,
            "batch": self.batch,
            "checkpoint": self.checkpoint.to_dict(),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a pending acknowledgement from durable JSON.

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
                "unsupported streaming acknowledgement schema"
            )
        layout = records.load("Acknowledgement", cls.migrations, record)
        return cls(
            source=str(layout["source"]),
            consumer=str(layout["consumer"]),
            batch=str(layout["batch"]),
            checkpoint=progress.Checkpoint.from_dict(layout["checkpoint"]),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Voucher:
    """A transport-neutral receipt bound to one streaming batch.

    A voucher carries no version of its own: the `Commitment` that holds it
    is versioned and applies `migrations` to it.

    Attributes:
        source: Identity of the source.
        batch: Identity of the committed batch.
        checkpoint: Checkpoint reached by the commit.
        receipt: Opaque, non-empty receipt text issued by the sink.
        migrations: The voucher's layout steps, applied by its holder.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={"source_identity": "source", "batch_id": "batch"},
            change=lift,
            revert=lower,
        ),
    )

    source: str
    batch: str
    checkpoint: progress.Checkpoint
    receipt: str

    def __post_init__(self) -> None:
        """Validates the receipt's basic identities and checkpoint source.

        Raises:
            ValueError: If the source identity or receipt text is blank, the
                batch identity does not start with "batch_", or the checkpoint
                belongs to another source.
        """
        if not self.source.strip():
            raise errors.InputError("source must not be empty")
        if not self.batch.startswith("batch_"):
            raise errors.InputError("batch must be a streaming batch identity")
        if not self.receipt.strip():
            raise errors.InputError("receipt must not be empty")
        if self.checkpoint.source != self.source:
            raise errors.InputError("receipt and checkpoint sources differ")

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible sink receipt evidence.

        Returns:
            A dictionary with the four receipt fields.
        """
        return {
            "source": self.source,
            "batch": self.batch,
            "checkpoint": self.checkpoint.to_dict(),
            "receipt": self.receipt,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a sink receipt from durable JSON.

        Args:
            record: Mapping in the form produced by `to_dict`.

        Returns:
            The receipt described by `record`.
        """
        return cls(
            source=str(record["source"]),
            batch=str(record["batch"]),
            checkpoint=progress.Checkpoint.from_dict(record["checkpoint"]),
            receipt=str(record["receipt"]),
        )


class Sink(abc.ABC):
    """Caller-owned sink contract for an idempotent streaming commit.

    A sink must commit or deduplicate the supplied ``batch_id`` and return a
    receipt containing the exact source, batch, and checkpoint identities. The
    ledger atomically records that receipt with checkpoint acknowledgement in
    its local JSON state, but this contract does not provide a broker
    transaction or distributed exactly-once delivery.
    """

    @abc.abstractmethod
    def commit(self, batch: progress.Batch) -> Voucher:
        """Commits one batch and returns its transport-owned receipt.

        Args:
            batch: The batch to commit; implementations deduplicate on
                `batch.batch_id`.

        Returns:
            A receipt carrying the batch's source, batch and checkpoint
            identities.
        """


@dataclasses.dataclass(frozen=True, slots=True)
class Commitment:
    """Durable binding of a sink receipt and checkpoint acknowledgement.

    Attributes:
        source: Identity of the source.
        batch: Identity of the committed batch.
        receipt: Receipt issued by the sink for the batch.
        acknowledgement: Acknowledgement bound to the batch; it carries its
            own schema version.
        migrations: The record's version steps, oldest first.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={
                "source_identity": "source",
                "batch_id": "batch",
                "sink_receipt": "receipt",
            },
            change=lambda layout: progress.shift(
                layout, "receipt", Voucher.migrations[0].upgrade
            ),
            revert=lambda layout: progress.shift(
                layout, "receipt", Voucher.migrations[0].downgrade
            ),
        ),
    )

    source: str
    batch: str
    receipt: Voucher
    acknowledgement: Acknowledgement

    def __post_init__(self) -> None:
        """Validates all identities in one durable transaction record.

        Raises:
            ValueError: If the receipt or acknowledgement differs from the
                evidence in source or batch, or their checkpoints differ.
        """
        if self.receipt.source != self.source:
            raise errors.InputError("evidence and sink receipt sources differ")
        if self.acknowledgement.source != self.source:
            raise errors.InputError(
                "evidence and acknowledgement sources differ"
            )
        if self.receipt.batch != self.batch:
            raise errors.InputError("evidence and sink receipt batches differ")
        if self.acknowledgement.batch != self.batch:
            raise errors.InputError(
                "evidence and acknowledgement batches differ"
            )
        if self.receipt.checkpoint != self.acknowledgement.checkpoint:
            raise errors.InputError(
                "sink receipt and acknowledgement checkpoints differ"
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible transaction evidence.

        Returns:
            A dictionary holding the schema version and the nested records.
        """
        return {
            "schema_version": STREAM_TRANSACTION_SCHEMA,
            "source": self.source,
            "batch": self.batch,
            "receipt": self.receipt.to_dict(),
            "acknowledgement": self.acknowledgement.to_dict(),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds transaction evidence from durable JSON.

        A version-1 record is migrated first; its acknowledgement carries
        its own version and migrates itself.

        Args:
            record: Mapping in the form produced by `to_dict`.

        Returns:
            The evidence described by `record`.

        Raises:
            ValueError: If the record's schema version is missing or newer
                than this build.
        """
        if not isinstance(record.get("schema_version"), int):
            raise errors.InputError("unsupported streaming transaction schema")
        layout = records.load("Commitment", cls.migrations, record)
        return cls(
            source=str(layout["source"]),
            batch=str(layout["batch"]),
            receipt=Voucher.from_dict(layout["receipt"]),
            acknowledgement=Acknowledgement.from_dict(
                layout["acknowledgement"]
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Settlement:
    """Durable idempotency and checkpoint state for one source snapshot.

    Attributes:
        checkpoint: Position after the last committed batch.
        committed: Identities of all committed batches, sorted and unique.
        commitments: Evidence for committed batches, sorted by batch
            identity; may cover only some committed batches. Each carries
            its own schema version.
        migrations: The record's version steps, oldest first. Version 2
            added the commitments (a version-1 state has none).
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(),
        records.Migration(
            renames={
                "committed_batch_ids": "committed",
                "transaction_evidence": "commitments",
            },
            change=lift,
            revert=lower,
        ),
    )

    checkpoint: progress.Checkpoint
    committed: tuple[str, ...] = ()
    commitments: tuple[Commitment, ...] = ()

    def __post_init__(self) -> None:
        """Validates deterministic committed batch identities.

        Raises:
            ValueError: If the committed identities or the evidence are not
                sorted and unique, or evidence names a batch that is not
                committed.
        """
        if tuple(sorted(set(self.committed))) != self.committed:
            raise errors.InputError("committed must be sorted and unique")
        evidence_ids = tuple(evidence.batch for evidence in self.commitments)
        if tuple(sorted(set(evidence_ids))) != evidence_ids:
            raise errors.InputError("commitments must be sorted by batch")
        if not set(evidence_ids).issubset(self.committed):
            raise errors.InputError(
                "transaction evidence must identify committed batches"
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible durable state.

        Returns:
            A dictionary holding the schema version, checkpoint, committed
            batch identities and transaction evidence.
        """
        return {
            "schema_version": STREAM_STATE_SCHEMA,
            "checkpoint": self.checkpoint.to_dict(),
            "committed": list(self.committed),
            "commitments": [
                evidence.to_dict() for evidence in self.commitments
            ],
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds state from durable JSON.

        Args:
            record: Mapping in the form produced by `to_dict`. A state stored
                at an older schema version is migrated first.

        Returns:
            The state described by `record`.

        Raises:
            ValueError: If the record's schema version is missing or newer
                than this build.
        """
        if not isinstance(record.get("schema_version"), int):
            raise errors.InputError("unsupported streaming state schema")
        layout = records.load("Settlement", cls.migrations, record)
        return cls(
            checkpoint=progress.Checkpoint.from_dict(layout["checkpoint"]),
            committed=tuple(str(value) for value in layout["committed"]),
            commitments=tuple(
                Commitment.from_dict(value)
                for value in layout.get("commitments", [])
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Receipt:
    """Evidence from one idempotent streaming commit attempt.

    Attributes:
        source_identity: Identity of the source.
        batch_id: Identity of the batch.
        accepted: True if this call committed the batch.
        duplicate: True if the batch had already been committed.
        checkpoint: Checkpoint after the batch.
        sink_receipt: Receipt issued by the sink, when one is involved.
        acknowledgement: Acknowledgement bound to the batch, when one is
            involved.
    """

    source_identity: str
    batch_id: str
    accepted: bool
    duplicate: bool
    checkpoint: progress.Checkpoint
    sink_receipt: Voucher | None = None
    acknowledgement: Acknowledgement | None = None

    def __post_init__(self) -> None:
        """Validates mutually exclusive commit classification.

        Raises:
            ValueError: If `accepted` and `duplicate` are equal, or the
                checkpoint, sink receipt or acknowledgement does not match the
                commit.
        """
        if self.accepted == self.duplicate:
            raise errors.InputError("a commit must be accepted or duplicate")
        if self.checkpoint.source != self.source_identity:
            raise errors.InputError("receipt and checkpoint sources differ")
        if self.sink_receipt is not None and (
            self.sink_receipt.source != self.source_identity
            or self.sink_receipt.batch != self.batch_id
            or self.sink_receipt.checkpoint != self.checkpoint
        ):
            raise errors.InputError("receipt does not match the commit")
        if self.acknowledgement is not None and (
            self.acknowledgement.source != self.source_identity
            or self.acknowledgement.batch != self.batch_id
            or self.acknowledgement.checkpoint != self.checkpoint
        ):
            raise errors.InputError("acknowledgement does not match the commit")
