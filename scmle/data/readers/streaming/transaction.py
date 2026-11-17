"""Acknowledgements, sink receipts and the durable commit state."""

import abc
from collections.abc import Mapping
import dataclasses
from typing import Any
from typing import Self

from scmle.contracts import errors
from scmle.data.readers.streaming import progress

STREAM_STATE_SCHEMA = 2
STREAM_ACKNOWLEDGEMENT_SCHEMA = 1
STREAM_TRANSACTION_SCHEMA = 1


@dataclasses.dataclass(frozen=True, slots=True)
class Acknowledgement:
    """Durable evidence awaiting delivery to an external transport.

    Attributes:
        source_identity: Identity of the source.
        consumer_group: Consumer group the acknowledgement is for.
        batch_id: Identity of the acknowledged batch.
        checkpoint: Checkpoint reached once the batch is committed.
    """

    source_identity: str
    consumer_group: str
    batch_id: str
    checkpoint: progress.Checkpoint

    def __post_init__(self) -> None:
        """Validates acknowledgement identity and checkpoint binding.

        Raises:
            ValueError: If the source identity or consumer group is blank, the
                batch identity does not start with "batch_", or the checkpoint
                belongs to another source.
        """
        if not self.source_identity.strip():
            raise errors.InputError("source_identity must not be empty")
        if not self.consumer_group.strip():
            raise errors.InputError("consumer_group must not be empty")
        if not self.batch_id.startswith("batch_"):
            raise errors.InputError(
                "batch_id must be a streaming batch identity"
            )
        if self.checkpoint.source_identity != self.source_identity:
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
            "source_identity": self.source_identity,
            "consumer_group": self.consumer_group,
            "batch_id": self.batch_id,
            "checkpoint": self.checkpoint.to_dict(),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a pending acknowledgement from durable JSON.

        Args:
            record: Mapping in the form produced by `to_dict`.

        Returns:
            The acknowledgement described by `record`.

        Raises:
            ValueError: If the record's schema version is not supported.
        """
        if record.get("schema_version") != STREAM_ACKNOWLEDGEMENT_SCHEMA:
            raise errors.InputError(
                "unsupported streaming acknowledgement schema"
            )
        return cls(
            source_identity=str(record["source_identity"]),
            consumer_group=str(record["consumer_group"]),
            batch_id=str(record["batch_id"]),
            checkpoint=progress.Checkpoint.from_dict(record["checkpoint"]),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Voucher:
    """A transport-neutral receipt bound to one streaming batch.

    Attributes:
        source_identity: Identity of the source.
        batch_id: Identity of the committed batch.
        checkpoint: Checkpoint reached by the commit.
        receipt: Opaque, non-empty receipt text issued by the sink.
    """

    source_identity: str
    batch_id: str
    checkpoint: progress.Checkpoint
    receipt: str

    def __post_init__(self) -> None:
        """Validates the receipt's basic identities and checkpoint source.

        Raises:
            ValueError: If the source identity or receipt text is blank, the
                batch identity does not start with "batch_", or the checkpoint
                belongs to another source.
        """
        if not self.source_identity.strip():
            raise errors.InputError("source_identity must not be empty")
        if not self.batch_id.startswith("batch_"):
            raise errors.InputError(
                "batch_id must be a streaming batch identity"
            )
        if not self.receipt.strip():
            raise errors.InputError("receipt must not be empty")
        if self.checkpoint.source_identity != self.source_identity:
            raise errors.InputError("receipt and checkpoint sources differ")

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible sink receipt evidence.

        Returns:
            A dictionary with the four receipt fields.
        """
        return {
            "source_identity": self.source_identity,
            "batch_id": self.batch_id,
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
            source_identity=str(record["source_identity"]),
            batch_id=str(record["batch_id"]),
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
        source_identity: Identity of the source.
        batch_id: Identity of the committed batch.
        sink_receipt: Receipt issued by the sink for the batch.
        acknowledgement: Acknowledgement bound to the batch.
    """

    source_identity: str
    batch_id: str
    sink_receipt: Voucher
    acknowledgement: Acknowledgement

    def __post_init__(self) -> None:
        """Validates all identities in one durable transaction record.

        Raises:
            ValueError: If the receipt or acknowledgement differs from the
                evidence in source or batch, or their checkpoints differ.
        """
        if self.sink_receipt.source_identity != self.source_identity:
            raise errors.InputError("evidence and sink receipt sources differ")
        if self.acknowledgement.source_identity != self.source_identity:
            raise errors.InputError(
                "evidence and acknowledgement sources differ"
            )
        if self.sink_receipt.batch_id != self.batch_id:
            raise errors.InputError("evidence and sink receipt batches differ")
        if self.acknowledgement.batch_id != self.batch_id:
            raise errors.InputError(
                "evidence and acknowledgement batches differ"
            )
        if self.sink_receipt.checkpoint != self.acknowledgement.checkpoint:
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
            "source_identity": self.source_identity,
            "batch_id": self.batch_id,
            "sink_receipt": self.sink_receipt.to_dict(),
            "acknowledgement": self.acknowledgement.to_dict(),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds transaction evidence from durable JSON.

        Args:
            record: Mapping in the form produced by `to_dict`.

        Returns:
            The evidence described by `record`.

        Raises:
            ValueError: If the record's schema version is not supported.
        """
        if record.get("schema_version") != STREAM_TRANSACTION_SCHEMA:
            raise errors.InputError("unsupported streaming transaction schema")
        return cls(
            source_identity=str(record["source_identity"]),
            batch_id=str(record["batch_id"]),
            sink_receipt=Voucher.from_dict(record["sink_receipt"]),
            acknowledgement=Acknowledgement.from_dict(
                record["acknowledgement"]
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Settlement:
    """Durable idempotency and checkpoint state for one source snapshot.

    Attributes:
        checkpoint: Position after the last committed batch.
        committed_batch_ids: Identities of all committed batches, sorted
            and unique.
        transaction_evidence: Evidence for committed batches, sorted by
            batch identity; may cover only some committed batches.
    """

    checkpoint: progress.Checkpoint
    committed_batch_ids: tuple[str, ...] = ()
    transaction_evidence: tuple[Commitment, ...] = ()

    def __post_init__(self) -> None:
        """Validates deterministic committed batch identities.

        Raises:
            ValueError: If the committed identities or the evidence are not
                sorted and unique, or evidence names a batch that is not
                committed.
        """
        if tuple(sorted(set(self.committed_batch_ids))) != (
            self.committed_batch_ids
        ):
            raise errors.InputError(
                "committed_batch_ids must be sorted and unique"
            )
        evidence_ids = tuple(
            evidence.batch_id for evidence in self.transaction_evidence
        )
        if tuple(sorted(set(evidence_ids))) != evidence_ids:
            raise errors.InputError(
                "transaction_evidence must be sorted by batch_id"
            )
        if not set(evidence_ids).issubset(self.committed_batch_ids):
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
            "committed_batch_ids": list(self.committed_batch_ids),
            "transaction_evidence": [
                evidence.to_dict() for evidence in self.transaction_evidence
            ],
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds state from durable JSON.

        Args:
            record: Mapping in the form produced by `to_dict`. Schema
                versions 1 and 2 are accepted.

        Returns:
            The state described by `record`.

        Raises:
            ValueError: If the record's schema version is not supported.
        """
        if record.get("schema_version") not in (1, STREAM_STATE_SCHEMA):
            raise errors.InputError("unsupported streaming state schema")
        return cls(
            checkpoint=progress.Checkpoint.from_dict(record["checkpoint"]),
            committed_batch_ids=tuple(
                str(value) for value in record["committed_batch_ids"]
            ),
            transaction_evidence=tuple(
                Commitment.from_dict(value)
                for value in record.get("transaction_evidence", [])
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
        if self.checkpoint.source_identity != self.source_identity:
            raise errors.InputError("receipt and checkpoint sources differ")
        if self.sink_receipt is not None and (
            self.sink_receipt.source_identity != self.source_identity
            or self.sink_receipt.batch_id != self.batch_id
            or self.sink_receipt.checkpoint != self.checkpoint
        ):
            raise errors.InputError("receipt does not match the commit")
        if self.acknowledgement is not None and (
            self.acknowledgement.source_identity != self.source_identity
            or self.acknowledgement.batch_id != self.batch_id
            or self.acknowledgement.checkpoint != self.checkpoint
        ):
            raise errors.InputError("acknowledgement does not match the commit")
