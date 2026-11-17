"""Signed stage-boundary checkpoints of a modality lifecycle run."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import hmac
import json
from pathlib import Path
from typing import Literal
from typing import Self

from scmle.contracts import errors
from scmle.contracts import modalities
from scmle.contracts import signing
from scmle.pipeline.modality_lifecycle import reporting


@dataclasses.dataclass(frozen=True, slots=True)
class Checkpoint:
    """Signed durable boundary for one modality lifecycle replay.

    A checkpoint records only JSON-native stage identity and evidence digests.
    Resume reconstructs the in-memory context by replaying completed stages
    against the caller-provided adapter, then verifies that every replayed
    stage is byte-for-byte equivalent to the checkpoint. This avoids trusting
    a serialized estimator or arbitrary adapter object as recovery state.

    Attributes:
        modality: Modality value of the run.
        dataset_hash: Hash identifying the training dataset.
        observed_dataset_hash: Hash identifying the dataset used for
            monitoring.
        package_path: Where the model package is written.
        report_path: Where the lifecycle report is written.
        seed: Random seed of the run.
        completed_stages: Values of the lifecycle stages finished so far, a
            prefix of the required order.
        stage_digests: Digest of each completed stage's evidence, in the same
            order.
        require_independent_ground_truth: Whether the run demands independently
            authored explanation ground truth.
        manifest_digest: Digest of the package manifest once it exists, else
            None.
        status: "running" until every stage is done, then "completed".
        signature: Signature value over the content, or None if unsigned.
        schema_version: Version of the checkpoint layout.
    """

    modality: str
    dataset_hash: str
    observed_dataset_hash: str
    package_path: str
    report_path: str
    seed: int
    completed_stages: tuple[str, ...]
    stage_digests: tuple[str, ...]
    require_independent_ground_truth: bool
    manifest_digest: str | None = None
    status: Literal["running", "completed"] = "running"
    signature: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        """Rejects incomplete, non-prefix, or unsigned-identity state.

        Raises:
            ValueError: If an identity field is blank, the status is unknown, a
                digest or signature is blank, the completed stages are not an
                ordered prefix of the lifecycle, the digests do not match the
                stages, or a completed checkpoint lacks a stage.
            TypeError: If the seed is not an integer or the ground-truth
                requirement is not a boolean.
        """
        for field_name in (
            "modality",
            "dataset_hash",
            "observed_dataset_hash",
            "package_path",
            "report_path",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise errors.InputError(f"{field_name} must not be empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise errors.KindError("seed must be an integer")
        if not isinstance(self.require_independent_ground_truth, bool):
            raise errors.KindError(
                "require_independent_ground_truth must be bool"
            )
        if self.status not in ("running", "completed"):
            raise errors.InputError(
                "checkpoint status must be running or completed"
            )
        if self.manifest_digest is not None and not (
            self.manifest_digest.strip()
        ):
            raise errors.InputError(
                "manifest_digest must be non-empty when provided"
            )
        if self.signature is not None and not self.signature.strip():
            raise errors.InputError("signature must be non-empty when provided")
        self.sequence()

    def sequence(self) -> None:
        """Rejects stages that are not an ordered prefix of the lifecycle.

        Raises:
            ValueError: If the completed stages are not an ordered prefix of
                the lifecycle, the digests do not match the stages, a digest
                is blank, or a completed checkpoint lacks a stage.
        """
        expected_stages = tuple(
            stage.value for stage in modalities.REQUIRED_LIFECYCLE_STAGES
        )[: len(self.completed_stages)]
        if tuple(self.completed_stages) != expected_stages:
            raise errors.InputError(
                "completed stages must be an ordered lifecycle prefix"
            )
        if len(self.stage_digests) != len(self.completed_stages):
            raise errors.InputError("stage digests must match completed stages")
        if any(not digest.strip() for digest in self.stage_digests):
            raise errors.InputError("stage digests must not be empty")
        if self.status == "completed" and len(self.completed_stages) != len(
            modalities.REQUIRED_LIFECYCLE_STAGES
        ):
            raise errors.InputError(
                "completed checkpoint must contain every lifecycle stage"
            )

    def preimage(self) -> dict[str, object]:
        """Builds the exact checkpoint content covered by its signature.

        Returns:
            JSON-native mapping of every field except the signature, plus a
            fixed `record_type` entry.
        """
        return {
            "schema_version": self.schema_version,
            "record_type": "modality_lifecycle_checkpoint",
            "modality": self.modality,
            "dataset_hash": self.dataset_hash,
            "observed_dataset_hash": self.observed_dataset_hash,
            "package_path": self.package_path,
            "report_path": self.report_path,
            "seed": self.seed,
            "completed_stages": list(self.completed_stages),
            "stage_digests": list(self.stage_digests),
            "require_independent_ground_truth": (
                self.require_independent_ground_truth
            ),
            "manifest_digest": self.manifest_digest,
            "status": self.status,
        }

    def sign(self, signing_key: str | bytes) -> Self:
        """Signs the checkpoint over its exact serialized identity.

        Args:
            signing_key: Key to sign with.

        Returns:
            A copy of this checkpoint carrying the signature.
        """
        return dataclasses.replace(
            self,
            signature=signing.mac(self.preimage(), signing_key),
        )

    def verify_signature(self, signing_key: str | bytes) -> bool:
        """Checks the persisted checkpoint signature.

        Args:
            signing_key: Key the signature should have been made with.

        Returns:
            True if the checkpoint is signed and the signature matches; False if
            it is unsigned or does not match.
        """
        if self.signature is None:
            return False
        expected = signing.mac(self.preimage(), signing_key)
        return hmac.compare_digest(self.signature, expected)

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> Self:
        """Restores a checkpoint without trusting its serialized claims.

        Args:
            record: Mapping produced by `to_dict`.

        Returns:
            The restored checkpoint.

        Raises:
            ValueError: If the schema version, record type, signature or any
                field is unsupported, missing or malformed.
        """
        if record.get("schema_version") != 1:
            raise errors.InputError(
                "unsupported lifecycle checkpoint schema version"
            )
        if record.get("record_type") != "modality_lifecycle_checkpoint":
            raise errors.InputError("invalid lifecycle checkpoint record type")
        completed = reporting.require(
            record.get("completed_stages"),
            "checkpoint completed_stages must be a list",
        )
        digests = reporting.require(
            record.get("stage_digests"),
            "checkpoint stage_digests must be a list",
        )
        seed = record.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise errors.InputError("checkpoint seed is invalid")
        require_ground_truth = record.get("require_independent_ground_truth")
        if not isinstance(require_ground_truth, bool):
            raise errors.InputError(
                "checkpoint ground-truth requirement is invalid"
            )
        status = record.get("status")
        if status not in ("running", "completed"):
            raise errors.InputError("checkpoint status is invalid")
        signature = record.get("signature")
        if not isinstance(signature, Mapping):
            raise errors.InputError("checkpoint signature is missing")
        if signature.get("algorithm") != signing.SIGNATURE_ALGORITHM:
            raise errors.InputError(
                "unsupported lifecycle checkpoint signature"
            )
        signature_value = signature.get("value")
        if not isinstance(signature_value, str) or not signature_value.strip():
            raise errors.InputError("lifecycle checkpoint signature is invalid")
        manifest_digest = record.get("manifest_digest")
        if manifest_digest is not None and not isinstance(manifest_digest, str):
            raise errors.InputError("checkpoint manifest_digest is invalid")
        return cls(
            modality=str(record.get("modality", "")),
            dataset_hash=str(record.get("dataset_hash", "")),
            observed_dataset_hash=str(record.get("observed_dataset_hash", "")),
            package_path=str(record.get("package_path", "")),
            report_path=str(record.get("report_path", "")),
            seed=seed,
            completed_stages=tuple(completed),
            stage_digests=tuple(digests),
            require_independent_ground_truth=require_ground_truth,
            manifest_digest=manifest_digest,
            status=status,
            signature=signature_value,
            schema_version=1,
        )

    def to_dict(self) -> dict[str, object]:
        """Converts the checkpoint to deterministic JSON-native evidence.

        Returns:
            The unsigned content plus a `signature` entry holding the algorithm
            and value.
        """
        record = self.preimage()
        record["signature"] = {
            "algorithm": signing.SIGNATURE_ALGORITHM,
            "value": self.signature,
        }
        return record

    @classmethod
    def read(cls, path: str | Path) -> Self:
        """Reads a persisted lifecycle checkpoint.

        Args:
            path: JSON file written from `to_dict`.

        Returns:
            The parsed checkpoint (signature not yet verified).

        Raises:
            ValueError: If the file does not hold a JSON object.
        """
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(record, Mapping):
            raise errors.InputError(
                "lifecycle checkpoint must be a JSON object"
            )
        return cls.from_dict(record)

    @classmethod
    def read_verified(cls, path: str | Path, signing_key: str | bytes) -> Self:
        """Reads a checkpoint and rejects it unless its signature verifies.

        Args:
            path: JSON file written from `to_dict`.
            signing_key: Key the signature should have been made with.

        Returns:
            The parsed checkpoint with a verified signature.

        Raises:
            ValueError: If the signature does not verify.
        """
        checkpoint = cls.read(path)
        if not checkpoint.verify_signature(signing_key):
            raise errors.InputError(
                "lifecycle checkpoint signature does not verify"
            )
        return checkpoint
