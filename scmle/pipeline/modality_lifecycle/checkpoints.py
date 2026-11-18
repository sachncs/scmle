"""Signed stage-boundary checkpoints of a modality lifecycle run.

Version 2 of the checkpoint layout renamed its keys to single words and
nested the four digests under `digests`. A checkpoint is signed over the
layout of the version it is stored at, so a version-1 checkpoint still
verifies after it is loaded: its content is laid out again as version 1
before the signature is checked, and it is never re-signed.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import functools
import hmac
import json
from pathlib import Path
from typing import Literal
from typing import Self

from scmle.contracts import errors
from scmle.contracts import modalities
from scmle.contracts import records
from scmle.contracts import signing
from scmle.evaluation import envelope
from scmle.pipeline.modality_lifecycle import reporting

NESTED = {
    "digests": {
        "dataset_hash": "dataset",
        "observed_dataset_hash": "observed",
        "stage_digests": "stages",
        "manifest_digest": "manifest",
    }
}
MIGRATION = records.Migration(
    renames={
        "package_path": "package",
        "report_path": "report",
        "completed_stages": "completed",
        "require_independent_ground_truth": "independent",
    },
    change=functools.partial(envelope.fold, groups=NESTED),
    revert=functools.partial(envelope.unfold, groups=NESTED),
)
VERSION = records.ORIGIN + 1


@dataclasses.dataclass(frozen=True, slots=True)
class Digests:
    """The digests a checkpoint binds.

    Attributes:
        dataset: Hash identifying the training dataset.
        observed: Hash identifying the dataset used for monitoring.
        stages: Digest of each completed stage's evidence, in stage order.
        manifest: Digest of the package manifest once it exists, else None.
    """

    dataset: str
    observed: str
    stages: tuple[str, ...]
    manifest: str | None = None

    @classmethod
    def restore(cls, record: object) -> Self:
        """Restores stored digests; blank values are rejected by `Checkpoint`.

        Args:
            record: The checkpoint's `digests` entry.

        Returns:
            The digests.

        Raises:
            ValueError: If the entry is not an object, its stages are not a
                list of strings, or its manifest is neither None nor text.
        """
        if not isinstance(record, Mapping):
            raise errors.InputError("checkpoint digests must be an object")
        stages = reporting.require(
            record.get("stages"), "checkpoint digests.stages must be a list"
        )
        manifest = record.get("manifest")
        if manifest is not None and not isinstance(manifest, str):
            raise errors.InputError("checkpoint digests.manifest is invalid")
        return cls(
            dataset=str(record.get("dataset", "")),
            observed=str(record.get("observed", "")),
            stages=tuple(stages),
            manifest=manifest,
        )


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
        digests: The dataset, monitoring dataset, stage and manifest
            digests.
        package: Where the model package is written.
        report: Where the lifecycle report is written.
        seed: Random seed of the run.
        completed: Values of the lifecycle stages finished so far, a prefix
            of the required order.
        independent: Whether the run demands independently authored
            explanation ground truth.
        status: "running" until every stage is done, then "completed".
        signature: Signature value over the content, or None if unsigned.
        schema_version: Version of the layout the checkpoint is stored and
            signed in.
    """

    modality: str
    digests: Digests
    package: str
    report: str
    seed: int
    completed: tuple[str, ...]
    independent: bool
    status: Literal["running", "completed"] = "running"
    signature: str | None = None
    schema_version: int = VERSION

    def __post_init__(self) -> None:
        """Rejects incomplete, non-prefix, or unsigned-identity state.

        Raises:
            ValueError: If an identity field is blank, the status or schema
                version is unknown, a digest or signature is blank, the
                completed stages are not an ordered prefix of the lifecycle,
                the digests do not match the stages, or a completed
                checkpoint lacks a stage.
            TypeError: If the digests are not `Digests`, the seed is not an
                integer or the ground-truth requirement is not a boolean.
        """
        if not isinstance(self.digests, Digests):
            raise errors.KindError("digests must be Digests")
        identity: tuple[tuple[str, object], ...] = (
            ("modality", self.modality),
            ("digests.dataset", self.digests.dataset),
            ("digests.observed", self.digests.observed),
            ("package", self.package),
            ("report", self.report),
        )
        for field_name, value in identity:
            if not isinstance(value, str) or not value.strip():
                raise errors.InputError(f"{field_name} must not be empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise errors.KindError("seed must be an integer")
        if not isinstance(self.independent, bool):
            raise errors.KindError("independent must be bool")
        if self.status not in ("running", "completed"):
            raise errors.InputError(
                "checkpoint status must be running or completed"
            )
        if self.schema_version not in (records.ORIGIN, VERSION):
            raise errors.InputError(
                "unsupported lifecycle checkpoint schema version"
            )
        manifest = self.digests.manifest
        if manifest is not None and not manifest.strip():
            raise errors.InputError(
                "digests.manifest must be non-empty when provided"
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
        )[: len(self.completed)]
        if tuple(self.completed) != expected_stages:
            raise errors.InputError(
                "completed stages must be an ordered lifecycle prefix"
            )
        if len(self.digests.stages) != len(self.completed):
            raise errors.InputError("stage digests must match completed stages")
        if any(not digest.strip() for digest in self.digests.stages):
            raise errors.InputError("stage digests must not be empty")
        if self.status == "completed" and len(self.completed) != len(
            modalities.REQUIRED_LIFECYCLE_STAGES
        ):
            raise errors.InputError(
                "completed checkpoint must contain every lifecycle stage"
            )

    def preimage(self) -> dict[str, object]:
        """Builds the exact checkpoint content covered by its signature.

        Returns:
            JSON-native mapping of every field except the signature, plus a
            fixed `record_type` entry, laid out as at `schema_version`.
        """
        content: records.Layout = {
            "record_type": "modality_lifecycle_checkpoint",
            "modality": self.modality,
            "digests": {
                **dataclasses.asdict(self.digests),
                "stages": list(self.digests.stages),
            },
            "package": self.package,
            "report": self.report,
            "seed": self.seed,
            "completed": list(self.completed),
            "independent": self.independent,
            "status": self.status,
        }
        if self.schema_version == records.ORIGIN:
            content = MIGRATION.downgrade(content)
        return {records.KEY: self.schema_version, **content}

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
        """Restores a checkpoint of either version without trusting its claims.

        A version-1 record is migrated to the current field names but keeps
        its stored version, so its signature is checked over the layout it
        was signed in.

        Args:
            record: Mapping produced by `to_dict`.

        Returns:
            The restored checkpoint.

        Raises:
            ValueError: If the schema version, record type, signature or any
                field is unsupported, missing or malformed.
        """
        version = record.get(records.KEY)
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version not in (records.ORIGIN, VERSION)
        ):
            raise errors.InputError(
                "unsupported lifecycle checkpoint schema version"
            )
        layout = dict(record)
        del layout[records.KEY]
        if version == records.ORIGIN:
            layout = MIGRATION.upgrade(layout)
        if layout.get("record_type") != "modality_lifecycle_checkpoint":
            raise errors.InputError("invalid lifecycle checkpoint record type")
        completed = reporting.require(
            layout.get("completed"), "checkpoint completed must be a list"
        )
        digests = Digests.restore(layout.get("digests"))
        seed = layout.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise errors.InputError("checkpoint seed is invalid")
        independent = layout.get("independent")
        if not isinstance(independent, bool):
            raise errors.InputError(
                "checkpoint ground-truth requirement is invalid"
            )
        status = layout.get("status")
        if status not in ("running", "completed"):
            raise errors.InputError("checkpoint status is invalid")
        signature = layout.get("signature")
        if not isinstance(signature, Mapping):
            raise errors.InputError("checkpoint signature is missing")
        if signature.get("algorithm") != signing.SIGNATURE_ALGORITHM:
            raise errors.InputError(
                "unsupported lifecycle checkpoint signature"
            )
        signature_value = signature.get("value")
        if not isinstance(signature_value, str) or not signature_value.strip():
            raise errors.InputError("lifecycle checkpoint signature is invalid")
        return cls(
            modality=str(layout.get("modality", "")),
            digests=digests,
            package=str(layout.get("package", "")),
            report=str(layout.get("report", "")),
            seed=seed,
            completed=tuple(completed),
            independent=independent,
            status=status,
            signature=signature_value,
            schema_version=version,
        )

    def to_dict(self) -> dict[str, object]:
        """Converts the checkpoint to deterministic JSON-native evidence.

        Returns:
            The unsigned content, laid out as at `schema_version`, plus a
            `signature` entry holding the algorithm and value.
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
