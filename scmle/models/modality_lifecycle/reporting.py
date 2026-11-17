"""The signed modality lifecycle report and its persisted form."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import hmac
import json
from pathlib import Path
from typing import cast
from typing import Self

from scmle.contracts import errors
from scmle.contracts import modalities
from scmle.contracts import signing
from scmle.models.modality_lifecycle import constants
from scmle.models.modality_lifecycle import containers


def rehydrate(
    raw_stage_evidence: object,
) -> tuple[modalities.Proof, ...]:
    """Restores the per-stage evidence of a persisted lifecycle report.

    Args:
        raw_stage_evidence: The report's `stage_evidence` entry.

    Returns:
        The restored stage evidence, in order.

    Raises:
        ValueError: If the entry is not a list or an item lacks a valid
            stage, evidence links, evidence mapping or notes.
    """
    if not isinstance(raw_stage_evidence, list):
        raise errors.InputError(
            "lifecycle report stage_evidence must be a list"
        )
    stage_evidence: list[modalities.Proof] = []
    for item in raw_stage_evidence:
        if not isinstance(item, Mapping):
            raise errors.InputError(
                "lifecycle report stage evidence is invalid"
            )
        stage = item.get("stage")
        evidence_links = item.get("evidence_links")
        evidence = item.get("evidence")
        notes = item.get("notes")
        if not isinstance(stage, str):
            raise errors.InputError("lifecycle report stage is invalid")
        if not isinstance(evidence_links, list) or not all(
            isinstance(value, str) for value in evidence_links
        ):
            raise errors.InputError(
                "lifecycle report evidence_links are invalid"
            )
        if not isinstance(evidence, Mapping):
            raise errors.InputError("lifecycle report evidence is invalid")
        if not isinstance(notes, list) or not all(
            isinstance(value, str) for value in notes
        ):
            raise errors.InputError("lifecycle report notes are invalid")
        stage_evidence.append(
            modalities.Proof(
                stage=modalities.Phase(stage),
                evidence_links=tuple(evidence_links),
                evidence=evidence,
                notes=tuple(notes),
            )
        )
    return tuple(stage_evidence)


def unpack(raw_package: object) -> containers.Package:
    """Restores the package identity of a persisted lifecycle report.

    Args:
        raw_package: The report's `package` entry.

    Returns:
        The restored package identity.

    Raises:
        ValueError: If the entry is not a mapping, its path, digest or
            signature is not a string, or its schema version is not an
            integer.
    """
    if not isinstance(raw_package, Mapping):
        raise errors.InputError("lifecycle report package is invalid")
    package_values = {
        "path": raw_package.get("path"),
        "model_sha256": raw_package.get("model_sha256"),
        "signature": raw_package.get("signature"),
        "schema_version": raw_package.get("schema_version"),
    }
    if not all(
        isinstance(package_values[field_name], str)
        for field_name in ("path", "model_sha256", "signature")
    ):
        raise errors.InputError("lifecycle report package identity is invalid")
    if not isinstance(package_values["schema_version"], int):
        raise errors.InputError("lifecycle report package schema is invalid")
    return containers.Package(
        path=cast("str", package_values["path"]),
        model_sha256=cast("str", package_values["model_sha256"]),
        signature=cast("str", package_values["signature"]),
        schema_version=package_values["schema_version"],
    )


def unseal(raw_signature: object) -> str | None:
    """Restores the content signature of a persisted lifecycle report.

    Args:
        raw_signature: The report's `signature` entry.

    Returns:
        The signature value, or None for an unsigned report.

    Raises:
        ValueError: If the entry is neither None nor a mapping with the
            supported algorithm and a non-blank string value.
    """
    if raw_signature is None:
        return None
    if not isinstance(raw_signature, Mapping):
        raise errors.InputError("lifecycle report signature is invalid")
    if raw_signature.get("algorithm") != signing.SIGNATURE_ALGORITHM:
        raise errors.InputError("unsupported lifecycle report signature")
    signature_value = raw_signature.get("value")
    if not isinstance(signature_value, str) or not signature_value.strip():
        raise errors.InputError("lifecycle report signature is invalid")
    return signature_value


def require(value: object, message: str) -> list[str]:
    """Checks that a persisted value is a list of strings.

    Args:
        value: The persisted value.
        message: Error message for a value that is not a list of strings.

    Returns:
        The same list.

    Raises:
        ValueError: If `value` is not a list or holds a non-string.
    """
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise errors.InputError(message)
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class Record:
    """All evidence emitted by one completed modality lifecycle.

    Attributes:
        modality: Modality value the lifecycle ran for.
        dataset_hash: Hash identifying the dataset.
        stage_evidence: Evidence of every required lifecycle stage, in order.
        evaluation: Held-out evaluation evidence.
        performance: Resource and timing evidence.
        package: Identity of the written model package.
        inference: Inference smoke-test evidence.
        monitoring: Drift-monitoring evidence.
        documentation: Documentation evidence.
        recovery: Recovery and resource evidence.
        signature: SHA-256 signature of the report content, or None if unsigned.
    """

    modality: str
    dataset_hash: str
    stage_evidence: tuple[modalities.Proof, ...]
    evaluation: dict[str, object]
    performance: dict[str, object]
    package: containers.Package
    inference: dict[str, object]
    monitoring: dict[str, object]
    documentation: dict[str, object]
    recovery: dict[str, object]
    signature: str | None = None

    def __post_init__(self) -> None:
        """Validates the report identity and optional signature.

        The six evidence mappings (`evaluation`, `performance`, `inference`,
        `monitoring`, `documentation`, `recovery`) are also normalized to
        sorted JSON-native dictionaries; a value that is not a mapping raises
        TypeError and one that is not finite JSON raises ValueError, both from
        `normalise`.

        Raises:
            ValueError: If the modality or dataset hash is blank, the stage
                evidence is not every required stage once in order or lacks
                evidence links, or the signature is not a SHA-256 digest.
            TypeError: If the package is not a `Package` or the stage
                evidence holds other objects.
        """
        for field_name in ("modality", "dataset_hash"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise errors.InputError(f"{field_name} must not be empty")
        if not isinstance(self.package, containers.Package):
            raise errors.KindError("package must be Package")
        stages = tuple(self.stage_evidence)
        if any(
            not isinstance(evidence, modalities.Proof) for evidence in stages
        ):
            raise errors.KindError("stage_evidence must contain Proof records")
        expected_stages = modalities.REQUIRED_LIFECYCLE_STAGES
        actual_stages = tuple(evidence.stage for evidence in stages)
        if actual_stages != expected_stages:
            raise errors.InputError(
                "lifecycle report stage evidence must contain every required "
                "stage exactly once in order"
            )
        if any(not evidence.evidenced for evidence in stages):
            raise errors.InputError(
                "lifecycle report stages must contain evidence links"
            )
        object.__setattr__(self, "stage_evidence", stages)
        for field_name in (
            "evaluation",
            "performance",
            "inference",
            "monitoring",
            "documentation",
            "recovery",
        ):
            object.__setattr__(
                self,
                field_name,
                containers.normalise(getattr(self, field_name), field_name),
            )
        if self.signature is not None and not containers.hashlike(
            self.signature
        ):
            raise errors.InputError("signature must be a SHA-256 digest")

    def preimage(self) -> dict[str, object]:
        """Builds the exact report content covered by the signature.

        Returns:
            JSON-native mapping of every field except the signature.
        """
        return {
            "schema_version": 2,
            "modality": self.modality,
            "dataset_hash": self.dataset_hash,
            "stage_evidence": [
                evidence.to_dict() for evidence in self.stage_evidence
            ],
            "evaluation": self.evaluation,
            "performance": self.performance,
            "package": self.package.to_dict(),
            "inference": self.inference,
            "monitoring": self.monitoring,
            "documentation": self.documentation,
            "recovery": self.recovery,
        }

    def sign(self, signing_key: str | bytes) -> Self:
        """Signs the report over its exact serialized evidence.

        Args:
            signing_key: Key to sign with.

        Returns:
            A copy of this report carrying the signature.
        """
        return dataclasses.replace(
            self,
            signature=signing.mac(self.preimage(), signing_key),
        )

    def verify_signature(self, signing_key: str | bytes) -> bool:
        """Checks the report signature against its current content.

        Args:
            signing_key: Key the signature should have been made with.

        Returns:
            True if the report is signed and the signature matches; False if it
            is unsigned or does not match.
        """
        if self.signature is None:
            return False
        expected = signing.mac(self.preimage(), signing_key)
        return hmac.compare_digest(self.signature, expected)

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> Self:
        """Restores a versioned report without trusting its evidence claims.

        Args:
            record: Mapping produced by `to_dict`.

        Returns:
            The restored report.

        Raises:
            ValueError: If the schema version is unsupported or any field is
                missing or has the wrong shape.
        """
        if record.get("schema_version") != constants.REPORT_SCHEMA_VERSION:
            raise errors.InputError(
                "unsupported lifecycle report schema version"
            )

        stage_evidence = rehydrate(record.get("stage_evidence"))
        package = unpack(record.get("package"))

        def pluck(name: str) -> dict[str, object]:
            value = record.get(name)
            if not isinstance(value, Mapping):
                raise errors.InputError(
                    f"lifecycle report {name} must be an object"
                )
            return dict(value)

        signature = unseal(record.get("signature"))

        modality = record.get("modality")
        dataset_hash = record.get("dataset_hash")
        if not isinstance(modality, str) or not isinstance(dataset_hash, str):
            raise errors.InputError("lifecycle report identity is invalid")
        return cls(
            modality=modality,
            dataset_hash=dataset_hash,
            stage_evidence=stage_evidence,
            evaluation=pluck("evaluation"),
            performance=pluck("performance"),
            package=package,
            inference=pluck("inference"),
            monitoring=pluck("monitoring"),
            documentation=pluck("documentation"),
            recovery=pluck("recovery"),
            signature=signature,
        )

    @classmethod
    def read(cls, path: str | Path) -> Self:
        """Reads and parses a persisted lifecycle report.

        Args:
            path: JSON file written from `to_dict`.

        Returns:
            The parsed report (signature not yet verified).

        Raises:
            ValueError: If the file does not hold a JSON object.
        """
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(record, Mapping):
            raise errors.InputError("lifecycle report must be a JSON object")
        return cls.from_dict(record)

    @classmethod
    def read_verified(cls, path: str | Path, signing_key: str | bytes) -> Self:
        """Reads a report and rejects it unless its content signature verifies.

        Args:
            path: JSON file written from `to_dict`.
            signing_key: Key the signature should have been made with.

        Returns:
            The parsed report with a verified signature.

        Raises:
            ValueError: If the signature is missing or does not verify.
        """
        report = cls.read(path)
        if not report.verify_signature(signing_key):
            raise errors.InputError(
                "lifecycle report signature does not verify"
            )
        return report

    def to_dict(self) -> dict[str, object]:
        """Converts the report to a deterministic JSON-compatible record.

        Returns:
            The unsigned content plus a `signature` entry that is None if
            unsigned, otherwise the algorithm and value.
        """
        record = self.preimage()
        record["signature"] = (
            None
            if self.signature is None
            else {
                "algorithm": signing.SIGNATURE_ALGORITHM,
                "value": self.signature,
            }
        )
        return record
