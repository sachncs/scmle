"""The signed modality lifecycle report and its persisted form.

Version 3 of the report layout renamed `stage_evidence` to `evidence` and
nested the dataset hash under `digests`. A report is signed over the layout
of the version it is stored at, so a version-2 report still verifies after
it is loaded and is never re-signed.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import functools
import hmac
import json
from pathlib import Path
from typing import cast
from typing import Self

from scmle.contracts import errors
from scmle.contracts import modalities
from scmle.contracts import records
from scmle.contracts import signing
from scmle.evaluation import envelope
from scmle.pipeline.modality_lifecycle import constants
from scmle.pipeline.modality_lifecycle import containers

DATASET = {"digests": {"dataset_hash": "dataset"}}
MIGRATION = records.Migration(
    renames={"stage_evidence": "evidence"},
    change=functools.partial(envelope.fold, groups=DATASET),
    revert=functools.partial(envelope.unfold, groups=DATASET),
)
PREVIOUS = constants.REPORT_SCHEMA_VERSION - 1


def rehydrate(
    raw_stage_evidence: object,
) -> tuple[modalities.Proof, ...]:
    """Restores the per-stage evidence of a persisted lifecycle report.

    Args:
        raw_stage_evidence: The report's `evidence` entry; each stage
            record may carry its links as `links` or, as written before
            the key was renamed, `evidence_links`.

    Returns:
        The restored stage evidence, in order.

    Raises:
        ValueError: If the entry is not a list or an item lacks a valid
            stage, evidence links, evidence mapping or notes.
    """
    if not isinstance(raw_stage_evidence, list):
        raise errors.InputError("lifecycle report evidence must be a list")
    stage_evidence: list[modalities.Proof] = []
    for item in raw_stage_evidence:
        if not isinstance(item, Mapping):
            raise errors.InputError(
                "lifecycle report stage evidence is invalid"
            )
        layout = records.load("Proof", modalities.Proof.migrations, item)
        stage = layout.get("stage")
        evidence_links = layout.get("links")
        evidence = layout.get("evidence")
        notes = layout.get("notes")
        if not isinstance(stage, str):
            raise errors.InputError("lifecycle report stage is invalid")
        if not isinstance(evidence_links, list) or not all(
            isinstance(value, str) for value in evidence_links
        ):
            raise errors.InputError("lifecycle report links are invalid")
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
        raw_package: The report's `package` entry, with the written keys
            (`digests.model`) or the version-1 ones (`model_sha256`).

    Returns:
        The restored package identity.

    Raises:
        ValueError: If the entry is not a mapping, its path, digest or
            signature is not a string, or its schema version is not an
            integer.
    """
    if not isinstance(raw_package, Mapping):
        raise errors.InputError("lifecycle report package is invalid")
    if isinstance(raw_package.get("digests"), Mapping):
        raw_package = containers.Package.KEYS.downgrade(dict(raw_package))
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
        digests: The hash identifying the dataset, under `dataset`.
        evidence: Evidence of every required lifecycle stage, in order.
        evaluation: Held-out evaluation evidence.
        performance: Resource and timing evidence.
        package: Identity of the written model package.
        inference: Inference smoke-test evidence.
        monitoring: Drift-monitoring evidence.
        documentation: Documentation evidence.
        recovery: Recovery and resource evidence.
        signature: SHA-256 signature of the report content, or None if unsigned.
        schema_version: Version of the layout the report is stored and
            signed in.
    """

    modality: str
    digests: Mapping[str, str]
    evidence: tuple[modalities.Proof, ...]
    evaluation: dict[str, object]
    performance: dict[str, object]
    package: containers.Package
    inference: dict[str, object]
    monitoring: dict[str, object]
    documentation: dict[str, object]
    recovery: dict[str, object]
    signature: str | None = None
    schema_version: int = constants.REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validates the report identity and optional signature.

        The six evidence mappings (`evaluation`, `performance`, `inference`,
        `monitoring`, `documentation`, `recovery`) are also normalized to
        sorted JSON-native dictionaries; a value that is not a mapping raises
        TypeError and one that is not finite JSON raises ValueError, both from
        `normalise`.

        Raises:
            ValueError: If the modality or dataset hash is blank, the digests
                are not exactly the dataset's, the schema version is
                unknown, the stage evidence is not every required stage once
                in order or lacks evidence links, or the signature is not a
                SHA-256 digest.
            TypeError: If the package is not a `Package` or the stage
                evidence holds other objects.
        """
        if set(self.digests) != {"dataset"}:
            raise errors.InputError("digests must hold exactly the dataset")
        identity: tuple[tuple[str, object], ...] = (
            ("modality", self.modality),
            ("digests.dataset", self.digests["dataset"]),
        )
        for field_name, value in identity:
            if not isinstance(value, str) or not value.strip():
                raise errors.InputError(f"{field_name} must not be empty")
        if self.schema_version not in (
            PREVIOUS,
            constants.REPORT_SCHEMA_VERSION,
        ):
            raise errors.InputError(
                "unsupported lifecycle report schema version"
            )
        object.__setattr__(self, "digests", dict(self.digests))
        if not isinstance(self.package, containers.Package):
            raise errors.KindError("package must be Package")
        self.order()
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

    def order(self) -> None:
        """Checks the stage evidence and freezes it as a tuple.

        Raises:
            ValueError: If the evidence is not every required stage once in
                order or a stage lacks evidence links.
            TypeError: If the evidence holds objects other than `Proof`.
        """
        stages = tuple(self.evidence)
        if any(not isinstance(proof, modalities.Proof) for proof in stages):
            raise errors.KindError("evidence must contain Proof records")
        expected_stages = modalities.REQUIRED_LIFECYCLE_STAGES
        actual_stages = tuple(proof.stage for proof in stages)
        if actual_stages != expected_stages:
            raise errors.InputError(
                "lifecycle report stage evidence must contain every required "
                "stage exactly once in order"
            )
        if any(not proof.evidenced for proof in stages):
            raise errors.InputError(
                "lifecycle report stages must contain evidence links"
            )
        object.__setattr__(self, "evidence", stages)

    def preimage(self, anchored: bool = True) -> dict[str, object]:
        """Lays out every field except the signature as at `schema_version`.

        Args:
            anchored: Whether the stages and the package are taken at their
                version 1 (`evidence_links`, `model_sha256`), as the signature
                covers them, so a key rename never changes the signature of a
                stored report. A version-2 report always lays them out so.

        Returns:
            JSON-native mapping of the report content.
        """
        evidence = [proof.to_dict() for proof in self.evidence]
        package = self.package.to_dict()
        if anchored or self.schema_version == PREVIOUS:
            evidence = [
                records.downgrade(modalities.Proof.migrations, stage)
                for stage in evidence
            ]
            package = containers.Package.KEYS.downgrade(package)
        content: records.Layout = {
            "modality": self.modality,
            "digests": dict(self.digests),
            "evidence": evidence,
            "evaluation": self.evaluation,
            "performance": self.performance,
            "package": package,
            "inference": self.inference,
            "monitoring": self.monitoring,
            "documentation": self.documentation,
            "recovery": self.recovery,
        }
        if self.schema_version == PREVIOUS:
            content = MIGRATION.downgrade(content)
        return {records.KEY: self.schema_version, **content}

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

    def verify(self, signing_key: str | bytes) -> bool:
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

        A version-2 report is migrated to the current field names but keeps
        its stored version, so its signature is checked over the layout it
        was signed in.

        Args:
            record: Mapping produced by `to_dict`.

        Returns:
            The restored report.

        Raises:
            ValueError: If the schema version is unsupported or any field is
                missing or has the wrong shape.
        """
        version = record.get(records.KEY)
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version not in (PREVIOUS, constants.REPORT_SCHEMA_VERSION)
        ):
            raise errors.InputError(
                "unsupported lifecycle report schema version"
            )
        layout = dict(record)
        del layout[records.KEY]
        if version == PREVIOUS:
            layout = MIGRATION.upgrade(layout)
        evidence = rehydrate(layout.get("evidence"))
        package = unpack(layout.get("package"))

        def pluck(name: str) -> dict[str, object]:
            value = layout.get(name)
            if not isinstance(value, Mapping):
                raise errors.InputError(
                    f"lifecycle report {name} must be an object"
                )
            return dict(value)

        signature = unseal(layout.get("signature"))

        modality = layout.get("modality")
        digests = layout.get("digests")
        if (
            not isinstance(modality, str)
            or not isinstance(digests, Mapping)
            or not isinstance(digests.get("dataset"), str)
        ):
            raise errors.InputError("lifecycle report identity is invalid")
        return cls(
            modality=modality,
            digests=digests,
            evidence=evidence,
            evaluation=pluck("evaluation"),
            performance=pluck("performance"),
            package=package,
            inference=pluck("inference"),
            monitoring=pluck("monitoring"),
            documentation=pluck("documentation"),
            recovery=pluck("recovery"),
            signature=signature,
            schema_version=version,
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
    def load(cls, path: str | Path, signing_key: str | bytes) -> Self:
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
        if not report.verify(signing_key):
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
        record = self.preimage(anchored=False)
        record["signature"] = (
            None
            if self.signature is None
            else {
                "algorithm": signing.SIGNATURE_ALGORITHM,
                "value": self.signature,
            }
        )
        return record
