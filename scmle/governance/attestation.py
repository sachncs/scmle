"""Content-linked release attestation for model, provenance, and SBOM."""

from collections.abc import Mapping
import dataclasses
import hashlib
import hmac
import json
from typing import Any
from typing import ClassVar
from typing import Self

from scmle.contracts import errors
from scmle.contracts import records
from scmle.contracts.signing import keying
from scmle.contracts.signing import mac
from scmle.contracts.signing import SIGNATURE_ALGORITHM


def checksum(record: Mapping[str, Any]) -> str:
    """Returns the SHA-256 digest of a JSON-native governance record.

    Args:
        record: The record; keys are sorted and the JSON is compact.

    Returns:
        The hex digest of the canonical JSON.
    """
    serialized = json.dumps(
        record,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def mandatory(record: Mapping[str, Any], path: str) -> str:
    """Returns a required string from a nested governance record.

    Args:
        record: The record to read.
        path: Dot-separated path of the field.

    Returns:
        The non-empty string at that path.

    Raises:
        ValueError: If the path is absent or its value is not non-empty
            text.
    """
    value: Any = record
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise errors.InputError(f"missing attestation field: {path}")
        value = value[component]
    if not isinstance(value, str) or not value.strip():
        raise errors.InputError(
            f"attestation field must be non-empty text: {path}"
        )
    return value


SUBJECTS = {
    "model_manifest_sha256": "manifest",
    "provenance_sha256": "provenance",
    "sbom_sha256": "sbom",
}
"""Version-1 subject keys and their names under `digests`."""


@dataclasses.dataclass(frozen=True, slots=True)
class Attestation:
    """Immutable digest binding for one packaged release candidate.

    An attestation keeps the version it was stored in: its signed content is
    that of the stored layout, so a version-1 attestation still verifies.

    Attributes:
        run: The run that packaged the candidate.
        digests: SHA-256 of the model `manifest`, the `provenance` record and
            the `sbom`.
        signed: Whether the provenance says the bundle is signed.
        signature: The HMAC over `preimage`, or None if unsigned.
        schema_version: The layout the attestation was stored (and signed)
            in.
        migrations: The attestation's version steps, oldest first.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            {
                "run_id": "run",
                "subjects": "digests",
                "signature_present": "signed",
            },
            parts={"digests": records.Migration(renames=SUBJECTS)},
        ),
    )

    run: str
    digests: Mapping[str, str]
    signed: bool
    signature: str | None = None
    schema_version: int = records.ORIGIN + len(migrations)

    def __post_init__(self) -> None:
        """Rejects incomplete identity or signature claims.

        Raises:
            TypeError: If `signed` is not a bool.
            ValueError: If the run or a digest is not non-blank text, a digest
                is missing, or the signature, when given, is not non-blank
                text.
        """
        values = {"run": self.run} | {
            f"digests.{name}": self.digests.get(name)
            for name in SUBJECTS.values()
        }
        for name, value in values.items():
            if not isinstance(value, str) or not value.strip():
                raise errors.InputError(f"{name} must not be empty")
        if not isinstance(self.signed, bool):
            raise errors.KindError("signed must be bool")
        signature: object = self.signature
        if signature is not None and (
            not isinstance(signature, str) or not signature.strip()
        ):
            raise errors.InputError("signature must be non-empty text")

    @classmethod
    def from_records(
        cls,
        provenance: Mapping[str, Any],
        sbom: Mapping[str, Any],
        signing_key: str | bytes | None = None,
    ) -> Self:
        """Builds an attestation from the records emitted by a package run.

        Args:
            provenance: The provenance record of the run.
            sbom: The software bill of materials of the run.
            signing_key: The key to sign with, or None for the configured
                default; the attestation is unsigned if no key exists.

        Returns:
            The attestation binding the run, its model manifest, provenance
            and SBOM digests, signed when a key is available.

        Raises:
            ValueError: If the provenance has no boolean signature presence or
                lacks a non-blank `run_id` or `model.manifest_sha256`.
        """
        signature = provenance.get("signature")
        if not isinstance(signature, Mapping) or not isinstance(
            signature.get("present"), bool
        ):
            raise errors.InputError("provenance.signature.present must be bool")
        record = cls(
            run=mandatory(provenance, "run_id"),
            digests={
                "manifest": mandatory(provenance, "model.manifest_sha256"),
                "provenance": checksum(provenance),
                "sbom": checksum(sbom),
            },
            signed=signature["present"],
        )
        return record.sign(signing_key)

    def sign(self, signing_key: str | bytes | None = None) -> Self:
        """Returns this attestation signed over its stored layout.

        Args:
            signing_key: The key to sign with, or None for the configured
                default.

        Returns:
            A signed copy, or this attestation unchanged if no key exists.
        """
        key = keying(signing_key)
        if key is None:
            return self
        return dataclasses.replace(self, signature=mac(self.preimage(), key))

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Parses a persisted release attestation without verifying it.

        The signature is carried over as stored; `verify` checks it. A
        version-1 attestation is migrated and remembers its version. A
        `signed` flag that is not a bool raises `TypeError` from the
        constructor, and a missing one reads as False.

        Args:
            record: The persisted attestation.

        Returns:
            The attestation.

        Raises:
            ValueError: If the version is unknown, the digests or signature
                are not objects, the signature algorithm is unsupported or its
                value is blank, or a required field is missing.
        """
        values = dict(record)
        stored = records.detect("attestation", values)
        if stored == records.ORIGIN and not isinstance(
            values.get("subjects"), Mapping
        ):
            raise errors.InputError("attestation.subjects must be an object")
        if stored > records.ORIGIN + len(cls.migrations):
            raise errors.InputError(f"unsupported attestation version {stored}")
        values = records.upgrade(cls.migrations, values, stored)
        digests = values.get("digests")
        if not isinstance(digests, Mapping):
            raise errors.InputError("attestation.digests must be an object")
        signature = values.get("signature")
        signature_value: str | None = None
        if signature is not None:
            if not isinstance(signature, Mapping):
                raise errors.InputError(
                    "attestation.signature must be an object"
                )
            if signature.get("algorithm") != SIGNATURE_ALGORITHM:
                raise errors.InputError(
                    "unsupported attestation signature algorithm"
                )
            value = signature.get("value")
            if not isinstance(value, str) or not value.strip():
                raise errors.InputError(
                    "attestation signature value must be non-empty"
                )
            signature_value = value
        return cls(
            run=mandatory(values, "run"),
            digests={
                name: mandatory(digests, name) for name in SUBJECTS.values()
            },
            signed=values.get("signed", False),
            signature=signature_value,
            schema_version=stored,
        )

    def preimage(self) -> dict[str, Any]:
        """Returns the canonical content the attestation signature covers.

        Returns:
            The content in the layout the attestation is stored in.
        """
        record: records.Layout = {
            "run": self.run,
            "digests": {name: self.digests[name] for name in SUBJECTS.values()},
            "signed": self.signed,
        }
        record = records.downgrade(self.migrations, record, self.schema_version)
        return {records.KEY: self.schema_version, **record}

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible attestation record."""
        record = self.preimage()
        if self.signature is not None:
            record["signature"] = {
                "algorithm": SIGNATURE_ALGORITHM,
                "value": self.signature,
            }
        return record

    def verify(
        self,
        provenance: Mapping[str, Any],
        sbom: Mapping[str, Any],
        signing_key: str | bytes | None = None,
    ) -> bool:
        """Verifies that the attestation still matches its source records.

        The records are attested again in this attestation's stored layout,
        so a version-1 attestation is checked against version-1 content.

        Args:
            provenance: The provenance record of the run.
            sbom: The software bill of materials of the run.
            signing_key: The key to verify with, or None for the configured
                default.

        Returns:
            True if the attestation matches records rebuilt from the inputs and
            its signature matches (or both are unsigned); False otherwise,
            including when the records are malformed.
        """
        try:
            fresh = type(self).from_records(provenance, sbom, signing_key)
        except (TypeError, ValueError):
            # handled: malformed records give False, documented
            return False
        expected = dataclasses.replace(
            fresh, signature=None, schema_version=self.schema_version
        )
        if fresh.signature is not None:
            expected = expected.sign(signing_key)
        if self.preimage() != expected.preimage():
            return False
        if self.signature is None:
            return expected.signature is None
        if expected.signature is None:
            return False
        return hmac.compare_digest(self.signature, expected.signature)
