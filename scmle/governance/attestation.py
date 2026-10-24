"""Content-linked release attestation for model, provenance, and SBOM."""

from collections.abc import Mapping
import dataclasses
import hashlib
import hmac
import json
from typing import Any
from typing import Self

from scmle.contracts.signing import resolve_key
from scmle.contracts.signing import sign_payload
from scmle.contracts.signing import SIGNATURE_ALGORITHM


def canonical_digest(record: Mapping[str, Any]) -> str:
    """Return the SHA-256 digest of a JSON-native governance record."""
    serialized = json.dumps(
        record,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def required_text(record: Mapping[str, Any], path: str) -> str:
    """Return a required string from a nested governance record."""
    value: Any = record
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"missing attestation field: {path}")
        value = value[component]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"attestation field must be non-empty text: {path}")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class ReleaseAttestation:
    """Immutable digest binding for one packaged release candidate."""

    run_id: str
    model_manifest_sha256: str
    provenance_sha256: str
    sbom_sha256: str
    signature_present: bool
    attestation_signature: str | None = None

    def __post_init__(self) -> None:
        """Reject incomplete identity or signature claims."""
        for field_name in (
            "run_id",
            "model_manifest_sha256",
            "provenance_sha256",
            "sbom_sha256",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        if not isinstance(self.signature_present, bool):
            raise TypeError("signature_present must be bool")
        if self.attestation_signature is not None and (
            not isinstance(self.attestation_signature, str)
            or not self.attestation_signature.strip()
        ):
            raise ValueError("attestation_signature must be non-empty text")

    @classmethod
    def from_records(
        cls,
        provenance: Mapping[str, Any],
        sbom: Mapping[str, Any],
        signing_key: str | bytes | None = None,
    ) -> Self:
        """Build an attestation from the records emitted by a package run."""
        signature = provenance.get("signature")
        if not isinstance(signature, Mapping) or not isinstance(
            signature.get("present"), bool
        ):
            raise ValueError("provenance.signature.present must be bool")
        record = cls(
            run_id=required_text(provenance, "run_id"),
            model_manifest_sha256=required_text(
                provenance, "model.manifest_sha256"
            ),
            provenance_sha256=canonical_digest(provenance),
            sbom_sha256=canonical_digest(sbom),
            signature_present=signature["present"],
        )
        key = resolve_key(signing_key)
        if key is None:
            return record
        return dataclasses.replace(
            record,
            attestation_signature=sign_payload(record.unsigned_dict(), key),
        )

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Parse a persisted release attestation without trusting its claims."""
        subjects = record.get("subjects")
        if not isinstance(subjects, Mapping):
            raise ValueError("attestation.subjects must be an object")
        signature = record.get("signature")
        signature_value: str | None = None
        if signature is not None:
            if not isinstance(signature, Mapping):
                raise ValueError("attestation.signature must be an object")
            if signature.get("algorithm") != SIGNATURE_ALGORITHM:
                raise ValueError("unsupported attestation signature algorithm")
            value = signature.get("value")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    "attestation signature value must be non-empty"
                )
            signature_value = value
        return cls(
            run_id=required_text(record, "run_id"),
            model_manifest_sha256=required_text(
                subjects, "model_manifest_sha256"
            ),
            provenance_sha256=required_text(subjects, "provenance_sha256"),
            sbom_sha256=required_text(subjects, "sbom_sha256"),
            signature_present=record.get("signature_present", False),
            attestation_signature=signature_value,
        )

    def unsigned_dict(self) -> dict[str, Any]:
        """Return the canonical content covered by the attestation signature."""
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "subjects": {
                "model_manifest_sha256": self.model_manifest_sha256,
                "provenance_sha256": self.provenance_sha256,
                "sbom_sha256": self.sbom_sha256,
            },
            "signature_present": self.signature_present,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible attestation record."""
        record = self.unsigned_dict()
        if self.attestation_signature is not None:
            record["signature"] = {
                "algorithm": SIGNATURE_ALGORITHM,
                "value": self.attestation_signature,
            }
        return record

    def verify(
        self,
        provenance: Mapping[str, Any],
        sbom: Mapping[str, Any],
        signing_key: str | bytes | None = None,
    ) -> bool:
        """Verify that the attestation still matches its source records."""
        try:
            expected = type(self).from_records(provenance, sbom, signing_key)
        except (TypeError, ValueError):
            return False
        if self.unsigned_dict() != expected.unsigned_dict():
            return False
        if self.attestation_signature is None:
            return expected.attestation_signature is None
        if expected.attestation_signature is None:
            return False
        return hmac.compare_digest(
            self.attestation_signature, expected.attestation_signature
        )
