"""Content-linked release attestation for model, provenance, and SBOM."""

from collections.abc import Mapping
import dataclasses
import hashlib
import json
from typing import Any
from typing import Self


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

    @classmethod
    def from_records(
        cls,
        provenance: Mapping[str, Any],
        sbom: Mapping[str, Any],
    ) -> Self:
        """Build an attestation from the records emitted by a package run."""
        signature = provenance.get("signature")
        if not isinstance(signature, Mapping) or not isinstance(
            signature.get("present"), bool
        ):
            raise ValueError("provenance.signature.present must be bool")
        return cls(
            run_id=required_text(provenance, "run_id"),
            model_manifest_sha256=required_text(
                provenance, "model.manifest_sha256"
            ),
            provenance_sha256=canonical_digest(provenance),
            sbom_sha256=canonical_digest(sbom),
            signature_present=signature["present"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible attestation record."""
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

    def verify(
        self,
        provenance: Mapping[str, Any],
        sbom: Mapping[str, Any],
    ) -> bool:
        """Verify that the attestation still matches its source records."""
        try:
            expected = type(self).from_records(provenance, sbom)
        except (TypeError, ValueError):
            return False
        return self == expected
