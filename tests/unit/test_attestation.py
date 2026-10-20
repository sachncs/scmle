"""Tests for content-linked release attestation."""

import dataclasses

import pytest

from scml.governance import attestation


def records() -> tuple[dict[str, object], dict[str, object]]:
    """Return representative provenance and SBOM records."""
    return (
        {
            "run_id": "run-1",
            "model": {"manifest_sha256": "model-digest"},
            "signature": {"present": True},
        },
        {
            "bomFormat": "CycloneDX",
            "components": [{"name": "scikit-learn", "version": "1.6.0"}],
        },
    )


def test_attestation_binds_model_provenance_and_sbom_digests():
    provenance, sbom = records()

    record = attestation.ReleaseAttestation.from_records(provenance, sbom)

    assert record.verify(provenance, sbom)
    assert record.to_dict()["subjects"] == {
        "model_manifest_sha256": "model-digest",
        "provenance_sha256": attestation.canonical_digest(provenance),
        "sbom_sha256": attestation.canonical_digest(sbom),
    }


def test_attestation_fails_after_a_source_record_changes():
    provenance, sbom = records()
    record = attestation.ReleaseAttestation.from_records(provenance, sbom)
    changed = dataclasses.replace(
        record, signature_present=False
    )

    assert not changed.verify(provenance, sbom)
    sbom["components"] = []
    assert not record.verify(provenance, sbom)


def test_attestation_rejects_missing_or_ambiguous_signature_claim():
    provenance, sbom = records()
    del provenance["signature"]

    with pytest.raises(ValueError, match=r"signature\.present"):
        attestation.ReleaseAttestation.from_records(provenance, sbom)


def test_attestation_signature_binds_all_release_subjects():
    provenance, sbom = records()

    record = attestation.ReleaseAttestation.from_records(
        provenance, sbom, signing_key="release-key"
    )

    assert record.attestation_signature
    assert record.verify(provenance, sbom, signing_key="release-key")
    assert not record.verify(provenance, sbom, signing_key="wrong-key")
    assert attestation.ReleaseAttestation.from_dict(record.to_dict()) == record


def test_attestation_signature_rejects_tampered_persisted_record():
    provenance, sbom = records()
    record = attestation.ReleaseAttestation.from_records(
        provenance, sbom, signing_key="release-key"
    )
    persisted = record.to_dict()
    persisted["subjects"]["sbom_sha256"] = "tampered"

    loaded = attestation.ReleaseAttestation.from_dict(persisted)

    assert not loaded.verify(provenance, sbom, signing_key="release-key")
