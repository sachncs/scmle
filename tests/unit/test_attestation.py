"""Tests for content-linked release attestation."""

import dataclasses

import pytest

from scmle.contracts import errors
from scmle.contracts import signing
from scmle.governance import attestation


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
    # Enforces website/src/pages/docs/operations.astro:60.
    provenance, sbom = records()

    record = attestation.Attestation.from_records(provenance, sbom)

    assert record.verify(provenance, sbom)
    assert record.to_dict()["digests"] == {
        "manifest": "model-digest",
        "provenance": attestation.checksum(provenance),
        "sbom": attestation.checksum(sbom),
    }


def test_attestation_fails_after_a_source_record_changes():
    provenance, sbom = records()
    record = attestation.Attestation.from_records(provenance, sbom)
    changed = dataclasses.replace(record, signed=False)

    assert not changed.verify(provenance, sbom)
    sbom["components"] = []
    assert not record.verify(provenance, sbom)


def test_attestation_rejects_missing_or_ambiguous_signature_claim():
    provenance, sbom = records()
    del provenance["signature"]

    with pytest.raises(ValueError, match=r"signature\.present"):
        attestation.Attestation.from_records(provenance, sbom)


def test_attestation_signature_binds_all_release_subjects():
    provenance, sbom = records()

    record = attestation.Attestation.from_records(
        provenance, sbom, signing_key="release-key"
    )

    assert record.signature
    assert record.verify(provenance, sbom, signing_key="release-key")
    assert not record.verify(provenance, sbom, signing_key="wrong-key")
    assert attestation.Attestation.from_dict(record.to_dict()) == record


def test_attestation_signature_rejects_tampered_persisted_record():
    # Enforces website/src/pages/docs/operations.astro:60.
    provenance, sbom = records()
    record = attestation.Attestation.from_records(
        provenance, sbom, signing_key="release-key"
    )
    persisted = record.to_dict()
    persisted["digests"]["sbom"] = "tampered"

    loaded = attestation.Attestation.from_dict(persisted)

    assert not loaded.verify(provenance, sbom, signing_key="release-key")


def test_an_empty_bytes_signing_key_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        signing.keying(b"")
    with pytest.raises(ValueError, match="empty"):
        signing.mac({"a": 1}, b"")
    # ERR-3: an empty string used to mean "no key" and left records unsigned.
    with pytest.raises(ValueError, match="empty"):
        signing.keying("")
    assert signing.keying(b"k") == b"k"


def test_an_unusable_configured_key_is_refused_and_named(tmp_path):
    # Enforces docs/SERVICE_SLO.md:123.
    # Enforces website/src/pages/docs/service.astro:83.
    with pytest.raises(errors.ConfigError, match="set but empty"):
        signing.lookup({signing.SIGNING_KEY_ENV: ""})
    with pytest.raises(errors.ConfigError, match="set but empty"):
        signing.lookup({signing.SIGNING_KEY_FILE_ENV: ""})
    missing = tmp_path / "absent.key"
    with pytest.raises(errors.ConfigError, match="cannot read"):
        signing.lookup({signing.SIGNING_KEY_FILE_ENV: str(missing)})
    empty = tmp_path / "empty.key"
    empty.write_text("  \n", encoding="utf-8")
    with pytest.raises(errors.ConfigError, match="is empty"):
        signing.lookup({signing.SIGNING_KEY_FILE_ENV: str(empty)})
    good = tmp_path / "good.key"
    good.write_text("secret\n", encoding="utf-8")
    assert signing.lookup({signing.SIGNING_KEY_FILE_ENV: str(good)}) == "secret"
    assert signing.lookup({}) is None
