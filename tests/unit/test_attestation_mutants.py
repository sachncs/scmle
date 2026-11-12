"""Targeted tests for attestation behaviours the main suite left unpinned."""

import hashlib

import pytest

from scmle.contracts import signing
from scmle.governance import attestation
from tests.unit.test_attestation import records


def test_canonical_digest_is_compact_sorted_ascii_sha256():
    expected = hashlib.sha256(b'{"a":[1,2],"b":"caf\\u00e9"}').hexdigest()

    assert attestation.canonical_digest({"b": "café", "a": [1, 2]}) == (
        expected
    )


def test_required_text_rejects_a_blank_value_and_a_non_mapping_step():
    with pytest.raises(ValueError, match="non-empty text: a"):
        attestation.required_text({"a": "  "}, "a")
    with pytest.raises(ValueError, match="non-empty text: a"):
        attestation.required_text({"a": 5}, "a")
    with pytest.raises(ValueError, match=r"missing attestation field: a.b"):
        attestation.required_text({"a": "text"}, "a.b")
    with pytest.raises(ValueError, match="missing attestation field: x"):
        attestation.required_text({"y": "v"}, "x")
    assert attestation.required_text({"a": {"b": "ok"}}, "a.b") == "ok"


def build_record(**changes):
    fields = {
        "run_id": "r",
        "model_manifest_sha256": "m",
        "provenance_sha256": "p",
        "sbom_sha256": "s",
        "signature_present": True,
    }
    fields.update(changes)
    return attestation.ReleaseAttestation(**fields)


def test_identity_fields_must_be_non_blank_text():
    for name in (
        "run_id",
        "model_manifest_sha256",
        "provenance_sha256",
        "sbom_sha256",
    ):
        with pytest.raises(ValueError, match=f"{name} must not be empty"):
            build_record(**{name: "   "})
        with pytest.raises(ValueError, match=f"{name} must not be empty"):
            build_record(**{name: 7})


def test_the_signature_must_be_non_blank_text_when_given():
    with pytest.raises(ValueError, match="attestation_signature"):
        build_record(attestation_signature="  ")
    with pytest.raises(ValueError, match="attestation_signature"):
        build_record(attestation_signature=5)
    assert (
        build_record(attestation_signature="sig").attestation_signature == "sig"
    )
    assert build_record().attestation_signature is None


def test_a_non_bool_signature_claim_is_a_type_error():
    with pytest.raises(TypeError, match="signature_present must be bool"):
        build_record(signature_present=1)


def test_from_records_message_and_signature_cover_the_unsigned_record():
    provenance, sbom = records()
    provenance["signature"] = {"present": "yes"}
    with pytest.raises(
        ValueError, match=r"^provenance\.signature\.present must be bool$"
    ):
        attestation.ReleaseAttestation.from_records(provenance, sbom)

    provenance, sbom = records()
    record = attestation.ReleaseAttestation.from_records(
        provenance, sbom, signing_key="k"
    )
    assert record.attestation_signature == signing.sign_payload(
        record.unsigned_dict(), b"k"
    )


def test_a_missing_signature_present_reads_as_false_and_true_is_kept():
    base = build_record(signature_present=False).to_dict()
    del base["signature_present"]
    loaded = attestation.ReleaseAttestation.from_dict(base)
    assert loaded.signature_present is False
    assert loaded.attestation_signature is None
    base["signature_present"] = True
    assert (
        attestation.ReleaseAttestation.from_dict(base).signature_present is True
    )


def test_from_dict_rejects_blank_or_missing_signature_values():
    provenance, sbom = records()
    persisted = attestation.ReleaseAttestation.from_records(
        provenance, sbom, signing_key="k"
    ).to_dict()
    for bad in ("  ", 5, None):
        persisted["signature"] = {
            "algorithm": signing.SIGNATURE_ALGORITHM,
            "value": bad,
        }
        with pytest.raises(ValueError, match="value must be non-empty"):
            attestation.ReleaseAttestation.from_dict(persisted)
    persisted["signature"] = "text"
    with pytest.raises(ValueError, match="signature must be an object"):
        attestation.ReleaseAttestation.from_dict(persisted)
    persisted["signature"] = {"algorithm": "nope", "value": "v"}
    with pytest.raises(
        ValueError, match=r"^unsupported attestation signature algorithm$"
    ):
        attestation.ReleaseAttestation.from_dict(persisted)


def test_unsigned_dict_is_exactly_the_schema_one_record():
    assert build_record().unsigned_dict() == {
        "schema_version": 1,
        "run_id": "r",
        "subjects": {
            "model_manifest_sha256": "m",
            "provenance_sha256": "p",
            "sbom_sha256": "s",
        },
        "signature_present": True,
    }
