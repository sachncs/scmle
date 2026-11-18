"""Targeted tests for attestation behaviours the main suite left unpinned."""

import hashlib

import pytest

from scmle.contracts import signing
from scmle.governance import attestation
from tests.unit.test_attestation import records


def test_canonical_digest_is_compact_sorted_ascii_sha256():
    expected = hashlib.sha256(b'{"a":[1,2],"b":"caf\\u00e9"}').hexdigest()

    assert attestation.checksum({"b": "café", "a": [1, 2]}) == (expected)


def test_required_text_rejects_a_blank_value_and_a_non_mapping_step():
    with pytest.raises(ValueError, match="non-empty text: a"):
        attestation.mandatory({"a": "  "}, "a")
    with pytest.raises(ValueError, match="non-empty text: a"):
        attestation.mandatory({"a": 5}, "a")
    with pytest.raises(ValueError, match=r"missing attestation field: a.b"):
        attestation.mandatory({"a": "text"}, "a.b")
    with pytest.raises(ValueError, match="missing attestation field: x"):
        attestation.mandatory({"y": "v"}, "x")
    assert attestation.mandatory({"a": {"b": "ok"}}, "a.b") == "ok"


DIGESTS = {"manifest": "m", "provenance": "p", "sbom": "s"}


def build_record(**changes):
    fields = {"run": "r", "digests": DIGESTS, "signed": True}
    fields.update(changes)
    return attestation.Attestation(**fields)


def test_identity_fields_must_be_non_blank_text():
    for bad in ("   ", 7):
        with pytest.raises(ValueError, match="run must not be empty"):
            build_record(run=bad)
        for name in DIGESTS:
            with pytest.raises(
                ValueError, match=f"digests.{name} must not be empty"
            ):
                build_record(digests={**DIGESTS, name: bad})


def test_the_signature_must_be_non_blank_text_when_given():
    with pytest.raises(ValueError, match="signature"):
        build_record(signature="  ")
    with pytest.raises(ValueError, match="signature"):
        build_record(signature=5)
    assert build_record(signature="sig").signature == "sig"
    assert build_record().signature is None


def test_a_non_bool_signature_claim_is_a_type_error():
    with pytest.raises(TypeError, match="signed must be bool"):
        build_record(signed=1)


def test_from_records_message_and_signature_cover_the_unsigned_record():
    provenance, sbom = records()
    provenance["signature"] = {"present": "yes"}
    with pytest.raises(
        ValueError, match=r"^provenance\.signature\.present must be bool$"
    ):
        attestation.Attestation.from_records(provenance, sbom)

    provenance, sbom = records()
    record = attestation.Attestation.from_records(
        provenance, sbom, signing_key="k"
    )
    assert record.signature == signing.mac(record.preimage(), b"k")


def test_a_missing_signed_flag_reads_as_false_and_true_is_kept():
    base = build_record(signed=False).to_dict()
    del base["signed"]
    loaded = attestation.Attestation.from_dict(base)
    assert loaded.signed is False
    assert loaded.signature is None
    base["signed"] = True
    assert attestation.Attestation.from_dict(base).signed is True


def test_from_dict_rejects_blank_or_missing_signature_values():
    provenance, sbom = records()
    persisted = attestation.Attestation.from_records(
        provenance, sbom, signing_key="k"
    ).to_dict()
    for bad in ("  ", 5, None):
        persisted["signature"] = {
            "algorithm": signing.SIGNATURE_ALGORITHM,
            "value": bad,
        }
        with pytest.raises(ValueError, match="value must be non-empty"):
            attestation.Attestation.from_dict(persisted)
    persisted["signature"] = "text"
    with pytest.raises(ValueError, match="signature must be an object"):
        attestation.Attestation.from_dict(persisted)
    persisted["signature"] = {"algorithm": "nope", "value": "v"}
    with pytest.raises(
        ValueError, match=r"^unsupported attestation signature algorithm$"
    ):
        attestation.Attestation.from_dict(persisted)


def test_unsigned_dict_is_exactly_the_schema_two_record():
    assert build_record().preimage() == {
        "schema_version": 2,
        "run": "r",
        "digests": DIGESTS,
        "signed": True,
    }


def test_a_version_one_record_keeps_its_stored_layout():
    assert build_record(schema_version=1).preimage() == {
        "schema_version": 1,
        "run_id": "r",
        "subjects": {
            "model_manifest_sha256": "m",
            "provenance_sha256": "p",
            "sbom_sha256": "s",
        },
        "signature_present": True,
    }
