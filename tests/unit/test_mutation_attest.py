"""The attestation file's refusals and the applied section's keys."""

import json

import pytest

from scmle.assurance.charter import causal
from scmle.causal.identification import attest
from scmle.contracts import errors
from scmle.contracts import signing


def test_a_document_without_a_list_is_refused(tmp_path) -> None:
    path = tmp_path / "attestations.json"
    path.write_text(
        json.dumps(
            {"format_version": attest.FORMAT_VERSION, "attestations": {}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        errors.InputError,
        match=r"^attestations must hold a list under 'attestations'$",
    ):
        attest.load(path)


def test_the_applied_section_names_its_evaluation_date() -> None:
    applied = attest.Applied({}, (), {"a": "none"}, "2026-01-02")
    assert applied.to_dict() == {
        "evaluated": "2026-01-02",
        "accepted": [],
        "rejected": {"a": "none"},
    }


def statement(**changes) -> attest.Attestation:
    values = {
        "assumption": "positivity",
        "attestor": "A. Rivera",
        "statement": "I judge this assumption to hold",
        "evidence": "design memo",
        "date": "2026-11-01",
        "expires": "2027-11-01",
        "scope": "ab" * 32,
    }
    values.update(changes)
    return attest.Attestation(**values)


def test_a_blank_field_is_refused_by_its_name() -> None:
    with pytest.raises(
        errors.InputError, match=r"^assumption must not be empty$"
    ):
        statement(assumption=" ")


def test_a_signed_record_does_not_verify_without_a_key(monkeypatch) -> None:
    monkeypatch.delenv(signing.SIGNING_KEY_ENV, raising=False)
    monkeypatch.delenv(signing.SIGNING_KEY_FILE_ENV, raising=False)
    assert statement().signed("key").verify(None) is False
    assert statement().verify("key") is False


def test_with_no_key_nothing_is_accepted_and_the_reason_says_so(
    monkeypatch,
) -> None:
    monkeypatch.delenv(signing.SIGNING_KEY_ENV, raising=False)
    monkeypatch.delenv(signing.SIGNING_KEY_FILE_ENV, raising=False)
    item = statement().signed("key")
    attestors = (
        causal.Attestor("A. Rivera", "statistician", (item.assumption,)),
    )
    applied = attest.apply(
        attestors, (item,), (item.assumption,), item.scope, "2026-11-06", None
    )
    assert applied.rejected == {
        item.assumption: "no signing key to verify the signature"
    }
