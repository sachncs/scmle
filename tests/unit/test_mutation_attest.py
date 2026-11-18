"""The attestation file's refusals and the applied section's keys."""

import json

import pytest

from scmle.causal.identification import attest
from scmle.contracts import errors


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
