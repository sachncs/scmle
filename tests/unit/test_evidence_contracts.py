"""Tests for deterministic release evidence contracts."""

import pytest

from scml.contracts import evidence


def make_record(
    criterion_id: str = "quality.contracts",
    result: evidence.EvidenceResult = evidence.EvidenceResult.PASS,
    decision: evidence.EvidenceDecision = evidence.EvidenceDecision.ACCEPTED,
) -> evidence.EvidenceRecord:
    """Builds a representative release evidence record."""
    return evidence.EvidenceRecord(
        criterion_id=criterion_id,
        claim="structured curation findings are persisted",
        scope={"release": "beta", "area": "data"},
        raw_inputs={"dataset": "sha256:dataset"},
        command_or_test="uv run pytest tests/unit/test_evidence_contracts.py",
        artifact={"path": "curation_report.json", "digest": "sha256:report"},
        result=result,
        reviewer="independent-reviewer",
        observed_at="2026-10-20T12:00:00+00:00",
        limitations=(
            "This fixture does not prove external connector behavior.",
        ),
        decision=decision,
    )


def test_record_round_trips_and_index_is_deterministic():
    index = evidence.EvidenceIndex(
        [make_record("z.last"), make_record("a.first")]
    )

    assert [record.criterion_id for record in index.ordered()] == [
        "a.first",
        "z.last",
    ]
    assert (
        evidence.EvidenceRecord.from_dict(make_record().to_dict())
        == make_record()
    )
    assert index.ready()
    assert (
        index.canonical_json()
        == evidence.EvidenceIndex(
            [make_record("a.first"), make_record("z.last")]
        ).canonical_json()
    )


def test_index_rejects_duplicate_and_pending_records():
    index = evidence.EvidenceIndex([make_record()])

    with pytest.raises(ValueError, match="duplicate"):
        index.register(make_record())

    blocked = evidence.EvidenceIndex(
        [
            make_record(
                result=evidence.EvidenceResult.FAIL,
                decision=evidence.EvidenceDecision.BLOCKED,
            )
        ]
    )
    assert not blocked.ready()
    assert [item.criterion_id for item in blocked.pending()] == [
        "quality.contracts"
    ]


@pytest.mark.parametrize("observed_at", ["2026-10-20T12:00:00", "not-a-time"])
def test_record_requires_aware_timestamp(observed_at):
    with pytest.raises(ValueError):
        evidence.EvidenceRecord(
            **{
                **make_record().to_dict(),
                "result": evidence.EvidenceResult.PASS,
                "decision": evidence.EvidenceDecision.ACCEPTED,
                "observed_at": observed_at,
            }
        )
