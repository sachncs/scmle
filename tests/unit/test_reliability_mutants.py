"""Targeted tests pinning the declared fault plan and its validation."""

import dataclasses
import json

import pytest

from scmle.contracts import reliability
from tests.unit.test_reliability_contracts import make_evidence
from tests.unit.test_reliability_contracts import make_plan

DECLARED = (
    (
        "PROCESS_DEATH",
        "RESUME_DURABLE_CHECKPOINT",
        "AUTOMATIC",
        ("process_termination", "checkpoint_resume", "artifact_integrity"),
    ),
    (
        "STALE_CHECKPOINT",
        "REJECT_STALE_CHECKPOINT",
        "OPERATOR_ASSISTED",
        ("checkpoint_identity", "stale_checkpoint_rejection", "run_state"),
    ),
    (
        "DISK_FULL",
        "PRESERVE_DURABLE_ARTIFACT",
        "OPERATOR_ASSISTED",
        ("disk_capacity_exhaustion", "artifact_integrity", "resource_error"),
    ),
    (
        "CORRUPT_ARTIFACT",
        "QUARANTINE_CORRUPT_ARTIFACT",
        "OPERATOR_ASSISTED",
        ("corruption_injection", "digest_mismatch", "quarantine_record"),
    ),
    (
        "WORKER_LOSS",
        "RETRY_IDEMPOTENT_WORK",
        "AUTOMATIC",
        ("worker_termination", "idempotent_retry", "event_deduplication"),
    ),
    (
        "TIMEOUT",
        "CANCEL_AND_RECORD_TERMINAL_FAILURE",
        "OPERATOR_ASSISTED",
        ("timeout_injection", "cancellation_record", "resource_error"),
    ),
    (
        "MEMORY_EXHAUSTION",
        "ABORT_BEFORE_PUBLICATION",
        "OPERATOR_ASSISTED",
        ("memory_pressure", "publication_absence", "resource_error"),
    ),
    (
        "PARTIAL_WRITE",
        "PRESERVE_DURABLE_ARTIFACT",
        "AUTOMATIC",
        ("write_interruption", "atomicity_check", "artifact_integrity"),
    ),
    (
        "DEPENDENCY_MISMATCH",
        "REJECT_INCOMPATIBLE_DEPENDENCY",
        "OPERATOR_ASSISTED",
        ("environment_identity", "dependency_rejection", "run_state"),
    ),
    (
        "INVALID_SIGNATURE",
        "REJECT_UNTRUSTED_ARTIFACT",
        "OPERATOR_ASSISTED",
        ("signature_tampering", "verification_failure", "quarantine_record"),
    ),
    (
        "RETRY_EXHAUSTION",
        "TERMINATE_AFTER_RETRY_EXHAUSTION",
        "TERMINAL_SAFE_FAILURE",
        ("retry_history", "terminal_event", "partial_publication_check"),
    ),
    (
        "DUPLICATE_EVENT",
        "IGNORE_DUPLICATE_EVENT",
        "AUTOMATIC",
        ("duplicate_event", "idempotency_key", "state_transition"),
    ),
    (
        "INCOMPLETE_LINEAGE",
        "BLOCK_RELEASE_FOR_MISSING_LINEAGE",
        "TERMINAL_SAFE_FAILURE",
        ("lineage_gap", "release_block", "audit_record"),
    ),
    (
        "REJECTED_APPROVAL",
        "BLOCK_RELEASE_FOR_REJECTED_APPROVAL",
        "TERMINAL_SAFE_FAILURE",
        ("approval_rejection", "release_block", "audit_record"),
    ),
    (
        "CHANGED_SNAPSHOT",
        "INVALIDATE_CHANGED_SNAPSHOT",
        "OPERATOR_ASSISTED",
        ("snapshot_digest_change", "run_invalidation", "lineage_record"),
    ),
)


def test_the_declared_fault_plan_is_exactly_the_reviewed_table():
    plans = reliability.default_fault_injection_plans()

    assert (
        tuple(
            (
                plan.scenario.name,
                plan.expected_safe_outcome.name,
                plan.recoverability.name,
                plan.evidence_requirements,
            )
            for plan in plans
        )
        == DECLARED
    )
    assert all(plan.release_blocking is True for plan in plans)


def test_plan_enum_fields_are_type_checked_under_their_own_names():
    good = {
        "scenario": reliability.FaultScenario.PROCESS_DEATH,
        "expected_safe_outcome": (
            reliability.SafeOutcome.RESUME_DURABLE_CHECKPOINT
        ),
        "recoverability": reliability.Recoverability.AUTOMATIC,
        "evidence_requirements": ("a",),
        "release_blocking": True,
        "verification_status": reliability.VerificationStatus.PLANNED,
    }
    for name, expected in (
        ("scenario", "FaultScenario"),
        ("expected_safe_outcome", "SafeOutcome"),
        ("recoverability", "Recoverability"),
        ("verification_status", "VerificationStatus"),
    ):
        with pytest.raises(TypeError, match=f"^{name} must be {expected}$"):
            reliability.Plan(**{**good, name: "wrong"})
    with pytest.raises(TypeError, match="release_blocking must be bool"):
        reliability.Plan(**{**good, "release_blocking": 1})


def test_blank_or_non_text_requirements_are_rejected_individually():
    with pytest.raises(ValueError, match="must contain text"):
        reliability.check_evidence_requirements(("ok", " "))
    with pytest.raises(ValueError, match="must contain text"):
        reliability.check_evidence_requirements(("ok", 3))
    with pytest.raises(ValueError, match="must be unique"):
        reliability.check_evidence_requirements(("ok", "ok"))
    with pytest.raises(ValueError, match="must not be empty"):
        reliability.check_evidence_requirements(())
    reliability.check_evidence_requirements(("ok", "also"))


def test_plan_error_messages_name_the_gap():
    with pytest.raises(
        ValueError, match=r"^verification status requires execution evidence$"
    ):
        make_plan(status=reliability.VerificationStatus.FAILED)
    with pytest.raises(
        ValueError,
        match=r"^passed scenario is missing evidence requirements: "
        "checkpoint_resume$",
    ):
        make_plan(
            status=reliability.VerificationStatus.PASSED,
            evidence=(make_evidence("process_termination"),),
        )
    with pytest.raises(
        ValueError,
        match=r"^passed scenario requires safe outcome evidence only$",
    ):
        make_plan(
            status=reliability.VerificationStatus.PASSED,
            evidence=(
                make_evidence("process_termination"),
                make_evidence(
                    "checkpoint_resume",
                    reliability.EvidenceOutcome.INCONCLUSIVE,
                ),
            ),
        )


def test_plan_round_trip_keeps_status_and_evidence():
    evidence = (
        make_evidence("process_termination"),
        make_evidence("checkpoint_resume"),
    )
    plan = make_plan(
        status=reliability.VerificationStatus.PASSED, evidence=evidence
    )

    restored = reliability.Plan.from_dict(plan.to_dict())

    assert restored == plan
    assert restored.verification_status is reliability.VerificationStatus.PASSED
    assert restored.evidence == evidence


def test_record_evidence_validates_its_input_with_specific_messages():
    registry = reliability.Registry.production_beta()
    scenario = reliability.FaultScenario.PROCESS_DEATH
    with pytest.raises(
        ValueError, match=r"^evidence must be a non-empty tuple$"
    ):
        registry.record_evidence(scenario, ())
    with pytest.raises(
        ValueError, match=r"^evidence must be a non-empty tuple$"
    ):
        registry.record_evidence(
            scenario, [make_evidence("process_termination")]
        )
    other = dataclasses.replace(
        make_evidence("checkpoint_resume"), execution_id="different"
    )
    with pytest.raises(
        ValueError,
        match=r"^evidence for one scenario must share an execution_id$",
    ):
        registry.record_evidence(
            scenario, (make_evidence("process_termination"), other)
        )
    with pytest.raises(
        ValueError,
        match=r"^evidence has unexpected requirements: a_extra, b_extra$",
    ):
        registry.record_evidence(
            scenario,
            (make_evidence("b_extra"), make_evidence("a_extra")),
        )


def test_incomplete_registry_names_missing_scenarios_in_value_order():
    registry = reliability.Registry()
    registry.register(make_plan())

    with pytest.raises(ValueError) as info:
        registry.validate_complete()

    names = str(info.value).split(": ", 1)[1].split(", ")
    assert names == sorted(names)
    assert "process_death" not in names
    assert len(names) == len(reliability.FaultScenario) - 1


def test_to_json_escapes_non_ascii_evidence_text():
    registry = reliability.Registry.production_beta()
    scenario = reliability.FaultScenario.PROCESS_DEATH
    evidence = dataclasses.replace(
        make_evidence("process_termination"), observation="résumé"
    )
    registry.record_evidence(scenario, (evidence,))

    text = registry.to_json()

    assert text.isascii()
    assert "r\\u00e9sum\\u00e9" in text


def test_to_json_is_compact_sorted_ascii():
    registry = reliability.Registry.production_beta()
    text = registry.to_json()

    assert ", " not in text and ": " not in text
    record = json.loads(text)
    assert text == json.dumps(
        record, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    unsorted_dump = json.dumps(registry.to_dict(), separators=(",", ":"))
    assert text != unsorted_dump
