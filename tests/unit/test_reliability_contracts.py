"""Tests for evidence-first reliability fault-injection contracts."""

import dataclasses
import json

import pytest

from scmle.contracts import reliability


def make_evidence(
    kind: str,
    outcome: reliability.Outcome = (reliability.Outcome.SAFE_OUTCOME_OBSERVED),
) -> reliability.Evidence:
    """Build one concrete evidence record for contract tests."""
    return reliability.Evidence(
        kind=kind,
        execution="fault-run-2026-10-20-001",
        uri=f"artifact://faults/{kind}.json",
        digests={"artifact": f"sha256:{kind}"},
        outcome=outcome,
        recorded="2026-10-20T10:00:00+00:00",
        reviewer="reliability-reviewer",
        observation="Observed safe terminal behavior with no partial publication.",
    )


def make_plan(
    status: reliability.Verification = reliability.Verification.PLANNED,
    evidence: tuple[reliability.Evidence, ...] = (),
) -> reliability.Plan:
    """Build one representative scenario plan."""
    return reliability.Plan(
        scenario=reliability.Contingency.PROCESS_DEATH,
        outcome=reliability.SafeOutcome.RESUME_DURABLE_CHECKPOINT,
        recoverability=reliability.Recoverability.AUTOMATIC,
        requirements=("process_termination", "checkpoint_resume"),
        blocking=True,
        verification=status,
        evidence=evidence,
    )


def test_default_plan_declares_every_required_scenario_without_claiming_execution():
    registry = reliability.Registry.prerelease()

    assert set(registry.plans) == set(reliability.Contingency)
    assert all(
        plan.verification is reliability.Verification.PLANNED
        for plan in registry.plans.values()
    )
    assert all(plan.blocking for plan in registry.plans.values())


def test_default_plan_has_explicit_safe_outcome_recovery_and_evidence_requirements():
    plans = reliability.drills()

    assert len(plans) == len(reliability.Contingency)
    assert all(plan.outcome for plan in plans)
    assert all(plan.recoverability for plan in plans)
    assert all(plan.requirements for plan in plans)


def test_passed_scenario_requires_all_safe_execution_evidence():
    evidence = (
        make_evidence("process_termination"),
        make_evidence("checkpoint_resume"),
    )

    plan = make_plan(
        status=reliability.Verification.PASSED,
        evidence=evidence,
    )

    assert plan.verification is reliability.Verification.PASSED


@pytest.mark.parametrize(
    "status",
    [
        reliability.Verification.PASSED,
        reliability.Verification.FAILED,
        reliability.Verification.INCONCLUSIVE,
    ],
)
def test_verification_status_cannot_claim_execution_without_evidence(status):
    with pytest.raises(ValueError, match="execution evidence"):
        make_plan(status=status)


def test_passed_scenario_rejects_missing_requirement():
    with pytest.raises(ValueError, match="missing evidence requirements"):
        make_plan(
            status=reliability.Verification.PASSED,
            evidence=(make_evidence("process_termination"),),
        )


def test_passed_scenario_rejects_unsafe_evidence():
    evidence = (
        make_evidence("process_termination"),
        make_evidence(
            "checkpoint_resume",
            reliability.Outcome.UNSAFE_OUTCOME_OBSERVED,
        ),
    )

    with pytest.raises(ValueError, match="safe outcome evidence"):
        make_plan(
            status=reliability.Verification.PASSED,
            evidence=evidence,
        )


def test_evidence_rejects_empty_review_fields():
    with pytest.raises(ValueError, match=r"digests\.artifact"):
        make_evidence("process_termination").__class__(
            kind="process_termination",
            execution="run-1",
            uri="artifact://run/evidence.json",
            digests={"artifact": " "},
            outcome=reliability.Outcome.SAFE_OUTCOME_OBSERVED,
            recorded="2026-10-20T10:00:00+00:00",
            reviewer="reviewer",
            observation="observed",
        )


def test_registry_rejects_duplicate_scenarios():
    registry = reliability.Registry()
    registry.register(make_plan())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(make_plan())


def test_registry_rejects_incomplete_plan_before_serialization():
    registry = reliability.Registry()
    registry.register(make_plan())

    with pytest.raises(ValueError, match="plan is incomplete"):
        registry.to_json()


def test_registry_serialization_is_deterministic_and_round_trips():
    registry = reliability.Registry.prerelease()

    first = registry.to_json()
    second = reliability.Registry.prerelease().to_json()
    restored = reliability.Registry.from_json(first)

    assert first == second
    assert json.loads(first) == json.loads(second)
    assert restored.to_json() == first
    assert first.index('"scenario":"changed_snapshot"') < first.index(
        '"scenario":"process_death"'
    )


def test_registry_rejects_missing_scenario_in_serialized_plan():
    record = reliability.Registry.prerelease().to_dict()
    record["scenarios"] = record["scenarios"][:-1]

    with pytest.raises(ValueError, match="plan is incomplete"):
        reliability.Registry.from_dict(record)


def test_evidence_round_trip_preserves_observed_result():
    evidence = make_evidence("checkpoint_resume")

    restored = reliability.Evidence.from_dict(evidence.to_dict())

    assert restored == evidence


def test_registry_marks_a_scenario_passed_only_after_full_safe_evidence():
    registry = reliability.Registry()
    registry.register(make_plan())

    plan = registry.log(
        reliability.Contingency.PROCESS_DEATH,
        (
            make_evidence("process_termination"),
            make_evidence("checkpoint_resume"),
        ),
    )

    assert plan.verification is reliability.Verification.PASSED
    assert registry.plans[reliability.Contingency.PROCESS_DEATH] == plan


def test_registry_marks_unsafe_evidence_failed_and_incomplete_safe_evidence_inconclusive():
    registry = reliability.Registry()
    registry.register(make_plan())

    inconclusive = registry.log(
        reliability.Contingency.PROCESS_DEATH,
        (make_evidence("process_termination"),),
    )
    assert inconclusive.verification is reliability.Verification.INCONCLUSIVE

    failed = registry.log(
        reliability.Contingency.PROCESS_DEATH,
        (
            make_evidence("process_termination"),
            make_evidence(
                "checkpoint_resume",
                reliability.Outcome.UNSAFE_OUTCOME_OBSERVED,
            ),
        ),
    )
    assert failed.verification is reliability.Verification.FAILED


def test_registry_rejects_evidence_not_declared_by_scenario():
    registry = reliability.Registry()
    registry.register(make_plan())

    with pytest.raises(ValueError, match="unexpected requirements"):
        registry.log(
            reliability.Contingency.PROCESS_DEATH,
            (make_evidence("unknown"),),
        )


def test_registry_rejects_evidence_mixed_across_executions():
    """A scenario result cannot combine unrelated execution evidence."""
    registry = reliability.Registry()
    registry.register(make_plan())
    second_execution = dataclasses.replace(
        make_evidence("checkpoint_resume"), execution="fault-run-002"
    )

    with pytest.raises(ValueError, match="execution"):
        registry.log(
            reliability.Contingency.PROCESS_DEATH,
            (make_evidence("process_termination"), second_execution),
        )
