"""Tests for durable fault-injection evidence recording."""

import hashlib

import pytest

from scmle.contracts import errors
from scmle.contracts import reliability
from scmle.governance import reliability as harness_module


def artifact(evidence_type: str) -> harness_module.FaultArtifact:
    """Build a safe fixture artifact for one requirement."""
    return harness_module.FaultArtifact(
        evidence_type=evidence_type,
        content=f"observed:{evidence_type}".encode(),
        observed_outcome=reliability.EvidenceOutcome.SAFE_OUTCOME_OBSERVED,
        observation="safe outcome observed in the injected execution",
    )


def harness(tmp_path):
    """Build a complete plan and deterministic evidence writer."""
    return harness_module.FaultInjectionHarness(
        reliability.FaultInjectionPlanRegistry.production_beta(),
        tmp_path,
        execution_id="fault-run-001",
        reviewer="reliability-reviewer",
        recorded_at="2026-10-20T10:00:00+00:00",
    )


def test_harness_persists_digest_bound_evidence_and_passes_scenario(tmp_path):
    runner = harness(tmp_path)
    scenario = reliability.FaultScenario.PARTIAL_WRITE
    requirements = runner.registry.plans[scenario].evidence_requirements

    plan = runner.run(
        scenario, lambda current_scenario: tuple(map(artifact, requirements))
    )

    assert plan.verification_status is reliability.VerificationStatus.PASSED
    for evidence in plan.evidence:
        path = tmp_path / evidence.artifact_uri.rsplit("/", 1)[-1]
        assert path.exists()
        assert (
            evidence.artifact_digest
            == hashlib.sha256(path.read_bytes()).hexdigest()
        )


def test_harness_keeps_partial_executor_output_inconclusive(tmp_path):
    runner = harness(tmp_path)
    scenario = reliability.FaultScenario.INVALID_SIGNATURE

    plan = runner.run(
        scenario,
        lambda current_scenario: (artifact("signature_tampering"),),
    )

    assert (
        plan.verification_status is reliability.VerificationStatus.INCONCLUSIVE
    )


def test_harness_rejects_duplicate_evidence_before_writing(tmp_path):
    """Duplicate evidence cannot overwrite a previously emitted artifact."""
    runner = harness(tmp_path)
    scenario = reliability.FaultScenario.INVALID_SIGNATURE

    with pytest.raises(ValueError, match="duplicate evidence types"):
        runner.run(
            scenario,
            lambda current_scenario: (
                artifact("signature_tampering"),
                artifact("signature_tampering"),
            ),
        )

    assert not list(tmp_path.iterdir())


def test_harness_rejects_unexpected_evidence_before_writing(tmp_path):
    """Evidence names outside the plan cannot leave orphaned artifacts."""
    runner = harness(tmp_path)
    scenario = reliability.FaultScenario.INVALID_SIGNATURE

    with pytest.raises(ValueError, match="unexpected=unexpected"):
        runner.run(
            scenario,
            lambda current_scenario: (
                artifact("signature_tampering"),
                artifact("verification_failure"),
                artifact("quarantine_record"),
                artifact("unexpected"),
            ),
        )

    assert not list(tmp_path.iterdir())


def test_harness_wraps_executor_failure_without_claiming_evidence(tmp_path):
    runner = harness(tmp_path)

    def fail(scenario: reliability.FaultScenario):
        del scenario
        raise OSError("worker disappeared")

    with pytest.raises(errors.ReliabilityError, match="before evidence"):
        runner.run(reliability.FaultScenario.WORKER_LOSS, fail)
    assert not list(tmp_path.iterdir())


def test_harness_rejects_path_injection_in_evidence_name(tmp_path):
    with pytest.raises(ValueError, match="path-safe"):
        harness_module.FaultArtifact(
            evidence_type="../escape",
            content=b"evidence",
            observed_outcome=reliability.EvidenceOutcome.INCONCLUSIVE,
            observation="not enough evidence",
        )
