"""Tests for evidence-backed release evaluation."""

import dataclasses
import json

import pytest

from scmle.contracts import evaluation
from scmle.contracts import evidence
from scmle.contracts import tasks
from scmle.evaluation import release


def make_record(
    criterion_id: str,
    result: evidence.EvidenceResult = evidence.EvidenceResult.PASS,
    decision: evidence.EvidenceDecision = evidence.EvidenceDecision.ACCEPTED,
) -> evidence.EvidenceRecord:
    """Build a complete evidence record for the release fixtures."""
    return evidence.EvidenceRecord(
        criterion_id=criterion_id,
        claim=f"evidence for {criterion_id}",
        scope={"release": "beta"},
        raw_inputs={"run": "sha256:run"},
        command_or_test="uv run pytest tests/unit/test_release_evaluation.py",
        artifact={"path": f"{criterion_id}.json", "digest": "sha256:item"},
        result=result,
        reviewer="release-reviewer",
        observed_at="2026-10-20T12:00:00+00:00",
        limitations=("Fixture evidence is local and synthetic.",),
        decision=decision,
    )


def make_policy(
    *,
    fairness: bool = False,
    diagnostic_action: evaluation.DiagnosticAction = evaluation.DiagnosticAction.BLOCK,
) -> evaluation.EvaluationPolicy:
    """Build a binary policy with representative release gates."""
    fairness_policy = (
        evaluation.FairnessPolicy(
            evaluation.FairnessMethod.GROUP_REPORT,
            sensitive_columns=("group",),
            metrics=("demographic_parity_difference",),
        )
        if fairness
        else evaluation.FairnessPolicy(evaluation.FairnessMethod.NONE)
    )
    fairness_gates = (
        (
            evaluation.FairnessGate(
                "demographic_parity_difference",
                maximum=0.10,
                required_evidence=("fairness.report",),
            ),
        )
        if fairness
        else ()
    )
    diagnostic_gates: tuple[evaluation.DiagnosticGate, ...] = (
        evaluation.DiagnosticGate(
            "metric_uncertainty",
            diagnostic_action,
            ("diagnostic.metric",),
        ),
    )
    if diagnostic_action is evaluation.DiagnosticAction.WARN:
        diagnostic_gates += (
            evaluation.DiagnosticGate(
                "robustness",
                evaluation.DiagnosticAction.BLOCK,
                ("diagnostic.robustness",),
            ),
        )
    return evaluation.EvaluationPolicy(
        task=tasks.TaskType.BINARY,
        primary_metric="roc_auc",
        metrics=(
            evaluation.MetricDeclaration(
                "roc_auc", evaluation.MetricDirection.MAXIMIZE
            ),
        ),
        uncertainty=evaluation.UncertaintyPolicy(
            evaluation.UncertaintyMethod.NONE
        ),
        calibration=evaluation.CalibrationPolicy(
            evaluation.CalibrationMethod.NONE
        ),
        fairness=fairness_policy,
        small_groups=evaluation.SmallGroupPolicy(30, 10),
        explanations=evaluation.ExplanationPolicy(
            evaluation.ExplanationAvailability.NONE
        ),
        release=evaluation.ReleasePolicy(
            metric_gates=(
                evaluation.MetricGate(
                    "roc_auc", evaluation.MetricDirection.MAXIMIZE, 0.75
                ),
            ),
            diagnostic_gates=diagnostic_gates,
            required_evidence=("release.manifest",),
            fairness_gates=fairness_gates,
        ),
    )


def make_index(
    policy: evaluation.EvaluationPolicy,
    *,
    result: evidence.EvidenceResult = evidence.EvidenceResult.PASS,
    decision: evidence.EvidenceDecision = evidence.EvidenceDecision.ACCEPTED,
) -> evidence.EvidenceIndex:
    """Build an index for every policy reference."""
    return evidence.EvidenceIndex(
        [
            make_record(reference, result, decision)
            for reference in release.required_evidence_references(policy)
        ]
    )


def test_release_evaluation_passes_and_serializes_deterministically():
    """A complete set of accepted evidence produces a stable pass."""
    policy = make_policy()
    first = release.evaluate_release(
        policy,
        {"roc_auc": 0.81},
        {"metric_uncertainty": {"estimate": 0.81}},
        {},
        make_index(policy),
    )
    second = release.evaluate_release(
        policy,
        {"roc_auc": 0.81},
        {"metric_uncertainty": {"estimate": 0.81}},
        {},
        make_index(policy),
    )

    assert first.passed
    assert not first.blocked
    assert first.blocking_reasons == ()
    assert first == second
    assert json.loads(first.canonical_json())["passed"] is True


def test_release_evidence_gate_requires_a_verified_signature():
    """A passing policy cannot bypass an unsigned evidence index."""
    policy = make_policy()
    outcome = release.evaluate_release(
        policy,
        {"roc_auc": 0.81},
        {"metric_uncertainty": {"estimate": 0.81}},
        {},
        make_index(policy),
    )
    unsigned = make_index(policy)
    blocked = release.apply_evidence_index_gate(outcome, unsigned, False)
    assert not blocked.passed
    assert any(
        finding.name == "release_evidence" for finding in blocked.findings
    )

    signed = make_index(policy)
    signed.sign("evidence-key")
    accepted = release.apply_evidence_index_gate(outcome, signed, True)
    assert accepted == outcome


def test_metric_failure_and_missing_evidence_block_release():
    """A failed metric and an absent required record are blocking findings."""
    policy = make_policy()
    result = release.evaluate_release(
        policy,
        {"roc_auc": 0.70},
        {"metric_uncertainty": {"estimate": 0.70}},
        {},
        evidence.EvidenceIndex(),
    )

    assert not result.passed
    assert any("roc_auc" in reason for reason in result.blocking_reasons)
    assert any(
        "release.manifest" in reason for reason in result.blocking_reasons
    )
    assert any(
        "diagnostic.metric" in reason for reason in result.blocking_reasons
    )


def test_warning_diagnostic_failure_is_visible_but_not_blocking():
    """A declared warning remains a failed finding without blocking release."""
    policy = make_policy(diagnostic_action=evaluation.DiagnosticAction.WARN)
    result = release.evaluate_release(
        policy,
        {"roc_auc": 0.81},
        {
            "metric_uncertainty": {"error": "bootstrap failed"},
            "robustness": {"scenarios": {}},
        },
        {},
        make_index(policy),
    )

    diagnostic = next(
        item
        for item in result.findings
        if item.kind is release.FindingKind.DIAGNOSTIC
    )
    assert not diagnostic.passed
    assert not diagnostic.blocking
    assert result.passed


def test_fairness_gate_uses_worst_value_across_columns():
    """Fairness thresholds fail closed on the worst reported group metric."""
    policy = make_policy(fairness=True)
    result = release.evaluate_release(
        policy,
        {"roc_auc": 0.81},
        {"metric_uncertainty": {"estimate": 0.81}},
        {
            "fairness": {
                "available": True,
                "columns": {
                    "group": {"demographic_parity_difference": 0.12},
                    "region": {"demographic_parity_difference": 0.03},
                },
            }
        },
        make_index(policy),
    )

    fairness = next(
        item
        for item in result.findings
        if item.kind is release.FindingKind.FAIRNESS
    )
    assert not fairness.passed
    assert fairness.observed == 0.12
    assert not result.passed


def test_unsupported_reports_and_unaccepted_evidence_block():
    """Unsupported report shapes and pending evidence cannot pass silently."""
    policy = make_policy()
    result = release.evaluate_release(
        policy,
        {"roc_auc": 0.81},
        {"metric_uncertainty": []},
        {},
        make_index(
            policy,
            result=evidence.EvidenceResult.PASS,
            decision=evidence.EvidenceDecision.REQUIRED,
        ),
    )

    assert not result.passed
    assert any(
        "unsupported report" in reason for reason in result.blocking_reasons
    )
    assert any("not accepted" in reason for reason in result.blocking_reasons)


def test_explanation_quality_threshold_is_a_blocking_release_gate():
    """A declared quality threshold must be present and pass."""
    policy = dataclasses.replace(
        make_policy(),
        explanations=evaluation.ExplanationPolicy(
            evaluation.ExplanationAvailability.GLOBAL,
            required_methods=("permutation_importance",),
            required_evidence=("explanation.json",),
            require_stability=True,
            stability_threshold=0.8,
        ),
    )
    reports = {
        "metric_uncertainty": {"estimate": 0.81},
        "permutation_importance": {"quality": {"stability_score": 0.75}},
    }

    result = release.evaluate_release(
        policy,
        {"roc_auc": 0.81},
        reports,
        {},
        make_index(policy),
    )

    finding = next(
        item
        for item in result.findings
        if item.kind is release.FindingKind.EXPLANATION
    )
    assert finding.name == "permutation_importance.stability"
    assert finding.observed == 0.75
    assert not finding.passed
    assert not result.passed


def test_explanation_quality_release_gate_passes_with_valid_evidence():
    """A quality report above its declared threshold permits this gate."""
    policy = dataclasses.replace(
        make_policy(),
        explanations=evaluation.ExplanationPolicy(
            evaluation.ExplanationAvailability.GLOBAL,
            required_methods=("permutation_importance",),
            required_evidence=("explanation.json",),
            require_stability=True,
            require_fidelity=True,
            stability_threshold=0.8,
            fidelity_threshold=0.9,
        ),
    )
    reports = {
        "metric_uncertainty": {"estimate": 0.81},
        "permutation_importance": {
            "quality": {"stability_score": 0.95, "fidelity_score": 0.92}
        },
    }

    result = release.evaluate_release(
        policy,
        {"roc_auc": 0.81},
        reports,
        {},
        make_index(policy),
    )

    assert result.passed


def test_release_is_blocked_when_fault_plan_is_unexecuted():
    """Statistical success does not override an unverified fault plan."""
    outcome = release.evaluate_release(
        make_policy(),
        {"roc_auc": 0.81},
        {"metric_uncertainty": {"estimate": 0.81}},
        {},
        make_index(make_policy()),
    )

    blocked = release.apply_reliability_gate(
        outcome,
        {
            "execution_status": "planned",
            "release_blocked": True,
        },
    )

    finding = next(
        item
        for item in blocked.findings
        if item.kind is release.FindingKind.RELIABILITY
    )
    assert not blocked.passed
    assert finding.observed == "planned"


def test_release_reliability_gate_requires_explicit_completed_plan():
    """Only an explicit completed, unblocked plan can pass this gate."""
    outcome = release.evaluate_release(
        make_policy(),
        {"roc_auc": 0.81},
        {"metric_uncertainty": {"estimate": 0.81}},
        {},
        make_index(make_policy()),
    )

    assert (
        release.apply_reliability_gate(
            outcome,
            {"execution_status": "passed", "release_blocked": False},
        )
        == outcome
    )


def test_results_are_immutable_and_findings_are_sorted():
    """The public result cannot be mutated and serializes in stable order."""
    policy = make_policy()
    result = release.evaluate_release(
        policy,
        {"roc_auc": 0.81, "unexpected": 1.0},
        {"metric_uncertainty": {"estimate": 0.81}},
        {},
        make_index(policy),
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.passed = True  # type: ignore[misc]
    assert list(result.findings) == sorted(
        result.findings,
        key=lambda item: (item.kind.value, item.name, item.reason),
    )


def test_readiness_summary_counts_findings_and_serializes_deterministically():
    """A readiness summary exposes stable aggregate and per-kind counts."""
    policy = make_policy(diagnostic_action=evaluation.DiagnosticAction.WARN)
    result = release.evaluate_release(
        policy,
        {"roc_auc": 0.81},
        {
            "metric_uncertainty": {"error": "bootstrap failed"},
            "robustness": {"scenarios": {}},
        },
        {},
        make_index(policy),
    )

    summary = result.readiness_summary()

    assert summary.status is release.ReleaseStatus.READY
    assert summary.total_findings == 6
    assert summary.passed_findings == 5
    assert summary.failed_findings == 1
    assert summary.blocking_findings == 5
    assert summary.blocking_failures == 0
    assert summary.non_blocking_failures == 1
    assert summary.counts_by_kind["diagnostic"] == release.FindingCounts(
        total=2, passed=1, failed=1, blocking=1, blocking_failures=0
    )
    assert set(summary.counts_by_kind) == {
        kind.value for kind in release.FindingKind
    }
    assert json.loads(summary.canonical_json()) == summary.to_dict()
    assert (
        summary.canonical_json() == result.readiness_summary().canonical_json()
    )
    assert result.to_dict()["readiness"] == summary.to_dict()


def test_blocked_readiness_summary_distinguishes_blocking_failures():
    """A failed blocking finding produces an explicit blocked status."""
    policy = make_policy()
    result = release.evaluate_release(
        policy,
        {"roc_auc": 0.70},
        {"metric_uncertainty": {"estimate": 0.70}},
        {},
        make_index(policy),
    )

    summary = result.readiness_summary()

    assert summary.status is release.ReleaseStatus.BLOCKED
    assert summary.failed_findings == 1 + len(
        [
            item
            for item in result.findings
            if item.kind is release.FindingKind.EVIDENCE and not item.passed
        ]
    )
    assert summary.blocking_failures == summary.failed_findings
    assert summary.counts_by_kind["metric"].blocking_failures == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", release.ReleaseStatus.READY, "status"),
        ("total_findings", 2, "total_findings"),
        ("counts_by_kind", {}, "counts_by_kind"),
    ],
)
def test_readiness_summary_rejects_inconsistent_values(
    field: str, value: object, message: str
):
    """Summary construction fails closed when its counts are inconsistent."""
    values: dict[str, object] = {
        "task": "binary_classification",
        "policy_schema_version": 1,
        "status": release.ReleaseStatus.BLOCKED,
        "total_findings": 1,
        "passed_findings": 0,
        "failed_findings": 1,
        "blocking_findings": 1,
        "blocking_failures": 1,
        "non_blocking_failures": 0,
        "counts_by_kind": {
            kind.value: release.FindingCounts(
                total=1 if kind is release.FindingKind.METRIC else 0,
                passed=0,
                failed=1 if kind is release.FindingKind.METRIC else 0,
                blocking=1 if kind is release.FindingKind.METRIC else 0,
                blocking_failures=(
                    1 if kind is release.FindingKind.METRIC else 0
                ),
            )
            for kind in release.FindingKind
        },
    }
    values[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        release.ReleaseReadinessSummary(**values)  # type: ignore[arg-type]


def test_attestation_gate_requires_a_signed_release_record():
    """A release record without an HMAC signature fails closed."""
    outcome = release.ReleaseEvaluation(
        task="classification",
        policy_schema_version=1,
        passed=True,
        findings=(),
        blocking_reasons=(),
    )

    blocked = release.apply_attestation_gate(outcome, {})
    signed = release.apply_attestation_gate(
        outcome,
        {
            "signature": {
                "algorithm": "hmac-sha256",
                "value": "digest",
            }
        },
        verified=True,
    )

    assert not blocked.passed
    assert blocked.findings[-1].kind is release.FindingKind.SECURITY
    assert signed is outcome


def test_attestation_gate_rejects_a_tampered_signature():
    """A present signature is insufficient when verification fails."""
    outcome = release.ReleaseEvaluation(
        task="classification",
        policy_schema_version=1,
        passed=True,
        findings=(),
        blocking_reasons=(),
    )

    blocked = release.apply_attestation_gate(
        outcome,
        {
            "signature": {
                "algorithm": "hmac-sha256",
                "value": "tampered",
            }
        },
        verified=False,
    )

    assert not blocked.passed
    assert blocked.findings[-1].observed == "invalid"
