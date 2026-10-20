"""Tests for evidence-backed release evaluation."""

import dataclasses
import json

import pytest

from scml.contracts import evaluation
from scml.contracts import evidence
from scml.contracts import tasks
from scml.evaluation import release


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
