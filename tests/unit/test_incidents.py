"""Tests for the auditable incident workflow."""

import pytest

from scmle.contracts import observability
from scmle.monitoring import incidents
from scmle.monitoring import slo


def evaluation() -> slo.SLOEvaluation:
    """Build one breached SLO evaluation."""
    target = slo.SLOTarget(
        name="latency",
        metric_name="p95_latency_ms",
        kind=observability.SignalKind.LATENCY,
        maximum=250.0,
        window_days=7,
        minimum_samples=1,
        owner="ml-on-call",
        runbook="https://runbooks.example/latency",
        alert_route="on-call",
    )
    sample = observability.MetricSample(
        name="p95_latency_ms",
        kind=observability.SignalKind.LATENCY,
        value=400.0,
        unit="ms",
        timestamp="2026-10-20T12:00:00+00:00",
    )
    return slo.SLOMonitor().evaluate(
        target, (sample,), observed_at="2026-10-20T12:00:00+00:00"
    )


def open_incident(workflow: incidents.IncidentWorkflow) -> incidents.Incident:
    """Open the shared fixture incident."""
    return workflow.open(
        evaluation(),
        opened_at="2026-10-20T12:01:00+00:00",
        run_id="run-42",
        rollback_target="model-previous",
    )


def test_workflow_is_idempotent_and_preserves_runbook_and_rollback(tmp_path):
    workflow = incidents.IncidentWorkflow(tmp_path / "incidents.jsonl")
    first = open_incident(workflow)
    second = open_incident(workflow)

    assert first == second
    assert first.status is incidents.IncidentStatus.OPEN
    assert first.run_id == "run-42"
    assert first.rollback_target == "model-previous"
    assert len(workflow.read_events()) == 1


def test_workflow_requires_acknowledgement_before_evidence_backed_resolution(
    tmp_path,
):
    workflow = incidents.IncidentWorkflow(tmp_path / "incidents.jsonl")
    opened = open_incident(workflow)

    with pytest.raises(ValueError, match="acknowledged"):
        workflow.resolve(
            opened.incident_id,
            actor="operator",
            resolved_at="2026-10-20T12:03:00+00:00",
            resolution="rolled back",
            evidence_digest="sha256:fix",
        )

    acknowledged = workflow.acknowledge(
        opened.incident_id,
        actor="operator",
        acknowledged_at="2026-10-20T12:02:00+00:00",
    )
    resolved = workflow.resolve(
        opened.incident_id,
        actor="operator",
        resolved_at="2026-10-20T12:03:00+00:00",
        resolution="rolled back to the previous model",
        evidence_digest="sha256:fix",
    )

    assert acknowledged.status is incidents.IncidentStatus.ACKNOWLEDGED
    assert resolved.status is incidents.IncidentStatus.RESOLVED
    assert resolved.evidence_digest == "sha256:fix"
    assert (
        workflow.resolve(
            opened.incident_id,
            actor="another-operator",
            resolved_at="2026-10-20T12:04:00+00:00",
            resolution="duplicate retry",
            evidence_digest="sha256:other",
        )
        == resolved
    )


def test_tampered_or_reordered_journal_fails_closed(tmp_path):
    path = tmp_path / "incidents.jsonl"
    workflow = incidents.IncidentWorkflow(path)
    open_incident(workflow)
    original = path.read_text(encoding="utf-8")
    path.write_text(
        original.replace('"status":"open"', '"status":"resolved"'),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="digest"):
        workflow.records()


def test_a_valid_digest_cannot_hide_an_invalid_transition(tmp_path):
    workflow = incidents.IncidentWorkflow(tmp_path / "incidents.jsonl")
    opened = open_incident(workflow)
    workflow.append_event("resolved", opened)

    with pytest.raises(ValueError, match="event type"):
        workflow.records()


def test_passing_evaluation_cannot_open_an_incident(tmp_path):
    current = evaluation()
    passing = slo.SLOEvaluation(
        target=current.target,
        observed_at=current.observed_at,
        status=slo.SLOStatus.PASS,
        sample_count=1,
        in_window_count=1,
        observed_maximum=100.0,
        error_budget_remaining=0.6,
        reason="maximum 100 is within 250",
    )

    with pytest.raises(ValueError, match="passing"):
        incidents.IncidentWorkflow(tmp_path / "incidents.jsonl").open(
            passing,
            opened_at="2026-10-20T12:01:00+00:00",
            run_id="run-42",
            rollback_target="model-previous",
        )
