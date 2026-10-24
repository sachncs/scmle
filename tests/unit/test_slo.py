"""Tests for service-level objective contracts."""

import pytest

from scmle.contracts import observability
from scmle.monitoring import slo


def target() -> slo.SLOTarget:
    """Build a target with an explicit operational response path."""
    return slo.SLOTarget(
        name="prediction-latency",
        metric_name="p95_latency_ms",
        kind=observability.SignalKind.LATENCY,
        maximum=250.0,
        window_days=7,
        minimum_samples=2,
        owner="ml-on-call",
        runbook="https://runbooks.example/prediction-latency",
        alert_route="on-call",
    )


def metric(value: float, timestamp: str) -> observability.MetricSample:
    """Build a timestamped latency observation."""
    return observability.MetricSample(
        name="p95_latency_ms",
        kind=observability.SignalKind.LATENCY,
        value=value,
        unit="ms",
        timestamp=timestamp,
    )


def test_slo_passes_and_records_owner_runbook_and_budget():
    """A complete healthy window is auditable by an operator."""
    result = slo.SLOMonitor().evaluate(
        target(),
        (
            metric(100.0, "2026-10-19T12:00:00+00:00"),
            metric(200.0, "2026-10-20T12:00:00+00:00"),
        ),
        observed_at="2026-10-20T12:00:00+00:00",
    )

    assert result.status is slo.SLOStatus.PASS
    assert result.incident_required is False
    assert result.error_budget_remaining == pytest.approx(0.2)
    assert result.to_dict()["owner"] == "ml-on-call"


def test_slo_breach_preserves_incident_path_and_uses_worst_sample():
    """One bad observation cannot be averaged away."""
    result = slo.SLOMonitor().evaluate(
        target(),
        (
            metric(100.0, "2026-10-19T12:00:00+00:00"),
            metric(400.0, "2026-10-20T12:00:00+00:00"),
        ),
        observed_at="2026-10-20T12:00:00+00:00",
    )

    assert result.status is slo.SLOStatus.BREACHED
    assert result.incident_required
    assert result.target.runbook.endswith("prediction-latency")
    assert result.observed_maximum == 400.0


def test_slo_is_not_green_when_data_is_missing_or_future():
    """Missing and temporally invalid telemetry fail closed."""
    monitor = slo.SLOMonitor()
    missing = monitor.evaluate(
        target(),
        (metric(100.0, "2026-10-20T12:00:00+00:00"),),
        observed_at="2026-10-20T12:00:00+00:00",
    )
    future = monitor.evaluate(
        target(),
        (
            metric(100.0, "2026-10-19T12:00:00+00:00"),
            metric(100.0, "2026-10-21T12:00:00+00:00"),
        ),
        observed_at="2026-10-20T12:00:00+00:00",
    )

    assert missing.status is slo.SLOStatus.INSUFFICIENT_DATA
    assert missing.incident_required
    assert future.status is slo.SLOStatus.INVALID
    assert future.incident_required


def test_slo_rejects_missing_operational_metadata_and_invalid_limits():
    """An SLO without an owner or runbook is not an operating contract."""
    with pytest.raises(ValueError, match="owner"):
        slo.SLOTarget(
            name="latency",
            metric_name="p95_latency_ms",
            kind=observability.SignalKind.LATENCY,
            maximum=250.0,
            window_days=7,
            minimum_samples=1,
            owner="",
            runbook="runbook",
            alert_route="on-call",
        )
    with pytest.raises(ValueError, match="maximum"):
        slo.SLOTarget(
            name="latency",
            metric_name="p95_latency_ms",
            kind=observability.SignalKind.LATENCY,
            maximum=-1.0,
            window_days=7,
            minimum_samples=1,
            owner="owner",
            runbook="runbook",
            alert_route="on-call",
        )
