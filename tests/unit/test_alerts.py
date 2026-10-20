"""Tests for deterministic observability alert routing."""

import pytest

from scml.contracts import observability
from scml.monitoring import alerts
from scml.monitoring import observability as monitor_observability


def evaluation() -> monitor_observability.ObservabilityEvaluation:
    """Build one breached observability evaluation."""
    policy = observability.ObservabilityPolicy(
        retention_days=30,
        required_signals=(observability.SignalKind.LATENCY,),
        redacted_fields=(),
        alert_thresholds={"p95_latency_ms": 250.0},
    )
    metric = observability.MetricSample(
        name="p95_latency_ms",
        kind=observability.SignalKind.LATENCY,
        value=300.0,
        unit="ms",
        timestamp="2026-10-20T12:00:00+00:00",
    )
    return monitor_observability.evaluate(
        (metric,), policy, observed_at="2026-10-20T12:00:00+00:00"
    )


def test_alert_routing_is_explicit_and_idempotent():
    first = alerts.route(evaluation(), {"p95_latency_ms": "on-call"})
    second = alerts.route(evaluation(), {"p95_latency_ms": "on-call"})

    assert first == second
    assert first[0].route == "on-call"
    assert first[0].alert_id


def test_breach_without_a_route_fails_closed():
    with pytest.raises(ValueError, match="no alert route"):
        alerts.route(evaluation(), {})


def test_non_breaching_evaluation_emits_no_alerts():
    policy = observability.ObservabilityPolicy(
        retention_days=30,
        required_signals=(observability.SignalKind.LATENCY,),
        redacted_fields=(),
        alert_thresholds={"p95_latency_ms": 250.0},
    )
    metric = observability.MetricSample(
        name="p95_latency_ms",
        kind=observability.SignalKind.LATENCY,
        value=200.0,
        unit="ms",
        timestamp="2026-10-20T12:00:00+00:00",
    )
    result = monitor_observability.evaluate(
        (metric,), policy, observed_at="2026-10-20T12:00:00+00:00"
    )

    assert alerts.route(result, {}) == ()
