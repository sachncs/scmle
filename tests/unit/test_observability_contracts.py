"""Tests for redacted observability contracts."""

import pytest

from scmle.contracts import observability


def policy() -> observability.ObservabilityPolicy:
    """Builds a representative observability policy."""
    return observability.ObservabilityPolicy(
        retention_days=30,
        required_signals=(
            observability.SignalKind.ERROR,
            observability.SignalKind.LATENCY,
        ),
        redacted_fields=("email", "phone"),
        alert_thresholds={"latency_p95_ms": 250.0},
    )


def test_metrics_events_and_policy_have_deterministic_json():
    metric = observability.MetricSample(
        name="p95_latency",
        kind=observability.SignalKind.LATENCY,
        value=12.5,
        unit="ms",
        timestamp="2026-10-20T12:00:00+00:00",
        labels={"model": "baseline"},
    )
    event = observability.InferenceEvent(
        request_id="req-1",
        run_id="run-1",
        model_fingerprint="model-sha",
        outcome="error",
        latency_ms=20.0,
        timestamp="2026-10-20T12:00:00+00:00",
        error_category="schema",
        redacted_fields=("email",),
    )

    payload = observability.canonical_json((metric,), (event,), policy())

    assert '"error_category":"schema"' in payload
    assert '"email"' in payload
    assert observability.MetricSample.from_dict(metric.to_dict()) == metric


def test_event_requires_error_category_for_failed_request():
    with pytest.raises(ValueError, match="error_category"):
        observability.InferenceEvent(
            request_id="req-1",
            run_id="run-1",
            model_fingerprint="model-sha",
            outcome="error",
            latency_ms=20.0,
            timestamp="2026-10-20T12:00:00+00:00",
        )


@pytest.mark.parametrize("value", [float("nan")])
def test_metric_rejects_invalid_values(value):
    with pytest.raises(ValueError, match=r"metric\.value"):
        observability.MetricSample(
            name="latency",
            kind=observability.SignalKind.LATENCY,
            value=value,
            unit="ms",
            timestamp="2026-10-20T12:00:00+00:00",
        )
