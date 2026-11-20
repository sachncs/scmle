"""Tests for redacted observability contracts."""

import pytest

from scmle.contracts import observability


def policy() -> observability.Requirements:
    """Builds a representative observability policy."""
    return observability.Requirements(
        retention=30,
        signals=(
            observability.Indicator.ERROR,
            observability.Indicator.LATENCY,
        ),
        fields=("email", "phone"),
        thresholds={"latency_p95_ms": 250.0},
    )


def test_metrics_events_and_policy_have_deterministic_json():
    metric = observability.Sample(
        name="p95_latency",
        kind=observability.Indicator.LATENCY,
        value=12.5,
        unit="ms",
        timestamp="2026-10-20T12:00:00+00:00",
        labels={"model": "baseline"},
    )
    event = observability.Inference(
        request="req-1",
        run="run-1",
        model_fingerprint="model-sha",
        outcome="error",
        latency=20.0,
        timestamp="2026-10-20T12:00:00+00:00",
        category="schema",
        fields=("email",),
    )

    payload = observability.canonize((metric,), (event,), policy())

    assert '"error":"schema"' in payload
    assert '"digests":{"model":"model-sha","payload":null}' in payload
    assert '"email"' in payload
    assert observability.Sample.from_dict(metric.to_dict()) == metric


def test_event_requires_error_category_for_failed_request():
    with pytest.raises(ValueError, match="error_category"):
        observability.Inference(
            request="req-1",
            run="run-1",
            model_fingerprint="model-sha",
            outcome="error",
            latency=20.0,
            timestamp="2026-10-20T12:00:00+00:00",
        )


@pytest.mark.parametrize("value", [float("nan")])
def test_metric_rejects_invalid_values(value):
    with pytest.raises(ValueError, match=r"metric\.value"):
        observability.Sample(
            name="latency",
            kind=observability.Indicator.LATENCY,
            value=value,
            unit="ms",
            timestamp="2026-10-20T12:00:00+00:00",
        )
