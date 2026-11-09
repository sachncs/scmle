"""Tests for policy-driven observability evaluation."""

import pytest

from scmle.contracts import observability
from scmle.monitoring import events
import scmle.monitoring.observability


def make_policy() -> observability.ObservabilityPolicy:
    """Build a policy with one required signal and one upper bound."""
    return observability.ObservabilityPolicy(
        retention_days=30,
        required_signals=(observability.SignalKind.LATENCY,),
        redacted_fields=("email",),
        alert_thresholds={"latency_p95_ms": 250.0},
    )


def make_metric(
    name: str = "latency_p95_ms",
    value: float = 100.0,
    kind: observability.SignalKind = observability.SignalKind.LATENCY,
    timestamp: str = "2026-10-20T10:00:00+00:00",
) -> observability.MetricSample:
    """Build a timestamped metric sample."""
    return observability.MetricSample(
        name=name,
        kind=kind,
        value=value,
        unit="ms",
        timestamp=timestamp,
    )


def test_evaluation_passes_fresh_metrics_under_threshold():
    result = scmle.monitoring.observability.evaluate(
        [make_metric()],
        make_policy(),
        observed_at="2026-10-20T10:01:00+00:00",
    )

    assert result.passed
    assert result.to_dict()["threshold_breaches"] == []


def test_evaluation_fails_closed_for_missing_stale_future_and_breached_metrics():
    policy = observability.ObservabilityPolicy(
        retention_days=1,
        required_signals=(
            observability.SignalKind.LATENCY,
            observability.SignalKind.ERROR,
        ),
        redacted_fields=(),
        alert_thresholds={"latency_p95_ms": 250.0},
    )
    metrics = [
        make_metric(value=300.0),
        make_metric(
            name="old_latency",
            timestamp="2026-10-18T09:00:00+00:00",
        ),
        make_metric(
            name="future_latency",
            timestamp="2026-10-20T11:00:00+00:00",
        ),
    ]

    result = scmle.monitoring.observability.evaluate(
        metrics,
        policy,
        observed_at="2026-10-20T10:00:00+00:00",
    )

    assert not result.passed
    assert result.missing_signals == ("error",)
    assert result.expired_metrics == ("old_latency",)
    assert result.future_metrics == ("future_latency",)
    assert result.threshold_breaches[0]["metric"] == "latency_p95_ms"


def test_prediction_log_enforces_sidecar_retention_atomically(tmp_path):
    policy = observability.ObservabilityPolicy(
        retention_days=1,
        required_signals=(observability.SignalKind.LATENCY,),
        redacted_fields=(),
        alert_thresholds={},
    )
    log = events.PredictionLog(
        tmp_path / "predictions.jsonl", observability_policy=policy
    )
    log.observability_path.write_text(
        "\n".join(
            (
                '{"type":"inference","timestamp":"2026-10-18T10:00:00+00:00"}',
                '{"type":"inference","timestamp":"2026-10-20T10:00:00+00:00"}',
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = log.enforce_retention("2026-10-20T10:00:00+00:00")

    assert result["removed"] == 1
    assert result["retained"] == 1
    assert (
        log.observability_path.read_text(encoding="utf-8").count("inference")
        == 1
    )


def test_prediction_log_does_not_replace_malformed_sidecar(tmp_path):
    policy = make_policy()
    log = events.PredictionLog(
        tmp_path / "predictions.jsonl", observability_policy=policy
    )
    original = "not-json\n"
    log.observability_path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError):
        log.enforce_retention("2026-10-20T10:00:00+00:00")

    assert log.observability_path.read_text(encoding="utf-8") == original
