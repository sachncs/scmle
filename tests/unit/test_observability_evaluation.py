"""Tests for policy-driven observability evaluation."""

import pytest

from scmle.contracts import observability
from scmle.monitoring import events
import scmle.monitoring.observability


def make_policy() -> observability.Requirements:
    """Build a policy with one required signal and one upper bound."""
    return observability.Requirements(
        retention_days=30,
        required_signals=(observability.Indicator.LATENCY,),
        redacted_fields=("email",),
        alert_thresholds={"latency_p95_ms": 250.0},
    )


def make_metric(
    name: str = "latency_p95_ms",
    value: float = 100.0,
    kind: observability.Indicator = observability.Indicator.LATENCY,
    timestamp: str = "2026-10-20T10:00:00+00:00",
) -> observability.Sample:
    """Build a timestamped metric sample."""
    return observability.Sample(
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
    assert result.to_dict()["breaches"] == []


def test_evaluation_fails_closed_for_missing_stale_future_and_breached_metrics():
    # Enforces website/src/pages/docs/operations.astro:61.
    policy = observability.Requirements(
        retention_days=1,
        required_signals=(
            observability.Indicator.LATENCY,
            observability.Indicator.ERROR,
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
    assert result.signals == ("error",)
    assert result.metrics == ("old_latency",)
    assert result.future == ("future_latency",)
    assert result.breaches[0]["metric"] == "latency_p95_ms"


def test_prediction_log_enforces_sidecar_retention_atomically(tmp_path):
    policy = observability.Requirements(
        retention_days=1,
        required_signals=(observability.Indicator.LATENCY,),
        redacted_fields=(),
        alert_thresholds={},
    )
    log = events.Journal(tmp_path / "predictions.jsonl", policy=policy)
    log.sidecar.write_text(
        '{"type":"inference","timestamp":"2026-10-18T10:00:00+00:00"}\n'
        '{"type":"inference","timestamp":"2026-10-20T10:00:00+00:00"}\n',
        encoding="utf-8",
    )

    result = log.purge("2026-10-20T10:00:00+00:00")

    assert result["removed"] == 1
    assert result["retained"] == 1
    assert log.sidecar.read_text(encoding="utf-8").count("inference") == 1


def test_prediction_log_does_not_replace_malformed_sidecar(tmp_path):
    policy = make_policy()
    log = events.Journal(tmp_path / "predictions.jsonl", policy=policy)
    original = "not-json\n"
    log.sidecar.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError):
        log.purge("2026-10-20T10:00:00+00:00")

    assert log.sidecar.read_text(encoding="utf-8") == original
