"""Evaluate operational metrics against an explicit observability policy."""

from collections.abc import Sequence
import dataclasses
import datetime
from typing import Any

from scml.contracts import observability


@dataclasses.dataclass(frozen=True, slots=True)
class ObservabilityEvaluation:
    """Fail-closed result of evaluating operational metric evidence."""

    passed: bool
    observed_at: str
    missing_signals: tuple[str, ...]
    expired_metrics: tuple[str, ...]
    future_metrics: tuple[str, ...]
    threshold_breaches: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible evaluation."""
        return {
            "passed": self.passed,
            "observed_at": self.observed_at,
            "missing_signals": list(self.missing_signals),
            "expired_metrics": list(self.expired_metrics),
            "future_metrics": list(self.future_metrics),
            "threshold_breaches": [
                dict(breach) for breach in self.threshold_breaches
            ],
        }


def evaluate(
    metrics: Sequence[observability.MetricSample],
    policy: observability.ObservabilityPolicy,
    *,
    observed_at: str,
) -> ObservabilityEvaluation:
    """Evaluate metric freshness, required signals, and upper-bound alerts.

    Thresholds are intentionally interpreted as upper bounds. A service can
    define a metric such as ``latency_p95_ms`` or ``error_rate`` and make a
    breach release- or incident-visible without inventing a default.
    """
    observed_timestamp = parse_timestamp(observed_at, "observed_at")
    cutoff = observed_timestamp - datetime.timedelta(days=policy.retention_days)
    observed_signals = {metric.kind for metric in metrics}
    missing = tuple(
        sorted(
            signal.value
            for signal in policy.required_signals
            if signal not in observed_signals
        )
    )
    expired = []
    future = []
    breaches = []
    for metric in metrics:
        timestamp = parse_timestamp(metric.timestamp, "metric.timestamp")
        if timestamp < cutoff:
            expired.append(metric.name)
        if timestamp > observed_timestamp:
            future.append(metric.name)
        threshold = policy.alert_thresholds.get(metric.name)
        if threshold is not None and metric.value > threshold:
            breaches.append(
                {
                    "metric": metric.name,
                    "observed": metric.value,
                    "threshold": threshold,
                    "unit": metric.unit,
                }
            )
    return ObservabilityEvaluation(
        passed=not missing and not expired and not future and not breaches,
        observed_at=observed_at,
        missing_signals=missing,
        expired_metrics=tuple(sorted(expired)),
        future_metrics=tuple(sorted(future)),
        threshold_breaches=tuple(
            sorted(breaches, key=lambda breach: breach["metric"])
        ),
    )


def parse_timestamp(value: str, field_name: str) -> datetime.datetime:
    """Parse a timezone-aware timestamp for temporal policy checks."""
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field_name} must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information")
    return parsed
