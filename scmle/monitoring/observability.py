"""Evaluate operational metrics against an explicit observability policy."""

from collections.abc import Sequence
import dataclasses
import datetime
from typing import Any
from typing import ClassVar

from scmle.contracts import errors
from scmle.contracts import observability
from scmle.contracts import records


@dataclasses.dataclass(frozen=True, slots=True)
class Evaluation:
    """Fail-closed result of evaluating operational metric evidence.

    Attributes:
        passed: True only if nothing is missing, expired, from the future or
            over a threshold.
        observed: Evaluation time as given.
        signals: Required signal kinds with no metric, sorted.
        metrics: Names of metrics older than the retention window,
            sorted.
        future: Names of metrics timestamped after `observed`,
            sorted.
        breaches: Metrics over their alert threshold, each with the
            metric name, observed value, threshold and unit, sorted by name.
        KEYS: Version-1 keys to the written ones (`observed`, `missing`,
            `expired`, `future`, `breaches`); `downgrade` restores version 1.
    """

    KEYS: ClassVar[records.Migration] = records.Migration(
        renames={
            "observed_at": "observed",
            "missing_signals": "missing",
            "expired_metrics": "expired",
            "future_metrics": "future",
            "threshold_breaches": "breaches",
        }
    )

    passed: bool
    observed: str
    signals: tuple[str, ...]
    metrics: tuple[str, ...]
    future: tuple[str, ...]
    breaches: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Returns a deterministic JSON-compatible evaluation.

        Returns:
            A dict of all fields, with tuples as lists.
        """
        return self.KEYS.upgrade(
            {
                "passed": self.passed,
                "observed_at": self.observed,
                "missing_signals": list(self.signals),
                "expired_metrics": list(self.metrics),
                "future_metrics": list(self.future),
                "threshold_breaches": [
                    dict(breach) for breach in self.breaches
                ],
            }
        )


def evaluate(
    metrics: Sequence[observability.Sample],
    policy: observability.Requirements,
    *,
    observed_at: str,
) -> Evaluation:
    """Evaluates metric freshness, required signals, and upper-bound alerts.

    Thresholds are intentionally interpreted as upper bounds. A service can
    define a metric such as ``latency_p95_ms`` or ``error_rate`` and make a
    breach release- or incident-visible without inventing a default.

    Args:
        metrics: The metric samples observed.
        policy: Required signals, retention window and alert thresholds.
        observed_at: Evaluation time, an ISO-8601 timestamp with timezone.

    Returns:
        The evaluation. It fails if a required signal has no sample, a sample is
        older than the retention window or later than `observed_at`, or a value
        exceeds its threshold.
    """
    observed_timestamp = parse(observed_at, "observed_at")
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
        timestamp = parse(metric.timestamp, "metric.timestamp")
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
    return Evaluation(
        passed=not missing and not expired and not future and not breaches,
        observed=observed_at,
        signals=missing,
        metrics=tuple(sorted(expired)),
        future=tuple(sorted(future)),
        breaches=tuple(sorted(breaches, key=lambda breach: breach["metric"])),
    )


def parse(value: str, field_name: str) -> datetime.datetime:
    """Parses a timezone-aware timestamp for temporal policy checks.

    Args:
        value: An ISO-8601 timestamp; a trailing `Z` means UTC.
        field_name: Name used in error messages.

    Returns:
        The timestamp as an aware datetime.

    Raises:
        ValueError: If the value is not ISO-8601 or has no timezone.
    """
    try:
        parsed = datetime.datetime.fromisoformat(value)
    except ValueError as error:
        raise errors.InputError(f"{field_name} must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise errors.InputError(
            f"{field_name} must include timezone information"
        )
    return parsed
