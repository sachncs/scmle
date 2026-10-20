"""Redacted observability contracts for runs and inference events."""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import datetime
import enum
import json
import math
from typing import Any
from typing import Self


class SignalKind(enum.StrEnum):
    """Operational or statistical signal emitted by a run or model."""

    LATENCY = "latency"
    ERROR = "error"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    DATA_QUALITY = "data_quality"
    DRIFT = "drift"
    NOVELTY = "novelty"
    CALIBRATION = "calibration"
    FAIRNESS = "fairness"
    COVERAGE = "coverage"


def require_text(value: str, field_name: str) -> str:
    """Returns trimmed non-empty text for a contract field."""
    result = value.strip()
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    return result


def validate_timestamp(value: str, field_name: str) -> str:
    """Validates an aware ISO-8601 timestamp without changing its text."""
    require_text(value, field_name)
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field_name} must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class MetricSample:
    """One numeric signal with bounded labels and an explicit timestamp."""

    name: str
    kind: SignalKind
    value: float
    unit: str
    timestamp: str
    labels: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate a metric before it is exported or persisted."""
        require_text(self.name, "metric.name")
        require_text(self.unit, "metric.unit")
        validate_timestamp(self.timestamp, "metric.timestamp")
        if not isinstance(self.kind, SignalKind):
            object.__setattr__(self, "kind", SignalKind(self.kind))
        if not math.isfinite(self.value):
            raise ValueError("metric.value must be finite")
        if any(
            not key.strip() or not value.strip()
            for key, value in self.labels.items()
        ):
            raise ValueError("metric labels must not contain empty text")

    def to_dict(self) -> dict[str, Any]:
        """Returns a deterministic JSON-compatible metric."""
        return {
            "name": self.name,
            "kind": self.kind.value,
            "value": float(self.value),
            "unit": self.unit,
            "timestamp": self.timestamp,
            "labels": dict(sorted(self.labels.items())),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a metric from serialized evidence."""
        return cls(
            name=str(record["name"]),
            kind=SignalKind(record["kind"]),
            value=float(record["value"]),
            unit=str(record["unit"]),
            timestamp=str(record["timestamp"]),
            labels=dict(record.get("labels", {})),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class InferenceEvent:
    """Redacted metadata for one prediction request."""

    request_id: str
    run_id: str
    model_fingerprint: str
    outcome: str
    latency_ms: float
    timestamp: str
    input_fingerprint: str | None = None
    error_category: str | None = None
    redacted_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate event identity and prevent unbounded sensitive payloads."""
        for field_name, value in (
            ("request_id", self.request_id),
            ("run_id", self.run_id),
            ("model_fingerprint", self.model_fingerprint),
            ("outcome", self.outcome),
        ):
            require_text(value, f"event.{field_name}")
        validate_timestamp(self.timestamp, "event.timestamp")
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("event.latency_ms must be finite and non-negative")
        if self.outcome == "error" and not self.error_category:
            raise ValueError("error events require error_category")
        if self.outcome != "error" and self.error_category is not None:
            raise ValueError("successful events must not carry error_category")
        if any(not field.strip() for field in self.redacted_fields):
            raise ValueError("redacted_fields must not contain empty values")

    def to_dict(self) -> dict[str, Any]:
        """Returns redacted, JSON-compatible event metadata."""
        return {
            "request_id": self.request_id,
            "run_id": self.run_id,
            "model_fingerprint": self.model_fingerprint,
            "outcome": self.outcome,
            "latency_ms": float(self.latency_ms),
            "timestamp": self.timestamp,
            "input_fingerprint": self.input_fingerprint,
            "error_category": self.error_category,
            "redacted_fields": list(self.redacted_fields),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class ObservabilityPolicy:
    """Retention, redaction, and alert requirements for emitted signals."""

    retention_days: int
    required_signals: tuple[SignalKind, ...]
    redacted_fields: tuple[str, ...]
    alert_thresholds: Mapping[str, float]

    def __post_init__(self) -> None:
        """Reject policies that silently omit required operational signals."""
        if self.retention_days < 1:
            raise ValueError("retention_days must be at least 1")
        if not self.required_signals:
            raise ValueError("required_signals must not be empty")
        if any(
            not isinstance(signal, SignalKind)
            for signal in self.required_signals
        ):
            raise TypeError("required_signals must contain SignalKind values")
        if len(self.required_signals) != len(set(self.required_signals)):
            raise ValueError("required_signals must be unique")
        if any(not field.strip() for field in self.redacted_fields):
            raise ValueError("redacted_fields must not contain empty values")
        if any(
            not name.strip() or not math.isfinite(value) or value < 0
            for name, value in self.alert_thresholds.items()
        ):
            raise ValueError(
                "alert thresholds must be named, finite, and non-negative"
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns a deterministic policy record."""
        return {
            "retention_days": self.retention_days,
            "required_signals": sorted(
                signal.value for signal in self.required_signals
            ),
            "redacted_fields": sorted(self.redacted_fields),
            "alert_thresholds": dict(sorted(self.alert_thresholds.items())),
        }


def canonical_json(
    metrics: Sequence[MetricSample],
    events: Sequence[InferenceEvent],
    policy: ObservabilityPolicy,
) -> str:
    """Returns deterministic redacted observability evidence."""
    payload = {
        "policy": policy.to_dict(),
        "metrics": [metric.to_dict() for metric in metrics],
        "events": [event.to_dict() for event in events],
    }
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
