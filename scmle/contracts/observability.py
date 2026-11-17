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

from scmle.contracts import errors


class Indicator(enum.StrEnum):
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
    """Returns trimmed non-empty text for a contract field.

    Args:
        value: The text to check.
        field_name: The field name, for the error message.

    Returns:
        `value` with surrounding whitespace removed.

    Raises:
        ValueError: If the trimmed text is empty.
    """
    result = value.strip()
    if not result:
        raise errors.InputError(f"{field_name} must not be empty")
    return result


def validate_timestamp(value: str, field_name: str) -> str:
    """Validates an aware ISO-8601 timestamp without changing its text.

    Args:
        value: The timestamp text.
        field_name: The field name, for the error message.

    Returns:
        `value`, unchanged.

    Raises:
        ValueError: If the text is empty, not ISO-8601, or has no timezone.
    """
    require_text(value, field_name)
    try:
        parsed = datetime.datetime.fromisoformat(value)
    except ValueError as error:
        raise errors.InputError(f"{field_name} must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise errors.InputError(
            f"{field_name} must include timezone information"
        )
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class Sample:
    """One numeric signal with bounded labels and an explicit timestamp."""

    name: str
    kind: Indicator
    value: float
    unit: str
    timestamp: str
    labels: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validates a metric and coerces `kind` to a `Indicator`.

        Raises:
            ValueError: If the name, unit or timestamp is invalid, `kind` is
                not a `Indicator` value, the value is not finite, or a label
                key or value is blank.
        """
        require_text(self.name, "metric.name")
        require_text(self.unit, "metric.unit")
        validate_timestamp(self.timestamp, "metric.timestamp")
        kind: Any = self.kind  # may be a raw value at runtime
        if not isinstance(kind, Indicator):
            object.__setattr__(self, "kind", Indicator(kind))
        if not math.isfinite(self.value):
            raise errors.InputError("metric.value must be finite")
        if any(
            not key.strip() or not value.strip()
            for key, value in self.labels.items()
        ):
            raise errors.InputError("metric labels must not contain empty text")

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
        """Rebuilds a metric from serialized evidence.

        Args:
            record: A mapping as produced by `to_dict`; `labels` may be
                absent.

        Returns:
            The metric sample. A missing required key raises `KeyError`, and
            invalid values raise the errors of `__post_init__`.
        """
        return cls(
            name=str(record["name"]),
            kind=Indicator(record["kind"]),
            value=float(record["value"]),
            unit=str(record["unit"]),
            timestamp=str(record["timestamp"]),
            labels=dict(record.get("labels", {})),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Inference:
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
        """Validates event identity, timestamp, latency and error category.

        Any outcome other than ``error`` counts as successful here.

        Raises:
            ValueError: If an identity field or the timestamp is invalid, the
                latency is not finite and non-negative, an ``error`` event has
                no error category, a successful event has one, or a redacted
                field name is empty.
        """
        for field_name, value in (
            ("request_id", self.request_id),
            ("run_id", self.run_id),
            ("model_fingerprint", self.model_fingerprint),
            ("outcome", self.outcome),
        ):
            require_text(value, f"event.{field_name}")
        validate_timestamp(self.timestamp, "event.timestamp")
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise errors.InputError(
                "event.latency_ms must be finite and non-negative"
            )
        if self.outcome == "error" and not self.error_category:
            raise errors.InputError("error events require error_category")
        if self.outcome != "error" and self.error_category is not None:
            raise errors.InputError(
                "successful events must not carry error_category"
            )
        if any(not field.strip() for field in self.redacted_fields):
            raise errors.InputError(
                "redacted_fields must not contain empty values"
            )

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
class Requirements:
    """Retention, redaction, and alert requirements for emitted signals."""

    retention_days: int
    required_signals: tuple[Indicator, ...]
    redacted_fields: tuple[str, ...]
    alert_thresholds: Mapping[str, float]

    def __post_init__(self) -> None:
        """Rejects policies that silently omit required operational signals.

        Raises:
            TypeError: If a required signal is not a `Indicator`.
            ValueError: If retention is under one day, no signals are required,
                signals repeat, a redacted field name is empty, or an alert
                threshold is unnamed, not finite or negative.
        """
        if self.retention_days < 1:
            raise errors.InputError("retention_days must be at least 1")
        if not self.required_signals:
            raise errors.InputError("required_signals must not be empty")
        if any(
            not isinstance(signal, Indicator)
            for signal in self.required_signals
        ):
            raise errors.KindError(
                "required_signals must contain Indicator values"
            )
        if len(self.required_signals) != len(set(self.required_signals)):
            raise errors.InputError("required_signals must be unique")
        if any(not field.strip() for field in self.redacted_fields):
            raise errors.InputError(
                "redacted_fields must not contain empty values"
            )
        if any(
            not name.strip() or not math.isfinite(value) or value < 0
            for name, value in self.alert_thresholds.items()
        ):
            raise errors.InputError(
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
    metrics: Sequence[Sample],
    events: Sequence[Inference],
    policy: Requirements,
) -> str:
    """Returns deterministic redacted observability evidence.

    Args:
        metrics: The metric samples.
        events: The inference events.
        policy: The observability policy.

    Returns:
        The compact, key-sorted ASCII JSON of the policy, metrics and events.
    """
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
