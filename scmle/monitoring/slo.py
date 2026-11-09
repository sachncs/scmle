"""Service-level objectives for measurable ML operations."""

from collections.abc import Sequence
import dataclasses
import datetime
import enum
import math
from typing import Any

from scmle.contracts import observability


class SLOStatus(enum.StrEnum):
    """Outcome of evaluating one SLO window.

    Attributes:
        PASS: The worst in-window value is within the target.
        BREACHED: The worst in-window value exceeds the target.
        INSUFFICIENT_DATA: Fewer in-window samples than required.
        INVALID: Samples from the future make the window unusable.
    """

    PASS = "pass"  # noqa: S105 -- operational status, not a credential  # nosec B105
    BREACHED = "breached"
    INSUFFICIENT_DATA = "insufficient_data"
    INVALID = "invalid"


def require_text(value: str, field_name: str) -> str:
    """Returns a non-empty SLO field.

    Args:
        value: The field value.
        field_name: Name used in the error message.

    Returns:
        The value with surrounding whitespace removed.

    Raises:
        ValueError: If the value is not a string or is blank.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value.strip()


def parse_timestamp(value: str, field_name: str) -> datetime.datetime:
    """Parses a timezone-aware timestamp for SLO window checks.

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
        raise ValueError(f"{field_name} must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information")
    return parsed


@dataclasses.dataclass(frozen=True, slots=True)
class SLOTarget:
    """One upper-bound operational objective and its incident ownership.

    Attributes:
        name: Objective name.
        metric_name: Name of the metric samples to evaluate.
        kind: Kind of signal the samples carry.
        maximum: Largest acceptable value.
        window_days: Length of the evaluation window, in days.
        minimum_samples: Fewest in-window samples for a decision.
        owner: Who answers for the objective.
        runbook: Where the response procedure is described.
        alert_route: Where alerts are sent.
    """

    name: str
    metric_name: str
    kind: observability.SignalKind
    maximum: float
    window_days: int
    minimum_samples: int
    owner: str
    runbook: str
    alert_route: str

    def __post_init__(self) -> None:
        """Validates the measurable target and response path.

        Raises:
            ValueError: If a text field is blank, `maximum` is negative or not
                finite, or `window_days` or `minimum_samples` is below 1.
        """
        for field_name in (
            "name",
            "metric_name",
            "owner",
            "runbook",
            "alert_route",
        ):
            require_text(getattr(self, field_name), field_name)
        if not isinstance(self.kind, observability.SignalKind):
            object.__setattr__(
                self, "kind", observability.SignalKind(self.kind)
            )
        if not math.isfinite(self.maximum) or self.maximum < 0:
            raise ValueError("maximum must be finite and non-negative")
        if self.window_days < 1:
            raise ValueError("window_days must be at least 1")
        if self.minimum_samples < 1:
            raise ValueError("minimum_samples must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        """Returns deterministic SLO policy metadata.

        Returns:
            A dict of all fields, with the kind as its value.
        """
        return {
            "name": self.name,
            "metric_name": self.metric_name,
            "kind": self.kind.value,
            "maximum": self.maximum,
            "window_days": self.window_days,
            "minimum_samples": self.minimum_samples,
            "owner": self.owner,
            "runbook": self.runbook,
            "alert_route": self.alert_route,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class SLOEvaluation:
    """Auditable result of one SLO evaluation window.

    Attributes:
        target: The objective evaluated.
        observed_at: Evaluation time as given.
        status: The outcome.
        sample_count: Samples matching the target's metric and kind.
        in_window_count: Matching samples inside the window.
        observed_maximum: Largest in-window value, or None without a decision.
        error_budget_remaining: Share of the allowed maximum not yet used, in
            [0, 1], or None without a decision.
        reason: One sentence explaining the status.
    """

    target: SLOTarget
    observed_at: str
    status: SLOStatus
    sample_count: int
    in_window_count: int
    observed_maximum: float | None
    error_budget_remaining: float | None
    reason: str

    def __post_init__(self) -> None:
        """Validates the result and its measurable bounds.

        Raises:
            TypeError: If `target` is not an `SLOTarget`.
            ValueError: If a count is negative, the in-window count exceeds the
                sample count, the observed maximum is negative or not finite,
                the error budget is outside [0, 1], or the reason is blank.
        """
        if not isinstance(self.target, SLOTarget):
            raise TypeError("target must be SLOTarget")
        parse_timestamp(self.observed_at, "observed_at")
        if not isinstance(self.status, SLOStatus):
            object.__setattr__(self, "status", SLOStatus(self.status))
        if self.sample_count < 0 or self.in_window_count < 0:
            raise ValueError("sample counts must not be negative")
        if self.in_window_count > self.sample_count:
            raise ValueError("in_window_count cannot exceed sample_count")
        if self.observed_maximum is not None and (
            not math.isfinite(self.observed_maximum)
            or self.observed_maximum < 0
        ):
            raise ValueError("observed_maximum must be finite and non-negative")
        if (
            self.error_budget_remaining is not None
            and not 0 <= self.error_budget_remaining <= 1
        ):
            raise ValueError("error_budget_remaining must be between 0 and 1")
        require_text(self.reason, "reason")

    @property
    def incident_required(self) -> bool:
        """Whether the operator must follow the target's incident path."""
        return self.status in {
            SLOStatus.BREACHED,
            SLOStatus.INSUFFICIENT_DATA,
            SLOStatus.INVALID,
        }

    def to_dict(self) -> dict[str, Any]:
        """Returns an operator-facing evaluation record.

        Returns:
            A dict with the target, status, counts, observed maximum, error
            budget, whether an incident is required, and the owner, runbook,
            alert route and reason.
        """
        return {
            "target": self.target.to_dict(),
            "observed_at": self.observed_at,
            "status": self.status.value,
            "sample_count": self.sample_count,
            "in_window_count": self.in_window_count,
            "observed_maximum": self.observed_maximum,
            "error_budget_remaining": self.error_budget_remaining,
            "incident_required": self.incident_required,
            "owner": self.target.owner,
            "runbook": self.target.runbook,
            "alert_route": self.target.alert_route,
            "reason": self.reason,
        }


def evaluate(
    target: SLOTarget,
    samples: Sequence[observability.MetricSample],
    *,
    observed_at: str,
) -> SLOEvaluation:
    """Evaluates one target over its declared time window.

    The maximum observed value is used deliberately: a single breach is not
    averaged away. Missing samples are an explicit operational state, not a
    passing result.

    Args:
        target: The objective.
        samples: Metric samples; only those with the target's metric name
            and kind count.
        observed_at: Evaluation time, an ISO-8601 timestamp with timezone.

    Returns:
        The evaluation: `invalid` if any matching sample is later than
        `observed_at`, `insufficient_data` if fewer than `minimum_samples`
        fall in the window, otherwise `pass` or `breached` by comparing the
        window maximum with the target.

    Raises:
        TypeError: If `target` is not an `SLOTarget`.
    """
    if not isinstance(target, SLOTarget):
        raise TypeError("target must be SLOTarget")
    observed_timestamp = parse_timestamp(observed_at, "observed_at")
    window_start = observed_timestamp - datetime.timedelta(
        days=target.window_days
    )
    matching = tuple(
        sample
        for sample in samples
        if sample.name == target.metric_name and sample.kind is target.kind
    )
    future = tuple(
        sample
        for sample in matching
        if parse_timestamp(sample.timestamp, "metric.timestamp")
        > observed_timestamp
    )
    in_window = tuple(
        sample
        for sample in matching
        if window_start
        <= parse_timestamp(sample.timestamp, "metric.timestamp")
        <= observed_timestamp
    )
    if future:
        return SLOEvaluation(
            target=target,
            observed_at=observed_at,
            status=SLOStatus.INVALID,
            sample_count=len(matching),
            in_window_count=len(in_window),
            observed_maximum=None,
            error_budget_remaining=None,
            reason="future metric samples cannot support an SLO decision",
        )
    if len(in_window) < target.minimum_samples:
        return SLOEvaluation(
            target=target,
            observed_at=observed_at,
            status=SLOStatus.INSUFFICIENT_DATA,
            sample_count=len(matching),
            in_window_count=len(in_window),
            observed_maximum=None,
            error_budget_remaining=None,
            reason=(
                f"{len(in_window)} in-window samples; "
                f"{target.minimum_samples} required"
            ),
        )
    observed_maximum = max(sample.value for sample in in_window)
    remaining = max(0.0, 1.0 - observed_maximum / max(target.maximum, 1e-12))
    status = (
        SLOStatus.PASS
        if observed_maximum <= target.maximum
        else SLOStatus.BREACHED
    )
    reason = (
        f"maximum {observed_maximum:g} is within {target.maximum:g}"
        if status is SLOStatus.PASS
        else f"maximum {observed_maximum:g} exceeds {target.maximum:g}"
    )
    return SLOEvaluation(
        target=target,
        observed_at=observed_at,
        status=status,
        sample_count=len(matching),
        in_window_count=len(in_window),
        observed_maximum=observed_maximum,
        error_budget_remaining=remaining,
        reason=reason,
    )


__all__ = [
    "SLOEvaluation",
    "SLOStatus",
    "SLOTarget",
    "evaluate",
    "parse_timestamp",
    "require_text",
]
