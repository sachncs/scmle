"""Deterministic alert routing for observability threshold breaches."""

from collections.abc import Mapping
import dataclasses
import hashlib
import json
from typing import Any

from scml.monitoring import observability as observability_monitor


@dataclasses.dataclass(frozen=True, slots=True)
class Alert:
    """One deduplicable alert emitted from an evaluated threshold breach."""

    alert_id: str
    metric: str
    observed: float
    threshold: float
    unit: str
    observed_at: str
    route: str

    def __post_init__(self) -> None:
        """Validate the alert envelope before delivery."""
        for name in ("alert_id", "metric", "unit", "observed_at", "route"):
            if (
                not isinstance(getattr(self, name), str)
                or not getattr(self, name).strip()
            ):
                raise ValueError(f"{name} must not be empty")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible alert."""
        return {
            "alert_id": self.alert_id,
            "metric": self.metric,
            "observed": self.observed,
            "threshold": self.threshold,
            "unit": self.unit,
            "observed_at": self.observed_at,
            "route": self.route,
        }


def alert_id(
    metric: str, observed_at: str, threshold: float, route: str
) -> str:
    """Return a stable identifier for one breach occurrence."""
    payload = json.dumps(
        {
            "metric": metric,
            "observed_at": observed_at,
            "threshold": threshold,
            "route": route,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def route(
    evaluation: Any,
    routes: Mapping[str, str],
) -> tuple[Alert, ...]:
    """Map evaluated breaches to explicit destinations with deduplication.

    A breach without a configured route raises instead of being silently
    dropped. The same metric, observation time, threshold, and route produce
    the same alert identifier, allowing downstream incident systems to handle
    retries idempotently.
    """
    if not isinstance(
        evaluation, observability_monitor.ObservabilityEvaluation
    ):
        raise TypeError("evaluation must be ObservabilityEvaluation")
    alerts: list[Alert] = []
    for breach in evaluation.threshold_breaches:
        metric = str(breach["metric"])
        destination = routes.get(metric)
        if destination is None or not destination.strip():
            raise ValueError(f"no alert route configured for metric '{metric}'")
        observed = float(breach["observed"])
        threshold = float(breach["threshold"])
        alerts.append(
            Alert(
                alert_id=alert_id(
                    metric, evaluation.observed_at, threshold, destination
                ),
                metric=metric,
                observed=observed,
                threshold=threshold,
                unit=str(breach["unit"]),
                observed_at=evaluation.observed_at,
                route=destination,
            )
        )
    return tuple(alerts)
