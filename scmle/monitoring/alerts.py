"""Idempotent alert delivery for SLO and incident workflows."""

import abc
from collections.abc import Mapping
import dataclasses
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib import error as url_error
from urllib import parse as url_parse
from urllib import request as url_request

from scmle import durability


class AlertDeliveryError(RuntimeError):
    """A sink could not deliver an alert after its bounded retry budget."""


@dataclasses.dataclass(frozen=True, slots=True)
class Alert:
    """One operator-facing alert bound to evidence."""

    alert_id: str
    route: str
    severity: str
    title: str
    details: Mapping[str, Any]
    evidence_digest: str
    created_at: str

    def __post_init__(self) -> None:
        """Validate identity, route, and timestamp fields."""
        for name in (
            "alert_id",
            "route",
            "severity",
            "title",
            "evidence_digest",
            "created_at",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        try:
            datetime.datetime.fromisoformat(
                self.created_at.replace("Z", "+00:00")
            )
        except ValueError as error:
            raise ValueError("created_at must be ISO-8601") from error
        if not isinstance(self.details, Mapping):
            raise TypeError("details must be a mapping")
        object.__setattr__(self, "details", dict(self.details))

    def to_dict(self) -> dict[str, object]:
        """Return JSON-native alert evidence."""
        return {
            "alert_id": self.alert_id,
            "route": self.route,
            "severity": self.severity,
            "title": self.title,
            "details": dict(self.details),
            "evidence_digest": self.evidence_digest,
            "created_at": self.created_at,
        }

    @classmethod
    def from_evidence(
        cls,
        route: str,
        severity: str,
        title: str,
        details: Mapping[str, Any],
        evidence: Mapping[str, Any],
        created_at: str,
    ) -> "Alert":
        """Build a deterministic ID from the complete alert evidence."""
        encoded_evidence = json.dumps(
            evidence, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        evidence_digest = hashlib.sha256(encoded_evidence).hexdigest()
        identity = json.dumps(
            {
                "route": route,
                "severity": severity,
                "title": title,
                "details": dict(details),
                "evidence_digest": evidence_digest,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        alert_id = hashlib.sha256(identity).hexdigest()
        return cls(
            alert_id=alert_id,
            route=route,
            severity=severity,
            title=title,
            details=details,
            evidence_digest=evidence_digest,
            created_at=created_at,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class AlertDelivery:
    """Auditable outcome of one sink delivery attempt."""

    alert_id: str
    route: str
    status: str
    attempts: int
    reason: str
    observed_at: str

    def __post_init__(self) -> None:
        """Validate delivery state."""
        if self.status not in {"delivered", "failed", "deduplicated"}:
            raise ValueError("unsupported alert delivery status")
        if self.attempts < 0:
            raise ValueError("attempts must not be negative")
        if not self.reason.strip():
            raise ValueError("reason must not be empty")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-native delivery evidence."""
        return dataclasses.asdict(self)


class AlertSink(abc.ABC):
    """Polymorphic destination for one alert route."""

    @abc.abstractmethod
    def deliver(self, alert: Alert) -> AlertDelivery:
        """Deliver an alert and return a bounded outcome."""


class JsonlAlertSink(AlertSink):
    """Append-only local sink for operator pickup and test evidence."""

    def __init__(self, path: str | Path) -> None:
        """Bind a local alert journal."""
        self.path = Path(path)

    def deliver(self, alert: Alert) -> AlertDelivery:
        """Append one alert record and flush it durably."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        delivered_at = datetime.datetime.now(datetime.UTC).isoformat()
        record = {
            "type": "alert",
            "alert": alert.to_dict(),
            "delivered_at": delivered_at,
        }
        with durability.path_lock(self.path):
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            durability.sync_directory(self.path.parent)
        return AlertDelivery(
            alert.alert_id,
            alert.route,
            "delivered",
            1,
            "local journal append committed",
            delivered_at,
        )


class WebhookAlertSink(AlertSink):
    """Deliver JSON alerts to an HTTPS-compatible pager or incident gateway."""

    def __init__(
        self,
        url: str,
        *,
        token: str | None = None,
        timeout_seconds: float = 5.0,
        max_attempts: int = 3,
        backoff_seconds: float = 0.1,
        allow_insecure_localhost: bool = False,
    ) -> None:
        """Validate the bounded webhook delivery policy."""
        parsed = url_parse.urlparse(url)
        if parsed.scheme != "https":
            local = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
            if not (
                allow_insecure_localhost and parsed.scheme == "http" and local
            ):
                raise ValueError("webhook URL must use HTTPS")
        if timeout_seconds <= 0 or max_attempts < 1 or backoff_seconds < 0:
            raise ValueError("webhook retry and timeout bounds are invalid")
        self.url = url
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds

    def deliver(self, alert: Alert) -> AlertDelivery:
        """POST an alert with bounded retries and no false success."""
        body = json.dumps(alert.to_dict(), sort_keys=True).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        last_reason = "webhook did not respond"
        for attempt in range(1, self.max_attempts + 1):
            request = url_request.Request(
                self.url, data=body, headers=headers, method="POST"
            )
            try:
                with url_request.urlopen(
                    request,
                    timeout=self.timeout_seconds,  # nosec B310 - URL scheme is validated in the constructor.
                ) as response:
                    status = int(response.status)
                    if 200 <= status < 300:
                        return AlertDelivery(
                            alert.alert_id,
                            alert.route,
                            "delivered",
                            attempt,
                            f"webhook returned HTTP {status}",
                            datetime.datetime.now(datetime.UTC).isoformat(),
                        )
                    last_reason = f"webhook returned HTTP {status}"
            except (
                url_error.HTTPError,
                url_error.URLError,
                TimeoutError,
            ) as error:
                last_reason = f"webhook delivery failed: {type(error).__name__}"
            if attempt < self.max_attempts and self.backoff_seconds:
                time.sleep(min(self.backoff_seconds * attempt, 1.0))
        raise AlertDeliveryError(
            f"alert {alert.alert_id} was not delivered after "
            f"{self.max_attempts} attempts: {last_reason}"
        )


class AlertDispatcher:
    """Route alerts once and persist delivery outcomes for replay safety."""

    def __init__(
        self, sinks: Mapping[str, AlertSink], journal: str | Path
    ) -> None:
        """Bind routes and an append-only delivery journal."""
        if not sinks:
            raise ValueError("at least one alert sink is required")
        if any(not route.strip() for route in sinks):
            raise ValueError("alert routes must not be empty")
        self.sinks = dict(sinks)
        self.journal = Path(journal)
        self.lock = durability.path_lock(self.journal)

    def dispatch(self, alert: Alert) -> AlertDelivery:
        """Deliver an alert or return a prior successful delivery.

        The journal check, the delivery and the journal write are one step
        per journal within this process, so concurrent threads cannot both
        send the same alert.
        """
        with self.lock:
            return self.dispatch_locked(alert)

    def dispatch_locked(self, alert: Alert) -> AlertDelivery:
        """Dispatches one alert while holding the journal lock."""
        previous = self.successful_alerts()
        if alert.alert_id in previous:
            return AlertDelivery(
                alert.alert_id,
                alert.route,
                "deduplicated",
                0,
                "successful delivery already recorded",
                datetime.datetime.now(datetime.UTC).isoformat(),
            )
        sink = self.sinks.get(alert.route)
        if sink is None:
            delivery = AlertDelivery(
                alert.alert_id,
                alert.route,
                "failed",
                0,
                "no sink is configured for the alert route",
                datetime.datetime.now(datetime.UTC).isoformat(),
            )
        else:
            try:
                delivery = sink.deliver(alert)
            except AlertDeliveryError as error:
                delivery = AlertDelivery(
                    alert.alert_id,
                    alert.route,
                    "failed",
                    0,
                    str(error),
                    datetime.datetime.now(datetime.UTC).isoformat(),
                )
        self.append(alert, delivery)
        return delivery

    def successful_alerts(self) -> set[str]:
        """Return alert IDs with a durable successful delivery record."""
        if not self.journal.exists():
            return set()
        successful: set[str] = set()
        for line in self.journal.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("delivery", {}).get("status") == "delivered":
                successful.add(str(record["alert"]["alert_id"]))
        return successful

    def append(self, alert: Alert, delivery: AlertDelivery) -> None:
        """Persist one alert and its delivery outcome durably."""
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        record = {"alert": alert.to_dict(), "delivery": delivery.to_dict()}
        with self.lock:
            with self.journal.open(
                "a", encoding="utf-8", newline="\n"
            ) as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            durability.sync_directory(self.journal.parent)


__all__ = [
    "Alert",
    "AlertDelivery",
    "AlertDeliveryError",
    "AlertDispatcher",
    "AlertSink",
    "JsonlAlertSink",
    "WebhookAlertSink",
]
