"""Idempotent alert delivery for SLO and incident workflows."""

import abc
from collections.abc import Mapping
import dataclasses
import datetime
import hashlib
import http
import json
import os
from pathlib import Path
import time
from typing import Any
from typing import ClassVar
from urllib import parse
import urllib.error
import urllib.request

from scmle import durability
from scmle.contracts import errors
from scmle.contracts import records


class DeliveryError(RuntimeError):
    """A sink could not deliver an alert after its bounded retry budget."""


@dataclasses.dataclass(frozen=True, slots=True)
class Alert:
    """One operator-facing alert bound to evidence.

    Attributes:
        alert_id: Deterministic identity of the alert.
        route: Name of the route (sink) it goes to.
        severity: Severity label.
        title: Short title.
        details: Further JSON-compatible facts.
        evidence_digest: Digest of the evidence the alert is bound to.
        created_at: ISO-8601 creation time.
        KEYS: Version-1 keys to the written ones (`identifier`,
            `digests.evidence`, `created`); `downgrade` restores version 1.
            The id is the digest of a fixed version-1 literal
            (`from_evidence`), so renaming keys leaves every id unchanged.
    """

    KEYS: ClassVar[records.Migration] = records.nest(
        {"alert_id": "identifier", "created_at": "created"},
        groups={"digests": {"evidence_digest": "evidence"}},
    )

    alert_id: str
    route: str
    severity: str
    title: str
    details: Mapping[str, Any]
    evidence_digest: str
    created_at: str

    def __post_init__(self) -> None:
        """Validates identity, route, and timestamp fields.

        Raises:
            ValueError: If a text field is blank or `created_at` is not
                ISO-8601.
            TypeError: If `details` is not a mapping.
        """
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
                raise errors.InputError(f"{name} must not be empty")
        try:
            datetime.datetime.fromisoformat(self.created_at)
        except ValueError as error:
            raise errors.InputError("created_at must be ISO-8601") from error
        if not isinstance(self.details, Mapping):
            raise errors.KindError("details must be a mapping")
        object.__setattr__(self, "details", dict(self.details))

    def to_dict(self) -> dict[str, object]:
        """Returns JSON-native alert evidence.

        Returns:
            A dict of all fields.
        """
        return self.KEYS.upgrade(
            {
                "alert_id": self.alert_id,
                "route": self.route,
                "severity": self.severity,
                "title": self.title,
                "details": dict(self.details),
                "evidence_digest": self.evidence_digest,
                "created_at": self.created_at,
            }
        )

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
        """Builds a deterministic ID from the complete alert evidence.

        Args:
            route: Name of the route the alert goes to.
            severity: Severity label.
            title: Short title.
            details: Further JSON-compatible facts.
            evidence: The evidence, digested into the alert.
            created_at: ISO-8601 creation time.

        Returns:
            The alert, whose ID is the digest of the route, severity, title,
            details and evidence digest, so the same evidence gives the same ID.
        """
        encoded_evidence = json.dumps(
            evidence, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        evidence_digest = hashlib.sha256(encoded_evidence).hexdigest()
        # The version-1 literal: the id must not move when keys are renamed.
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
class Delivery:
    """Auditable outcome of one sink delivery attempt.

    Attributes:
        alert_id: The alert delivered.
        route: Its route.
        status: `delivered`, `failed` or `deduplicated`.
        attempts: Delivery attempts made.
        reason: One sentence explaining the status.
        observed_at: ISO-8601 time of the outcome.
        KEYS: Version-1 keys to the written ones (`alert`, `observed`);
            `downgrade` restores version 1.
    """

    KEYS: ClassVar[records.Migration] = records.Migration(
        renames={"alert_id": "alert", "observed_at": "observed"}
    )

    alert_id: str
    route: str
    status: str
    attempts: int
    reason: str
    observed_at: str

    def __post_init__(self) -> None:
        """Validates delivery state.

        Raises:
            ValueError: If the status is unsupported, `attempts` is negative or
                the reason is blank.
        """
        if self.status not in {"delivered", "failed", "deduplicated"}:
            raise errors.InputError("unsupported alert delivery status")
        if self.attempts < 0:
            raise errors.InputError("attempts must not be negative")
        if not self.reason.strip():
            raise errors.InputError("reason must not be empty")

    def to_dict(self) -> dict[str, object]:
        """Returns JSON-native delivery evidence.

        Returns:
            A dict of all fields, with the written keys.
        """
        return self.KEYS.upgrade(dataclasses.asdict(self))


class Sink(abc.ABC):
    """Polymorphic destination for one alert route."""

    @abc.abstractmethod
    def deliver(self, alert: Alert) -> Delivery:
        """Delivers an alert and returns a bounded outcome.

        Args:
            alert: The alert.

        Returns:
            The outcome of the delivery.
        """


class Jsonl(Sink):
    """Append-only local sink for operator pickup and test evidence.

    Attributes:
        path: The journal file.
    """

    def __init__(self, path: str | Path) -> None:
        """Binds a local alert journal.

        Args:
            path: The journal file; its directory is created on delivery.
        """
        self.path = Path(path)

    def deliver(self, alert: Alert) -> Delivery:
        """Appends one alert record and flushes it durably.

        Args:
            alert: The alert.

        Returns:
            A `delivered` outcome after one attempt.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        delivered_at = datetime.datetime.now(datetime.UTC).isoformat()
        record = {
            "type": "alert",
            "alert": alert.to_dict(),
            "delivered_at": delivered_at,
        }
        with durability.latch(self.path):
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            durability.sync_directory(self.path.parent)
        return Delivery(
            alert.alert_id,
            alert.route,
            "delivered",
            1,
            "local journal append committed",
            delivered_at,
        )


class Pager(Sink):
    """Deliver JSON alerts to an HTTPS-compatible pager or incident gateway.

    Attributes:
        url: Webhook URL.
        token: Bearer token sent with each request, if any.
        timeout_seconds: Seconds to wait for each request.
        max_attempts: Most delivery attempts.
        backoff_seconds: Base delay between attempts.
    """

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
        """Validates the bounded webhook delivery policy.

        Args:
            url: Webhook URL; must be HTTPS.
            token: Bearer token to send, or None.
            timeout_seconds: Seconds to wait for each request.
            max_attempts: Most delivery attempts.
            backoff_seconds: Base delay between attempts; the wait grows with
                the attempt number and is capped at one second.
            allow_insecure_localhost: Allow plain HTTP to a loopback host.

        Raises:
            ValueError: If the URL is not HTTPS (or allowed loopback HTTP), the
                timeout is not positive, `max_attempts` is below 1 or the
                backoff is negative.
        """
        parsed = parse.urlparse(url)
        if parsed.scheme != "https":
            local = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
            if not (
                allow_insecure_localhost and parsed.scheme == "http" and local
            ):
                raise errors.InputError("webhook URL must use HTTPS")
        if timeout_seconds <= 0 or max_attempts < 1 or backoff_seconds < 0:
            raise errors.InputError(
                "webhook retry and timeout bounds are invalid"
            )
        self.url = url
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds

    def deliver(self, alert: Alert) -> Delivery:
        """POST an alert with bounded retries and no false success.

        Args:
            alert: The alert.

        Returns:
            A `delivered` outcome if some attempt gets an HTTP 2xx answer.

        Raises:
            DeliveryError: If no attempt succeeds within `max_attempts`.
        """
        body = json.dumps(alert.to_dict(), sort_keys=True).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        last_reason = "webhook did not respond"
        for attempt in range(1, self.max_attempts + 1):
            request = urllib.request.Request(  # noqa: S310 -- the scheme is validated in the constructor
                self.url, data=body, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(  # noqa: S310 -- the scheme is validated in the constructor
                    request,
                    timeout=self.timeout_seconds,  # nosec B310 - URL scheme is validated in the constructor.
                ) as response:
                    status = int(response.status)
                    if (
                        http.HTTPStatus.OK
                        <= status
                        < http.HTTPStatus.MULTIPLE_CHOICES
                    ):
                        return Delivery(
                            alert.alert_id,
                            alert.route,
                            "delivered",
                            attempt,
                            f"webhook returned HTTP {status}",
                            datetime.datetime.now(datetime.UTC).isoformat(),
                        )
                    last_reason = f"webhook returned HTTP {status}"
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
            ) as error:
                # handled: failure kept as last_reason; DeliveryError after
                # retries
                last_reason = f"webhook delivery failed: {type(error).__name__}"
            if attempt < self.max_attempts and self.backoff_seconds:
                time.sleep(min(self.backoff_seconds * attempt, 1.0))
        raise DeliveryError(
            f"alert {alert.alert_id} was not delivered after "
            f"{self.max_attempts} attempts: {last_reason}"
        )


class Dispatcher:
    """Routes alerts once and persists delivery outcomes for replay safety.

    Attributes:
        sinks: Sink by route name.
        journal: The delivery journal file.
        lock: Serializes dispatching per journal.
    """

    def __init__(self, sinks: Mapping[str, Sink], journal: str | Path) -> None:
        """Binds routes and an append-only delivery journal.

        Args:
            sinks: Sink by route name.
            journal: The delivery journal file.

        Raises:
            ValueError: If there are no sinks or a route name is blank.
        """
        if not sinks:
            raise errors.InputError("at least one alert sink is required")
        if any(not route.strip() for route in sinks):
            raise errors.InputError("alert routes must not be empty")
        self.sinks = dict(sinks)
        self.journal = Path(journal)
        self.lock = durability.latch(self.journal)

    def dispatch(self, alert: Alert) -> Delivery:
        """Delivers an alert unless a successful delivery is already journaled.

        The journal check, the delivery and the journal write are one step per
        journal within this process, so concurrent threads cannot both send the
        same alert.

        Args:
            alert: The alert.

        Returns:
            The delivery outcome: `deduplicated` if a `delivered` record for
            the alert is already in the journal, `failed` if the route has no
            sink or the sink raised `DeliveryError`, otherwise the
            outcome the sink returned.
        """
        with self.lock:
            return self.relay(alert)

    def relay(self, alert: Alert) -> Delivery:
        """Dispatches one alert while the caller holds the journal lock.

        Args:
            alert: The alert.

        Returns:
            The delivery outcome, which is also appended to the journal unless
            the alert was deduplicated.
        """
        previous = self.delivered()
        if alert.alert_id in previous:
            return Delivery(
                alert.alert_id,
                alert.route,
                "deduplicated",
                0,
                "successful delivery already recorded",
                datetime.datetime.now(datetime.UTC).isoformat(),
            )
        sink = self.sinks.get(alert.route)
        if sink is None:
            delivery = Delivery(
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
            except DeliveryError as error:
                # handled: delivery error becomes journaled 'failed'
                # Delivery
                delivery = Delivery(
                    alert.alert_id,
                    alert.route,
                    "failed",
                    0,
                    str(error),
                    datetime.datetime.now(datetime.UTC).isoformat(),
                )
        self.append(alert, delivery)
        return delivery

    def delivered(self) -> set[str]:
        """Returns alert IDs with a durable successful delivery record.

        Returns:
            The IDs of alerts whose journal record has status `delivered`; empty
            if there is no journal. A record written before the keys were
            renamed names its alert `alert_id`.
        """
        if not self.journal.exists():
            return set()
        successful: set[str] = set()
        for line in self.journal.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("delivery", {}).get("status") == "delivered":
                alert = record["alert"]
                successful.add(
                    str(alert.get("identifier", alert.get("alert_id")))
                )
        return successful

    def append(self, alert: Alert, delivery: Delivery) -> None:
        """Persists one alert and its delivery outcome durably.

        Args:
            alert: The alert.
            delivery: Its delivery outcome.
        """
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
    "Delivery",
    "DeliveryError",
    "Dispatcher",
    "Jsonl",
    "Pager",
    "Sink",
]
