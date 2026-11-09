"""Pager and ticket webhooks: signed payloads, bounded retry, injected send.

A `WebhookSender` posts one JSON document. The body is signed with
HMAC-SHA256 over ``"<timestamp>.<body>"`` and sent as
``X-Scmle-Signature: sha256=<hex>`` beside ``X-Scmle-Timestamp``, so a
receiver can reject a forged or replayed payload (`verify_signature`). One
idempotency key is used for every attempt of a notification, so a retry after
a timeout does not page twice.

Delivery goes through a ``transport`` callable that takes a `Request` and
returns the HTTP status or raises ``OSError``. Tests pass a recorder; the
default sends over HTTPS with the standard library. 5xx, 429 and transport
errors are retried up to ``max_attempts`` with exponential backoff; other 4xx
are final. The outcome is returned, never swallowed: a failed page is a
`Delivery` with ``delivered=False``.
"""

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
import hashlib
import hmac
import json
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

SIGNATURE_HEADER = "X-Scmle-Signature"
TIMESTAMP_HEADER = "X-Scmle-Timestamp"
IDEMPOTENCY_HEADER = "X-Scmle-Idempotency-Key"
KINDS = ("pager", "ticket")
SEVERITIES = ("info", "warning", "critical")
RETRY_STATUSES = frozenset({408, 425, 429})
MAX_BODY_BYTES = 64 * 1024


@dataclasses.dataclass(frozen=True, slots=True)
class Request:
    """One POST to send."""

    url: str
    headers: Mapping[str, str]
    body: bytes


Transport = Callable[[Request], int]


@dataclasses.dataclass(frozen=True, slots=True)
class Delivery:
    """The outcome of one notification.

    Attributes:
        kind: ``pager`` or ``ticket``.
        delivered: Whether a 2xx answer was received.
        attempts: Sends made.
        http_status: Last status (None if the last attempt raised).
        reason: Short text for the outcome (``delivered``, ``HTTP <status>``,
            the name of the transport error, or that no sender was
            configured), with the attempt count added after a final failure
            that used every attempt; never a credential.
        idempotency_key: The key sent on every attempt; empty if no sender
            was configured.
    """

    kind: str
    delivered: bool
    attempts: int
    http_status: int | None
    reason: str
    idempotency_key: str

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready record.

        Returns:
            The delivery's fields as a dict.
        """
        return dataclasses.asdict(self)


def sign(secret: str | bytes, timestamp: str, body: bytes) -> str:
    """``sha256=<hex>`` over ``<timestamp>.<body>``.

    Args:
        secret: The shared secret.
        timestamp: The timestamp sent in the timestamp header.
        body: The request body.

    Returns:
        The HMAC-SHA256 of the timestamp, a dot and the body, prefixed with
        ``sha256=``.
    """
    key = secret.encode() if isinstance(secret, str) else secret
    mac = hmac.new(key, timestamp.encode() + b"." + body, hashlib.sha256)
    return "sha256=" + mac.hexdigest()


def verify_signature(
    secret: str | bytes,
    headers: Mapping[str, str],
    body: bytes,
    now: float,
    tolerance_seconds: float = 300.0,
) -> bool:
    """What a receiver checks: a valid signature on a recent timestamp.

    Args:
        secret: The shared secret.
        headers: The request headers; the timestamp and signature are looked
            up under their exact header names.
        body: The request body.
        now: The receiver's current time, in seconds.
        tolerance_seconds: Largest accepted distance in seconds between
            `now` and the timestamp, in the past or the future.

    Returns:
        True if the timestamp parses as a number within the tolerance of
        `now` and the signature matches; False if the timestamp is missing,
        not a number or too far from `now`, or the signature differs.
    """
    timestamp = headers.get(TIMESTAMP_HEADER, "")
    given = headers.get(SIGNATURE_HEADER, "")
    try:
        age = abs(now - float(timestamp))
    except ValueError:
        return False
    if age > tolerance_seconds:
        return False
    return hmac.compare_digest(given, sign(secret, timestamp, body))


def urllib_transport(timeout_seconds: float = 5.0) -> Transport:
    """A transport that POSTs over HTTPS with the standard library.

    Args:
        timeout_seconds: Timeout of each request.

    Returns:
        A function that sends a request and returns the HTTP status code,
        including error codes. When called, it raises `ValueError` for a URL
        that is not HTTPS and `OSError` when the connection fails.
    """

    def send(request: Request) -> int:
        parsed = urllib.parse.urlparse(request.url)
        if parsed.scheme != "https":
            raise ValueError("webhook URL must use HTTPS")
        outgoing = urllib.request.Request(
            request.url,
            data=request.body,
            headers=dict(request.headers),
            method="POST",
        )
        try:
            with urllib.request.urlopen(  # nosec B310
                outgoing, timeout=timeout_seconds
            ) as reply:
                return int(reply.status)
        except urllib.error.HTTPError as error:
            return int(error.code)
        except urllib.error.URLError as error:
            raise OSError(str(error.reason)) from error

    return send


class WebhookSender:
    """Sends signed notifications of one kind to one endpoint."""

    def __init__(
        self,
        kind: str,
        url: str,
        secret: str | bytes,
        transport: Transport | None = None,
        *,
        max_attempts: int = 3,
        backoff_seconds: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Creates a sender.

        Args:
            kind: ``pager`` or ``ticket``.
            url: HTTPS endpoint (the default transport refuses others).
            secret: HMAC key shared with the receiver.
            transport: Sends a `Request`; default is HTTPS.
            max_attempts: Total sends allowed, at least one.
            backoff_seconds: First wait; doubled after each failure.
            sleep: Waits between attempts.
            clock: Epoch seconds, for the timestamp header.

        Raises:
            ValueError: For an unknown kind, an empty secret or bad bounds.
        """
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if not secret:
            raise ValueError("a signing secret is required")
        if max_attempts < 1 or backoff_seconds < 0:
            raise ValueError("max_attempts >= 1 and backoff >= 0 required")
        self.kind = kind
        self.url = url
        self.secret = secret
        self.transport = transport or urllib_transport()
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.sleep = sleep
        self.clock = clock

    def payload(self, notification: Mapping[str, Any]) -> bytes:
        """Builds the canonical JSON body of a notification.

        Args:
            notification: The notification, which must carry a known
                ``severity`` and be JSON-serialisable. A ``kind`` entry in it
                replaces this sender's kind in the body.

        Returns:
            The compact, key-sorted JSON of the notification with a ``kind``
            entry holding this sender's kind, as ASCII-only UTF-8 bytes.

        Raises:
            ValueError: If the severity is unknown or the body is larger than
                64 KiB.
            TypeError: If the notification cannot be serialised to JSON.
        """
        severity = notification.get("severity", "")
        if severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}")
        body = json.dumps(
            {"kind": self.kind, **notification},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(body) > MAX_BODY_BYTES:
            raise ValueError("notification is too large")
        return body

    def send(self, notification: Mapping[str, Any]) -> Delivery:
        """Posts the notification, retrying what is worth retrying.

        Retries 5xx, 408, 425 and 429 statuses and transport errors up to
        ``max_attempts`` sends in all, waiting ``backoff_seconds`` doubled
        after each failure between sends. Any other non-2xx status ends the
        attempts at once. Each attempt is stamped and signed afresh; the
        idempotency key is a digest of the kind and body, so it is the same
        for every attempt and for any identical notification.

        Args:
            notification: The notification to post.

        Returns:
            The delivery record: whether it was delivered (a 2xx status), the
            number of attempts, the last status and the idempotency key.

        Raises:
            ValueError: If the payload is invalid (see `payload`) or the
                default transport is given a URL that is not HTTPS. Nothing
                has been sent for a payload error.
            TypeError: If the notification cannot be serialised to JSON.
        """
        body = self.payload(notification)
        key = hashlib.sha256(self.kind.encode() + b"|" + body).hexdigest()[:32]
        status: int | None = None
        reason = "not sent"
        for attempt in range(1, self.max_attempts + 1):
            stamp = f"{self.clock():.3f}"
            request = Request(
                self.url,
                {
                    "Content-Type": "application/json",
                    TIMESTAMP_HEADER: stamp,
                    SIGNATURE_HEADER: sign(self.secret, stamp, body),
                    IDEMPOTENCY_HEADER: key,
                },
                body,
            )
            try:
                status = self.transport(request)
            except OSError as error:
                status = None
                reason = f"transport error: {type(error).__name__}"
            else:
                if 200 <= status < 300:
                    return Delivery(
                        self.kind, True, attempt, status, "delivered", key
                    )
                reason = f"HTTP {status}"
                final = status < 500 and status not in RETRY_STATUSES
                if final:
                    return Delivery(
                        self.kind, False, attempt, status, reason, key
                    )
            if attempt < self.max_attempts:
                self.sleep(self.backoff_seconds * 2 ** (attempt - 1))
        return Delivery(
            self.kind,
            False,
            self.max_attempts,
            status,
            f"{reason} after {self.max_attempts} attempts",
            key,
        )


class Notifier:
    """Routes a notification to the senders its severity calls for."""

    def __init__(
        self,
        pager: WebhookSender | None = None,
        ticket: WebhookSender | None = None,
    ) -> None:
        """Takes the two senders; either may be absent.

        Args:
            pager: Sender for pages, or None.
            ticket: Sender for tickets, or None.
        """
        self.pager = pager
        self.ticket = ticket

    def notify(self, notification: Mapping[str, Any]) -> list[Delivery]:
        """Sends to the senders the severity calls for.

        Critical pages and files a ticket, warning files a ticket, and info
        or an unrecognised severity sends nothing. A missing sender is
        reported as an undelivered notification, not skipped silently.

        Args:
            notification: The notification, with its ``severity``.

        Returns:
            One delivery record for each sender the severity calls for, the
            pager first; empty if none is called for.

        Raises:
            ValueError: If a sender's body would be larger than 64 KiB.
        """
        severity = notification.get("severity", "")
        wanted = {
            "critical": ("pager", "ticket"),
            "warning": ("ticket",),
        }.get(severity, ())
        results = []
        for kind in wanted:
            sender = self.pager if kind == "pager" else self.ticket
            if sender is None:
                results.append(
                    Delivery(kind, False, 0, None, "no sender configured", "")
                )
            else:
                results.append(sender.send(notification))
        return results
