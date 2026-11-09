"""Structured JSON logs with trace correlation and model attribution.

Every line is one JSON object with a fixed set of keys: ``ts``, ``level``,
``message``, ``service``, ``trace_id``, ``model_version`` and
``artifact_digest``. A request carries a W3C ``traceparent`` header or gets a
new trace id; the same id appears on the request log, on the response and in
anything written for that request, so a line can be joined to an alert, an
audit row or a downstream service.

Fields whose names look like credentials are replaced, never written. Request
bodies are not logged: the caller passes only counts and fingerprints.
"""

from collections.abc import Callable
from collections.abc import Mapping
import json
import pathlib
import re
import secrets
import threading
import time
from typing import Any

from scmle import durability

LEVELS = ("debug", "info", "warning", "error", "critical")
RESERVED = frozenset(
    {
        "ts",
        "level",
        "message",
        "service",
        "trace_id",
        "model_version",
        "artifact_digest",
    }
)
SECRET_NAME = re.compile(
    r"(secret|token|password|passwd|authorization|api[_-]?key|credential)",
    re.IGNORECASE,
)
REDACTED = "[redacted]"
TRACEPARENT = re.compile(
    r"^00-(?P<trace>[0-9a-f]{32})-(?P<span>[0-9a-f]{16})-[0-9a-f]{2}$"
)
ZERO_TRACE = "0" * 32


def new_trace_id() -> str:
    """A random 128-bit trace id as 32 lowercase hex digits.

    Returns:
        A trace id that is not the all-zero invalid id.
    """
    while True:
        found = secrets.token_hex(16)
        if found != ZERO_TRACE:
            return found


def parse_traceparent(header: str | None) -> str | None:
    """The trace id of a valid W3C ``traceparent`` header, else None.

    Args:
        header: The header value, or None.

    Returns:
        The 32-digit trace id, or None if the header is absent, malformed or
        carries the all-zero trace id.
    """
    if not header:
        return None
    match = TRACEPARENT.match(header.strip())
    if match is None or match.group("trace") == ZERO_TRACE:
        return None
    return match.group("trace")


def traceparent_for(trace_id: str) -> str:
    """A ``traceparent`` header for ``trace_id`` with a fresh span id.

    Args:
        trace_id: A trace id of 32 lowercase hex digits.

    Returns:
        The header, with sampling flag set.

    Raises:
        ValueError: If `trace_id` is not 32 lowercase hex digits or is all
            zeros.
    """
    if not re.fullmatch(r"[0-9a-f]{32}", trace_id) or trace_id == ZERO_TRACE:
        raise ValueError("trace_id must be 32 lowercase hex digits")
    return f"00-{trace_id}-{secrets.token_hex(8)}-01"


def trace_for(header: str | None) -> str:
    """The incoming trace id, or a new one when the header is absent/bad.

    Args:
        header: The incoming ``traceparent`` header, or None.

    Returns:
        The header's trace id if valid, else a new trace id.
    """
    return parse_traceparent(header) or new_trace_id()


def scrub(value: Any) -> Any:
    """Replaces values under credential-like names, recursively.

    Args:
        value: A value, possibly a nested mapping, list or tuple.

    Returns:
        The structure with the values of credential-like keys replaced by a
        redaction marker; tuples become lists, other values are unchanged.
    """
    if isinstance(value, Mapping):
        return {
            str(key): (
                REDACTED if SECRET_NAME.search(str(key)) else scrub(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [scrub(item) for item in value]
    return value


class StructuredLog:
    """Writes JSON lines to a file or a callable."""

    def __init__(
        self,
        service: str,
        model_version: str,
        artifact_digest: str,
        *,
        path: str | pathlib.Path | None = None,
        sink: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Creates a log bound to one model.

        Args:
            service: Service name.
            model_version: Version of the served model.
            artifact_digest: Digest of the served artifact.
            path: File that lines are appended to.
            sink: Called with each line instead (for tests and shippers).
            clock: Epoch seconds.

        Raises:
            ValueError: If neither or both of ``path`` and ``sink`` are given.
        """
        if (path is None) == (sink is None):
            raise ValueError("give exactly one of path and sink")
        self.service = service
        self.model_version = model_version
        self.artifact_digest = artifact_digest
        self.path = None if path is None else pathlib.Path(path)
        self.sink = sink
        self.clock = clock
        self.lock = threading.Lock()

    def emit(
        self,
        level: str,
        message: str,
        trace_id: str | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        """Writes one line and returns the record.

        Args:
            level: The log level; one of the allowed levels.
            message: The message.
            trace_id: Trace id for the line; a new one if None.
            **fields: Extra fields, scrubbed of credential-like values.

        Returns:
            The record that was written.

        Raises:
            ValueError: If the level is unknown or a field name is reserved.
        """
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}")
        clash = RESERVED & set(fields)
        if clash:
            raise ValueError(f"reserved field names: {sorted(clash)}")
        record = {
            "ts": round(self.clock(), 3),
            "level": level,
            "message": message,
            "service": self.service,
            "trace_id": trace_id or new_trace_id(),
            "model_version": self.model_version,
            "artifact_digest": self.artifact_digest,
            **scrub(fields),
        }
        line = json.dumps(record, sort_keys=True, default=str)
        with self.lock:
            if self.sink is not None:
                self.sink(line)
            elif self.path is not None:
                durability.append_text(self.path, line + "\n")
        return record

    def request(
        self,
        trace_id: str,
        route: str,
        status: int,
        seconds: float,
        **fields: Any,
    ) -> dict[str, Any]:
        """Writes the line for one finished request.

        Args:
            trace_id: Trace id of the request.
            route: The route served.
            status: The HTTP status; 500 and above logs at error level.
            seconds: Duration of the request in seconds.
            **fields: Extra fields.

        Returns:
            The record that was written.
        """
        level = "error" if status >= 500 else "info"
        return self.emit(
            level,
            "request",
            trace_id,
            route=route,
            status=status,
            duration_ms=round(seconds * 1000.0, 3),
            **fields,
        )
