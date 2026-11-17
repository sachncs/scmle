"""The request handlers behind `HttpServing`."""

from __future__ import annotations

from http import server
import json
import logging
import time
from typing import Any
from typing import TYPE_CHECKING

import scmle.artifacts.bundle
from scmle.contracts import errors
from scmle.contracts import serving
from scmle.inference import guard
from scmle.runtime import logs
from scmle.security import access

LOGGER = logging.getLogger("scmle.serving")

if TYPE_CHECKING:
    from scmle.serving.server import web


class Exchange(server.BaseHTTPRequestHandler):
    """Reads one request and answers it with safe JSON and its request ID.

    Attributes:
        adapter: The serving adapter whose bundle, limits and gateway answer
            the requests.
        request_id: The ID of the request being handled.
        started: Monotonic time at which the current request began.
    """

    request_id = ""
    started = 0.0

    def __init__(
        self, *args: Any, adapter: web.HttpServing, **kwargs: Any
    ) -> None:
        """Binds the handler to its adapter and handles the request.

        Args:
            *args: The request, client address and server that the socket
                server passes to every handler.
            adapter: The serving adapter that answers the request.
            **kwargs: Further keyword arguments of the base handler.
        """
        self.adapter = adapter
        super().__init__(*args, **kwargs)

    def setup(self) -> None:
        """Applies the request I/O timeout to the connection."""
        super().setup()
        self.connection.settimeout(self.adapter.request_timeout_seconds)

    def version_string(self) -> str:
        """Names the server without leaking library versions.

        Returns:
            The fixed server name.
        """
        return "scmle-reference"

    def transmit(
        self,
        status: int,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> None:
        """Sends a JSON response with safe headers and records the request.

        Args:
            status: HTTP status code.
            payload: Response body; the request ID is added to it.
            headers: Extra response headers.
        """
        body = json.dumps({**payload, "request_id": self.request_id}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Request-ID", self.request_id)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)
        self.adapter.observe(self.path, status, time.monotonic() - self.started)

    def fail(
        self,
        status: int,
        message: str,
        category: str,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Sends a structured error response.

        Args:
            status: HTTP status code.
            message: Human-readable reason.
            category: Machine-readable error category.
            details: Extra evidence, included when not None.
            headers: Extra response headers.
        """
        payload: dict[str, Any] = {
            "schema_version": serving.SCHEMA_VERSION,
            "error": message,
            "category": category,
        }
        if details is not None:
            payload["details"] = details
        self.transmit(status, payload, headers)

    def identify(self) -> bool:
        """Resolves the request ID and rejects an invalid caller ID.

        Returns:
            True when the request may continue; False after a 400 was sent
            for an invalid `X-Request-ID`.
        """
        self.started = time.monotonic()
        caller = self.headers.get("X-Request-ID")
        if caller is None:
            self.request_id = guard.mint()
            return True
        self.request_id = caller if guard.wellformed(caller) else guard.mint()
        if self.request_id != caller:
            self.fail(
                400,
                "X-Request-ID must be 1-128 ASCII characters from "
                "A-Z, a-z, 0-9, '.', '_', ':', or '-': "
                "request rejected",
                "request",
            )
            return False
        return True

    def authorize(self, method: str) -> bool:
        """Applies the gateway; answers a refusal with a safe body.

        Args:
            method: The HTTP method.

        Returns:
            True when the request may proceed (no gateway, or the gateway
            accepted it); False after a refusal was sent.
        """
        gateway = self.adapter.gateway
        if gateway is None:
            return True
        try:
            gateway.check(
                method,
                self.path,
                self.headers.get("Authorization"),
                self.client_address[0],
                self.request_id,
            )
        except access.DeniedError as denied:
            # handled: gateway denial sent as refusal response; returns False
            self.fail(
                denied.status,
                denied.body["error"],
                denied.body["category"],
                None,
                denied.headers,
            )
            return False
        return True

    def log_message(
        self,
        format: str,  # noqa: A002 -- name fixed by BaseHTTPRequestHandler
        *args: Any,
    ) -> None:
        """Silences per-request logging.

        Args:
            format: Unused log format string.
            *args: Unused format arguments.
        """


class Responder(Exchange):
    """Handles the requests of one `HttpServing` adapter."""

    def do_GET(self) -> None:
        """Serves the probes, health and schema endpoints."""
        if not self.identify() or not self.authorize("GET"):
            return
        with logs.bind(request=self.request_id):
            self.probe()

    def probe(self) -> None:
        """Answers a GET for the probes, health and schema endpoints."""
        adapter = self.adapter
        readiness = adapter.health()
        if self.path == "/livez":
            alive = adapter.admission.alive
            self.transmit(
                200 if alive else 503,
                {
                    "status": "alive" if alive else "wedged",
                    **adapter.admission.state(),
                },
            )
        elif self.path == "/readyz":
            self.transmit(
                200 if readiness["ready"] else 503,
                {**readiness, **adapter.identity()},
                None if readiness["ready"] else {"Retry-After": "1"},
            )
        elif not readiness["ready"]:
            self.fail(
                503,
                "service is not ready",
                "not_ready",
                dict(readiness),
                {"Retry-After": "1"},
            )
        elif self.path == "/health":
            info = adapter.bundle.info
            self.transmit(
                200,
                {
                    "status": "ok",
                    "model_version": info.model_version,
                    "model_sha256": info.files.get(
                        scmle.artifacts.bundle.MODEL_FILE, ""
                    ),
                    "artifact_digest": adapter.artifact_digest,
                    "task": adapter.bundle.schema.task,
                },
            )
        elif self.path == "/schema":
            self.transmit(
                200,
                {
                    "schema_version": serving.SCHEMA_VERSION,
                    "schema": adapter.bundle.schema.to_dict(),
                    "schema_digest": adapter.schema_digest,
                    "request_schema": serving.REQUEST_JSON_SCHEMA,
                    "response_schema": serving.RESPONSE_JSON_SCHEMA,
                    "limits": adapter.limits(),
                    **adapter.identity(),
                },
            )
        else:
            self.fail(404, "not found", "request")

    def receive(self) -> bytes | None:
        """Reads the request body after checking its headers.

        Returns:
            The body, or None after an error response was sent for a wrong
            `Content-Type`, a missing, malformed or oversized
            `Content-Length`, or an incomplete body.
        """
        content_type = self.headers.get("Content-Type", "")
        if content_type.split(";", 1)[0].strip().lower() != (
            "application/json"
        ):
            self.fail(415, "Content-Type must be application/json", "request")
            return None
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self.fail(411, "Content-Length is required", "request")
            return None
        try:
            length = int(raw_length)
        except ValueError:
            # handled: non-integer Content-Length answered with HTTP 400
            self.fail(400, "Content-Length must be an integer", "request")
            return None
        if length < 0 or length > self.adapter.max_request_bytes:
            self.fail(
                413,
                f"request body exceeds {self.adapter.max_request_bytes} bytes",
                "request",
            )
            return None
        body = self.rfile.read(length)
        if len(body) != length:
            self.fail(400, "request body is incomplete", "request")
            return None
        return body

    def do_POST(self) -> None:
        """Scores a prediction request under the admission limit."""
        if not self.identify() or not self.authorize("POST"):
            return
        with logs.bind(request=self.request_id):
            self.accept()

    def accept(self) -> None:
        """Routes a POST to scoring under the admission limit."""
        adapter = self.adapter
        if self.path not in serving.PREDICT_PATHS:
            self.fail(404, "not found", "request")
            return
        readiness = adapter.health()
        if not readiness["ready"]:
            self.fail(
                503,
                "service is not ready",
                "not_ready",
                dict(readiness),
                {"Retry-After": "1"},
            )
            return
        try:
            with adapter.admission.admit():
                self.predict()
        except guard.OverloadedError as error:
            # handled: admission overload answered with HTTP 429 and Retry-After
            self.fail(
                429,
                str(error),
                error.category,
                None,
                {"Retry-After": "1"},
            )

    def predict(self) -> None:
        """Scores the request and answers it, mapping failures to statuses."""
        try:
            self.answer()
        except guard.DeadlineError as error:
            # handled: scoring deadline exceeded answered with HTTP 504
            self.fail(504, str(error), error.category)
        except guard.UnreadyError as error:
            # handled: not-ready scorer answered with HTTP 503
            self.fail(503, str(error), error.category)
        except serving.MismatchError as error:
            # handled: schema change answered with HTTP 409 and details
            self.fail(409, str(error), error.category, error.details)
        except errors.OrderError:
            # ERR-20: a step called out of order inside the server was a
            # RuntimeError, answered 500 before it became an InputError;
            # it is still a server defect, not the caller's input.
            LOGGER.exception("internal serving failure")
            self.fail(500, "internal serving failure", "internal")
        except errors.InputError as error:
            # ERR-20: these were ValueError or TypeError and answered 400
            # as malformed; the DataError branch below would make them 422,
            # which the service documents only for schema and policy.
            # handled: malformed request answered with HTTP 400
            self.fail(400, f"malformed request: {error}", "request")
        except errors.ScmleError as error:
            # Rows that break the schema are well-formed requests the
            # server cannot process: 422. Anything else is a 400.
            # handled: scmle error answered with HTTP 422 or 400
            self.fail(
                422 if isinstance(error, errors.DataError) else 400,
                str(error),
                error.category,
                error.details,
            )
        except OverflowError as error:
            # handled: overflow answered with HTTP 413
            self.fail(413, str(error), "request")
        except (KeyError, ValueError, TypeError) as error:
            # handled: malformed request answered with HTTP 400 naming the error
            self.fail(400, f"malformed request: {error}", "request")
        except Exception:
            LOGGER.exception("internal serving failure")
            self.fail(500, "internal serving failure", "internal")

    def answer(self) -> None:
        """Reads, parses and scores the request body and sends the answer.

        An error response is sent, and nothing is scored, when the body is
        unreadable or is not JSON. Other failures propagate to `predict`.
        """
        body = self.receive()
        if body is None:
            return
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as error:
            # handled: invalid JSON body answered with HTTP 400, documented
            self.fail(400, f"invalid JSON request: {error.msg}", "request")
            return
        adapter = self.adapter
        request = serving.parse(payload, adapter.max_records)
        response = adapter.admission.run(lambda: adapter.score(request))
        response["request_id"] = self.request_id
        self.transmit(200, response)
