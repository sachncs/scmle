"""Reference HTTP serving for signed experimental modality packages.

The core package does not guess how an arbitrary document, image, audio file,
video container, graph, or geospatial object should be decoded. A caller-owned
``ModalitySampleDecoder`` makes that trust boundary explicit while this module
provides the same bounded, inspectable HTTP surface as tabular serving.
"""

from __future__ import annotations

import abc
import functools
from http import server
import json
import re
import threading
from typing import Any
from typing import TYPE_CHECKING
import uuid

import numpy

from scmle.contracts import errors
from scmle.contracts import modalities
from scmle.contracts import serving
from scmle.inference import guard
from scmle.security import access

if TYPE_CHECKING:
    from scmle.models import modality_lifecycle

MAX_MODALITY_REQUEST_BYTES = 16 * 1024 * 1024
MAX_MODALITY_REQUEST_SAMPLES = 10_000
MAX_REQUEST_ID_LENGTH = 128
REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


class ModalitySampleDecoder(abc.ABC):
    """Decodes one request payload into caller-owned modality samples."""

    @abc.abstractmethod
    def decode(self, payload: object) -> tuple[Any, ...]:
        """Decodes and validates a JSON payload into finite sample objects.

        Args:
            payload: The parsed JSON request body.

        Returns:
            The decoded samples, one object per sample.
        """

    @abc.abstractmethod
    def schema(self) -> dict[str, object]:
        """Describes the decoder's input contract as JSON evidence.

        Returns:
            A JSON-serializable description of the accepted input.
        """


class ModalityHttpServing:
    """Serves a verified modality predictor over a bounded HTTP reference API.

    Endpoints are ``GET /livez``, ``GET /readyz``, ``GET /health``,
    ``GET /schema``, and ``POST /predict`` with ``{"samples": ...}``. The
    predictor is verified when it is loaded, so the service is ready once
    constructed; the warm-up step is recorded as skipped because a caller-owned
    decoder has no synthetic sample. At most 8 requests run at once (429
    beyond) and a scoring call past 30 seconds answers 504. Every response
    carries the schema version and the package digest, which doubles as the
    model version because modality packages carry no separate label.
    Authentication, TLS, transport-specific decoding, and operational paging
    remain deployment-owner responsibilities.
    """

    def __init__(
        self,
        predictor: modality_lifecycle.Predictor,
        decoder: ModalitySampleDecoder,
        host: str = "127.0.0.1",
        port: int = 0,
        max_request_bytes: int = MAX_MODALITY_REQUEST_BYTES,
        max_samples: int = MAX_MODALITY_REQUEST_SAMPLES,
        require_lifecycle_report: bool = False,
    ) -> None:
        """Binds a verified predictor and an explicit sample decoder.

        Marks the verify, load and warm-up steps complete at once. Sets the
        admission limits to 8 requests in flight and a 30 second scoring
        time. Authentication is added with `set_gateway`, and `start`
        refuses a non-loopback ``host`` without one.

        Args:
            predictor: The verified modality predictor.
            decoder: Caller-owned decoder for request payloads.
            host: Interface to bind.
            port: Port to bind; 0 picks a free port.
            max_request_bytes: Largest accepted request body, in bytes.
            max_samples: Largest accepted number of samples per request.
            require_lifecycle_report: Whether to enable the
                production-boundary mode, in which the predictor must have
                been loaded with a verified, identity-matching signed
                lifecycle report.

        Raises:
            ValueError: If a request bound is below 1, or
                `require_lifecycle_report` is set and the predictor carries
                no lifecycle report digest.
        """
        if max_request_bytes < 1 or max_samples < 1:
            raise ValueError("request bounds must be positive")
        self.gateway: access.Gateway | None = None
        if require_lifecycle_report and (
            predictor.lifecycle_report_digest is None
        ):
            raise ValueError(
                "release-bound serving requires a verified lifecycle report"
            )
        self.predictor = predictor
        self.decoder = decoder
        self.host = host
        self.port = port
        self.max_request_bytes = max_request_bytes
        self.max_samples = max_samples
        self.require_lifecycle_report = require_lifecycle_report
        self.lock = threading.Lock()
        self.admission = guard.AdmissionControl(8, 30.0)
        self.readiness = guard.Readiness()
        for step in guard.STEPS[:2]:
            self.readiness.begin(step)
            self.readiness.complete(step, "verified when the predictor loaded")
        self.readiness.begin(guard.Phase.WARMING)
        self.readiness.complete(
            guard.Phase.WARMING, "skipped: the decoder is owned by the caller"
        )
        self.httpd: server.ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def set_gateway(self, gateway: access.Gateway | None) -> None:
        """Sets the gateway that guards every request except the probes.

        Args:
            gateway: Authenticates, authorizes, rate-limits and audits
                requests (`GET /livez` and `GET /readyz` stay public); None
                removes authentication.
        """
        self.gateway = gateway

    @property
    def address(self) -> str:
        """The base URL of the running server.

        Raises:
            ScmleError: If the server is not running.
        """
        if self.httpd is None:
            raise errors.ScmleError("server is not running")
        return f"http://{self.host}:{self.httpd.server_address[1]}"

    @property
    def artifact_digest(self) -> str:
        """The `sha256:` digest of the verified model package."""
        return "sha256:" + self.predictor.package.model_sha256

    def response(self, payload: object) -> dict[str, object]:
        """Decodes samples and returns predictions with their probabilities.

        Scoring runs under a lock, so concurrent calls are serialized.

        Args:
            payload: The parsed JSON request body.

        Returns:
            The response body: schema and model versions, the package
            digest, modality, predictions and probabilities as lists, the
            sample count, and the verification flags.

        Raises:
            ValueError: If the decoder returns no samples.
            OverflowError: If there are more samples than `max_samples`.
                Errors raised by the decoder or the predictor also
                propagate.
        """
        samples = self.decoder.decode(payload)
        if not samples:
            raise ValueError("samples must not be empty")
        if len(samples) > self.max_samples:
            raise OverflowError(
                f"request contains more than {self.max_samples} samples"
            )
        with self.lock:
            predictions = self.predictor.predict(samples)
            probabilities = self.predictor.predict_proba(samples)
        return {
            "schema_version": serving.SCHEMA_VERSION,
            "model_version": self.artifact_digest,
            "artifact_digest": self.artifact_digest,
            "modality": modalities.modality_value(
                self.predictor.adapter.modality
            ),
            "predictions": numpy.asarray(predictions).tolist(),
            "probabilities": numpy.asarray(probabilities).tolist(),
            "sample_count": len(samples),
            "package_verified": True,
            "lifecycle_report_verified": (
                self.predictor.lifecycle_report_digest is not None
            ),
            "lifecycle_report_digest": self.predictor.lifecycle_report_digest,
        }

    @staticmethod
    def is_valid_request_id(request_id: str) -> bool:
        """Checks whether a caller request ID is safe to echo in headers.

        Args:
            request_id: The ID supplied by the caller.

        Returns:
            True when it has 1 to 128 allowed characters.
        """
        return (
            1 <= len(request_id) <= MAX_REQUEST_ID_LENGTH
            and REQUEST_ID_PATTERN.fullmatch(request_id) is not None
        )

    @staticmethod
    def new_server_request_id() -> str:
        """Returns a bounded, opaque request ID with no request data.

        Returns:
            A new random ID of the form `srv-<32 hex digits>`.
        """
        return f"srv-{uuid.uuid4().hex}"

    def start(self) -> None:
        """Binds the socket and serves in a daemon thread.

        A non-loopback host without a gateway is refused with a
        `ConfigError` by `access.require_authentication_for`.

        Raises:
            ScmleError: If the server is already running.
        """
        if self.httpd is not None:
            raise errors.ScmleError("server is already running")
        handler = functools.partial(Responder, adapter=self)
        access.require_authentication_for(self.host, self.gateway)
        self.httpd = server.ThreadingHTTPServer((self.host, self.port), handler)
        self.thread = threading.Thread(
            target=self.httpd.serve_forever,
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        """Shuts down the server, joins its thread and closes the pool."""
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)
        self.admission.close()
        self.httpd = None
        self.thread = None


class RejectedRequestError(Exception):
    """Raised when a request is refused with a client-error status.

    Attributes:
        status: The HTTP status code to answer with.
    """

    def __init__(self, status: int, message: str) -> None:
        """Creates the error.

        Args:
            status: The HTTP status code to answer with.
            message: Explanation sent to the caller.
        """
        super().__init__(message)
        self.status = status


class Responder(server.BaseHTTPRequestHandler):
    """Handles one modality reference-server request.

    Attributes:
        adapter: The serving adapter whose predictor, gateway and limits
            answer the requests.
    """

    def __init__(
        self, *args: Any, adapter: ModalityHttpServing, **kwargs: Any
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

    def request_identity(self) -> tuple[str, str | None]:
        """Resolves the request ID of the current request.

        Returns:
            The caller's `X-Request-ID` when it is valid, else a new
            server ID, and an error message when the caller's ID was
            invalid (None otherwise).
        """
        caller_request_id = self.headers.get("X-Request-ID")
        if caller_request_id is None:
            return self.adapter.new_server_request_id(), None
        if self.adapter.is_valid_request_id(caller_request_id):
            return caller_request_id, None
        return (
            self.adapter.new_server_request_id(),
            (
                "X-Request-ID must be 1-128 ASCII characters from "
                "A-Z, a-z, 0-9, '.', '_', ':', or '-': request rejected"
            ),
        )

    def send_json(
        self,
        status: int,
        payload: dict[str, object],
        request_id: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Sends a JSON response with safe reference-server headers.

        Args:
            status: HTTP status code.
            payload: Response body; `request_id` is added to it.
            request_id: The ID echoed in the body and header.
            headers: Extra response headers.
        """
        response_payload = dict(payload)
        response_payload["request_id"] = request_id
        body = json.dumps(response_payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Request-ID", request_id)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def authorized(self, method: str, request_id: str) -> bool:
        """Applies the gateway; answers a refusal with a safe body.

        Args:
            method: The HTTP method.
            request_id: The ID of the current request.

        Returns:
            True when the request may proceed (no gateway, or the
            gateway accepted it); False after a refusal was sent.
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
                request_id,
            )
        except access.DeniedError as denied:
            self.send_json(
                denied.status, denied.body, request_id, denied.headers
            )
            return False
        return True

    def admit_request(self, method: str) -> str | None:
        """Resolves the request ID and applies the gateway.

        Args:
            method: The HTTP method.

        Returns:
            The request ID when the request may proceed; None after a 400
            for an invalid `X-Request-ID` or a gateway refusal was sent.
        """
        request_id, identity_error = self.request_identity()
        if identity_error is not None:
            self.send_json(
                400,
                {"error": identity_error, "category": "request"},
                request_id,
            )
            return None
        if not self.authorized(method, request_id):
            return None
        return request_id

    def do_GET(self) -> None:
        """Serves the probes, health and decoder-schema endpoints.

        An invalid `X-Request-ID` is answered with 400, a refused
        gateway check with its own status, and any other path
        with 404.
        """
        request_id = self.admit_request("GET")
        if request_id is None:
            return
        adapter = self.adapter
        if self.path == "/livez":
            self.send_json(200, {"status": "alive"}, request_id)
        elif self.path == "/readyz":
            state = adapter.readiness.snapshot()
            self.send_json(
                200 if state["ready"] else 503,
                {**state, "artifact_digest": adapter.artifact_digest},
                request_id,
            )
        elif self.path == "/health":
            self.send_json(
                200,
                {
                    **adapter.predictor.health(),
                    "artifact_digest": adapter.artifact_digest,
                },
                request_id,
            )
        elif self.path == "/schema":
            self.send_json(
                200,
                {
                    "schema_version": serving.SCHEMA_VERSION,
                    "modality": (
                        modalities.modality_value(
                            adapter.predictor.adapter.modality
                        )
                    ),
                    "decoder": adapter.decoder.schema(),
                    "feature_names": list(adapter.predictor.feature_names),
                },
                request_id,
            )
        else:
            self.send_json(404, {"error": "not found"}, request_id)

    def read_payload(self) -> object:
        """Reads and parses the JSON body after checking its headers.

        Returns:
            The parsed JSON request body.

        Raises:
            RejectedRequestError: If the `Content-Type` is not JSON (415),
                `Content-Length` is missing (411) or not an integer (400),
                the body is incomplete (400) or is not valid JSON (400).
            OverflowError: If `Content-Length` is negative or above the
                limit.
        """
        content_type = self.headers.get("Content-Type", "")
        if content_type.split(";", 1)[0].strip().lower() != (
            "application/json"
        ):
            raise RejectedRequestError(
                415, "Content-Type must be application/json"
            )
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise RejectedRequestError(411, "Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise RejectedRequestError(
                400, "Content-Length must be an integer"
            ) from error
        if length < 0 or length > self.adapter.max_request_bytes:
            raise OverflowError(
                f"request body exceeds {self.adapter.max_request_bytes} bytes"
            )
        body = self.rfile.read(length)
        if len(body) != length:
            raise RejectedRequestError(400, "request body is incomplete")
        try:
            return json.loads(body)
        except json.JSONDecodeError as error:
            raise RejectedRequestError(
                400, f"invalid JSON request: {error.msg}"
            ) from error

    def do_POST(self) -> None:
        """Decodes and scores a bounded prediction request.

        Only `POST /predict` is served; another path gets 404 and a
        refused gateway check gets its own status. Failures are
        answered with an error response (415, 411, 400, 413, 422,
        429, 504 or 500 as appropriate) and not propagated.
        """
        request_id = self.admit_request("POST")
        if request_id is None:
            return
        if self.path != "/predict":
            self.send_json(404, {"error": "not found"}, request_id)
            return
        adapter = self.adapter
        try:
            payload = self.read_payload()
            with adapter.admission.admit():
                response = adapter.admission.run(
                    lambda: adapter.response(payload)
                )
            self.send_json(200, response, request_id)
        except RejectedRequestError as rejected:
            self.send_json(
                rejected.status,
                {"error": str(rejected), "category": "request"},
                request_id,
            )
        except guard.OverloadedError as error:
            self.send_json(
                429,
                {"error": str(error), "category": error.category},
                request_id,
            )
        except guard.DeadlineError as error:
            self.send_json(
                504,
                {"error": str(error), "category": error.category},
                request_id,
            )
        except OverflowError as error:
            self.send_json(
                413,
                {"error": str(error), "category": "request"},
                request_id,
            )
        except (TypeError, ValueError) as error:
            self.send_json(
                422,
                {
                    "error": f"invalid modality samples: {error}",
                    "category": "samples",
                },
                request_id,
            )
        except Exception:  # noqa: BLE001 -- last boundary: answer with a 500
            self.send_json(
                500,
                {
                    "error": "internal modality serving failure",
                    "category": "internal",
                },
                request_id,
            )

    def log_message(
        self,
        format: str,  # noqa: A002 -- name fixed by BaseHTTPRequestHandler
        *args: Any,
    ) -> None:
        """Suppresses per-request reference-server logging.

        Args:
            format: Unused log format string.
            *args: Unused format arguments.
        """


__all__ = ["ModalityHttpServing", "ModalitySampleDecoder"]
