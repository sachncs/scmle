"""Reference HTTP serving for signed experimental modality packages.

The core package does not guess how an arbitrary document, image, audio file,
video container, graph, or geospatial object should be decoded. A caller-owned
``ModalitySampleDecoder`` makes that trust boundary explicit while this module
provides the same bounded, inspectable HTTP surface as tabular serving.
"""

from __future__ import annotations

import abc
from http import server
import json
import threading
from typing import Any

import numpy as np

from scml.contracts import errors
from scml.contracts import modalities
from scml.models import modality_lifecycle

MAX_MODALITY_REQUEST_BYTES = 16 * 1024 * 1024
MAX_MODALITY_REQUEST_SAMPLES = 10_000


class ModalitySampleDecoder(abc.ABC):
    """Decode one request payload into caller-owned modality samples."""

    @abc.abstractmethod
    def decode(self, payload: object) -> tuple[Any, ...]:
        """Decode and validate a JSON payload into finite sample objects."""

    @abc.abstractmethod
    def schema(self) -> dict[str, object]:
        """Describe the decoder's input contract as JSON evidence."""


class ModalityHttpServing:
    """Serve a verified modality predictor through a bounded HTTP reference API.

    Endpoints are ``GET /health``, ``GET /schema``, and ``POST /predict`` with
    ``{"samples": ...}``. Authentication, TLS, transport-specific decoding,
    and operational paging remain deployment-owner responsibilities.
    """

    def __init__(
        self,
        predictor: modality_lifecycle.ModalityPredictor,
        decoder: ModalitySampleDecoder,
        host: str = "127.0.0.1",
        port: int = 0,
        max_request_bytes: int = MAX_MODALITY_REQUEST_BYTES,
        max_samples: int = MAX_MODALITY_REQUEST_SAMPLES,
    ) -> None:
        """Bind a verified predictor and an explicit sample decoder."""
        if max_request_bytes < 1 or max_samples < 1:
            raise ValueError("request bounds must be positive")
        self.predictor = predictor
        self.decoder = decoder
        self.host = host
        self.port = port
        self.max_request_bytes = max_request_bytes
        self.max_samples = max_samples
        self.lock = threading.Lock()
        self.httpd: server.ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    @property
    def address(self) -> str:
        """Return the base URL after the server has started."""
        if self.httpd is None:
            raise errors.ScmlError("server is not running")
        return f"http://{self.host}:{self.httpd.server_address[1]}"

    def response(self, payload: object) -> dict[str, object]:
        """Decode samples and return deterministic predictions."""
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
            "modality": modalities.modality_value(
                self.predictor.adapter.modality
            ),
            "predictions": np.asarray(predictions).tolist(),
            "probabilities": np.asarray(probabilities).tolist(),
            "sample_count": len(samples),
            "package_verified": True,
        }

    def start(self) -> None:
        """Bind the socket and serve in a daemon thread."""
        if self.httpd is not None:
            raise errors.ScmlError("server is already running")
        adapter = self

        class Handler(server.BaseHTTPRequestHandler):
            """Handle one modality reference-server request."""

            def send_json(
                self, status: int, payload: dict[str, object]
            ) -> None:
                """Send a JSON response with safe reference-server headers."""
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                """Serve health and decoder-schema evidence."""
                if self.path == "/health":
                    self.send_json(200, adapter.predictor.health())
                elif self.path == "/schema":
                    self.send_json(
                        200,
                        {
                            "modality": (
                                modalities.modality_value(
                                    adapter.predictor.adapter.modality
                                )
                            ),
                            "decoder": adapter.decoder.schema(),
                            "feature_names": list(
                                adapter.predictor.feature_names
                            ),
                        },
                    )
                else:
                    self.send_json(404, {"error": "not found"})

            def do_POST(self) -> None:
                """Decode and score a bounded prediction request."""
                if self.path != "/predict":
                    self.send_json(404, {"error": "not found"})
                    return
                try:
                    content_type = self.headers.get("Content-Type", "")
                    if content_type.split(";", 1)[0].strip().lower() != (
                        "application/json"
                    ):
                        self.send_json(
                            415,
                            {
                                "error": (
                                    "Content-Type must be application/json"
                                ),
                                "category": "request",
                            },
                        )
                        return
                    raw_length = self.headers.get("Content-Length")
                    if raw_length is None:
                        self.send_json(
                            411,
                            {
                                "error": "Content-Length is required",
                                "category": "request",
                            },
                        )
                        return
                    try:
                        length = int(raw_length)
                    except ValueError:
                        self.send_json(
                            400,
                            {
                                "error": "Content-Length must be an integer",
                                "category": "request",
                            },
                        )
                        return
                    if length < 0 or length > adapter.max_request_bytes:
                        raise OverflowError(
                            "request body exceeds "
                            f"{adapter.max_request_bytes} bytes"
                        )
                    body = self.rfile.read(length)
                    if len(body) != length:
                        self.send_json(
                            400,
                            {
                                "error": "request body is incomplete",
                                "category": "request",
                            },
                        )
                        return
                    try:
                        payload = json.loads(body)
                    except json.JSONDecodeError as error:
                        self.send_json(
                            400,
                            {
                                "error": f"invalid JSON request: {error.msg}",
                                "category": "request",
                            },
                        )
                        return
                    self.send_json(200, adapter.response(payload))
                except OverflowError as error:
                    self.send_json(
                        413,
                        {
                            "error": str(error),
                            "category": "request",
                        },
                    )
                except (TypeError, ValueError) as error:
                    self.send_json(
                        422,
                        {
                            "error": f"invalid modality samples: {error}",
                            "category": "samples",
                        },
                    )
                except Exception:
                    self.send_json(
                        500,
                        {
                            "error": "internal modality serving failure",
                            "category": "internal",
                        },
                    )

            def log_message(self, format: str, *args: Any) -> None:
                """Suppress per-request reference-server logging."""

        self.httpd = server.ThreadingHTTPServer((self.host, self.port), Handler)
        self.thread = threading.Thread(
            target=self.httpd.serve_forever,
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        """Shut down the server and join its worker thread."""
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)
        self.httpd = None
        self.thread = None


__all__ = ["ModalityHttpServing", "ModalitySampleDecoder"]
