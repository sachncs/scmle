"""Serving adapters: how a bundle is exposed to callers.

`ServingAdapter` is the contract; `BatchServing` scores files and
`HttpServing` exposes a JSON API using only the standard library, so serving
needs no extra dependencies.
"""

import abc
from http import server
import json
import pathlib
import threading
from typing import Any

import pandas as pd

from scml.artifacts import bundle as bundle_module
from scml.contracts import errors

MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_REQUEST_RECORDS = 10_000


class ServingAdapter(abc.ABC):
    """Exposes a model bundle to callers."""

    def __init__(self, bundle: bundle_module.ModelBundle) -> None:
        """Initializes the adapter.

        Args:
            bundle: Model to serve.
        """
        self.bundle = bundle
        self.lock = threading.Lock()

    def respond(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Scores records and builds the structured response.

        Args:
            records: Feature dictionaries.

        Returns:
            Predictions, probabilities (classifiers), model version and
            schema diagnostics.

        Raises:
            SchemaError: If the records do not satisfy the model's schema.
            ValueError: If no records were supplied.
        """
        if not records:
            raise ValueError("records must not be empty")
        with self.lock:
            return self.respond_locked(records)

    def respond_locked(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Scores records, then remembers them as series history."""
        frame = pd.DataFrame(records)
        response = self.bundle.score(frame)
        self.bundle.remember(frame)
        return response

    @abc.abstractmethod
    def start(self) -> None:
        """Begins serving."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Stops serving and releases resources."""


class BatchServing(ServingAdapter):
    """Scores CSV / Parquet files."""

    def start(self) -> None:
        """Nothing to start for batch scoring."""

    def stop(self) -> None:
        """Nothing to stop for batch scoring."""

    def score_file(
        self, source: str | pathlib.Path, destination: str | pathlib.Path
    ) -> int:
        """Scores a file and writes predictions (and probabilities).

        Args:
            source: Input CSV / Parquet.
            destination: Output CSV.

        Returns:
            Number of rows scored.
        """
        frame = self.bundle.read_frame(source)
        result = pd.DataFrame({"prediction": self.bundle.predict(frame)})
        if self.bundle.schema.classes is not None:
            probabilities = self.bundle.predict_proba(frame)
            for index, label in enumerate(self.bundle.schema.classes):
                result[f"proba_{label}"] = probabilities[:, index]
        self.bundle.remember(frame)
        result.to_csv(destination, index=False)
        return len(result)


class HttpServing(ServingAdapter):
    """JSON-over-HTTP serving on a background thread.

    Endpoints: `GET /health` (status, model version and checksum),
    `GET /schema`, and `POST /predict` with `{"records": [{...}, ...]}`.
    Records that violate the schema get a 422 with the reason; malformed
    requests get a 400.
    """

    def __init__(
        self,
        bundle: bundle_module.ModelBundle,
        host: str = "127.0.0.1",
        port: int = 0,
        max_request_bytes: int = MAX_REQUEST_BYTES,
        max_records: int = MAX_REQUEST_RECORDS,
    ) -> None:
        """Initializes the adapter.

        Args:
            bundle: Model to serve.
            host: Interface to bind (loopback by default).
            port: Port; 0 picks a free one.
            max_request_bytes: Maximum JSON request body size.
            max_records: Maximum records accepted by one prediction request.

        Raises:
            ValueError: If a request bound is not positive.
        """
        if max_request_bytes < 1 or max_records < 1:
            raise ValueError("request bounds must be positive")
        super().__init__(bundle)
        self.host = host
        self.port = port
        self.max_request_bytes = max_request_bytes
        self.max_records = max_records
        self.httpd: server.ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    @property
    def address(self) -> str:
        """Base URL once started."""
        if self.httpd is None:
            raise errors.ScmlError("server is not running")
        return f"http://{self.host}:{self.httpd.server_address[1]}"

    def start(self) -> None:
        """Binds the socket and serves in a daemon thread."""
        if self.httpd is not None:
            raise errors.ScmlError("server is already running")
        adapter = self

        class Handler(server.BaseHTTPRequestHandler):
            def send_json(self, status: int, payload: dict[str, Any]) -> None:
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
                if self.path == "/health":
                    info = adapter.bundle.info
                    self.send_json(
                        200,
                        {
                            "status": "ok",
                            "model_version": info.model_version,
                            "model_sha256": info.files.get(
                                bundle_module.MODEL_FILE, ""
                            ),
                            "task": adapter.bundle.schema.task,
                        },
                    )
                elif self.path == "/schema":
                    self.send_json(
                        200,
                        {
                            "schema": adapter.bundle.schema.to_dict(),
                            "model_version": adapter.bundle.info.model_version,
                        },
                    )
                else:
                    self.send_json(404, {"error": "not found"})

            def do_POST(self) -> None:
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
                        self.send_json(
                            413,
                            {
                                "error": "request body exceeds "
                                f"{adapter.max_request_bytes} bytes",
                                "category": "request",
                            },
                        )
                        return
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
                    records = payload["records"]
                    if not isinstance(records, list):
                        raise TypeError("'records' must be a list")
                    if len(records) > adapter.max_records:
                        self.send_json(
                            413,
                            {
                                "error": "request contains more than "
                                f"{adapter.max_records} records",
                                "category": "request",
                            },
                        )
                        return
                    if not all(isinstance(record, dict) for record in records):
                        raise TypeError("every record must be an object")
                    self.send_json(200, adapter.respond(records))
                except errors.ScmlError as error:
                    # Rows that break the schema are well-formed requests the
                    # server cannot process: 422. Anything else is a 400.
                    self.send_json(
                        422 if isinstance(error, errors.DataError) else 400,
                        {
                            "error": str(error),
                            "category": error.category,
                            "details": error.details,
                        },
                    )
                except (KeyError, ValueError, TypeError) as error:
                    self.send_json(
                        400,
                        {
                            "error": f"malformed request: {error}",
                            "category": "request",
                        },
                    )
                except Exception:
                    self.send_json(
                        500,
                        {
                            "error": "internal serving failure",
                            "category": "internal",
                        },
                    )

            def log_message(self, format: str, *args: Any) -> None:
                """Silences per-request logging."""

        self.httpd = server.ThreadingHTTPServer((self.host, self.port), Handler)
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, daemon=True
        )
        self.thread.start()

    def stop(self) -> None:
        """Shuts the server down and joins its thread."""
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)
        self.httpd = self.thread = None
