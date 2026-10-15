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

from mml.artifacts import bundle as bundle_module
from mml.contracts import errors


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
        """
        with self.lock:
            return self.respond_locked(records)

    def respond_locked(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Scores records, then remembers them as series history."""
        frame = pd.DataFrame(records)
        report = self.bundle.validate(frame)
        predictions = self.bundle.predict(frame)
        response: dict[str, Any] = {
            "predictions": [
                item.item() if hasattr(item, "item") else item
                for item in predictions
            ],
            "model_version": self.bundle.info.model_version,
            "schema_valid": report.valid,
            "diagnostics": {
                "ignored_columns": list(report.extra),
                "missing_values": report.missing_value_counts,
                "unseen_categories": report.unseen_categories,
            },
        }
        if self.bundle.schema.classes is not None:
            response["classes"] = list(self.bundle.schema.classes)
            response["probabilities"] = self.bundle.predict_proba(
                frame
            ).tolist()
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

    Endpoints: `GET /health`, `GET /schema`, `POST /predict` with
    `{"records": [{...}, ...]}`.
    """

    def __init__(
        self,
        bundle: bundle_module.ModelBundle,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        """Initializes the adapter.

        Args:
            bundle: Model to serve.
            host: Interface to bind (loopback by default).
            port: Port; 0 picks a free one.
        """
        super().__init__(bundle)
        self.host = host
        self.port = port
        self.httpd: server.ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    @property
    def address(self) -> str:
        """Base URL once started."""
        if self.httpd is None:
            raise errors.MagicMLError("server is not running")
        return f"http://{self.host}:{self.httpd.server_address[1]}"

    def start(self) -> None:
        """Binds the socket and serves in a daemon thread."""
        adapter = self

        class Handler(server.BaseHTTPRequestHandler):
            def send_json(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path == "/health":
                    self.send_json(200, {"status": "ok"})
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
                    length = int(self.headers.get("Content-Length", 0))
                    records = json.loads(self.rfile.read(length))["records"]
                    self.send_json(200, adapter.respond(records))
                except errors.MagicMLError as error:
                    self.send_json(
                        400,
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
