"""Serving adapters: how a bundle is exposed to callers.

`Server` is the contract; `BatchServing` scores files and
`HttpServing` exposes a JSON API using only the standard library, so serving
needs no extra dependencies. Online and batch scoring share one path,
`Server.respond_frame`, so they cannot drift apart: the same
validation, the same preprocessing, the same model call, the same response.
"""

import abc
from collections.abc import Callable
import functools
from http import server
import json
import logging
import pathlib
import threading
import time
from typing import Any
from typing import Protocol

import pandas

from scmle import durability
import scmle.artifacts.bundle
from scmle.contracts import errors
from scmle.contracts import serving
from scmle.deploy import provenance
from scmle.inference import guard
from scmle.inference import validation
from scmle.security import access
from scmle.security import audit

LOGGER = logging.getLogger("scmle.serving")

MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_REQUEST_RECORDS = 10_000
MAX_CONCURRENT_REQUESTS = 8
SCORING_TIMEOUT_SECONDS = 30.0
BATCH_CHUNK_ROWS = 10_000
Loader = Callable[[guard.Readiness], scmle.artifacts.bundle.Bundle]


class Server(abc.ABC):
    """Exposes a model bundle to callers."""

    def __init__(
        self,
        bundle: scmle.artifacts.bundle.Bundle | None,
        on_null: str = "impute",
        on_unseen: str = "report",
        warmup_records: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initializes the adapter and its readiness tracker.

        With a bundle, the verify and load steps are marked complete at once;
        the warm-up step is left to `warm_up`.

        Args:
            bundle: Model to serve. `None` defers loading to a subclass that
                drives `readiness` itself (see `HttpServing`).
            on_null: `impute` lets the fitted imputers handle null features
                and reports them per row; `reject` answers 422 instead.
            on_unseen: `report` scores rows with categories never seen in
                training and flags them per row; `reject` answers 422.
            warmup_records: Records for the warm-up call; by default one is
                derived from the schema.

        Raises:
            ConfigError: If a policy name is unknown.
        """
        if on_null not in serving.NULL_POLICIES:
            raise errors.ConfigError(
                f"on_null must be in {serving.NULL_POLICIES}"
            )
        if on_unseen not in serving.UNSEEN_POLICIES:
            raise errors.ConfigError(
                f"on_unseen must be in {serving.UNSEEN_POLICIES}"
            )
        self.loaded: scmle.artifacts.bundle.Bundle | None = bundle
        self.lock = threading.Lock()
        self.on_null = on_null
        self.on_unseen = on_unseen
        self.warmup_records = warmup_records
        self.readiness = guard.Readiness()
        if bundle is not None:
            note = "bundle provided by the caller"
            self.readiness.begin(guard.Phase.VERIFYING)
            self.readiness.complete(guard.Phase.VERIFYING, note)
            self.readiness.begin(guard.Phase.LOADING)
            self.readiness.complete(guard.Phase.LOADING, note)

    @property
    def bundle(self) -> scmle.artifacts.bundle.Bundle:
        """The loaded bundle.

        Raises:
            NotReadyError: Before verification and loading finished.
        """
        if self.loaded is None:
            raise guard.NotReadyError("the model is not loaded")
        return self.loaded

    @property
    def artifact_digest(self) -> str:
        """Digest of the manifest of the bundle being served."""
        return provenance.manifest_digest(self.bundle.info.to_dict())

    @property
    def schema_digest(self) -> str:
        """Digest of the input schema of the bundle being served."""
        return serving.schema_digest(self.bundle.schema.to_dict())

    def synthetic_record(self) -> dict[str, Any] | None:
        """Derives a record that is valid for the schema.

        Returns:
            A record with 0.0 for every numeric feature and the first known
            category (an empty string if none is known) for every
            categorical one; None when the schema has series, extra or
            optional inputs.
        """
        schema = self.bundle.schema
        if schema.series or schema.extra_inputs or schema.optional_inputs:
            return None
        record: dict[str, Any] = dict.fromkeys(schema.numeric, 0.0)
        for name in schema.categorical:
            known = schema.known_categories.get(name) or [""]
            record[name] = known[0]
        return record

    def warm_up(self) -> None:
        """Scores once so the first request does not pay for initialization.

        Marks the warm-up step complete, or skipped when no record can be
        derived, or marks the service failed if scoring raises. Warm-up
        records come from `warmup_records` or `synthetic_record`. The warm-up
        call never feeds series history. A `ScmleError` from
        `Readiness.begin` (warm-up started out of order) is not caught.
        """
        self.readiness.begin(guard.Phase.WARMING)
        try:
            records = self.warmup_records
            note = "explicit warm-up records"
            if records is None:
                record = self.synthetic_record()
                records = None if record is None else [record]
                note = "synthetic record from the schema"
            if records is None:
                self.readiness.complete(
                    guard.Phase.WARMING,
                    "skipped: no synthetic record for series or extra inputs",
                )
                return
            with self.lock:
                self.bundle.score(pandas.DataFrame(records))
        except Exception as error:
            LOGGER.exception("warm-up failed")
            self.readiness.fail(f"warm-up failed: {type(error).__name__}")
            return
        self.readiness.complete(guard.Phase.WARMING, note)

    def respond(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Scores records and builds the structured response.

        A row that breaks a policy set to `reject` makes `enforce_policy`
        raise `PolicyError`.

        Args:
            records: Feature dictionaries. Every record must carry every
                required feature (null is allowed, a missing key is not).

        Returns:
            The version 1 response: predictions, probabilities (classifiers),
            model version, artifact digest, schema digest, schema diagnostics,
            `record_count` and one `rows` entry per record.

        Raises:
            SchemaError: If the records do not satisfy the model's schema.
            ValueError: If no records were supplied.
        """
        if not records:
            raise ValueError("records must not be empty")
        required = self.bundle.features
        absent = sorted(
            {
                name
                for record in records
                for name in required
                if name not in record
            }
        )
        if absent:
            raise errors.SchemaError(
                f"records are missing required features: {absent}",
                {
                    "missing": absent,
                    "records": [
                        index
                        for index, record in enumerate(records)
                        if any(name not in record for name in required)
                    ][:20],
                },
            )
        return self.respond_frame(pandas.DataFrame(records))

    def respond_frame(
        self, frame: pandas.DataFrame, remember: bool = True
    ) -> dict[str, Any]:
        """Scores a frame; the one path used by online and batch serving.

        Args:
            frame: Rows to score, with the model's feature columns.
            remember: Whether to add the rows to the bundle's remembered
                series history after scoring.

        The policies are enforced after scoring and before the rows are
        remembered, so a rejected batch leaves the history unchanged and
        `PolicyError` propagates.

        Returns:
            The bundle's score response extended with the schema version,
            artifact digest, schema digest, `record_count` and the per-row
            `rows` report.
        """
        with self.lock:
            bundle = self.bundle
            rows = validation.InputValidator(bundle.schema).row_report(frame)
            response = bundle.score(frame)
            self.enforce_policy(rows)
            if remember:
                bundle.remember(frame)
        response.update(
            {
                "schema_version": serving.SCHEMA_VERSION,
                "artifact_digest": self.artifact_digest,
                "schema_digest": self.schema_digest,
                "record_count": len(rows),
                "rows": rows,
            }
        )
        return response

    def enforce_policy(self, rows: list[dict[str, Any]]) -> None:
        """Applies the null and unseen-category policies to scored rows.

        Args:
            rows: Per-row reports with `missing` and `unseen` entries.

        Raises:
            PolicyError: If `on_null` is `reject` and a row has null
                features, or `on_unseen` is `reject` and a row has unseen
                categories; the details list up to 20 record indexes.
        """
        if self.on_null == "reject":
            nulls = [i for i, row in enumerate(rows) if row["missing"]]
            if nulls:
                raise serving.PolicyError(
                    "null features are rejected by policy",
                    {"records": nulls[:20]},
                )
        if self.on_unseen == "reject":
            unseen = [i for i, row in enumerate(rows) if row["unseen"]]
            if unseen:
                raise serving.PolicyError(
                    "unseen categories are rejected by policy",
                    {"records": unseen[:20]},
                )

    @abc.abstractmethod
    def start(self) -> None:
        """Begins serving."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Stops serving and releases resources."""


class BatchServing(Server):
    """Scores CSV / Parquet files through the same path as online serving."""

    def start(self) -> None:
        """Nothing to start for batch scoring."""

    def stop(self) -> None:
        """Nothing to stop for batch scoring."""

    def score_file(
        self,
        source: str | pathlib.Path,
        destination: str | pathlib.Path,
        chunk_rows: int = BATCH_CHUNK_ROWS,
    ) -> int:
        """Scores a file and writes predictions (and probabilities).

        The output CSV has a `prediction` column and, for classifiers, one
        `proba_<class>` column per class. It is accompanied by
        `<destination>.manifest.json`, which carries the model version,
        artifact digest, schema digest, row count, the SHA-256 of the CSV and
        the counts of rows with nulls and with unseen categories. Chunks are
        scored through `respond_frame`, so a policy set to `reject` raises
        `PolicyError` before anything is written, and the scored rows are
        added to the bundle's remembered series history.

        Args:
            source: Input CSV / Parquet.
            destination: Output CSV.
            chunk_rows: Rows scored per call.

        Returns:
            Number of rows scored.

        Raises:
            ConfigError: If `chunk_rows` is not positive.
        """
        if chunk_rows < 1:
            raise errors.ConfigError("chunk_rows must be positive")
        frame = self.bundle.read_frame(source)
        pieces = []
        flagged = {"rows_with_nulls": 0, "rows_with_unseen_categories": 0}
        response: dict[str, Any] = {}
        for start in range(0, max(len(frame), 1), chunk_rows):
            chunk = frame.iloc[start : start + chunk_rows].reset_index(
                drop=True
            )
            response = self.respond_frame(chunk)
            result = pandas.DataFrame({"prediction": response["predictions"]})
            if "classes" in response:
                probabilities = response["probabilities"]
                for index, label in enumerate(response["classes"]):
                    result[f"proba_{label}"] = [
                        row[index] for row in probabilities
                    ]
            pieces.append(result)
            flagged["rows_with_nulls"] += sum(
                1 for row in response["rows"] if row["missing"]
            )
            flagged["rows_with_unseen_categories"] += sum(
                1 for row in response["rows"] if row["unseen"]
            )
        output = pandas.concat(pieces, ignore_index=True)
        content = output.to_csv(index=False).encode("utf-8")
        target = pathlib.Path(destination)
        durability.write_atomically(target, content)
        manifest = {
            "schema_version": serving.SCHEMA_VERSION,
            "model_version": response.get(
                "model_version", self.bundle.info.model_version
            ),
            "artifact_digest": self.artifact_digest,
            "schema_digest": self.schema_digest,
            "rows": len(output),
            "output_sha256": provenance.sha256_digest(content),
            **flagged,
        }
        durability.write_atomically(
            target.with_name(target.name + ".manifest.json"),
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
        )
        return len(output)


class RequestObserver(Protocol):
    """What `HttpServing` tells an operations monitor (duck-typed)."""

    def record_request(
        self, at: float, route: str, status: int, seconds: float
    ) -> None:
        """Records one finished request.

        Args:
            at: Unix time at which the request finished.
            route: The request path.
            status: The HTTP status returned.
            seconds: Time the request took.
        """

    def observe_features(self, at: float, frame: pandas.DataFrame) -> int:
        """Observes the rows of a scored batch.

        Args:
            at: Unix time of the batch.
            frame: The scored feature rows.

        Returns:
            The number of rows the observer kept from the batch.
        """
        ...


class HttpServing(Server):
    """JSON-over-HTTP serving on a background thread.

    Endpoints:

    * `GET /livez`: 200 while the process runs.
    * `GET /readyz`: 200 only after verify, load and warm-up; else 503.
    * `GET /health`: status, model version, digest (503 until ready).
    * `GET /schema`: input schema, schema digest, request and response
      schemas, limits and policies.
    * `POST /predict` (alias `/v1/predict`) with `{"records": [{...}]}`.

    Records that violate the schema get a 422 with the reason; malformed
    requests get a 400; a schema the caller does not expect gets a 409;
    more than `max_concurrent` requests in flight get a 429; a request that
    runs past `scoring_timeout_seconds` gets a 504; before readiness 503.
    """

    def __init__(
        self,
        bundle: scmle.artifacts.bundle.Bundle | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        max_request_bytes: int = MAX_REQUEST_BYTES,
        max_records: int = MAX_REQUEST_RECORDS,
        request_timeout_seconds: float = 10.0,
        *,
        loader: Loader | None = None,
        max_concurrent: int = MAX_CONCURRENT_REQUESTS,
        scoring_timeout_seconds: float = SCORING_TIMEOUT_SECONDS,
        on_null: str = "impute",
        on_unseen: str = "report",
        warmup_records: list[dict[str, Any]] | None = None,
        gateway: access.Gateway | None = None,
        monitor: RequestObserver | None = None,
    ) -> None:
        """Initializes the HTTP adapter and its admission limits.

        Args:
            bundle: Model to serve, already verified and loaded by the
                caller; the service is ready once `start` warmed it up.
            host: Interface to bind (loopback by default).
            port: Port; 0 picks a free one.
            max_request_bytes: Maximum JSON request body size.
            max_records: Maximum records accepted by one prediction request.
            request_timeout_seconds: Maximum time allowed for request I/O.
            loader: Instead of `bundle`: a callable that verifies and loads
                the bundle while driving the `Readiness` it is given. The
                server listens at once; `/livez` answers 200 and `/readyz`
                503 until the loader and the warm-up finish, and stays 503
                with the reason if either fails.
            max_concurrent: Requests allowed in flight.
            scoring_timeout_seconds: Time one scoring call may take.
            on_null: `impute` or `reject` (see `Server`).
            on_unseen: `report` or `reject` (see `Server`).
            warmup_records: Records for the warm-up call.
            gateway: Authenticates, authorizes, rate-limits and audits every
                request except `/livez` and `/readyz`. Without it the server
                may only bind a loopback address.
            monitor: Receives every finished request (route, status, seconds)
                and the rows of every scored batch, for the operations
                signals. A monitor that raises is logged and ignored: it
                never changes a response.

        Raises:
            ConfigError: If neither or both of `bundle` and `loader` are
                given, a policy name is unknown, or `host` is not a loopback
                address and there is no `gateway`.
            ValueError: If a request bound or timeout is not positive.
        """
        if (bundle is None) == (loader is None):
            raise errors.ConfigError("give exactly one of bundle and loader")
        if (
            max_request_bytes < 1
            or max_records < 1
            or request_timeout_seconds <= 0
        ):
            raise ValueError("request bounds and timeout must be positive")
        access.require_authentication_for(host, gateway)
        super().__init__(bundle, on_null, on_unseen, warmup_records)
        self.gateway = gateway
        self.monitor = monitor
        self.loader = loader
        self.host = host
        self.port = port
        self.max_request_bytes = max_request_bytes
        self.max_records = max_records
        self.request_timeout_seconds = request_timeout_seconds
        self.admission = guard.AdmissionControl(
            max_concurrent, scoring_timeout_seconds
        )
        self.httpd: server.ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.loading: threading.Thread | None = None

    @property
    def address(self) -> str:
        """The base URL of the running server.

        Raises:
            ScmleError: If the server is not running.
        """
        if self.httpd is None:
            raise errors.ScmleError("server is not running")
        return f"http://{self.host}:{self.httpd.server_address[1]}"

    def limits(self) -> dict[str, Any]:
        """Reports the declared limits and policies, for `/schema`.

        Returns:
            The request, record, concurrency and timeout limits, and the
            null and unseen-category policies.
        """
        return {
            "max_request_bytes": self.max_request_bytes,
            "max_records": self.max_records,
            "max_concurrent": self.admission.max_concurrent,
            "scoring_timeout_seconds": self.admission.timeout_seconds,
            "request_timeout_seconds": self.request_timeout_seconds,
            "on_null": self.on_null,
            "on_unseen": self.on_unseen,
        }

    def identity(self) -> dict[str, Any]:
        """Reports the model version and digest.

        Returns:
            `model_version` and `artifact_digest`; an empty mapping before
            the model is loaded.
        """
        if self.loaded is None:
            return {}
        return {
            "model_version": self.loaded.info.model_version,
            "artifact_digest": self.artifact_digest,
        }

    def load_in_background(self) -> None:
        """Runs the loader, then the warm-up; failure is kept as readiness.

        Raises:
            ScmleError: If the adapter has no loader.
        """
        loader = self.loader
        if loader is None:
            raise errors.ScmleError("no loader to run")
        try:
            self.loaded = loader(self.readiness)
        except Exception as error:
            LOGGER.exception("verification or load failed")
            reason = (
                f"{error.category}: {error}"
                if isinstance(error, errors.ScmleError)
                else type(error).__name__
            )
            self.readiness.fail(reason)
            self.announce()
            return
        self.warm_up()
        self.announce()

    def announce(self) -> None:
        """Audits the outcome of verify, load and warm-up (activation).

        Does nothing without a gateway that has an audit log.
        """
        gateway = self.gateway
        if gateway is None or gateway.audit is None:
            return
        snapshot = self.readiness.snapshot()
        if snapshot["ready"]:
            gateway.audit.activation(
                self.bundle.info.model_version, self.artifact_digest
            )
        else:
            gateway.audit.record(
                audit.ACTIVATION,
                actor="service",
                outcome=audit.DENIED,
                reason=str(snapshot.get("reason", "not ready")),
            )

    def observe_request(self, route: str, status: int, seconds: float) -> None:
        """Passes a finished request to the monitor; never raises.

        Args:
            route: The request path.
            status: The HTTP status returned.
            seconds: Time the request took.
        """
        if self.monitor is None:
            return
        try:
            self.monitor.record_request(time.time(), route, status, seconds)
        except Exception:  # monitoring must not change a response
            LOGGER.exception("the signal monitor failed on a request")

    def observe_rows(self, records: list[dict[str, Any]]) -> None:
        """Passes the rows of a scored batch to the monitor; never raises.

        Args:
            records: The scored feature records.
        """
        if self.monitor is None:
            return
        try:
            self.monitor.observe_features(
                time.time(), pandas.DataFrame(records)
            )
        except Exception:  # monitoring must not change a response
            LOGGER.exception("the signal monitor failed on features")

    def score(self, request: serving.ScoreRequest) -> dict[str, Any]:
        """Scores a parsed request.

        Args:
            request: The parsed request; its expected schema digest, when
                present, must equal the served one.

        Returns:
            The response of `respond` for the request's records.

        Raises:
            SchemaChangeError: If the caller's expected schema digest differs
                from the served schema's.
        """
        if (
            request.expected_schema_digest is not None
            and request.expected_schema_digest != self.schema_digest
        ):
            raise serving.SchemaChangeError(
                "the model's input schema differs from the one the caller "
                "expects",
                {
                    "expected": request.expected_schema_digest,
                    "served": self.schema_digest,
                },
            )
        response = self.respond(request.records)
        self.observe_rows(request.records)
        return response

    def start(self) -> None:
        """Binds the socket and serves in a daemon thread.

        With a loader, verification, loading and warm-up continue in a
        background thread; otherwise the bundle is warmed up before
        returning.

        Raises:
            ScmleError: If the server is already running.
        """
        if self.httpd is not None:
            raise errors.ScmleError("server is already running")
        handler = functools.partial(Responder, adapter=self)
        self.httpd = ReferenceServer((self.host, self.port), handler)
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, daemon=True
        )
        self.thread.start()
        if self.loader is not None:
            self.loading = threading.Thread(
                target=self.load_in_background, daemon=True
            )
            self.loading.start()
        else:
            self.warm_up()
            self.announce()

    def stop(self) -> None:
        """Shuts the server down, joins its thread and closes the pool."""
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)
        self.admission.close()
        self.httpd = self.thread = None


class ReferenceServer(server.ThreadingHTTPServer):
    """The threaded HTTP server that backs `HttpServing`."""

    daemon_threads = True
    allow_reuse_address = True


class Responder(server.BaseHTTPRequestHandler):
    """Handles the requests of one `HttpServing` adapter.

    Attributes:
        adapter: The serving adapter whose bundle, limits and gateway answer
            the requests.
        request_id: The ID of the request being handled.
        started: Monotonic time at which the current request began.
    """

    request_id = ""
    started = 0.0

    def __init__(self, *args: Any, adapter: HttpServing, **kwargs: Any) -> None:
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

    def send_json(
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
        self.adapter.observe_request(
            self.path, status, time.monotonic() - self.started
        )

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
        self.send_json(status, payload, headers)

    def identify(self) -> bool:
        """Resolves the request ID and rejects an invalid caller ID.

        Returns:
            True when the request may continue; False after a 400 was sent
            for an invalid `X-Request-ID`.
        """
        self.started = time.monotonic()
        caller = self.headers.get("X-Request-ID")
        if caller is None:
            self.request_id = guard.new_request_id()
            return True
        self.request_id = (
            caller
            if guard.is_valid_request_id(caller)
            else guard.new_request_id()
        )
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
            self.fail(
                denied.status,
                denied.body["error"],
                denied.body["category"],
                None,
                denied.headers,
            )
            return False
        return True

    def do_GET(self) -> None:
        """Serves the probes, health and schema endpoints."""
        if not self.identify() or not self.authorize("GET"):
            return
        adapter = self.adapter
        readiness = adapter.readiness.snapshot()
        if self.path == "/livez":
            self.send_json(200, {"status": "alive"})
        elif self.path == "/readyz":
            self.send_json(
                200 if readiness["ready"] else 503,
                {**readiness, **adapter.identity()},
                None if readiness["ready"] else {"Retry-After": "1"},
            )
        elif not readiness["ready"]:
            self.fail(
                503,
                "service is not ready",
                "not_ready",
                readiness,
                {"Retry-After": "1"},
            )
        elif self.path == "/health":
            info = adapter.bundle.info
            self.send_json(
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
            self.send_json(
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

    def read_body(self) -> bytes | None:
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
        adapter = self.adapter
        if self.path not in serving.PREDICT_PATHS:
            self.fail(404, "not found", "request")
            return
        if not adapter.readiness.ready:
            self.fail(
                503,
                "service is not ready",
                "not_ready",
                adapter.readiness.snapshot(),
                {"Retry-After": "1"},
            )
            return
        try:
            with adapter.admission.admit():
                self.predict()
        except guard.OverloadedError as error:
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
            self.answer_prediction()
        except guard.DeadlineError as error:
            self.fail(504, str(error), error.category)
        except guard.NotReadyError as error:
            self.fail(503, str(error), error.category)
        except serving.SchemaChangeError as error:
            self.fail(409, str(error), error.category, error.details)
        except errors.ScmleError as error:
            # Rows that break the schema are well-formed requests the
            # server cannot process: 422. Anything else is a 400.
            self.fail(
                422 if isinstance(error, errors.DataError) else 400,
                str(error),
                error.category,
                error.details,
            )
        except OverflowError as error:
            self.fail(413, str(error), "request")
        except (KeyError, ValueError, TypeError) as error:
            self.fail(400, f"malformed request: {error}", "request")
        except Exception:
            LOGGER.exception("internal serving failure")
            self.fail(500, "internal serving failure", "internal")

    def answer_prediction(self) -> None:
        """Reads, parses and scores the request body and sends the answer.

        An error response is sent, and nothing is scored, when the body is
        unreadable or is not JSON. Other failures propagate to `predict`.
        """
        body = self.read_body()
        if body is None:
            return
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as error:
            self.fail(400, f"invalid JSON request: {error.msg}", "request")
            return
        adapter = self.adapter
        request = serving.parse_request(payload, adapter.max_records)
        response = adapter.admission.run(lambda: adapter.score(request))
        response["request_id"] = self.request_id
        self.send_json(200, response)

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
