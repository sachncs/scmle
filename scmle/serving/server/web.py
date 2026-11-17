"""JSON-over-HTTP serving of a bundle on a background thread."""

from collections.abc import Mapping
import functools
from http import server
import logging
import threading
import time
from typing import Any
from typing import Protocol

import pandas

import scmle.artifacts.bundle
from scmle.contracts import errors
from scmle.contracts import serving
from scmle.inference import guard
from scmle.security import access
from scmle.security import audit
from scmle.serving.server import contract
from scmle.serving.server import handler

LOGGER = logging.getLogger("scmle.serving")


class Observer(Protocol):
    """What `HttpServing` tells an operations monitor (duck-typed)."""

    def tally(self, at: float, route: str, status: int, seconds: float) -> None:
        """Records one finished request.

        Args:
            at: Unix time at which the request finished.
            route: The request path.
            status: The HTTP status returned.
            seconds: Time the request took.
        """

    def sense(self, at: float, frame: pandas.DataFrame) -> int:
        """Observes the rows of a scored batch.

        Args:
            at: Unix time of the batch.
            frame: The scored feature rows.

        Returns:
            The number of rows the observer kept from the batch.
        """
        ...


class HttpServing(contract.Server):
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
        max_request_bytes: int = contract.MAX_REQUEST_BYTES,
        max_records: int = contract.MAX_REQUEST_RECORDS,
        request_timeout_seconds: float = 10.0,
        *,
        loader: contract.Loader | None = None,
        max_concurrent: int = contract.MAX_CONCURRENT_REQUESTS,
        scoring_timeout_seconds: float = contract.SCORING_TIMEOUT_SECONDS,
        on_null: str = "impute",
        on_unseen: str = "report",
        warmup_records: list[dict[str, Any]] | None = None,
        gateway: access.Gateway | None = None,
        monitor: Observer | None = None,
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
            raise errors.InputError(
                "request bounds and timeout must be positive"
            )
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
        self.admission = guard.Admission(
            max_concurrent, scoring_timeout_seconds
        )
        self.httpd: server.ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.loading: threading.Thread | None = None

    def health(self) -> Mapping[str, Any]:
        """Readiness, which a timed-out call still running takes away.

        Returns:
            See `guard.assess`.
        """
        return guard.assess(self.readiness, self.admission)

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

    def boot(self) -> None:
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
            # ERR-20: an InputError was a bare ValueError (or TypeError,
            # RuntimeError) here, published by type name only; readiness is
            # public, so its message (paths, values) stays private.
            reason = (
                f"{error.category}: {error}"
                if isinstance(error, errors.ScmleError)
                and not isinstance(error, errors.InputError)
                else type(error).__name__
            )
            self.readiness.fail(reason)
            self.announce()
            return
        self.warm()
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

    def observe(self, route: str, status: int, seconds: float) -> None:
        """Passes a finished request to the monitor; never raises.

        Args:
            route: The request path.
            status: The HTTP status returned.
            seconds: Time the request took.
        """
        if self.monitor is None:
            return
        try:
            self.monitor.tally(time.time(), route, status, seconds)
        except Exception:  # monitoring must not change a response
            LOGGER.exception("the signal monitor failed on a request")

    def feed(self, records: list[dict[str, Any]]) -> None:
        """Passes the rows of a scored batch to the monitor; never raises.

        Args:
            records: The scored feature records.
        """
        if self.monitor is None:
            return
        try:
            self.monitor.sense(time.time(), pandas.DataFrame(records))
        except Exception:  # monitoring must not change a response
            LOGGER.exception("the signal monitor failed on features")

    def score(self, request: serving.Inquiry) -> dict[str, Any]:
        """Scores a parsed request.

        Args:
            request: The parsed request; its expected schema digest, when
                present, must equal the served one.

        Returns:
            The response of `respond` for the request's records.

        Raises:
            MismatchError: If the caller's expected schema digest differs
                from the served schema's.
        """
        if (
            request.expected_schema_digest is not None
            and request.expected_schema_digest != self.schema_digest
        ):
            raise serving.MismatchError(
                "the model's input schema differs from the one the caller "
                "expects",
                {
                    "expected": request.expected_schema_digest,
                    "served": self.schema_digest,
                },
            )
        response = self.respond(request.records)
        self.feed(request.records)
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
        responder = functools.partial(handler.Responder, adapter=self)
        self.httpd = Listener((self.host, self.port), responder)
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, daemon=True
        )
        self.thread.start()
        if self.loader is not None:
            self.loading = threading.Thread(target=self.boot, daemon=True)
            self.loading.start()
        else:
            self.warm()
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


class Listener(server.ThreadingHTTPServer):
    """The threaded HTTP server that backs `HttpServing`."""

    daemon_threads = True
    allow_reuse_address = True
