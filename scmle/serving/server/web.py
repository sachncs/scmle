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
import scmle.data.reference
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

    def adopt(self, profile: scmle.data.reference.Baseline | None) -> None:
        """Judges drift against the training reference of a served bundle.

        Args:
            profile: The bundle's reference, or None if it has none.
        """


class Watch:
    """How a server tells its operations monitor; it never fails serving.

    Every call is passed on to `monitor`; one that raises is logged and
    ignored, since monitoring must not change a response.

    Attributes:
        monitor: The monitor, or None for none.
    """

    monitor: Observer | None

    def observe(self, route: str, status: int, seconds: float) -> None:
        """Passes on a finished request.

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
        """Passes on the rows of a scored batch.

        Args:
            records: The scored feature records.
        """
        if self.monitor is None:
            return
        try:
            self.monitor.sense(time.time(), pandas.DataFrame(records))
        except Exception:  # monitoring must not change a response
            LOGGER.exception("the signal monitor failed on features")

    def follow(self, model: scmle.artifacts.bundle.Bundle) -> None:
        """Passes on the training reference of the bundle now served.

        Args:
            model: The bundle now served.
        """
        if self.monitor is None:
            return
        try:
            self.monitor.adopt(model.reference)
        except Exception:  # monitoring must not change serving
            LOGGER.exception("the signal monitor failed on a new bundle")


class HttpServing(contract.Server, Watch):
    """JSON-over-HTTP serving on a background thread.

    Endpoints:

    * `GET /livez`: 200 while the process runs.
    * `GET /readyz`: 200 only after verify, load and warm-up; else 503.
    * `GET /health`: status, model version, digest (503 until ready).
    * `GET /schema`: input schema, schema digest, request and response
      schemas, limits and policies.
    * `POST /predict` (alias `/v1/predict`) with `{"records": [{...}]}`.

    Invalid records get a 422; malformed requests a 400; an unexpected
    schema a 409; too many in flight a 429; a scoring timeout a 504; before
    readiness a 503. `reload` swaps bundles without downtime and `stop`
    drains the requests in flight (`Listener.close`).
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
        drain_seconds: float | None = None,
    ) -> None:
        """Initializes the HTTP adapter and its admission limits.

        Args:
            bundle: Model to serve, verified and loaded by the caller.
            host: Interface to bind (loopback by default).
            port: Port; 0 picks a free one.
            max_request_bytes: Maximum JSON request body size.
            max_records: Maximum records accepted by one prediction request.
            request_timeout_seconds: Maximum time allowed for request I/O.
            loader: Instead of `bundle`: loads it in the background, driving
                the `Readiness` it is given (`/readyz` is 503 meanwhile).
            max_concurrent: Requests allowed in flight.
            scoring_timeout_seconds: Time one scoring call may take.
            on_null: `impute` or `reject` (see `Server`).
            on_unseen: `report` or `reject` (see `Server`).
            warmup_records: Records for the warm-up call.
            gateway: Authenticates, authorizes, rate-limits and audits every
                request except `/livez` and `/readyz`. Without it the server
                may only bind a loopback address.
            monitor: Told every finished request, scored batch and bundle
                served (see `Watch`, which never lets it fail serving).
            drain_seconds: Time `stop` waits for requests in flight; see
                `patience` for the default.

        Raises:
            ConfigError: If neither or both of `bundle` and `loader` are
                given, a policy name is unknown, or `host` is not a loopback
                address and there is no `gateway`.
            ValueError: If a request bound, timeout or the drain time is
                not positive.
        """
        if (bundle is None) == (loader is None):
            raise errors.ConfigError("give exactly one of bundle and loader")
        if min(max_request_bytes, max_records, request_timeout_seconds) <= 0:
            raise errors.InputError(
                "request bounds and timeouts must be positive"
            )
        access.require_authentication_for(host, gateway)
        super().__init__(bundle, on_null, on_unseen, warmup_records)
        self.gateway = gateway
        self.monitor = monitor
        if bundle is not None:
            self.follow(bundle)
        self.loader = loader
        self.host = host
        self.port = port
        self.max_request_bytes = max_request_bytes
        self.max_records = max_records
        self.request_timeout_seconds = request_timeout_seconds
        self.admission = guard.Admission(
            max_concurrent, scoring_timeout_seconds
        )
        self.httpd: Listener | None = None
        self.thread: threading.Thread | None = None
        self.loading: threading.Thread | None = None
        self.drain_seconds = patience(
            request_timeout_seconds, scoring_timeout_seconds, drain_seconds
        )
        self.draining = False
        self.swap = threading.Lock()

    def health(self) -> Mapping[str, Any]:
        """Readiness, which a timed-out call or draining takes away.

        Returns:
            See `guard.assess`; while `stop` drains, `ready` is false and
            the reason is `draining`.
        """
        found = guard.assess(self.readiness, self.admission)
        if self.draining:
            return {**found, "ready": False, "reason": "draining"}
        return found

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
            self.readiness.fail(publish(error))
            self.announce()
            return
        self.follow(self.loaded)
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

    def reload(self, loader: contract.Loader | None = None) -> str:
        """Loads and warms a new bundle, then serves it instead of the old.

        The old bundle answers until the new one is warmed up, and a
        request already scoring finishes on it. The new load's readiness
        replaces the old (a failed first load is recovered); a failure
        changes nothing and propagates.

        Args:
            loader: Loads the replacement, as `loader` in `__init__`; the
                adapter's own by default.

        Returns:
            The artifact digest now served.

        Raises:
            ConfigError: If there is no loader to reload from.
            UnreadyError: If the first load is still running or the server
                is draining.
        """
        chosen = self.loader if loader is None else loader
        if chosen is None:
            raise errors.ConfigError("no loader to reload the bundle from")
        with self.swap:
            if self.draining or (self.loading and self.loading.is_alive()):
                raise guard.UnreadyError(
                    "cannot reload while the first load runs or while draining"
                )
            staged = guard.Readiness()
            fresh = chosen(staged)
            staged.begin(guard.Phase.WARMING)
            staged.complete(guard.Phase.WARMING, self.rehearse(fresh))
            before = self.identity().get("artifact_digest")
            self.loaded = fresh
            self.readiness = staged
            self.follow(fresh)
            self.announce()
        LOGGER.info(
            "reloaded the bundle: %s -> %s", before, self.artifact_digest
        )
        return self.artifact_digest

    def start(self) -> None:
        """Binds the socket and serves in a daemon thread.

        A loader's load and warm-up continue in the background; a given
        bundle is warmed up before returning.

        Raises:
            ScmleError: If the server is already running.
        """
        if self.httpd is not None:
            raise errors.ScmleError("server is already running")
        self.draining = False
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

    def stop(self, grace: float | None = None) -> None:
        """Stops accepting, drains the requests in flight, then shuts down.

        New connections are refused at once and `/readyz` reports
        `draining` to the requests still running. Requests in flight get
        up to `grace` seconds to finish; any still running then are left
        to their daemon threads and logged.

        Args:
            grace: Seconds to wait for requests in flight; `drain_seconds`
                by default.
        """
        if self.httpd is not None:
            self.draining = True
            self.httpd.close(self.drain_seconds if grace is None else grace)
        if self.thread is not None:
            self.thread.join(timeout=5)
        self.admission.close()
        self.httpd = self.thread = None


def publish(error: Exception) -> str:
    """The reason a failed load gives at the public `/readyz`.

    ERR-20: an `InputError` (once a bare ValueError, TypeError or
    RuntimeError) is published by type name only, so its message (paths,
    values) stays private.

    Args:
        error: Why verification or loading failed.

    Returns:
        `<category>: <message>` for a package error, else the type name.
    """
    if isinstance(error, errors.ScmleError) and not isinstance(
        error, errors.InputError
    ):
        return f"{error.category}: {error}"
    return type(error).__name__


def patience(request: float, scoring: float, chosen: float | None) -> float:
    """The time `HttpServing.stop` gives the requests in flight.

    Args:
        request: The request I/O timeout, in seconds.
        scoring: The scoring timeout, in seconds.
        chosen: The drain time asked for, or None for the default.

    Returns:
        `chosen`, or by default the longest an admitted request can still
        take: `request` to read its body, `scoring` (after which it is
        answered 504) and `request` to write the answer, so every request
        that can still finish does.

    Raises:
        InputError: If `chosen` is not positive.
    """
    if chosen is not None and chosen <= 0:
        raise errors.InputError("the drain time must be positive")
    return 2 * request + scoring if chosen is None else chosen


class Listener(server.ThreadingHTTPServer):
    """The threaded HTTP server that backs `HttpServing`.

    Attributes:
        flight: Connections accepted and not yet answered.
        settled: Condition notified whenever a connection finishes.
    """

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], handler: Any) -> None:
        """Binds the socket and starts with nothing in flight.

        Args:
            address: Host and port to bind.
            handler: The request handler factory.
        """
        super().__init__(address, handler)
        self.flight = 0
        self.settled = threading.Condition()

    def process_request(self, request: Any, client_address: Any) -> None:
        """Counts the connection in flight, then hands it to a thread.

        The count rises on the serving thread, before `shutdown` can
        return, so `drain` never misses an accepted connection.

        Args:
            request: The accepted socket.
            client_address: The peer address.
        """
        with self.settled:
            self.flight += 1
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.land()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        """Handles one connection and counts it out of flight.

        Args:
            request: The accepted socket.
            client_address: The peer address.
        """
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.land()

    def land(self) -> None:
        """Counts one connection out of flight."""
        with self.settled:
            self.flight -= 1
            self.settled.notify_all()

    def close(self, grace: float) -> None:
        """Stops accepting, then waits for the connections in flight.

        Args:
            grace: Longest wait; connections still in flight then are left
                to their daemon threads and logged.
        """
        self.shutdown()
        self.server_close()
        if not self.drain(grace):
            LOGGER.warning(
                "%d request(s) still in flight after %.1f s of draining",
                self.flight,
                grace,
            )

    def drain(self, seconds: float) -> bool:
        """Waits until no connection is in flight.

        Args:
            seconds: Longest wait.

        Returns:
            Whether every connection finished in time.
        """
        with self.settled:
            return self.settled.wait_for(lambda: self.flight == 0, seconds)
