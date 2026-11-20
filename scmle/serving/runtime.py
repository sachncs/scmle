"""The process a generated service runs: verify, load, warm up, serve.

`serve.py` in a service folder calls `run`. The HTTP server starts first so
`/livez` answers at once; `/readyz` turns true only after `inspect`,
the bundle load and the warm-up finished. A folder that fails verification
leaves the process alive and unready, with the reason at `/readyz`, so the
orchestrator keeps it out of rotation and an operator can read why.

`hold` is the serving loop `run` and `scmle serve` share: SIGTERM or
SIGINT takes the service out of rotation, serves on for the linger
(`SCMLE_LINGER_SECONDS`), drains the requests in flight and stops, SIGHUP
reloads the bundle without downtime, and a background thread writes the
operations signals.
"""

from collections.abc import Mapping
import json
import logging
import os
import pathlib
import queue
import signal
import sys
import threading
import time
from typing import Any
import urllib.request

from scmle.artifacts import bundle
from scmle.contracts import errors
from scmle.deploy import verify
from scmle.inference import guard
from scmle.operations import signals
from scmle.runtime import logs
from scmle.runtime import tracing
from scmle.security import access
from scmle.serving import server

LOGGER = logging.getLogger(__name__)
# Set (to any non-empty value) for JSON-lines logs, as `scmle --log-json`.
LOG_JSON = "SCMLE_LOG_JSON"
DEFAULT_HOST = "0.0.0.0"  # noqa: S104 -- a container service binds all interfaces  # nosec B104
# Seconds between two evaluations of the signals: the shortest window the
# operations signals evaluate (one minute), so each window is written once.
INTERVAL = 60.0
# The signals that stop the service (drained) and that reload its bundle.
STOPS = (signal.SIGTERM, signal.SIGINT)
RELOAD = getattr(signal, "SIGHUP", None)
# Seconds a stopping service stays out of rotation (`/readyz` 503) while it
# still serves, before it closes its socket: set it to the longest the
# platform takes to stop routing to a replica that is not ready (failure
# threshold times probe period plus timeout, 36 s for the deployment spec's
# probes). The default, 0, closes the socket at once: a container stop
# (`docker stop` waits 10 s) must not kill a service still lingering.
LINGER = "SCMLE_LINGER_SECONDS"


def number(environ: Mapping[str, str], name: str, default: float) -> float:
    """Reads a numeric setting; a malformed value is a configuration error.

    Args:
        environ: The environment.
        name: Setting name.
        default: Value used when the setting is absent or empty.

    Returns:
        The setting as a float, or `default`.

    Raises:
        ConfigError: If the setting is not a number.
    """
    raw = environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as error:
        raise errors.ConfigError(f"{name} must be a number") from error


def lingering(environ: Mapping[str, str]) -> float:
    """Reads how long a stopping service keeps serving out of rotation.

    Args:
        environ: The environment (`SCMLE_LINGER_SECONDS`).

    Returns:
        The seconds; 0 when unset, which closes the socket at once.

    Raises:
        ConfigError: If the setting is not a number or is negative.
    """
    found = number(environ, LINGER, 0.0)
    if found < 0:
        raise errors.ConfigError(f"{LINGER}: the linger must not be negative")
    return found


def withdraw(
    adapter: server.Service, inbox: queue.SimpleQueue[int], linger: float
) -> None:
    """Takes the service out of rotation and serves on for `linger` seconds.

    `/readyz` answers 503 (`draining`) at once, so the platform stops
    routing here, while every request that still arrives is served. A
    second stop signal ends the wait early; a reload signal is ignored.

    Args:
        adapter: The running server.
        inbox: The signals received.
        linger: Seconds to keep serving.
    """
    adapter.draining = True
    deadline = time.monotonic() + linger
    while (left := deadline - time.monotonic()) > 0:
        try:
            got = inbox.get(timeout=left)
        except queue.Empty:
            # handled: the linger ran out with no further signal
            return
        if got != RELOAD:
            return


def settings(environ: Mapping[str, str]) -> dict[str, Any]:
    """Builds the `Service` keywords an environment asks for.

    The default host is every interface, so a service started without the
    `SCMLE_AUTH_*` settings is refused by `Service` unless `HOST` is a
    loopback address: no container exposes an unauthenticated backend.

    Args:
        environ: The environment (`HOST`, `PORT`, `SCMLE_*` settings).

    Returns:
        Keyword arguments for `Service`: host, port, request and concurrency
        limits, scoring timeout, null and unseen-value policies, the access
        gateway and the drain time (`SCMLE_DRAIN_SECONDS`; unset or 0 keeps
        the derived default).
    """
    return {
        "host": environ.get("HOST", DEFAULT_HOST),
        "port": int(number(environ, "PORT", 8000)),
        "payload": int(
            number(environ, "SCMLE_MAX_REQUEST_BYTES", server.MAX_REQUEST_BYTES)
        ),
        "batch": int(
            number(environ, "SCMLE_MAX_RECORDS", server.MAX_REQUEST_RECORDS)
        ),
        "max_concurrent": int(
            number(
                environ,
                "SCMLE_MAX_CONCURRENT",
                server.MAX_CONCURRENT_REQUESTS,
            )
        ),
        "scoring_timeout_seconds": number(
            environ, "SCMLE_SCORING_TIMEOUT", server.SCORING_TIMEOUT_SECONDS
        ),
        "on_null": environ.get("SCMLE_ON_NULL", "impute"),
        "on_unseen": environ.get("SCMLE_ON_UNSEEN", "report"),
        "gateway": access.assemble(environ),
        "drain": (number(environ, "SCMLE_DRAIN_SECONDS", 0.0) or None),
    }


def cadence(environ: Mapping[str, str]) -> dict[str, Any]:
    """Reads where and how often the evaluated signals are written.

    Args:
        environ: The environment (`SCMLE_SIGNALS_FILE`,
            `SCMLE_SIGNALS_INTERVAL`).

    Returns:
        A dict with `path` (None to use the service log), `interval` in
        seconds (`INTERVAL` by default) and `window`, the seconds of traffic
        each evaluation covers (`SCMLE_SIGNALS_WINDOW`, the `Window`
        default otherwise).
    """
    return {
        "path": environ.get("SCMLE_SIGNALS_FILE") or None,
        "interval": number(environ, "SCMLE_SIGNALS_INTERVAL", INTERVAL),
        "window": number(
            environ, "SCMLE_SIGNALS_WINDOW", signals.Window().window
        ),
    }


def emit(
    monitor: signals.Monitor,
    now: float,
    path: pathlib.Path | str | None,
) -> int:
    """Evaluates the monitor and appends one JSON line.

    The line goes to `path` when given and to the service log otherwise.

    Args:
        monitor: The signal monitor to evaluate.
        now: Evaluation time, in seconds since the epoch.
        path: File to append to, or None for the log.

    Returns:
        The number of signals evaluated.
    """
    found = [item.to_dict() for item in monitor.evaluate(now)]
    line = json.dumps({"at": now, "signals": found}, sort_keys=True)
    if path is None:
        LOGGER.info("signals %s", line)
    else:
        with pathlib.Path(path).open(
            "a", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(line + "\n")
    return len(found)


def pulse(
    monitor: signals.Monitor,
    stop: threading.Event,
    path: pathlib.Path | str | None,
    interval: float,
) -> None:
    """Evaluates and writes the signals every `interval` seconds.

    Runs until `stop` is set. A failed evaluation is logged and the loop
    continues.

    Args:
        monitor: The signal monitor to evaluate.
        stop: Event that ends the loop when set.
        path: File to append to, or None for the log.
        interval: Seconds between evaluations.
    """
    while not stop.wait(interval):
        try:
            emit(monitor, time.time(), path)
        except Exception:  # the loop must outlive one bad evaluation
            LOGGER.exception("could not write the signals")


def prime(folder: pathlib.Path, environ: Mapping[str, str]) -> server.Loader:
    """Builds the loader that drives the verify and load readiness steps.

    Args:
        folder: Service folder holding the `model` directory.
        environ: The environment (including the signing key).

    Returns:
        A function that, given the readiness tracker, verifies the folder and
        loads the model bundle.
    """

    def loader(readiness: guard.Readiness) -> bundle.Bundle:
        """Verifies the folder, then loads the model.

        Args:
            readiness: The readiness tracker updated at each step.

        Returns:
            The loaded model bundle.
        """
        readiness.begin(guard.Phase.VERIFYING)
        verified = verify.inspect(folder, environ)
        readiness.complete(
            guard.Phase.VERIFYING,
            f"bundle {verified.digest}; "
            f"signed={verified.signed}; "
            f"provenance={verified.verified}",
        )
        readiness.begin(guard.Phase.LOADING)
        model = bundle.Bundle.load(
            folder / "model",
            trusted=verified.trusted,
            signing_key=environ.get(verify.SIGNING_KEY) or None,
        )
        readiness.complete(guard.Phase.LOADING)
        return model

    return loader


def fetch(path: pathlib.Path, trusted: bool) -> server.Loader:
    """Builds the loader that reads a bundle directory, for `scmle serve`.

    Args:
        path: The bundle directory.
        trusted: Whether an unsigned bundle is vouched for.

    Returns:
        A function that, given the readiness tracker, loads the bundle;
        `Bundle.load` checks its signature, so that is the verify step.
    """

    def loader(readiness: guard.Readiness) -> bundle.Bundle:
        """Loads the bundle, checking its signature.

        Args:
            readiness: The readiness tracker updated at each step.

        Returns:
            The loaded model bundle.
        """
        readiness.begin(guard.Phase.VERIFYING)
        readiness.complete(
            guard.Phase.VERIFYING, "signature checked while loading"
        )
        readiness.begin(guard.Phase.LOADING)
        model = bundle.Bundle.load(path, trusted=trusted)
        readiness.complete(guard.Phase.LOADING, str(path))
        return model

    return loader


def refresh(adapter: server.Service, loader: server.Loader | None) -> None:
    """Reloads the served bundle; a failure keeps the old one serving.

    Args:
        adapter: The running server.
        loader: The loader of the replacement; the adapter's own if None.
    """
    try:
        digest = adapter.reload(loader)
    except Exception:  # the old model keeps serving whatever went wrong
        LOGGER.exception("reload failed; the previous bundle keeps serving")
        return
    LOGGER.info("serving %s", digest)


def hold(
    adapter: server.Service,
    monitor: signals.Monitor,
    path: pathlib.Path | str | None,
    interval: float,
    loader: server.Loader | None = None,
    linger: float | None = None,
) -> None:
    """Serves until SIGTERM or SIGINT, reloading on SIGHUP, then drains.

    Must run on the main thread, which receives the signals. Starts the
    server and the thread that writes the signals; each SIGHUP reloads the
    bundle while the old one keeps serving (`Service.reload`). SIGTERM
    or SIGINT makes `/readyz` answer 503 and keeps serving for the linger
    (`withdraw`), then stops accepting connections, lets the requests in
    flight finish (`Service.stop`), writes the signals one last time and
    restores the previous signal handlers.

    Args:
        adapter: The server to run; not yet started.
        monitor: The signal monitor the server feeds.
        path: File the signals are appended to, or None for the log.
        interval: Seconds between two evaluations of the signals.
        loader: The loader SIGHUP reloads from; the adapter's own if None.
        linger: Seconds the stopping service stays out of rotation while
            still serving (`withdraw`) before it closes its socket;
            `SCMLE_LINGER_SECONDS` (default 0) when None.

    Raises:
        ConfigError: If `interval` is not positive or this is not the main
            thread.
    """
    if interval <= 0:
        raise errors.ConfigError("the signals interval must be positive")
    if threading.current_thread() is not threading.main_thread():
        raise errors.ConfigError("serve from the main thread")
    wait = lingering(os.environ) if linger is None else linger
    inbox: queue.SimpleQueue[int] = queue.SimpleQueue()
    wanted = [*STOPS, *([] if RELOAD is None else [RELOAD])]
    previous = {
        number: signal.signal(number, lambda *got: inbox.put(got[0]))
        for number in wanted
    }
    stop = threading.Event()
    writer = threading.Thread(
        target=pulse, args=(monitor, stop, path, interval), daemon=True
    )
    try:
        adapter.start()
        writer.start()
        LOGGER.info("listening on %s", adapter.address)
        while inbox.get() == RELOAD:
            refresh(adapter, loader)
        LOGGER.info("stopping: draining the requests in flight")
        withdraw(adapter, inbox, wait)
    finally:
        stop.set()
        adapter.stop()
        if writer.is_alive():
            writer.join(timeout=interval)
        emit(monitor, time.time(), path)
        for number, handler in previous.items():
            signal.signal(number, handler)


def run(folder: pathlib.Path, environ: Mapping[str, str]) -> None:
    """Serves the folder until the process is told to stop.

    Serves through `hold`: SIGHUP verifies and loads the folder again and
    swaps the model in without downtime; SIGTERM or SIGINT drains and
    returns once the server is stopped. With `SCMLE_TRACING` set the
    requests are traced (see `scmle.runtime.tracing`).

    Args:
        folder: Service folder.
        environ: The environment (settings, signing key, access settings,
            `SCMLE_LOG_JSON` and `SCMLE_TRACING`).
    """
    logs.configure(logging.INFO, structured=bool(environ.get(LOG_JSON)))
    if environ.get(tracing.SWITCH):
        tracing.TRACING.enable(environ=environ)
    chosen = cadence(environ)
    monitor = signals.Monitor(signals.Window(window=chosen["window"]))
    adapter = server.Service(
        loader=prime(folder, environ),
        monitor=monitor,
        **settings(environ),
    )
    try:
        hold(
            adapter,
            monitor,
            chosen["path"],
            chosen["interval"],
            linger=lingering(environ),
        )
    finally:
        tracing.TRACING.disable()


def healthcheck(environ: Mapping[str, str]) -> int:
    """Checks readiness for a container health check.

    Args:
        environ: The environment; `PORT` names the service port.

    Returns:
        0 only when `/readyz` on the loopback address answers with success,
        otherwise 1.
    """
    port = int(number(environ, "PORT", 8000))
    try:
        with urllib.request.urlopen(  # nosec B310 - fixed loopback URL
            f"http://127.0.0.1:{port}/readyz", timeout=2
        ):
            return 0
    except OSError:
        # handled: healthcheck exits 1 (unhealthy)
        return 1


if __name__ == "__main__":
    if sys.argv[1:] == ["healthcheck"]:
        sys.exit(healthcheck(os.environ))
    sys.exit("usage: python -m scmle.serving.runtime healthcheck")
