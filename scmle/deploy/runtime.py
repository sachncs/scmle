"""The process a generated service runs: verify, load, warm up, serve.

`serve.py` in a service folder calls `run`. The HTTP server starts first so
`/livez` answers at once; `/readyz` turns true only after `verify_service`,
the bundle load and the warm-up finished. A folder that fails verification
leaves the process alive and unready, with the reason at `/readyz`, so the
orchestrator keeps it out of rotation and an operator can read why.
"""

from collections.abc import Mapping
import json
import logging
import os
import pathlib
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
from scmle.inference import serving
from scmle.operations import signals
from scmle.security import access

LOGGER = logging.getLogger("service")
DEFAULT_HOST = "0.0.0.0"  # nosec B104 - a container service binds all interfaces


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


def settings(environ: Mapping[str, str]) -> dict[str, Any]:
    """Builds the `HttpServing` keywords an environment asks for.

    The default host is every interface, so a service started without the
    `SCMLE_AUTH_*` settings is refused by `HttpServing` unless `HOST` is a
    loopback address: no container exposes an unauthenticated backend.

    Args:
        environ: The environment (`HOST`, `PORT`, `SCMLE_*` settings).

    Returns:
        Keyword arguments for `HttpServing`: host, port, request and concurrency
        limits, scoring timeout, null and unseen-value policies, and the access
        gateway.
    """
    return {
        "host": environ.get("HOST", DEFAULT_HOST),
        "port": int(number(environ, "PORT", 8000)),
        "max_request_bytes": int(
            number(
                environ, "SCMLE_MAX_REQUEST_BYTES", serving.MAX_REQUEST_BYTES
            )
        ),
        "max_records": int(
            number(environ, "SCMLE_MAX_RECORDS", serving.MAX_REQUEST_RECORDS)
        ),
        "max_concurrent": int(
            number(
                environ,
                "SCMLE_MAX_CONCURRENT",
                serving.MAX_CONCURRENT_REQUESTS,
            )
        ),
        "scoring_timeout_seconds": number(
            environ, "SCMLE_SCORING_TIMEOUT", serving.SCORING_TIMEOUT_SECONDS
        ),
        "on_null": environ.get("SCMLE_ON_NULL", "impute"),
        "on_unseen": environ.get("SCMLE_ON_UNSEEN", "report"),
        "gateway": access.gateway_from_environment(environ),
    }


def signal_settings(environ: Mapping[str, str]) -> dict[str, Any]:
    """Reads where and how often the evaluated signals are written.

    Args:
        environ: The environment (`SCMLE_SIGNALS_FILE`,
            `SCMLE_SIGNALS_INTERVAL`).

    Returns:
        A dict with `path` (None to use the service log) and `interval` in
        seconds (60 by default).
    """
    return {
        "path": environ.get("SCMLE_SIGNALS_FILE") or None,
        "interval": number(environ, "SCMLE_SIGNALS_INTERVAL", 60.0),
    }


def emit_signals(
    monitor: signals.SignalMonitor,
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
        with open(path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
    return len(found)


def signal_loop(
    monitor: signals.SignalMonitor,
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
            emit_signals(monitor, time.time(), path)
        except Exception:  # the loop must outlive one bad evaluation
            LOGGER.exception("could not write the signals")


def make_loader(
    folder: pathlib.Path, environ: Mapping[str, str]
) -> serving.Loader:
    """Builds the loader that drives the verify and load readiness steps.

    Args:
        folder: Service folder holding the `model` directory.
        environ: The environment (including the signing key).

    Returns:
        A function that, given the readiness tracker, verifies the folder and
        loads the model bundle.
    """

    def loader(readiness: guard.Readiness) -> bundle.ModelBundle:
        """Verifies the folder, then loads the model.

        Args:
            readiness: The readiness tracker updated at each step.

        Returns:
            The loaded model bundle.
        """
        readiness.begin(guard.Phase.VERIFYING)
        verified = verify.verify_service(folder, environ)
        readiness.complete(
            guard.Phase.VERIFYING,
            f"bundle {verified.bundle_digest}; "
            f"signed={verified.bundle_signed}; "
            f"provenance={verified.provenance_verified}",
        )
        readiness.begin(guard.Phase.LOADING)
        model = bundle.ModelBundle.load(
            folder / "model",
            trusted=verified.trusted,
            signing_key=environ.get(verify.SIGNING_KEY) or None,
        )
        readiness.complete(guard.Phase.LOADING)
        return model

    return loader


def run(folder: pathlib.Path, environ: Mapping[str, str]) -> None:
    """Serves the folder until the process is told to stop.

    Starts the HTTP server and a background thread that writes the signals, and
    returns after SIGTERM or a keyboard interrupt, once the server is stopped.

    Args:
        folder: Service folder.
        environ: The environment (settings, signing key and access settings).
    """
    logging.basicConfig(level=logging.INFO)
    monitor = signals.SignalMonitor()
    adapter = serving.HttpServing(
        loader=make_loader(folder, environ),
        monitor=monitor,
        **settings(environ),
    )
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda signum, frame: stop.set())
    chosen = signal_settings(environ)
    threading.Thread(
        target=signal_loop,
        args=(monitor, stop, chosen["path"], chosen["interval"]),
        daemon=True,
    ).start()
    adapter.start()
    LOGGER.info("listening on %s; not ready until verified", adapter.address)
    try:
        stop.wait()
    except KeyboardInterrupt:
        LOGGER.info("interrupted")
    finally:
        adapter.stop()


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
        return 1


if __name__ == "__main__":
    if sys.argv[1:] == ["healthcheck"]:
        sys.exit(healthcheck(os.environ))
    sys.exit("usage: python -m scmle.deploy.runtime healthcheck")
