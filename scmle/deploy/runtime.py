"""The process a generated service runs: verify, load, warm up, serve.

`serve.py` in a service folder calls `run`. The HTTP server starts first so
`/livez` answers at once; `/readyz` turns true only after `verify_service`,
the bundle load and the warm-up finished. A folder that fails verification
leaves the process alive and unready, with the reason at `/readyz`, so the
orchestrator keeps it out of rotation and an operator can read why.
"""

from collections.abc import Mapping
import logging
import os
import pathlib
import signal
import sys
import threading
from typing import Any
import urllib.request

from scmle.artifacts import bundle as bundle_module
from scmle.contracts import errors
from scmle.deploy import verify
from scmle.inference import guard
from scmle.inference import serving

LOGGER = logging.getLogger("service")
DEFAULT_HOST = "0.0.0.0"  # nosec B104 - a container service binds all interfaces


def number(environ: Mapping[str, str], name: str, default: float) -> float:
    """Reads a numeric setting; a malformed value is a configuration error."""
    raw = environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as error:
        raise errors.ConfigError(f"{name} must be a number") from error


def settings(environ: Mapping[str, str]) -> dict[str, Any]:
    """The `HttpServing` keywords an environment asks for."""
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
    }


def make_loader(
    folder: pathlib.Path, environ: Mapping[str, str]
) -> serving.Loader:
    """Returns the loader that drives the verify and load readiness steps."""

    def loader(readiness: guard.Readiness) -> bundle_module.ModelBundle:
        readiness.begin(guard.Phase.VERIFYING)
        verified = verify.verify_service(folder, environ)
        readiness.complete(
            guard.Phase.VERIFYING,
            f"bundle {verified.bundle_digest}; "
            f"signed={verified.bundle_signed}; "
            f"provenance={verified.provenance_verified}",
        )
        readiness.begin(guard.Phase.LOADING)
        model = bundle_module.ModelBundle.load(
            folder / "model",
            trusted=verified.trusted,
            signing_key=environ.get(verify.SIGNING_KEY) or None,
        )
        readiness.complete(guard.Phase.LOADING)
        return model

    return loader


def run(folder: pathlib.Path, environ: Mapping[str, str]) -> None:
    """Serves the folder until the process is told to stop."""
    logging.basicConfig(level=logging.INFO)
    adapter = serving.HttpServing(
        loader=make_loader(folder, environ), **settings(environ)
    )
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda signum, frame: stop.set())
    adapter.start()
    LOGGER.info("listening on %s; not ready until verified", adapter.address)
    try:
        stop.wait()
    except KeyboardInterrupt:
        LOGGER.info("interrupted")
    finally:
        adapter.stop()


def healthcheck(environ: Mapping[str, str]) -> int:
    """Exit code for a container health check: 0 only when `/readyz` is 200."""
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
