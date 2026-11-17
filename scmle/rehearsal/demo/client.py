"""Talking to the demo's services: processes, HTTP, the store and the sink."""

from collections.abc import Mapping
import http
import json
import os
import pathlib
import socket
import statistics
import subprocess  # nosec B404 - runs the generated serve.py and ps
import sys
import time
from typing import Any
from typing import NoReturn
import urllib.error
import urllib.request

import pandas

from scmle.contracts import errors
from scmle.contracts import observability
from scmle.deploy import mocks
from scmle.rehearsal.demo import fixture
from scmle.security import tokens


def launch(
    folder: pathlib.Path, port: int, environment: str = "production"
) -> subprocess.Popen[bytes]:
    """Starts a generated `serve.py` as the container would.

    `SCMLE_TRUSTED` is removed from the environment, and the published demo key
    is allowed.

    Args:
        folder: Service folder holding `serve.py`.
        port: Local port to listen on, on 127.0.0.1.
        environment: Value of `SCMLE_ENV` for the service.

    Returns:
        The running process; its stderr is piped and its stdout discarded.
    """
    env = {
        **os.environ,
        "HOST": "127.0.0.1",
        "PORT": str(port),
        "SCMLE_ENV": environment,
        "SCMLE_SIGNING_KEY": fixture.DEMO_SIGNING_KEY,
        "SCMLE_ALLOW_PUBLISHED_KEY": "1",
    }
    env.pop("SCMLE_TRUSTED", None)
    folder = folder.resolve()
    return subprocess.Popen(  # noqa: S603 -- our own interpreter and script  # nosec B603
        [sys.executable, str(folder / "serve.py")],
        cwd=folder,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


class Authenticator(urllib.request.BaseHandler):
    """Adds one bearer token to requests for the listed services only.

    Attributes:
        token: The bearer token.
        prefixes: URL prefixes the token may be sent to.
    """

    handler_order = 100

    def __init__(self, token: str, prefixes: tuple[str, ...]) -> None:
        """Remembers the token and the URL prefixes it may be sent to.

        Args:
            token: The bearer token.
            prefixes: URL prefixes the token may be sent to.
        """
        self.token = token
        self.prefixes = prefixes

    def http_request(
        self, request: urllib.request.Request
    ) -> urllib.request.Request:
        """Attaches the token if the URL is one of the listed services.

        Args:
            request: The outgoing request.

        Returns:
            The same request, with an `Authorization` header (not carried over
            redirects) when its URL starts with one of the prefixes.
        """
        if request.full_url.startswith(self.prefixes):
            request.add_unredirected_header(
                "Authorization", f"Bearer {self.token}"
            )
        return request


def authenticate(environ: Mapping[str, str], prefixes: tuple[str, ...]) -> bool:
    """Makes `urllib` authenticate to the compose services.

    The token is minted from the published demo key in `deploy/local/demo.env`
    and sent only to URLs that start with one of `prefixes`.

    Args:
        environ: Environment holding `SCMLE_AUTH_HS256_SECRET` and, optionally,
            `SCMLE_AUTH_ISSUER` and `SCMLE_AUTH_AUDIENCE`.
        prefixes: URL prefixes the token may be sent to.

    Returns:
        Whether a token was installed (the demo key was configured).
    """
    secret = environ.get("SCMLE_AUTH_HS256_SECRET")
    if not secret:
        return False
    now = time.time()
    token = tokens.sign_hs256(
        secret.encode(),
        "env",
        {
            "iss": environ.get("SCMLE_AUTH_ISSUER", ""),
            "aud": environ.get("SCMLE_AUTH_AUDIENCE", ""),
            "sub": "demo-test-runner",
            "iat": int(now),
            "exp": int(now) + 3600,
            "scope": "model:read model:predict",
        },
    )
    opener = urllib.request.build_opener(Authenticator(token, prefixes))
    urllib.request.install_opener(opener)
    return True


def retrieve(url: str, timeout: float = 10.0) -> tuple[int, dict[str, Any]]:
    """GETs JSON; HTTP errors return their status and body.

    Args:
        url: URL to request.
        timeout: Seconds to wait.

    Returns:
        The HTTP status and the parsed JSON body.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as reply:  # noqa: S310 -- URLs are http://127.0.0.1 addresses built by the demo  # nosec B310
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as error:
        # handled: HTTP error status returned to caller
        return error.code, json.loads(error.read())


def post_json(
    url: str, payload: dict[str, Any], timeout: float = 30.0
) -> tuple[int, dict[str, Any]]:
    """POSTs JSON; HTTP errors return their status and body.

    Args:
        url: URL to post to.
        payload: Object sent as the JSON body.
        timeout: Seconds to wait.

    Returns:
        The HTTP status and the parsed JSON body.
    """
    request = urllib.request.Request(  # noqa: S310 -- URLs are http://127.0.0.1 addresses built by the demo
        url,
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:  # noqa: S310 -- URLs are http://127.0.0.1 addresses built by the demo  # nosec B310
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as error:
        # handled: HTTP error status returned to caller
        return error.code, json.loads(error.read())


def settle(
    base: str,
    process: subprocess.Popen[bytes] | None = None,
    timeout: float = fixture.READY_TIMEOUT_SECONDS,
) -> float | None:
    """Waits until `/readyz` answers 200.

    Args:
        base: Base URL of the service.
        process: The service process; if it exits, waiting stops.
        timeout: Seconds to wait.

    Returns:
        Seconds until `/readyz` was 200, or None if it never was in time or the
        process exited.
    """
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if process is not None and process.poll() is not None:
            return None
        try:
            status = retrieve(f"{base}/readyz", timeout=2.0)[0]
        except OSError:
            # handled: not-yet-ready poll retried; timeout returns None
            status = 0
        if status == http.HTTPStatus.OK:
            return time.monotonic() - started
        time.sleep(0.2)
    return None


def halt(process: subprocess.Popen[bytes] | None) -> None:
    """Terminates a started service.

    The process is terminated, then killed if it has not exited after 15
    seconds, and its stderr pipe is closed.

    Args:
        process: The process to stop; None does nothing.
    """
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        # handled: graceful stop timed out; process killed
        process.kill()
        process.wait(timeout=15)
    if process.stderr is not None:
        process.stderr.close()


def unused() -> int:
    """Finds a free local TCP port.

    Returns:
        A port on 127.0.0.1 that was free when probed.
    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def sync(base: str, folder: pathlib.Path, prefix: str) -> dict[str, str]:
    """Uploads every file of `folder`.

    Args:
        base: Base URL of the object store.
        folder: Folder whose files are uploaded recursively.
        prefix: Key prefix under which the files are stored.

    Returns:
        A mapping from each file's path relative to `folder` to the entity tag
        the store returned.
    """
    return {
        path.relative_to(folder).as_posix(): mocks.deposit(
            base,
            f"{prefix}/{path.relative_to(folder).as_posix()}",
            path.read_bytes(),
        )
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def download(
    base: str, prefix: str, expected: dict[str, str], destination: pathlib.Path
) -> None:
    """Fetches the folder back, refusing any file whose digest differs.

    Args:
        base: Base URL of the object store.
        prefix: Key prefix under which the files are stored.
        expected: Mapping from relative file path to the expected entity tag.
        destination: Folder to write the files to; created as needed.
    """
    for name, etag in expected.items():
        data = mocks.get_object(base, f"{prefix}/{name}", expected=etag)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def serialize(frame: pandas.DataFrame) -> list[dict[str, Any]]:
    """Converts rows to JSON-ready dictionaries with string keys.

    Args:
        frame: The data frame to convert.

    Returns:
        One dict per row.
    """
    return [
        {str(key): value for key, value in row.items()}
        for row in frame.to_dict("records")
    ]


def footprint(process: subprocess.Popen[bytes] | None) -> float | None:
    """Reads the resident memory of a child process.

    Args:
        process: The process to measure; None gives None.

    Returns:
        The resident set size in MiB rounded to one decimal, or None if no
        process was given or the platform did not report a number (read with
        `ps` on POSIX and `psutil` on Windows).
    """
    if process is None:
        return None
    if sys.platform == "win32":
        import psutil  # noqa: PLC0415 -- Windows-only dependency  # pyright: ignore[reportMissingModuleSource]  # psutil is a Windows-only dependency and ships no stub source here

        try:
            return round(
                psutil.Process(process.pid).memory_info().rss / 1048576, 1
            )
        except psutil.Error:
            # handled: documented None when platform reports no number
            return None
    result = subprocess.run(  # noqa: S603 -- fixed arguments  # nosec B603 B607
        ["ps", "-o", "rss=", "-p", str(process.pid)],  # noqa: S607 -- ps is looked up on PATH, as on every POSIX host
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return round(int(result.stdout.strip()) / 1024, 1)
    except ValueError:
        # handled: documented None when platform reports no number
        return None


def measure(
    base: str, rows: list[dict[str, Any]], count: int = fixture.LATENCY_REQUESTS
) -> dict[str, float]:
    """Measures latency for sequential requests of `rows`.

    Args:
        base: Base URL of the service.
        rows: Records sent in each request.
        count: Number of requests.

    Returns:
        The request count and the p50, p95, p99 and mean latency in milliseconds
        (percentiles by index into the sorted timings).

    Raises:
        ScmleError: If a request is not answered 200.
    """
    timings = []
    for attempt in range(count):
        started = time.perf_counter()
        status = post_json(f"{base}/predict", {"records": rows})[0]
        if status != http.HTTPStatus.OK:
            raise errors.ScmleError(
                f"measurement request {attempt} failed: {status}"
            )
        timings.append((time.perf_counter() - started) * 1000)
    timings.sort()

    def percentile(share: float) -> float:
        """Looks up a latency percentile.

        Args:
            share: Fraction in [0, 1].

        Returns:
            The timing at that share of the sorted timings, in milliseconds,
            rounded to two decimals.
        """
        return round(
            timings[min(len(timings) - 1, int(share * len(timings)))], 2
        )

    return {
        "requests": float(count),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "mean_ms": round(statistics.fmean(timings), 2),
    }


def push(
    sink: str, values: dict[str, tuple[float, str]], model_version: str
) -> int:
    """Sends named values to the metrics sink.

    Args:
        sink: Base URL of the metrics sink.
        values: Mapping from metric name to its value and unit. A name that
            starts with `latency` or `memory` is sent as that kind of signal;
            any other is sent as drift.
        model_version: Version label attached to every sample.

    Returns:
        How many samples the sink accepted.
    """
    kinds = {
        "latency": observability.Indicator.LATENCY,
        "memory": observability.Indicator.MEMORY,
    }
    samples = [
        observability.Sample(
            name=name,
            kind=kinds.get(name.split("_")[0], observability.Indicator.DRIFT),
            value=value,
            unit=unit,
            timestamp=fixture.now(),
            labels={"model": fixture.MODEL, "model_version": model_version},
        )
        for name, (value, unit) in values.items()
    ]
    return mocks.transmit(sink, samples)


def unready(process: subprocess.Popen[bytes]) -> NoReturn:
    """Raises the error for a service that did not become ready.

    Args:
        process: The service process; the tail of its stderr is the message.

    Raises:
        ArtifactError: Always, carrying the last 800 characters of stderr.
    """
    raise errors.ArtifactError(
        (process.stderr.read().decode() if process.stderr else "")[-800:]
    )
