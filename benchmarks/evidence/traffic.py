"""Real HTTP traffic against `scmle serve` in a child process (C7 evidence).

The client side of `benchmarks.evidence.service`: a child running the real
command, clients that send requests on new connections and tell an answer
from a reset, a refusal and a timeout, and the counting rules (`minimum`
and `bound`) the measurements decide with.
"""

import collections
import contextlib
import dataclasses
import http
import http.client
import json
import math
import os
import pathlib
import re
import signal
import socket
import subprocess  # nosec B404
import sys
import threading
import time
from typing import Any
from typing import Self

from scipy import stats

from scmle.contracts import errors
from scmle.serving import server

ROOT = pathlib.Path(__file__).resolve().parents[2]
# Seconds a child may take to listen and to become ready: twice the 30 s
# startup objective of docs/SERVICE_SLO.md, so a slow start is measured
# against the objective instead of ending the measurement.
WAIT = 60.0
# The drain time `scmle serve` defaults to (`web.patience`): request I/O
# twice plus the scoring timeout, the longest an admitted request takes.
DRAIN = 2 * 10.0 + server.SCORING_TIMEOUT_SECONDS
# Slack on top of a wait that is already the longest the service may take.
MARGIN = 10.0
# A client gives up after the drain time and twice the margin.
CALL_SECONDS = DRAIN + 2 * MARGIN
# Seconds a client waits after a refused connection before it tries again,
# as a caller backs off from a replica that has stopped listening.
PAUSE = 0.01
LISTENING = re.compile(r"listening on (http://\S+)")
RESETS = (
    ConnectionResetError,
    BrokenPipeError,
    ConnectionAbortedError,
    http.client.IncompleteRead,
    http.client.BadStatusLine,
)


@dataclasses.dataclass(frozen=True)
class Outcome:
    """What one request got.

    Attributes:
        started: Monotonic time the request began.
        seconds: Time until the answer or the failure.
        status: The HTTP status, or None when there was no answer.
        kind: `answered`, `reset`, `refused`, `timeout` or an error name.
    """

    started: float
    seconds: float
    status: int | None
    kind: str


def send(
    address: tuple[str, int],
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float = CALL_SECONDS,
) -> Outcome:
    """Sends one request on a new connection and classifies what came back.

    Args:
        address: Host and port.
        path: The request path.
        payload: A JSON body to post, or None for a GET.
        timeout: Seconds before the client gives up.

    Returns:
        The outcome; a reset, a refusal and a timeout are told apart.
    """
    started = time.monotonic()
    connection = http.client.HTTPConnection(*address, timeout=timeout)
    status, kind = None, "answered"
    try:
        if payload is None:
            connection.request("GET", path)
        else:
            body = json.dumps(payload).encode()
            headers = {"Content-Type": "application/json"}
            connection.request("POST", path, body, headers)
        response = connection.getresponse()
        response.read()
        status = response.status
    except ConnectionRefusedError:
        kind = "refused"
    except RESETS:
        kind = "reset"
    except TimeoutError:
        kind = "timeout"
    except OSError as error:
        kind = type(error).__name__
    finally:
        connection.close()
    return Outcome(started, time.monotonic() - started, status, kind)


class Load:
    """Clients sending one payload over and over until stopped.

    Attributes:
        outcomes: Every request's outcome, in completion order.
        stop: Set to end the clients.
    """

    def __init__(
        self,
        address: tuple[str, int],
        payload: dict[str, Any],
        clients: int,
        gate: threading.Event | None = None,
    ) -> None:
        """Prepares the clients without starting them.

        Args:
            address: The service.
            payload: The request every client sends.
            clients: How many clients send at once.
            gate: While set, the clients send; once cleared they stop, as
                callers do when the platform stops routing to the replica.
                None sends until `stop`.
        """
        self.address = address
        self.payload = payload
        self.gate = gate
        self.outcomes: list[Outcome] = []
        self.stop = threading.Event()
        self.threads = [
            threading.Thread(target=self.score, daemon=True)
            for ignored in range(clients)
        ]

    def score(self) -> None:
        """Sends requests until `stop` is set."""
        while not self.stop.is_set() and (
            self.gate is None or self.gate.is_set()
        ):
            outcome = send(self.address, "/predict", self.payload)
            self.outcomes.append(outcome)
            if outcome.kind == "refused":
                time.sleep(PAUSE)

    def __enter__(self) -> Self:
        """Starts the clients."""
        for thread in self.threads:
            thread.start()
        return self

    def __exit__(self, *ignored: object) -> None:
        """Stops the clients and waits for their last requests."""
        self.stop.set()
        for thread in self.threads:
            thread.join(CALL_SECONDS + MARGIN)


class Replica:
    """`scmle serve` in a child process, its log read all along.

    Attributes:
        process: The child.
        lines: Its log lines so far.
        address: Host and port, once it listens.
        started: Monotonic time the child was started.
    """

    def __init__(
        self, bundle: pathlib.Path, settings: dict[str, str] | None = None
    ) -> None:
        """Starts the child and waits until it listens.

        Args:
            bundle: The bundle directory served.
            settings: Environment variables set for the child.

        Raises:
            ScmleError: If it does not listen within `WAIT` seconds.
        """
        environment = {**os.environ, **(settings or {})}
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT), environment.get("PYTHONPATH", "")]
        )
        self.started = time.monotonic()
        command = [sys.executable, "-m", "scmle", "serve", str(bundle)]
        self.process = subprocess.Popen(  # nosec B603
            [*command, "--trusted", "--port", "0"],
            cwd=ROOT,
            env=environment,
            stderr=subprocess.PIPE,
        )
        self.lines: list[str] = []
        self.listening = threading.Event()
        self.address: tuple[str, int] = ("", 0)
        threading.Thread(target=self.read, daemon=True).start()
        if not self.listening.wait(WAIT):
            self.close()
            raise errors.ScmleError("the service never listened: " + self.log)

    @property
    def log(self) -> str:
        """The last lines of the child's log."""
        return "".join(self.lines[-40:])

    def read(self) -> None:
        """Drains the log (a full pipe would block the service)."""
        stream = self.process.stderr
        if stream is None:
            return
        for raw in stream:
            line = raw.decode("utf-8", "replace")
            self.lines.append(line)
            found = LISTENING.search(line)
            if found and not self.listening.is_set():
                host, port = found.group(1).removeprefix("http://").split(":")
                self.address = (host, int(port))
                self.listening.set()

    def ready(self) -> float:
        """Polls `/readyz` until 200.

        Returns:
            Seconds from the child's start to the first 200.

        Raises:
            ScmleError: If it is not ready within `WAIT` seconds.
        """
        while time.monotonic() - self.started < WAIT:
            if (
                send(self.address, "/readyz", timeout=5).status
                == http.HTTPStatus.OK
            ):
                return time.monotonic() - self.started
            time.sleep(0.05)
        raise errors.ScmleError("the service never became ready: " + self.log)

    def terminate(self, timeout: float) -> int | None:
        """Sends SIGTERM and waits for the exit.

        Args:
            timeout: Seconds to wait.

        Returns:
            The exit status, or None if it had not exited (it is killed).
        """
        self.process.send_signal(signal.SIGTERM)
        try:
            return self.process.wait(timeout)
        except subprocess.TimeoutExpired:
            self.close()
            return None

    def close(self) -> None:
        """Kills the child if it still runs."""
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(30)


def minimum(failure: float, alpha: float) -> int:
    """Requests with zero failures that bound the failure rate.

    Args:
        failure: The largest rate to bound.
        alpha: One-sided error rate.

    Returns:
        The smallest n with 1 - alpha ** (1 / n) <= failure.
    """
    return math.ceil(math.log(alpha) / math.log(1 - failure))


def bound(latencies: list[float], quantile: float, alpha: float) -> float:
    """Distribution-free upper confidence bound of a latency quantile.

    The order statistic X_(j+1), with j the (1 - alpha) quantile of
    Binomial(n, quantile), lies above the true quantile with probability at
    least 1 - alpha, whatever the latency distribution.

    Args:
        latencies: Observed latencies.
        quantile: The quantile bounded, such as 0.95.
        alpha: One-sided error rate.

    Returns:
        The bound; infinity when there are too few observations.
    """
    ordered = sorted(latencies)
    if not ordered:
        return math.inf
    index = int(stats.binom.ppf(1 - alpha, len(ordered), quantile))
    return ordered[index] if index < len(ordered) else math.inf


def tally(outcomes: list[Outcome]) -> dict[str, int]:
    """Counts outcomes by status or kind.

    Args:
        outcomes: The outcomes.

    Returns:
        `200`, `504`, ... for answers, the kind otherwise.
    """
    return dict(
        collections.Counter(
            str(item.status) if item.kind == "answered" else item.kind
            for item in outcomes
        )
    )


def crawl(address: tuple[str, int], count: int) -> list[socket.socket]:
    """Opens connections that send a body too slowly to ever finish it.

    Args:
        address: The service.
        count: How many.

    Returns:
        The open sockets, each with its headers and one byte of body sent.
    """
    opened = []
    for ignored in range(count):
        connection = socket.create_connection(address, timeout=CALL_SECONDS)
        connection.sendall(
            b"POST /predict HTTP/1.0\r\nContent-Type: application/json\r\n"
            b"Content-Length: 1000\r\n\r\n{"
        )
        opened.append(connection)
    return opened


def answered(connection: socket.socket) -> int | None:
    """Reads the status a slow connection was answered with.

    Args:
        connection: A socket from `crawl`.

    Returns:
        The status, or None if it was closed or reset without one.
    """
    with contextlib.closing(connection):
        try:
            head = connection.makefile("rb").readline()
        except OSError:
            # handled: no status line counts as no answer
            return None
    parts = head.split()
    return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None


def watch(
    address: tuple[str, int], stop: threading.Event, seen: list[Any]
) -> None:
    """Polls `/readyz` and `/livez` every 0.2 s until stopped.

    Args:
        address: The service.
        stop: Ends the polling.
        seen: Gets (path, status) per poll.
    """
    while not stop.wait(0.2):
        seen.extend(
            (path, send(address, path, timeout=5).status)
            for path in ("/readyz", "/livez")
        )


class Request(threading.Thread):
    """One request sent on its own thread.

    Attributes:
        outcome: What it got, once finished.
    """

    def __init__(
        self, address: tuple[str, int], payload: dict[str, Any]
    ) -> None:
        """Starts the request."""
        super().__init__(daemon=True)
        self.address = address
        self.payload = payload
        self.outcome = Outcome(time.monotonic(), 0.0, None, "unsent")
        self.start()

    def run(self) -> None:
        """Sends it."""
        self.outcome = send(self.address, "/predict", self.payload)

    def finish(self) -> Outcome:
        """Waits for the answer.

        Returns:
            The outcome; `unsent` if it never finished.
        """
        self.join(CALL_SECONDS + MARGIN)
        return self.outcome


class Router(threading.Thread):
    """Routes callers to a replica while its readiness probe passes.

    The platform's part in a stop: it probes `/readyz` every `period`
    seconds, gives each probe `timeout` seconds to answer, and stops routing
    after `threshold` failures in a row, so it stops at most `threshold`
    times `period + timeout` after the replica turns unready.

    Attributes:
        routes: Set while callers are routed to the replica.
        probes: The status of every probe (None when it got no answer).
    """

    def __init__(
        self,
        address: tuple[str, int],
        period: float,
        timeout: float,
        threshold: int,
    ) -> None:
        """Starts probing; the replica is routed to from the start.

        Args:
            address: The replica.
            period: Seconds between two probes.
            timeout: Seconds a probe may take before it counts as failed.
            threshold: Failures in a row that stop the routing.
        """
        super().__init__(daemon=True)
        self.address = address
        self.period = period
        self.timeout = timeout
        self.threshold = threshold
        self.routes = threading.Event()
        self.routes.set()
        self.probes: list[int | None] = []
        self.start()

    def run(self) -> None:
        """Probes until `threshold` failures in a row, then stops routing."""
        failures = 0
        while failures < self.threshold:
            time.sleep(self.period)
            status = send(self.address, "/readyz", timeout=self.timeout).status
            self.probes.append(status)
            failures = 0 if status == http.HTTPStatus.OK else failures + 1
        self.routes.clear()
