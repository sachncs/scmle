"""Claim C7, service clauses: objectives under load, a hung request, SIGTERM.

Every measurement drives the real command, `python -m scmle serve BUNDLE
--trusted --port 0`, in a child process over real HTTP, one new connection
per request (the server speaks HTTP/1.0), the way a caller sees it. The
served bundle is a random forest the product's own preprocessing fits on
`reliability.binary_data` (`ROWS` rows): a real tabular bundle of about
8 MB, inside the 50 MiB of the objectives' profile.

* Objectives (`docs/SERVICE_SLO.md`, "Objectives"). `CLIENTS` clients score
  1-record requests, then 100-record requests, each profile until
  `minimum()` requests are sent. A profile passes when no request failed
  (anything but a 200, valid requests only, so every 4xx and 5xx counts),
  and the distribution-free upper confidence bound of the 95th percentile
  latency (`bound`) is within the objective. Startup to `/readyz` 200 and
  resident memory after the load are checked against their objectives.
* A hung request (`hang`). The bundle gets a `Stall` step that hangs for
  `HANG_SECONDS` on a row whose `a` is `SENTINEL`, inside the model's
  predict. While `CLIENTS` clients keep scoring and `max_concurrent`
  clients send a body too slowly to finish, one hung request is sent: it
  must be answered 504 at the scoring timeout, every other request must be
  answered 200, before and after the 504, the slow clients 408, and
  `/readyz` and `/livez` 200 throughout. Then every remaining worker is
  hung: the replica must say it is not alive (`/livez` 503), so the
  platform restarts it, and SIGTERM must still end the process.
* SIGTERM (`stop`). `minimum(STOP_FAILURE, alpha)` rounds: the child runs
  with `SCMLE_LINGER_SECONDS` = `LINGER`; `DRAIN_CLIENTS` clients (twice
  the admission limit, so the listen backlog is never empty) score
  100-record requests through a `Router` that probes `/readyz` as the
  platform does and stops routing after `PROBES.threshold` failures;
  SIGTERM is sent after `WARM_SECONDS` and the clients go on until routing
  stops. A connection reset (or closed with no answer) counts against the claim;
  so does a client timeout or an exit status other than 0. A round tests
  nothing unless the router still routed when the signal was sent and a
  request started after the signal was answered, so both are required. A
  refused connection (the socket closed before the client connected) is
  counted and reported, not a reset: the caller sent nothing. `DEFAULT`
  rounds run the same way with the product's default environment (no
  linger) and decide too. `DIRECT` further
  rounds with clients that ignore readiness and connect straight to the
  replica until it exits are reported and do not decide: a connection the
  kernel completes between the last accept and the close is reset by the
  kernel whatever the server does.

NOT measured here: the 24-hour soak (`--soak SECONDS` runs it for any
duration and prints its result), idle CPU, the deployment spec's 1 CPU
limit (the child runs unconstrained on the measuring machine, whose CPU
count and load are filed), and modality packages.

    python -m benchmarks.evidence.service --write
    python -m benchmarks.evidence.service --check
    python -m benchmarks.evidence.service --soak 86400 --output soak.json
"""

import argparse
import collections
import hashlib
import http
import json
import math
import os
import pathlib
import platform
import statistics
import tempfile
import threading
import time
from typing import Any

import pandas
from sklearn import base
from sklearn import pipeline

from benchmarks import acceptance
from benchmarks import evidence
from benchmarks import reliability
from benchmarks.evidence import traffic
import scmle.artifacts.bundle
from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.deploy import contract
from scmle.inference import guard
from scmle.models import registry
from scmle.preparation import preprocessing
from scmle.rehearsal.demo import client
from scmle.serving import runtime
from scmle.serving import server

PATH = pathlib.Path(__file__).with_suffix(".json")
ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "scmle"
# What a request to `scmle serve` runs, from the socket to the model.
PRODUCT = (
    "artifacts",
    "cli",
    "contracts",
    "inference",
    "operations",
    "preparation",
    "runtime",
    "security",
    "serving",
)
# The objectives of docs/SERVICE_SLO.md, declared before measurement.
P95_ONE_SECONDS = 0.1
P95_HUNDRED_SECONDS = 0.5
STARTUP_SECONDS = 30.0
RESIDENT_MIB = 400.0
QUANTILE = 0.95
# Largest failure rate a profile must bound: zero failures in n requests
# bound it at 1 - alpha ** (1 / n), the exact one-sided Clopper-Pearson
# limit, so n >= log(alpha) / log(1 - FAILURE) (2995 at alpha 0.05).
FAILURE = 0.001
# Largest share of SIGTERMs that may reset a connection, bounded the same
# way over the rounds (29 at alpha 0.05).
STOP_FAILURE = 0.1
# Clients scoring at once: half the admission limit, so a valid request is
# never refused with a 429 and every failure is the service's.
CLIENTS = server.MAX_CONCURRENT_REQUESTS // 2
DRAIN_CLIENTS = 2 * server.MAX_CONCURRENT_REQUESTS
WARM_SECONDS = 2.0
ROWS = 5000
SEED = 7
# Longer than the whole measurement: the hung call never returns in it.
HANG_SECONDS = 3600.0
SENTINEL = 1e12
# The platform's probe in the SIGTERM rounds: the deployment spec's probe
# timeout and failure threshold (`deploy.contract.Probes`), so a probe slowed
# by the load is not counted as failed early; the period is a twentieth of
# the spec's 10 s so a round takes seconds, not a minute.
PROBES = contract.Probes()
PERIOD = PROBES.period / 20
# The linger the configured rounds give the service (`SCMLE_LINGER_SECONDS`):
# the latest the router can stop routing after the replica turns unready,
# the threshold of probes that each wait a period and then time out.
LINGER = PROBES.threshold * (PERIOD + PROBES.timeout)
# Rounds run with the product's own default environment (no
# `SCMLE_LINGER_SECONDS`): they decide too, so the verdict covers the
# service as it ships, not only as configured for the drain.
DEFAULT = 1
# Rounds with clients that ignore readiness and keep connecting straight to
# the replica: reported, not deciding (no server can answer a connection
# the kernel completes after its socket closed; the platform's routing is
# what keeps callers away by then).
DIRECT = 3
# Most seconds one load profile may run before it is reported short.
PROFILE_SECONDS = 600.0
# Soak: seconds between two memory samples, and the growth in KiB per
# request answered that counts as a leak (tests/integration/test_serve_soak).
SAMPLE_SECONDS = 10.0
LEAK = 12.0


class Stall(base.BaseEstimator, base.TransformerMixin):  # type: ignore[misc]
    """A pipeline step that hangs on a sentinel row and passes rows through.

    Attributes:
        column: The column it watches.
        sentinel: The value that makes it hang.
        seconds: How long it hangs.
    """

    def __init__(
        self,
        column: str = "a",
        sentinel: float = SENTINEL,
        seconds: float = HANG_SECONDS,
    ) -> None:
        """Keeps the parameters, as every scikit-learn estimator does."""
        self.column = column
        self.sentinel = sentinel
        self.seconds = seconds

    def fit(self, frame: Any, target: Any = None) -> "Stall":
        """Learns nothing."""
        del frame, target
        return self

    def transform(self, frame: pandas.DataFrame) -> pandas.DataFrame:
        """Hangs when a row carries the sentinel, then passes the rows on."""
        if (frame[self.column] == self.sentinel).any():
            time.sleep(self.seconds)
        return frame

    def __sklearn_is_fitted__(self) -> bool:
        """Always fitted: it has nothing to learn."""
        return True


def forest(folder: pathlib.Path, stall: bool) -> pathlib.Path:
    """Fits and saves the served bundle.

    Args:
        folder: Where to save it.
        stall: Whether to put a `Stall` step in front of the pipeline.

    Returns:
        The bundle directory.
    """
    frame = reliability.binary_data(rows=ROWS, seed=SEED)
    target = frame.pop("y")
    plan = data.Blueprint(("a", "b"), ("c",))
    spec = registry.Catalog.stock().get("random_forest")
    fitted = preprocessing.instantiate(plan, spec, tasks.Task.BINARY, 0)
    fitted.fit(frame, target)
    if stall:
        fitted = pipeline.Pipeline([("stall", Stall()), *fitted.steps])
    model = scmle.artifacts.bundle.Bundle.create(
        fitted, plan, frame, "y", "binary_classification", "roc_auc", "forest"
    )
    return pathlib.Path(model.save(folder / ("stall" if stall else "forest")))


def records(count: int) -> list[dict[str, Any]]:
    """Valid records drawn from the training distribution.

    Args:
        count: How many.

    Returns:
        Records with `a`, `b` and `c`.
    """
    frame = reliability.binary_data(rows=count, seed=SEED + 1)
    return [
        {str(key): value for key, value in row.items()}
        for row in frame.drop(columns="y").to_dict("records")
    ]


def profile(
    address: tuple[str, int], size: int, target: float, alpha: float
) -> dict[str, Any]:
    """Scores requests of `size` records until the minimum is reached.

    Args:
        address: The service.
        size: Records per request.
        target: The objective for the 95th percentile, in seconds.
        alpha: One-sided error rate.

    Returns:
        Counts, latency quantiles, the bound and whether the profile passes.
    """
    least = traffic.minimum(FAILURE, alpha)
    started = time.monotonic()
    with traffic.Load(address, {"records": records(size)}, CLIENTS) as load:
        while (
            len(load.outcomes) < least
            and time.monotonic() - started < PROFILE_SECONDS
        ):
            time.sleep(0.2)
    outcomes = list(load.outcomes)
    good = sorted(
        item.seconds for item in outcomes if item.status == http.HTTPStatus.OK
    )
    failed = len(outcomes) - len(good)
    upper = traffic.bound(good, QUANTILE, alpha)
    return {
        "records": size,
        "requests": len(outcomes),
        "minimum": least,
        "failed": failed,
        "outcomes": traffic.tally(outcomes),
        "p50": statistics.median(good) if good else None,
        "p95": statistics.quantiles(good, n=100)[94] if len(good) > 1 else None,
        "p99": statistics.quantiles(good, n=100)[98] if len(good) > 1 else None,
        "p95_bound": upper,
        "p95_target": target,
        "passed": len(outcomes) >= least and failed == 0 and upper <= target,
    }


def objectives(folder: pathlib.Path, alpha: float) -> dict[str, Any]:
    """Startup, the two latency profiles and resident memory.

    Args:
        folder: Scratch directory.
        alpha: One-sided error rate.

    Returns:
        The measurements and whether every objective is met.
    """
    replica = traffic.Replica(forest(folder, stall=False))
    try:
        startup = replica.ready()
        found = [
            profile(replica.address, 1, P95_ONE_SECONDS, alpha),
            profile(replica.address, 100, P95_HUNDRED_SECONDS, alpha),
        ]
        resident = client.footprint(replica.process)
        code = replica.terminate(traffic.DRAIN + traffic.MARGIN)
    finally:
        replica.close()
    return {
        "startup_seconds": startup,
        "resident_mib": resident,
        "profiles": found,
        "exit": code,
        "passed": startup <= STARTUP_SECONDS
        and resident is not None
        and resident <= RESIDENT_MIB
        and all(item["passed"] for item in found)
        and code == 0,
    }


def hang(
    folder: pathlib.Path, timeout: float = server.SCORING_TIMEOUT_SECONDS
) -> dict[str, Any]:
    """One hung request among steady load, then every worker hung.

    Args:
        folder: Scratch directory.
        timeout: The scoring timeout the child runs with
            (`SCMLE_SCORING_TIMEOUT`); the product default unless a test
            shortens it.

    Returns:
        What each part got and whether the replica stayed unwedged.
    """
    replica = traffic.Replica(
        forest(folder, stall=True), {"SCMLE_SCORING_TIMEOUT": str(timeout)}
    )
    stuck = {"records": [{**records(1)[0], "a": SENTINEL}]}
    limit = server.MAX_CONCURRENT_REQUESTS
    try:
        replica.ready()
        address = replica.address
        polls: list[Any] = []
        quiet = threading.Event()
        poller = threading.Thread(
            target=traffic.watch, args=(address, quiet, polls)
        )
        with traffic.Load(address, {"records": records(1)}, CLIENTS) as load:
            poller.start()
            slow = traffic.crawl(address, limit)
            first = traffic.send(address, "/predict", stuck)
            time.sleep(traffic.MARGIN)
            quiet.set()
            poller.join(traffic.CALL_SECONDS)
        slowly = [traffic.answered(connection) for connection in slow]
        done = first.started + first.seconds
        after = [item for item in load.outcomes if item.started > done]
        rest = [traffic.Request(address, stuck) for ignored in range(limit - 1)]
        hung = [item.finish() for item in rest]
        alive = traffic.send(address, "/livez", timeout=5).status
        ready = traffic.send(address, "/readyz", timeout=5).status
        code = replica.terminate(traffic.DRAIN + traffic.MARGIN)
    finally:
        replica.close()
    plain = traffic.tally(load.outcomes)
    return {
        "hung": first.status,
        "hung_seconds": first.seconds,
        "load": plain,
        "after_hung": len(after),
        "slow": dict(collections.Counter(str(item) for item in slowly)),
        "probes": dict(collections.Counter(f"{p} {s}" for p, s in polls)),
        "scoring_timeout_seconds": timeout,
        "all_hung": dict(
            collections.Counter(str(item.status) for item in hung)
        ),
        "livez_all_hung": alive,
        "readyz_all_hung": ready,
        "exit": code,
        "passed": first.status == http.HTTPStatus.GATEWAY_TIMEOUT
        and set(plain) == {str(http.HTTPStatus.OK.value)}
        and len(after) > 0
        and slowly == [http.HTTPStatus.REQUEST_TIMEOUT] * limit
        and bool(polls)
        and {status for path, status in polls} == {http.HTTPStatus.OK}
        and [item.status for item in hung]
        == [http.HTTPStatus.GATEWAY_TIMEOUT] * (limit - 1)
        and alive == http.HTTPStatus.SERVICE_UNAVAILABLE
        and ready == http.HTTPStatus.SERVICE_UNAVAILABLE
        and code == 0,
    }


def drained(
    bundle: pathlib.Path,
    payload: dict[str, Any],
    routed: bool,
    linger: float | None = LINGER,
) -> dict[str, Any]:
    """One SIGTERM sent while clients keep scoring.

    Args:
        bundle: The bundle served.
        payload: The request every client sends.
        routed: Whether the clients go through a `Router`, as callers on a
            platform do, or straight to the replica whatever it reports.
        linger: `SCMLE_LINGER_SECONDS` for the child; None runs it with the
            product's default (the setting cleared).

    Returns:
        Outcomes before and after the signal, the router's probes, whether
        it still routed at the signal, the exit status, and whether the
        round tested the drain and no connection was reset.
    """
    setting = "" if linger is None else str(linger)
    waited = runtime.lingering({runtime.LINGER: setting})
    replica = traffic.Replica(bundle, {runtime.LINGER: setting})
    try:
        replica.ready()
        router = traffic.Router(
            replica.address, PERIOD, PROBES.timeout, PROBES.threshold
        )
        gate = router.routes if routed else None
        with traffic.Load(
            replica.address, payload, DRAIN_CLIENTS, gate
        ) as load:
            time.sleep(WARM_SECONDS)
            routing = router.routes.is_set()
            signalled = time.monotonic()
            code = replica.terminate(waited + traffic.DRAIN + traffic.MARGIN)
    finally:
        replica.close()
    before = [item for item in load.outcomes if item.started < signalled]
    late = [item for item in load.outcomes if item.started >= signalled]
    after = traffic.tally(late)
    served = sum(count for kind, count in after.items() if kind.isdigit())
    bad = {"reset", "timeout"}
    kinds = [item.kind for item in load.outcomes]
    return {
        "linger": waited,
        "routed_at_signal": routing,
        "probes": dict(
            collections.Counter(str(item) for item in list(router.probes))
        ),
        "before": traffic.tally(before),
        "after": after,
        "answered_after": served,
        "resets": kinds.count("reset"),
        "timeouts": kinds.count("timeout"),
        "exit": code,
        "passed": code == 0
        and routing
        and served > 0
        and not bad & set(kinds)
        and set(kinds) <= {"answered", "refused"},
    }


def stop(folder: pathlib.Path, alpha: float) -> dict[str, Any]:
    """SIGTERM under load: routed rounds decide, direct rounds are reported.

    The routed rounds with `LINGER` and the routed rounds with the
    product's default environment both decide.

    Args:
        folder: Scratch directory.
        alpha: One-sided error rate.

    Returns:
        Every round and the totals.
    """
    bundle = forest(folder, stall=False)
    payload = {"records": records(100)}
    rounds = [
        drained(bundle, payload, routed=True)
        for ignored in range(traffic.minimum(STOP_FAILURE, alpha))
    ]
    plain = [
        drained(bundle, payload, routed=True, linger=None)
        for ignored in range(DEFAULT)
    ]
    direct = [
        drained(bundle, payload, routed=False) for ignored in range(DIRECT)
    ]
    return {
        "rounds": rounds,
        "default": plain,
        "direct": direct,
        "direct_resets": sum(item["resets"] for item in direct),
        "default_resets": sum(item["resets"] for item in plain),
        "default_answered_after": sum(item["answered_after"] for item in plain),
        "resets": sum(item["resets"] for item in rounds),
        "timeouts": sum(item["timeouts"] for item in rounds),
        "refused": sum(item["after"].get("refused", 0) for item in rounds),
        "answered_after": sum(item["answered_after"] for item in rounds),
        "passed": all(item["passed"] for item in [*rounds, *plain]),
    }


def growth(
    samples: list[tuple[float, int, float | None]], seconds: float
) -> tuple[float | None, float | None]:
    """Resident memory growth over a soak, in total and per request.

    Samples taken in the first third of the run (warm-up) or without a
    memory reading are dropped; the medians of the first and the last third
    of the rest are compared (medians ignore the garbage collector's
    sawtooth), and the growth is divided by the requests answered between
    those two medians.

    Args:
        samples: (seconds elapsed, requests so far, resident MiB or None).
        seconds: The length of the soak.

    Returns:
        The growth in MiB and in KiB per request; None for both with fewer
        than three samples kept, and None per request when no request was
        answered in between.
    """
    kept = [
        (elapsed, count, float(resident))
        for elapsed, count, resident in samples
        if resident is not None and elapsed >= seconds / 3
    ]
    third = len(kept) // 3
    if not third:
        return None, None
    first, last = kept[:third], kept[-third:]
    count = statistics.median(n for t, n, m in last) - statistics.median(
        n for t, n, m in first
    )
    found = statistics.median(m for t, n, m in last) - statistics.median(
        m for t, n, m in first
    )
    return found, 1024 * found / count if count else None


def soak(seconds: float) -> dict[str, Any]:
    """Steady 1-record load for `seconds`, sampling resident memory.

    Memory is compared between the medians of the first and last third of
    the samples taken after the first third of the run (medians ignore
    the garbage collector's sawtooth), per request answered in between.

    Args:
        seconds: How long to soak.

    Returns:
        Requests, failures, samples, growth and whether memory stayed flat
        (at most `LEAK` KiB per request answered) with no failed request.
    """
    with tempfile.TemporaryDirectory() as scratch:
        replica = traffic.Replica(forest(pathlib.Path(scratch), stall=False))
        samples: list[tuple[float, int, float | None]] = []
        try:
            replica.ready()
            started = time.monotonic()
            with traffic.Load(
                replica.address, {"records": records(1)}, CLIENTS
            ) as load:
                while (elapsed := time.monotonic() - started) < seconds:
                    resident = client.footprint(replica.process)
                    samples.append((elapsed, len(load.outcomes), resident))
                    time.sleep(SAMPLE_SECONDS)
            code = replica.terminate(traffic.DRAIN + traffic.MARGIN)
        finally:
            replica.close()
    found, per = growth(samples, seconds)
    outcomes = traffic.tally(load.outcomes)
    return {
        "seconds": seconds,
        "requests": len(load.outcomes),
        "outcomes": outcomes,
        "samples": samples,
        "growth_mib": found,
        "kib_per_request": per,
        "leak_kib_per_request": LEAK,
        "exit": code,
        "passed": set(outcomes) == {str(http.HTTPStatus.OK.value)}
        and per is not None
        and per <= LEAK
        and code == 0,
    }


def settings(alpha: float) -> dict[str, Any]:
    """Every constant that changes what is measured.

    Args:
        alpha: One-sided error rate.

    Returns:
        The design of the measurement.
    """
    return {
        "p95_one_seconds": P95_ONE_SECONDS,
        "p95_hundred_seconds": P95_HUNDRED_SECONDS,
        "startup_seconds": STARTUP_SECONDS,
        "resident_mib": RESIDENT_MIB,
        "quantile": QUANTILE,
        "failure": FAILURE,
        "stop_failure": STOP_FAILURE,
        "alpha": alpha,
        "requests_min": traffic.minimum(FAILURE, alpha),
        "rounds": traffic.minimum(STOP_FAILURE, alpha),
        "clients": CLIENTS,
        "drain_clients": DRAIN_CLIENTS,
        "period": PERIOD,
        "probe_timeout": PROBES.timeout,
        "threshold": PROBES.threshold,
        "linger": LINGER,
        "default_linger": runtime.lingering({}),
        "default": DEFAULT,
        "direct": DIRECT,
        "rows": ROWS,
        "seed": SEED,
        "hang_seconds": HANG_SECONDS,
        "scoring_timeout_seconds": server.SCORING_TIMEOUT_SECONDS,
        "strikes": guard.STRIKES,
    }


def current() -> str:
    """Hash of this module and of the product code a request runs.

    Returns:
        SHA-256 over this file and `evidence.source(PRODUCT)`.
    """
    digest = hashlib.sha256(pathlib.Path(__file__).read_bytes())
    digest.update(evidence.source(PRODUCT, PACKAGE).encode())
    return digest.hexdigest()


def alpha() -> float:
    """The one-sided error rate of `acceptance.toml`."""
    return float(acceptance.load()["alpha"].value)


def measure() -> dict[str, Any]:
    """Runs the three measurements.

    Returns:
        The document stored at `PATH`.
    """
    level = alpha()
    with tempfile.TemporaryDirectory() as scratch:
        folder = pathlib.Path(scratch)
        found = {
            "objectives": objectives(folder, level),
            "hang": hang(folder),
            "stop": stop(folder, level),
        }
    return {
        "code": current(),
        "settings": settings(level),
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpus": os.cpu_count(),
            "load": round(os.getloadavg()[0], 1),
        },
        **found,
    }


def read(document: dict[str, Any] | None = None) -> dict[str, Any]:
    """The stored measurements.

    Args:
        document: Used as is when given.

    Returns:
        The document.

    Raises:
        InputError: If the file is missing or not a JSON object.
    """
    if document is not None:
        return document
    try:
        found = json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as caught:
        raise errors.InputError(f"cannot read {PATH.name}: {caught}") from None
    if not isinstance(found, dict):
        raise errors.InputError(f"{PATH.name} is not a JSON object")
    return found


def summary(document: dict[str, Any] | None = None) -> tuple[str, str]:
    """The scorecard value and verdict of the three service clauses.

    Args:
        document: The measurements; read from `PATH` when None.

    Returns:
        The value text and `pass` or `fail`; the soak is not part of it.
    """
    found = read(document)
    profiles = found["objectives"]["profiles"]
    latency = "; ".join(
        f"{item['records']} record(s): "
        f"p95 {1000 * (item['p95'] or math.inf):.1f} ms "
        f"(bound {1000 * item['p95_bound']:.1f} <= "
        f"{1000 * item['p95_target']:.0f} ms), "
        f"{item['failed']} of {item['requests']} failed"
        for item in profiles
    )
    held = found["hang"]
    stopped = found["stop"]
    design = found["settings"]
    value = (
        f"{latency}; a hung request answered {held['hung']} while "
        f"{sum(held['load'].values())} others got {sorted(held['load'])}; "
        f"every worker hung: /livez {held['livez_all_hung']}; SIGTERM under "
        f"load for callers routed by readiness, with SCMLE_LINGER_SECONDS "
        f"{design['linger']:g} >= probe threshold x (period + timeout) "
        f"x{len(stopped['rounds'])}: {stopped['resets']} resets, "
        f"{stopped['refused']} refused after close, "
        f"{stopped['answered_after']} answered after the signal; with the "
        f"default linger {design['default_linger']:g} "
        f"x{len(stopped['default'])}: {stopped['default_resets']} resets, "
        f"{stopped['default_answered_after']} answered after the signal "
        f"(clients that ignore readiness: {stopped['direct_resets']} resets "
        f"in {len(stopped['direct'])} rounds, not deciding); soak not "
        f"measured"
    )
    passed = all(
        found[part]["passed"] for part in ("objectives", "hang", "stop")
    )
    return value, "pass" if passed else "fail"


def problems(document: dict[str, Any] | None = None) -> list[str]:
    """Why the stored measurements are not for the current code.

    Args:
        document: The measurements; read from `PATH` when None.

    Returns:
        A message per problem; empty when current.
    """
    if document is None and not PATH.exists():
        return [f"{PATH.name} is missing"]
    found = read(document)
    issues = []
    if found.get("code") != current():
        issues.append("service: measurements are for other code")
    if found.get("settings") != settings(alpha()):
        issues.append("service: measurements are for other settings")
    return issues


def main(arguments: list[str] | None = None) -> int:
    """Writes (`--write`), checks (`--check`) or soaks (`--soak SECONDS`).

    Args:
        arguments: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        0 after `--write`; for `--check` and `--soak`, 0 when current or
        passed and 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--soak", type=float, metavar="SECONDS")
    parser.add_argument("--output", help="with --soak, also write it here")
    options = parser.parse_args(arguments)
    if options.soak is not None:
        if options.soak <= 0:
            parser.error("--soak needs a positive number of seconds")
        result = soak(options.soak)
        text = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if options.output:
            pathlib.Path(options.output).write_text(text, encoding="utf-8")
        print(text, end="")
        return 0 if result["passed"] else 1
    if options.write:
        document = measure()
        PATH.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        value, verdict = summary(document)
        print(json.dumps({"value": value, "verdict": verdict}))
        return 0
    issues = problems()
    print(json.dumps({"ok": not issues, "problems": issues}, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
