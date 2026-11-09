"""Local scenarios for the fault catalogue.

Each scenario starts the real local service (the generated ``serve.py`` in a
subprocess, as ``scmle.deploy.demo`` does), the local object-store and
metrics emulators, a real ledger or a real workflow executor, breaks it the
way the fault says and measures recovery with a monotonic clock. Everything
the scenarios start is owned by a `Fleet`, which stops it in ``close`` and
counts anything still alive afterwards.

These are local processes and emulators. The control plane in ``bad_release``
runs revisions as processes. None of this is a cloud, a container
orchestrator, a network partition or a machine crash; the catalogue
(`scmle.operations.recovery`) says so per fault.
"""

from collections.abc import Callable
from collections.abc import Mapping
import contextlib
import copy
import dataclasses
import json
import os
import pathlib
import shutil
import subprocess  # nosec B404 - runs our own interpreter and serve.py
import sys
import tempfile
import threading
import time
from typing import Any

import numpy

from scmle.artifacts import bundle
from scmle.contracts import errors
from scmle.contracts import observability
from scmle.core import limits
from scmle.data.store import objectstore
from scmle.deploy import control
from scmle.deploy import demo
from scmle.deploy import mocks
from scmle.deploy import observation
from scmle.deploy import provenance
from scmle.governance import deployment_ledger
from scmle.governance import registry
from scmle.governance import release_state
from scmle.inference import parity
from scmle.operations import classify
from scmle.operations import recovery
from scmle.operations import signals
from scmle.operations import spool
from scmle.security import audit

ROTATED_KEY = "scmle-rotated-signing-key-local-recovery-run"
WRONG_KEY = "scmle-wrong-signing-key-local-recovery-run"
MODEL = "recovery-model"
ENVIRONMENT = "production"
GOLDEN = pathlib.Path("deploy/local/golden")
READY_SECONDS = 60.0  # before `recovery.scaled`
METHOD_CLOCK = "time.monotonic() in the runner"


def records(count: int = 3) -> list[dict[str, Any]]:
    """The first ordinary golden records.

    Args:
        count: Number of records to return.

    Returns:
        The first `count` records of the golden record file.
    """
    loaded = json.loads((GOLDEN / "records.json").read_text(encoding="utf-8"))
    return list(loaded["records"][:count])


def golden_spec() -> dict[str, Any]:
    """The hand-written reference model's specification.

    Returns:
        The specification of the hand-written reference model.
    """
    return parity.load_golden(GOLDEN).spec


def inverted_spec() -> dict[str, Any]:
    """The same model with every coefficient negated: a wrong model.

    Returns:
        A copy of the golden specification with negated coefficients and
        intercept, versioned ``golden-inverted``.
    """
    spec = copy.deepcopy(golden_spec())
    spec["coefficients"] = {k: -v for k, v in spec["coefficients"].items()}
    spec["intercept"] = -spec["intercept"]
    spec["model_version"] = "golden-inverted"
    return dict(spec)


def export_service(
    folder: pathlib.Path, spec: dict[str, Any], key: str = demo.DEMO_SIGNING_KEY
) -> pathlib.Path:
    """Writes a signed service folder for a specification.

    Args:
        folder: The folder to write.
        spec: The reference model's specification.
        key: The signing key.

    Returns:
        `folder`.
    """
    parity.reference_bundle(spec).export_service(folder, signing_key=key)
    return folder


@dataclasses.dataclass
class Instance:
    """One running service process."""

    process: subprocess.Popen[bytes]
    port: int
    folder: pathlib.Path

    @property
    def base(self) -> str:
        """Base URL of the instance."""
        return f"http://127.0.0.1:{self.port}"

    def alive(self) -> bool:
        """Whether the process is still running.

        Returns:
            True while the process has not exited.
        """
        return self.process.poll() is None


class Fleet:
    """Owns every process and emulator a scenario starts."""

    def __init__(self) -> None:
        """Starts with nothing."""
        self.instances: list[Instance] = []
        self.servers: list[mocks.MockServer] = []
        self.children: list[subprocess.Popen[str]] = []

    def start(
        self,
        folder: pathlib.Path,
        key: str = demo.DEMO_SIGNING_KEY,
        port: int | None = None,
    ) -> Instance:
        """Starts ``serve.py`` of a folder in a subprocess, in production mode.

        The service runs with ``SCMLE_ENV=production`` on 127.0.0.1, with the
        published-key allowance set and ``SCMLE_TRUSTED`` removed from its
        environment. Its output is discarded.

        Args:
            folder: A service folder.
            key: The signing key given to the service.
            port: The port to listen on; a free one if None or 0.

        Returns:
            The running instance, which the fleet remembers.
        """
        chosen = port or demo.free_port()
        env = {
            **os.environ,
            "HOST": "127.0.0.1",
            "PORT": str(chosen),
            "SCMLE_ENV": "production",
            "SCMLE_SIGNING_KEY": key,
            "SCMLE_ALLOW_PUBLISHED_KEY": "1",
        }
        env.pop("SCMLE_TRUSTED", None)
        folder = folder.resolve()
        process = subprocess.Popen(  # nosec B603 - our interpreter, our file
            [sys.executable, str(folder / "serve.py")],
            cwd=folder,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        instance = Instance(process, chosen, folder)
        self.instances.append(instance)
        return instance

    def serve(self, server: mocks.MockServer) -> mocks.MockServer:
        """Starts an emulator and remembers it.

        Args:
            server: The emulator to start.

        Returns:
            `server`, started.
        """
        server.start()
        self.servers.append(server)
        return server

    def child(self, code: str, *arguments: str) -> subprocess.Popen[str]:
        """Starts a Python child process whose standard output is piped.

        Args:
            code: The Python source to run with ``python -c``.
            *arguments: Command-line arguments for the code.

        Returns:
            The child process (standard output piped as text, standard error
            discarded), which the fleet remembers.
        """
        process = subprocess.Popen(  # nosec B603 - our interpreter, our code
            [sys.executable, "-c", code, *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.children.append(process)
        return process

    def close(self) -> int:
        """Stops the emulators, kills every process and closes their pipes.

        Returns:
            The number of service and child processes that had not exited 15
            seconds after being killed. Errors from stopping an emulator are
            ignored.
        """
        for server in self.servers:
            with contextlib.suppress(OSError, errors.ScmleError):
                server.stop()
        for instance in self.instances:
            if instance.alive():
                instance.process.kill()
        for process in self.children:
            if process.poll() is None:
                process.kill()
        stray = 0
        for handle in [i.process for i in self.instances] + self.children:
            try:
                handle.wait(timeout=15)
            except subprocess.TimeoutExpired:
                stray += 1
            for stream in (getattr(handle, "stdout", None), handle.stderr):
                if stream is not None:
                    stream.close()
        return stray


def predict(base: str, count: int = 3) -> tuple[int, dict[str, Any]]:
    """POSTs the golden records to ``/predict``; unreachable is ``(0, {})``.

    Args:
        base: Base URL of the service.
        count: Number of golden records to send.

    Returns:
        The HTTP status (also for HTTP errors) and the JSON body; ``(0, {})``
        if the connection fails or times out.
    """
    try:
        return demo.post_json(
            f"{base}/predict",
            {"records": records(count)},
            timeout=recovery.scaled(5.0),
        )
    except OSError:
        return 0, {}


def ready_after(
    instance: Instance, timeout: float | None = None
) -> float | None:
    """Seconds until ``/readyz`` is 200, or None (time-scaled default).

    Args:
        instance: The running instance.
        timeout: Seconds to wait; the time-scaled default if None.

    Returns:
        The seconds it took to become ready, or None if it did not within
        the timeout or the process exited first.
    """
    limit = recovery.scaled(READY_SECONDS) if timeout is None else timeout
    return demo.wait_ready(instance.base, instance.process, limit)


def await_unready(
    instance: Instance, timeout: float | None = None
) -> dict[str, Any]:
    """Polls ``/readyz`` until the service reports phase ``failed``.

    Polling stops at the first ``failed`` phase or when the wait ends. The
    default wait is 20 s, multiplied by the run's time scale.

    Args:
        instance: The running instance.
        timeout: Seconds to wait; the time-scaled default if None.

    Returns:
        A dict with the last ``/readyz`` status (``readyz``), ``phase`` and
        ``reason`` observed (0, None and None if it never answered), then the
        ``/livez`` status (``livez``) and the ``/predict`` status
        (``predict``) taken afterwards; 0 where the service cannot be reached.
    """
    deadline = time.monotonic() + (
        recovery.scaled(20.0) if timeout is None else timeout
    )
    seen: dict[str, Any] = {"readyz": 0, "phase": None, "reason": None}
    while time.monotonic() < deadline:
        try:
            status, body = demo.get_json(f"{instance.base}/readyz", timeout=2.0)
        except OSError:
            time.sleep(0.2)
            continue
        seen = {
            "readyz": status,
            "phase": body.get("phase"),
            "reason": body.get("reason"),
        }
        if body.get("phase") == "failed":
            break
        time.sleep(0.2)
    try:
        seen["livez"] = demo.get_json(f"{instance.base}/livez", timeout=2.0)[0]
    except OSError:
        seen["livez"] = 0
    seen["predict"] = predict(instance.base)[0]
    return seen


def corrupt_copy(service: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
    """A copy of a service folder whose model file had bytes appended.

    Args:
        service: The service folder to copy.
        target: The folder to create; it must not exist yet.

    Returns:
        `target`.
    """
    shutil.copytree(service, target)
    with (target / "model" / bundle.MODEL_FILE).open("ab") as handle:
        handle.write(b"tampered")
    return target


def model_file_digest(service: pathlib.Path) -> str:
    """SHA-256 of the model file's bytes (what tampering changes).

    Args:
        service: A service folder.

    Returns:
        The digest of the folder's model file.
    """
    path = service / "model" / bundle.MODEL_FILE
    return provenance.sha256_digest(path.read_bytes())


def work_folder(prefix: str) -> pathlib.Path:
    """A fresh scratch folder.

    Args:
        prefix: Name part to tell the folder apart.

    Returns:
        The path of a new temporary directory.
    """
    return pathlib.Path(tempfile.mkdtemp(prefix=f"scmle-recovery-{prefix}-"))


# ---- process_kill ------------------------------------------------------

STATE_WRITER = """
import json, sys
from scmle import durability
n = 0
while True:
    body = json.dumps({"counter": n}).encode()
    durability.write_atomically(sys.argv[1], body)
    print(n, flush=True)
    n += 1
"""
LEDGER_WRITER = """
import sys
from scmle.security import audit
log = audit.Log.open(sys.argv[1])
n = 0
while True:
    log.admin("writer", "tick", n=n)
    print(n, flush=True)
    n += 1
"""


def kill_after(
    fleet: Fleet, code: str, path: pathlib.Path, acknowledged: int
) -> int:
    """Kills a writer after some acknowledged units.

    Runs the writer, SIGKILLs it after ``acknowledged`` lines and returns the
    highest unit it acknowledged on stdout.

    Args:
        fleet: The fleet that owns the writer process.
        code: The Python source of the writer.
        path: The file the writer works on.
        acknowledged: Number of acknowledged units after which to kill it.

    Returns:
        The highest unit the writer acknowledged, or -1 if none.

    Raises:
        ScmleError: If the writer process has no stdout.
    """
    process = fleet.child(code, str(path))
    last = -1
    stream = process.stdout
    if stream is None:
        raise errors.ScmleError("the writer has no stdout")
    for line in stream:
        last = int(line)
        if last + 1 >= acknowledged:
            break
    process.kill()
    process.wait(timeout=15)
    return last


def state_writer_trials(fleet: Fleet, base: pathlib.Path) -> dict[str, Any]:
    """Kills an atomic state writer ten times and reads its file back.

    Each trial kills the writer after a number of acknowledged writes drawn
    from 5 to 60 with a fixed seed, so the points are random but the same on
    every run.

    Args:
        fleet: The fleet that owns the writer processes.
        base: Directory for the writers' state files.

    Returns:
        A dict with ``trials`` (always 10), ``acknowledged_writes_lost`` (the
        sum over trials of how far the stored counter is below the last
        acknowledged one), ``unreadable_files`` (trials whose file could not
        be read back) and ``read_seconds_max`` (the longest read time).
    """
    rng = numpy.random.default_rng(7)
    lost = 0
    torn = 0
    seconds = []
    for trial in range(10):
        path = base / f"state-{trial}.json"
        acknowledged = kill_after(
            fleet, STATE_WRITER, path, int(rng.integers(5, 61))
        )
        started = time.monotonic()
        try:
            found = json.loads(path.read_text(encoding="utf-8"))["counter"]
        except (ValueError, KeyError, OSError):
            torn += 1
            found = -1
        seconds.append(time.monotonic() - started)
        lost += max(0, acknowledged - found)
    return {
        "trials": 10,
        "acknowledged_writes_lost": lost,
        "unreadable_files": torn,
        "read_seconds_max": max(seconds),
    }


def process_kill() -> recovery.Outcome:
    """Kills the service process three times, restarting it each time.

    After the three kills it also runs `state_writer_trials` on atomic state
    files.

    Returns:
        The recovery result of this fault: its checks, its RTO and RPO
        measurements and the details observed.

    Raises:
        ScmleError: If the service did not become ready.
    """
    fleet = Fleet()
    base = work_folder("kill")
    result = recovery.Outcome("process_kill", environment="local subprocess")
    try:
        service = export_service(base / "service", golden_spec())
        times = []
        same = []
        for ignored in range(3):
            first = fleet.start(service)
            if ready_after(first) is None:
                raise errors.ScmleError("the service did not become ready")
            before = predict(first.base)[1].get("predictions")
            killed = time.monotonic()
            first.process.kill()
            first.process.wait(timeout=15)
            gone = predict(first.base)[0] == 0
            second = fleet.start(service)
            after = None
            while time.monotonic() - killed < recovery.scaled(READY_SECONDS):
                status, body = predict(second.base)
                if status == 200:
                    after = body.get("predictions")
                    break
                time.sleep(0.1)
            times.append(time.monotonic() - killed)
            same.append(bool(gone) and after is not None and after == before)
            fleet.close()
        state = state_writer_trials(fleet, base)
        result.checks = {
            "killed process stopped answering and the restart answers the "
            "same predictions": all(same),
            "no state file was ever unreadable": state["unreadable_files"] == 0,
        }
        result.measurements = [
            recovery.Measurement(
                "service",
                max(times),
                0 if all(same) else sum(not s for s in same),
                "SIGKILL to the first 200 from /predict of the restarted "
                "process, 3 trials, maximum; RPO is answers that differ from "
                "before the kill. " + METHOD_CLOCK,
            ),
            recovery.Measurement(
                "registry_state",
                state["read_seconds_max"],
                state["acknowledged_writes_lost"],
                "10 trials: a child rewrites a state file atomically in a "
                "loop and prints each acknowledged counter; SIGKILL at a "
                "random point; the file is read back and its counter must "
                "not be below the last acknowledged one",
            ),
        ]
        result.detail = {
            "service_rto_seconds_per_trial": [round(t, 3) for t in times],
            "state_trials": state,
        }
    finally:
        result.detail["processes_still_alive_after_close"] = fleet.close()
        shutil.rmtree(base, ignore_errors=True)
    return result


# ---- unhealthy_container_rolling_restart -------------------------------


DRAIN_REQUESTS = 3
DRAIN_SECONDS = 10.0


class Client(threading.Thread):
    """Sends requests to whatever instance the router points at."""

    def __init__(self, router: dict[str, Instance]) -> None:
        """Starts idle; ``start`` begins sending.

        Args:
            router: Maps ``current`` to the instance that receives requests.
        """
        super().__init__(daemon=True)
        self.router = router
        self.stop_flag = threading.Event()
        self.sent = 0
        self.failed = 0
        self.longest_gap = 0.0

    def run(self) -> None:
        """Sends one request every 50 ms until stopped."""
        gap_start: float | None = None
        while not self.stop_flag.is_set():
            status = predict(self.router["current"].base, 1)[0]
            self.sent += 1
            now = time.monotonic()
            if status == 200:
                if gap_start is not None:
                    self.longest_gap = max(self.longest_gap, now - gap_start)
                    gap_start = None
            else:
                self.failed += 1
                gap_start = gap_start or now
            time.sleep(0.05)


def rolling_restart() -> recovery.Outcome:
    """Replaces an instance under client load: a tampered copy, then a good one.

    The tampered replacement must stay alive but unready and never receive
    traffic. The good replacement takes over once its ``/readyz`` is 200; the
    old instance is then terminated and the failures a client saw are counted.

    Returns:
        The recovery result of this fault: its checks, its RTO and RPO
        measurements and the details observed.

    Raises:
        ScmleError: If the first instance did not become ready.
    """
    fleet = Fleet()
    base = work_folder("rolling")
    result = recovery.Outcome(
        "unhealthy_container_rolling_restart", environment="local subprocess"
    )
    client = None
    try:
        service = export_service(base / "service", golden_spec())
        broken = corrupt_copy(service, base / "broken")
        old = fleet.start(service)
        if ready_after(old) is None:
            raise errors.ScmleError("the first instance did not become ready")
        router = {"current": old}
        client = Client(router)
        client.start()
        time.sleep(0.5)
        bad = fleet.start(broken)
        verdict = await_unready(bad)
        admitted = verdict["readyz"] == 200
        bad.process.kill()
        started = time.monotonic()
        new = fleet.start(service)
        became_ready = ready_after(new)
        if became_ready is not None:
            router["current"] = new
            # An orchestrator drains the old instance before stopping it: a
            # request already sent to it must finish, which is certain once
            # two requests that began after the switch have been answered.
            drained = client.sent + DRAIN_REQUESTS
            deadline = time.monotonic() + DRAIN_SECONDS
            while client.sent < drained and time.monotonic() < deadline:
                time.sleep(0.05)
        old.process.terminate()
        old.process.wait(timeout=15)
        time.sleep(0.8)
        client.stop_flag.set()
        client.join(timeout=10)
        window = time.monotonic() - started
        result.checks = {
            "the unready replacement stayed alive and unready": (
                verdict["livez"] == 200
                and verdict["readyz"] == 503
                and verdict["predict"] == 503
            ),
            "the unready replacement was never given traffic": not admitted,
            "the ready replacement took over": became_ready is not None,
            "no request failed during the replacement": client.failed == 0,
        }
        result.measurements = [
            recovery.Measurement(
                "service",
                client.longest_gap,
                client.failed,
                "a client sends one /predict every 50 ms to the instance "
                "the runner has declared current; RTO is the longest run of "
                "failures, RPO is failed requests; the runner stands in for "
                "the orchestrator: it switches only after /readyz is 200",
            )
        ]
        result.detail = {
            "requests_sent": client.sent,
            "requests_failed": client.failed,
            "unready_replacement": verdict,
            "replacement_ready_seconds": None
            if became_ready is None
            else round(became_ready, 2),
            "window_seconds": round(window, 2),
        }
    finally:
        if client is not None:
            client.stop_flag.set()
        result.detail["processes_still_alive_after_close"] = fleet.close()
        shutil.rmtree(base, ignore_errors=True)
    return result


# ---- credentials and secrets -------------------------------------------


def invalid_credentials() -> recovery.Outcome:
    """A wrong signing key leaves the service unready until corrected.

    Returns:
        The recovery result of this fault: its checks, its RTO and RPO
        measurements and the details observed.
    """
    fleet = Fleet()
    base = work_folder("credentials")
    result = recovery.Outcome(
        "invalid_credentials", environment="local subprocess"
    )
    try:
        service = export_service(base / "service", golden_spec())
        wrong = fleet.start(service, key=WRONG_KEY)
        verdict = await_unready(wrong)
        detected = time.monotonic()
        wrong.process.kill()
        wrong.process.wait(timeout=15)
        right = fleet.start(service, key=demo.DEMO_SIGNING_KEY)
        seconds = ready_after(right)
        elapsed = time.monotonic() - detected
        served = predict(right.base)[0] == 200 if seconds is not None else False
        result.checks = {
            "the wrong key leaves the service alive and unready": (
                verdict["livez"] == 200
                and verdict["readyz"] == 503
                and verdict["predict"] == 503
            ),
            "the failure reason is reported without the key": WRONG_KEY
            not in json.dumps(verdict),
            "the corrected key serves": served,
        }
        result.measurements = [
            recovery.Measurement(
                "secrets",
                elapsed if seconds is not None else None,
                0,
                "from the service reporting the credential failure to the "
                "first 200 from /predict under the corrected key (restart "
                "included). " + METHOD_CLOCK,
            )
        ]
        result.detail = {"with_wrong_key": verdict}
    finally:
        result.detail["processes_still_alive_after_close"] = fleet.close()
        shutil.rmtree(base, ignore_errors=True)
    return result


def secret_rotation() -> recovery.Outcome:
    """A bundle re-signed under a rotated key is refused under the old key.

    The service is shown to run under the old key, the re-signed bundle to be
    unready under the old key and to serve under the rotated key, and the
    rotation to be recorded in an audit ledger.

    Returns:
        The recovery result of this fault: its checks, its RTO and RPO
        measurements and the details observed.
    """
    fleet = Fleet()
    base = work_folder("rotation")
    result = recovery.Outcome(
        "secret_rotation_recovery", environment="local subprocess"
    )
    try:
        old_service = export_service(
            base / "old", golden_spec(), demo.DEMO_SIGNING_KEY
        )
        first = fleet.start(old_service, key=demo.DEMO_SIGNING_KEY)
        before = ready_after(first)
        first.process.kill()
        first.process.wait(timeout=15)
        new_service = export_service(base / "new", golden_spec(), ROTATED_KEY)
        stale = fleet.start(new_service, key=demo.DEMO_SIGNING_KEY)
        verdict = await_unready(stale)
        detected = time.monotonic()
        stale.process.kill()
        stale.process.wait(timeout=15)
        fresh = fleet.start(new_service, key=ROTATED_KEY)
        seconds = ready_after(fresh)
        elapsed = time.monotonic() - detected
        log = audit.Log.open(base / "audit.jsonl")
        log.admin("operator-1", "key_rotation", from_key="old", to_key="new")
        log.ledger.verify()
        result.checks = {
            "the service ran under the old key": before is not None,
            "the rotated bundle is refused under the old key": (
                verdict["readyz"] == 503 and verdict["predict"] == 503
            ),
            "the rotated key serves": seconds is not None
            and predict(fresh.base)[0] == 200,
            "the rotation is in the audit ledger": bool(log.rows("admin")),
        }
        result.measurements = [
            recovery.Measurement(
                "secrets",
                elapsed if seconds is not None else None,
                0,
                "from the stale key being refused to the first 200 under "
                "the rotated key with the re-signed bundle. " + METHOD_CLOCK,
            )
        ]
        result.detail = {"stale_key_verdict": verdict}
    finally:
        result.detail["processes_still_alive_after_close"] = fleet.close()
        shutil.rmtree(base, ignore_errors=True)
    return result


# ---- storage -----------------------------------------------------------


def upload(base: str, folder: pathlib.Path, prefix: str) -> dict[str, str]:
    """Uploads every file of a service folder to the object store.

    Args:
        base: Base URL of the object store.
        folder: The service folder to upload.
        prefix: Key prefix for the uploaded objects.

    Returns:
        A mapping from each file's path relative to `folder` to the entity
        tag the store returned (not the full object key with `prefix`).
    """
    return demo.upload_folder(base, folder, prefix)


def object_store_loss() -> recovery.Outcome:
    """The store goes away; the service keeps serving; access returns.

    Returns:
        The recovery result of this fault: its checks, its RTO and RPO
        measurements and the details observed.

    Raises:
        ScmleError: If the service did not become ready.
    """
    fleet = Fleet()
    base = work_folder("store")
    result = recovery.Outcome(
        "lost_object_store_access", environment="local emulator + subprocess"
    )
    try:
        service = export_service(base / "service", golden_spec())
        port = demo.free_port()
        store = mocks.ObjectStore(base / "objects", port=port)
        fleet.serve(store)
        url = store.address
        etags = upload(url, service, "models/m/v1")
        fetched = base / "fetched"
        demo.download_folder(url, "models/m/v1", etags, fetched)
        running = fleet.start(fetched)
        if ready_after(running) is None:
            raise errors.ScmleError("the service did not become ready")
        store.stop()
        fetch_failed = False
        try:
            mocks.get_object(url, "models/m/v1/serve.py", timeout=2.0)
        except OSError:
            fetch_failed = True
        served_during = predict(running.base)[0] == 200
        restored = time.monotonic()
        again = mocks.ObjectStore(base / "objects", port=port)
        fleet.serve(again)
        intact = {
            name: again.root.joinpath("models", "m", "v1", name).exists()
            for name in etags
        }
        lost = sum(not found for found in intact.values())
        second = base / "refetched"
        demo.download_folder(url, "models/m/v1", etags, second)
        same = provenance.bundle_digest(
            second / "model"
        ) == provenance.bundle_digest(service / "model")
        replacement = fleet.start(second)
        seconds = ready_after(replacement)
        elapsed = time.monotonic() - restored
        victim = again.root.joinpath("models", "m", "v1", "serve.py")
        victim.write_bytes(victim.read_bytes() + b"# altered")
        mismatch = False
        try:
            mocks.get_object(
                url, "models/m/v1/serve.py", expected=etags["serve.py"]
            )
        except errors.ArtifactError:
            mismatch = True
        result.checks = {
            "the fetch failed while the store was down": fetch_failed,
            "the running service kept serving": served_during,
            "the bundle fetched after recovery has the original digest": same,
            "an altered object is refused by its digest": mismatch,
        }
        result.measurements = [
            recovery.Measurement(
                "object_store",
                elapsed if seconds is not None else None,
                lost,
                "from the emulator being started again on the same port and "
                "root to a refetched, digest-verified bundle serving; RPO is "
                "uploaded objects missing from the root afterwards. "
                + METHOD_CLOCK,
            )
        ]
        result.detail = {"objects": len(etags)}
    finally:
        result.detail["processes_still_alive_after_close"] = fleet.close()
        shutil.rmtree(base, ignore_errors=True)
    return result


def metrics_destination_outage() -> recovery.Outcome:
    """The metrics sink is down; samples are spooled and drained later.

    Returns:
        The recovery result of this fault: its checks, its RTO and RPO
        measurements and the details observed.

    Raises:
        ScmleError: If the service did not become ready.
    """
    fleet = Fleet()
    base = work_folder("spool")
    result = recovery.Outcome("dependency_outage", environment="local emulator")
    try:
        service = export_service(base / "service", golden_spec())
        running = fleet.start(service)
        if ready_after(running) is None:
            raise errors.ScmleError("the service did not become ready")
        port = demo.free_port()
        url = f"http://127.0.0.1:{port}"

        def send(samples: list[observability.MetricSample]) -> int:
            return mocks.post_metrics(url, samples, timeout=2.0)

        stamp = "2026-11-05T10:00:00+00:00"
        samples = [
            observability.MetricSample(
                f"probe_{n}",
                observability.SignalKind.THROUGHPUT,
                float(n),
                "count",
                stamp,
            )
            for n in range(50)
        ]
        first = spool.MetricsSpool(base / "spool.jsonl", send, batch_size=20)
        first.add(samples)
        down = first.flush()
        serving_during = (
            predict(running.base)[0] == 200
            and demo.get_json(f"{running.base}/readyz")[0] == 200
        )
        restarted = spool.MetricsSpool(
            base / "spool.jsonl", send, batch_size=20
        )
        survived = restarted.pending()
        sink = mocks.MetricsSink(base / "sink.jsonl", port=port)
        fleet.serve(sink)
        back = time.monotonic()
        done = restarted.flush()
        elapsed = time.monotonic() - back
        stored = {s["name"] for s in sink.samples()}
        lost = len({s.name for s in samples} - stored)
        result.checks = {
            "the failed flush kept every sample": down.sent == 0
            and down.remaining == 50
            and down.error is not None,
            "serving and readiness were unaffected": serving_during,
            "a restarted spool still held every sample": survived == 50,
            "the spool drained": done.remaining == 0 and done.error is None,
        }
        result.measurements = [
            recovery.Measurement(
                "telemetry_spool",
                elapsed,
                lost,
                "50 samples spooled while the sink was down, the spool object "
                "recreated (a restart), the sink started, then flushed; RTO is "
                "the flush time, RPO is sample names absent from the sink",
            )
        ]
        result.detail = {"flush_while_down": dataclasses.asdict(down)}
    finally:
        result.detail["processes_still_alive_after_close"] = fleet.close()
        shutil.rmtree(base, ignore_errors=True)
    return result


def corrupted_bundle() -> recovery.Outcome:
    """A tampered bundle never serves; a digest-checked copy restores it.

    Returns:
        The recovery result of this fault: its checks, its RTO and RPO
        measurements and the details observed.
    """
    fleet = Fleet()
    base = work_folder("corrupt")
    result = recovery.Outcome(
        "corrupted_bundle", environment="local subprocess"
    )
    try:
        service = export_service(base / "service", golden_spec())
        expected = model_file_digest(service)
        pristine = base / "pristine"
        shutil.copytree(service, pristine)
        tampered = corrupt_copy(service, base / "tampered")
        broken = fleet.start(tampered)
        verdict = await_unready(broken)
        detected = time.monotonic()
        broken.process.kill()
        broken.process.wait(timeout=15)
        differs = model_file_digest(tampered) != expected
        shutil.rmtree(tampered)
        shutil.copytree(pristine, tampered)
        restored = model_file_digest(tampered) == expected
        fixed = fleet.start(tampered)
        seconds = ready_after(fixed)
        elapsed = time.monotonic() - detected
        result.checks = {
            "the tampered bundle was alive but never ready": (
                verdict["livez"] == 200
                and verdict["readyz"] == 503
                and verdict["predict"] == 503
            ),
            "the tampered bundle's digest differs": differs,
            "the restored copy has the recorded digest": restored,
            "the restored bundle serves": seconds is not None
            and predict(fixed.base)[0] == 200,
        }
        result.measurements = [
            recovery.Measurement(
                "service",
                elapsed if seconds is not None else None,
                0,
                "from the service reporting the tampered bundle to the first "
                "200 after restoring a copy whose digest equals the recorded "
                "one. " + METHOD_CLOCK,
            )
        ]
        result.detail = {"tampered_verdict": verdict}
    finally:
        result.detail["processes_still_alive_after_close"] = fleet.close()
        shutil.rmtree(base, ignore_errors=True)
    return result


def incompatible_schema() -> recovery.Outcome:
    """Requests with a wrong schema digest, version or column are refused.

    A different expected digest and an unserved schema version are checked to
    be refused with 409, records missing a column with 422, and the refusals
    to classify as data quality. The service must stay ready, and a caller
    that reads ``/schema`` and sends the digest must get a 200.

    Returns:
        The recovery result of this fault: its checks, its RTO and RPO
        measurements and the details observed.

    Raises:
        ScmleError: If the service did not become ready.
    """
    fleet = Fleet()
    base = work_folder("schema")
    result = recovery.Outcome(
        "incompatible_schema", environment="local subprocess"
    )
    try:
        service = export_service(base / "service", golden_spec())
        running = fleet.start(service)
        if ready_after(running) is None:
            raise errors.ScmleError("the service did not become ready")
        url = f"{running.base}/predict"
        wrong_digest = demo.post_json(
            url,
            {
                "records": records(1),
                "expected_schema_digest": "sha256:" + "0" * 64,
            },
        )[0]
        wrong_version = demo.post_json(
            url, {"records": records(1), "schema_version": "9"}
        )[0]
        missing = [
            {k: v for k, v in r.items() if k != "age"} for r in records(3)
        ]
        started = time.monotonic()
        watch = signals.SignalMonitor(signals.Window(min_samples=30))
        statuses = []
        for step in range(40):
            code = demo.post_json(url, {"records": missing})[0]
            statuses.append(code)
            watch.record_request(100.0 + step, "/predict", code, 0.01)
        verdict = classify.classify(watch.evaluate(140.0))
        digest = demo.get_json(f"{running.base}/schema")[1]["schema_digest"]
        adapted = demo.post_json(
            url, {"records": records(1), "expected_schema_digest": digest}
        )[0]
        elapsed = time.monotonic() - started
        result.checks = {
            "a different expected schema digest is refused with 409": (
                wrong_digest == 409
            ),
            "an unserved schema version is refused with 409": wrong_version
            == 409,
            "a record missing a required column is refused with 422": set(
                statuses
            )
            == {422},
            "the refusals classify as data quality, not infrastructure": (
                verdict.verdict is classify.Verdict.DATA_QUALITY
            ),
            "the caller recovers after reading /schema": adapted == 200,
            "the service stayed ready": demo.get_json(f"{running.base}/readyz")[
                0
            ]
            == 200,
        }
        result.measurements = [
            recovery.Measurement(
                "service",
                elapsed,
                0,
                "from the first refused request to a 200 after the client "
                "read /schema and sent its digest; the service itself was "
                "never down. " + METHOD_CLOCK,
            )
        ]
        result.detail = {
            "classification": verdict.to_dict(),
            "statuses": {
                "expected_schema_digest": wrong_digest,
                "schema_version": wrong_version,
            },
        }
    finally:
        result.detail["processes_still_alive_after_close"] = fleet.close()
        shutil.rmtree(base, ignore_errors=True)
    return result


# ---- a control plane made of local processes ---------------------------


class ProcessPlane(control.ControlPlane):
    """Revisions are local service processes; the interface is the same.

    ``deploy(digest)`` starts a revision of the folder registered for the
    digest and leaves earlier ones running, as a managed platform keeps
    earlier revisions; ``set_traffic`` points the single entry at one of
    them; ``observe`` reports the revision that holds the traffic, with its
    readiness from ``/readyz``.
    """

    provider = "local"

    def __init__(self, fleet: Fleet) -> None:
        """Starts with no images and no revisions.

        Args:
            fleet: The fleet that owns the processes.
        """
        self.fleet = fleet
        self.images: dict[str, pathlib.Path] = {}
        self.revisions: dict[str, Instance] = {}
        self.digests: dict[str, str] = {}
        self.primary: str | None = None
        self.count = 0

    def register_image(self, folder: pathlib.Path) -> str:
        """Starts a folder's service once to learn its digest and registers it.

        The service is started, its ``/health`` ``artifact_digest`` is read, and
        the process is killed; the digest then maps to the folder for `deploy`.

        Args:
            folder: A service folder.

        Returns:
            The artifact digest the service reports.

        Raises:
            ScmleError: If the service does not become ready.
        """
        probe = self.fleet.start(folder)
        if ready_after(probe) is None:
            raise errors.ScmleError("the image does not start")
        digest = str(
            demo.get_json(f"{probe.base}/health")[1]["artifact_digest"]
        )
        probe.process.kill()
        probe.process.wait(timeout=15)
        self.images[digest] = folder
        return digest

    def deploy(self, image_digest: str) -> str:
        """Starts a revision of an image; the first one gets the traffic.

        Earlier revisions are left running.

        Args:
            image_digest: The digest of a registered image.

        Returns:
            The name of the new revision, ``local-<n>``.
        """
        self.count += 1
        revision = f"local-{self.count}"
        self.revisions[revision] = self.fleet.start(self.images[image_digest])
        self.digests[revision] = image_digest
        if self.primary is None:
            self.primary = revision
        return revision

    def set_traffic(self, allocation: Mapping[str, int]) -> None:
        """Points all traffic at one revision.

        Args:
            allocation: A mapping from revision to percentage.

        Raises:
            Unsupported: If the allocation is not 100 percent to a single
                revision.
        """
        if len(allocation) != 1 or sum(allocation.values()) != 100:
            raise control.Unsupported("only all traffic to one revision")
        self.primary = next(iter(allocation))

    def ready(self, revision: str) -> bool:
        """Whether a revision answers /readyz with 200.

        Args:
            revision: The revision to ask.

        Returns:
            True if it answers with status 200; False otherwise.
        """
        try:
            return (
                demo.get_json(f"{self.revisions[revision].base}/readyz", 2.0)[0]
                == 200
            )
        except OSError:
            return False

    def observe(self) -> observation.Observation:
        """What is running and where traffic goes.

        Returns:
            An observation of the revision that holds the traffic, with its
            image digest, its readiness (from ``/readyz``) and the sorted
            (revision, digest) pairs of every revision deployed.

        Raises:
            ArtifactError: If nothing is deployed.
        """
        if self.primary is None:
            raise errors.ArtifactError("nothing is deployed")
        return observation.Observation(
            provider=self.provider,
            service="local-service",
            revision=self.primary,
            image_digest=self.digests[self.primary],
            ready=self.ready(self.primary),
            traffic=(observation.Traffic(self.primary, 100),),
            observed_at=demo.now(),
            revisions=tuple(sorted(self.digests.items())),
        )

    def base(self) -> str:
        """URL of the revision that holds the traffic.

        Returns:
            The base URL of the primary revision.

        Raises:
            ArtifactError: If nothing is deployed.
        """
        if self.primary is None:
            raise errors.ArtifactError("nothing is deployed")
        return self.revisions[self.primary].base


def register(
    store: registry.ModelRegistryStore,
    release: release_state.ReleaseRegistry,
    spec: dict[str, Any],
    work: pathlib.Path,
    label: str,
) -> registry.VersionRecord:
    """Registers and approves one signed bundle.

    Args:
        store: The model registry.
        release: The release registry.
        spec: The reference model's specification.
        work: Directory in which the bundle is saved, under `label`.
        label: Name of the bundle, also used as the run id.

    Returns:
        The record of the registered version.
    """
    exported = parity.reference_bundle(spec).save(
        work / label, demo.DEMO_SIGNING_KEY
    )
    record = store.register(
        MODEL,
        exported,
        metrics={},
        run_id=label,
        signing_key=demo.DEMO_SIGNING_KEY,
    )
    release.approve(
        MODEL,
        record.version,
        demo.approval_for(record, f"recovery run {label}", "recovery-run"),
        demo.DEMO_SIGNING_KEY,
    )
    return record


def bad_release() -> recovery.Outcome:
    """Promotes a wrong release, detects it from labels and rolls it back.

    A good and an inverted-coefficient version are promoted in turn. Noisy
    labels taken from the good model's answers are fed to a signal monitor
    until the loss is classified as model performance, then the release is
    rolled back and the registry, ledger and runtime are reconciled. A third
    release whose image cannot become ready must then be refused, leave the
    registry unchanged and leave the previous revision serving.

    Returns:
        The recovery result of this fault: its checks, its RTO and RPO
        measurements and the details observed.
    """
    fleet = Fleet()
    base = work_folder("release")
    result = recovery.Outcome("bad_release", environment="local processes")
    try:
        store = registry.ModelRegistryStore(base / "registry")
        ledger = deployment_ledger.DeploymentLedger.create(
            None, charter_hash="recovery", policy={}, alpha_promo="1/20"
        )
        release = release_state.ReleaseRegistry(
            store,
            ledger,
            control.Polling(attempts=40, delay=recovery.scaled(0.5)),
        )
        good_spec = golden_spec()
        third_spec = {**good_spec, "model_version": "golden-3"}
        v1 = register(store, release, good_spec, base, "v1")
        v2 = register(store, release, inverted_spec(), base, "v2")
        v3 = register(store, release, third_spec, base, "v3")
        plane = ProcessPlane(fleet)
        d1 = plane.register_image(export_service(base / "img1", good_spec))
        d2 = plane.register_image(
            export_service(base / "img2", inverted_spec())
        )
        d3 = plane.register_image(export_service(base / "img3", third_spec))
        first = release.promote(
            MODEL,
            v1.version,
            plane,
            ENVIRONMENT,
            d1,
            signing_key=demo.DEMO_SIGNING_KEY,
        )
        known_good = predict(plane.base())[1].get("predictions")
        promoted = release.promote(
            MODEL,
            v2.version,
            plane,
            ENVIRONMENT,
            d2,
            signing_key=demo.DEMO_SIGNING_KEY,
        )
        wrong = predict(plane.base())[1].get("predictions")
        # delayed labels: the truth is what the good model says, with 5% noise
        rng = numpy.random.default_rng(3)
        watch = signals.SignalMonitor(
            signals.Window(window_seconds=1000.0, min_samples=30),
            signals.SignalThresholds(reference_loss=0.05, loss_tolerance=0.05),
        )
        detected_after = None
        detected_at = None
        for step in range(60):
            served = predict(plane.base(), 3)[1].get("predictions") or []
            if not known_good or len(served) != len(known_good):
                # A request that failed or came back short has no labels to
                # compare with; it must not abort the run with a mismatch.
                continue
            truth = [p if rng.random() > 0.05 else 1 - p for p in known_good]
            watch.observe_outcomes(
                float(step), signals.zero_one_losses(served, truth)
            )
            verdict = classify.classify(watch.evaluate(float(step)))
            if verdict.verdict is classify.Verdict.MODEL_PERFORMANCE:
                detected_after = (step + 1) * 3
                detected_at = time.monotonic()
                break
        rolled = release.rollback(
            MODEL, ENVIRONMENT, plane, signing_key=demo.DEMO_SIGNING_KEY
        )
        elapsed = time.monotonic() - detected_at if detected_at else None
        restored = predict(plane.base())[1].get("predictions")
        consistent = release.reconcile(
            MODEL, ENVIRONMENT, plane.observe()
        ).consistent
        plane.images[d3] = corrupt_copy(plane.images[d3], base / "img3-broken")
        before = json.dumps(release.read(), sort_keys=True)
        unready = release.promote(
            MODEL,
            v3.version,
            plane,
            ENVIRONMENT,
            d3,
            signing_key=demo.DEMO_SIGNING_KEY,
        )
        after = json.dumps(release.read(), sort_keys=True)
        still = predict(plane.base())[1].get("predictions")
        result.checks = {
            "the first and the bad release were promoted": first.done
            and promoted.done,
            "the known good release answered the labelling requests": bool(
                known_good
            ),
            "the bad release served different answers": wrong != known_good,
            "the loss signal classified it as model performance": (
                detected_after is not None
            ),
            "the rollback was confirmed by the platform": rolled.done,
            "the rolled-back service answers as before": restored == known_good,
            "registry, ledger and runtime agree after the rollback": consistent,
            "a release that never becomes ready was not promoted": (
                not unready.done and unready.restored is True
            ),
            "the refused release changed nothing in the registry": before
            == after,
            "the previous revision still serves": still == known_good,
        }
        result.measurements = [
            recovery.Measurement(
                "service",
                elapsed,
                0,
                "from the labelled-outcome window classifying the release as "
                "model performance to the platform confirming the previous "
                "revision with all traffic (rollback and read-back). Detection "
                "delay is reported separately. " + METHOD_CLOCK,
            )
        ]
        result.detail = {
            "detected_after_labelled_cases": detected_after,
            "rollback": rolled.to_dict(),
            "unready_release": unready.to_dict(),
            "control_plane": "ProcessPlane: revisions are local processes",
        }
    finally:
        result.detail["processes_still_alive_after_close"] = fleet.close()
        shutil.rmtree(base, ignore_errors=True)
    return result


# ---- jobs and ledgers --------------------------------------------------

# POSIX reports death by signal N as -N; Windows has no signals, so
# `os.kill(pid, N)` terminates the process with exit code N.
KILLED_RETURN_CODE = (
    int(limits.KILL_SIGNAL) if limits.WINDOWS else -int(limits.KILL_SIGNAL)
)

JOB_CHILD = """
import json, os, sys, time
from scmle.core import limits
from scmle.data.store import objectstore
from scmle.workflow import definition as wd
from scmle.workflow import executor as ex

root, marker, kill = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
flow = wd.Workflow(
    "recovery-job",
    "1",
    (
        wd.Step("ingest", wd.StepKind.INGEST),
        wd.Step("train", wd.StepKind.TRAIN, ("ingest",)),
        wd.Step("evaluate", wd.StepKind.EVALUATE, ("train",)),
    ),
)

def handler(name):
    def run(context):
        with open(marker, "a") as handle:
            handle.write(name + "\\n")
            handle.flush()
            os.fsync(handle.fileno())
        if name == "train" and kill:
            os.kill(os.getpid(), limits.KILL_SIGNAL)
            time.sleep(60)  # delivery is asynchronous; do not return
        return {"step": name}
    return run

store = objectstore.LocalObjectStore(root)
runner = ex.Executor(
    flow,
    store,
    {n: handler(n) for n in ("ingest", "train", "evaluate")},
    lease_seconds=1.0,
)
record = runner.start("trigger-1")
print(json.dumps({"state": record["state"]}), flush=True)
"""


def job_interruption() -> recovery.Outcome:
    """A three-step workflow run kills itself in step two; a rerun resumes it.

    A second process started after the one-second lease runs the same trigger
    from the stored state.

    Returns:
        The recovery result of this fault: its checks, its RTO and RPO
        measurements and the details observed.
    """
    fleet = Fleet()
    base = work_folder("job")
    result = recovery.Outcome(
        "job_interruption", environment="local subprocess"
    )
    try:
        marker = base / "marker.txt"
        state = base / "state"
        first = fleet.child(JOB_CHILD, str(state), str(marker), "1")
        first.wait(timeout=60)
        killed = first.returncode == KILLED_RETURN_CODE
        after_kill = marker.read_text(encoding="utf-8").split()
        started = time.monotonic()
        time.sleep(1.5)  # the lease is 1 s; a live run would hold it
        second = fleet.child(JOB_CHILD, str(state), str(marker), "0")
        output = (second.stdout.read() if second.stdout else "").strip()
        second.wait(timeout=60)
        elapsed = time.monotonic() - started
        final = marker.read_text(encoding="utf-8").split()
        finished = '"succeeded"' in output
        persisted = objectstore.LocalObjectStore(state)
        keys = persisted.list_objects("workflows/recovery-job/")
        repeated_finished = final.count("ingest") - 1
        result.checks = {
            "the first run was killed during train": killed
            and after_kill == ["ingest", "train"],
            "the continued run finished": finished,
            "the finished step was not repeated": final.count("ingest") == 1,
            "the interrupted step ran again once": final.count("train") == 2,
            "the remaining step ran once": final.count("evaluate") == 1,
        }
        result.measurements = [
            recovery.Measurement(
                "job_state",
                elapsed,
                max(0, repeated_finished),
                "a child runs a 3-step workflow and SIGKILLs itself in step "
                "2; after the 1 s lease a second process starts the same "
                "trigger; RTO includes the lease wait; RPO counts completed "
                "steps run again or lost",
            )
        ]
        result.detail = {
            "handler_calls": final,
            "state_objects": len(list(keys)),
            "lease_seconds": 1.0,
        }
    finally:
        result.detail["processes_still_alive_after_close"] = fleet.close()
        shutil.rmtree(base, ignore_errors=True)
    return result


def audit_continuity() -> recovery.Outcome:
    """Kills an audit-log writer five times and reopens each log.

    Each reopened log must verify, hold every acknowledged row, continue after
    three more rows and verify again.

    Returns:
        The recovery result of this fault: its checks, its RTO and RPO
        measurements and the details observed.
    """
    fleet = Fleet()
    base = work_folder("ledger")
    result = recovery.Outcome(
        "audit_log_continuity", environment="local subprocess"
    )
    try:
        rng = numpy.random.default_rng(11)
        lost = 0
        broken = 0
        reload_seconds = []
        contiguous = True
        for trial in range(5):
            path = base / f"audit-{trial}.jsonl"
            acknowledged = kill_after(
                fleet, LEDGER_WRITER, path, int(rng.integers(3, 61))
            )
            started = time.monotonic()
            try:
                reopened = audit.Log.open(path)
            except deployment_ledger.DeploymentLedgerError:
                broken += 1
                continue
            reload_seconds.append(time.monotonic() - started)
            ticks = [r["body"]["n"] for r in reopened.rows("admin")]
            lost += max(0, acknowledged + 1 - len(ticks))
            for extra in range(3):
                reopened.admin("operator", "after_restart", n=extra)
            again = audit.Log.open(path)
            again.ledger.verify()
            contiguous &= ticks == list(range(len(ticks)))
            contiguous &= [r["seq"] for r in again.ledger.rows] == list(
                range(len(again.ledger.rows))
            )
        result.checks = {
            "every killed ledger reopened with a verified chain": broken == 0,
            "recovered rows are contiguous and the chain continues": contiguous,
        }
        result.measurements = [
            recovery.Measurement(
                "ledger",
                max(reload_seconds) if reload_seconds else None,
                lost,
                "5 trials: a child appends audit rows and prints each "
                "acknowledged one; SIGKILL at a random point; the ledger is "
                "reopened (chain verified) and must hold at least every "
                "acknowledged row; then it is extended and verified again. "
                "Machine crash and power loss are not covered.",
            )
        ]
        result.detail = {"trials": 5, "torn_or_broken": broken}
    finally:
        result.detail["processes_still_alive_after_close"] = fleet.close()
        shutil.rmtree(base, ignore_errors=True)
    return result


SCENARIOS: dict[str, Callable[[], recovery.Outcome]] = {
    "process_kill": process_kill,
    "unhealthy_container_rolling_restart": rolling_restart,
    "invalid_credentials": invalid_credentials,
    "lost_object_store_access": object_store_loss,
    "dependency_outage": metrics_destination_outage,
    "corrupted_bundle": corrupted_bundle,
    "incompatible_schema": incompatible_schema,
    "bad_release": bad_release,
    "job_interruption": job_interruption,
    "audit_log_continuity": audit_continuity,
    "secret_rotation_recovery": secret_rotation,
}


def run_all(only: list[str] | None = None) -> dict[str, Any]:
    """Runs the local scenarios of the fault catalogue and builds the report.

    Args:
        only: Fault ids to run; all if None. Faults that are not selected or
            have no local scenario are reported as not run, and ids outside
            the catalogue are ignored.

    Returns:
        The report built by `Runner.run`. A scenario that raises is recorded
        in it as a failed fault rather than propagated.
    """
    return recovery.Runner(SCENARIOS).run(only)
