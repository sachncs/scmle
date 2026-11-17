"""What every fault scenario shares: the fleet, the service and its probes."""

from collections.abc import Iterator
from collections.abc import Mapping
import contextlib
import copy
import dataclasses
import http
import json
import os
import pathlib
import shutil
import subprocess  # nosec B404 - runs our own interpreter and serve.py
import sys
import tempfile
import time
from typing import Any

from scmle.artifacts import bundle
from scmle.contracts import errors
from scmle.deploy import mocks
from scmle.deploy import packaging
from scmle.deploy import parity
from scmle.deploy import provenance
from scmle.operations import recovery
from scmle.rehearsal import demo

ROTATED_KEY = "scmle-rotated-signing-key-local-recovery-run"
WRONG_KEY = "scmle-wrong-signing-key-local-recovery-run"
MODEL = "recovery-model"
ENVIRONMENT = "production"
GOLDEN = pathlib.Path("deploy/local/golden")
READY_SECONDS = 60.0  # before `recovery.scaled`
METHOD_CLOCK = "time.monotonic() in the runner"
SPOOLED_SAMPLES = 50  # samples the outage scenario writes to the spool
FULL_TRAFFIC_PERCENT = 100  # a single revision takes all the traffic
LABEL_NOISE_RATE = 0.05  # share of delayed labels that disagree with the truth
INTERRUPTED_STEP_RUNS = 2  # the killed step runs once before and once after


def records(count: int = 3) -> list[dict[str, Any]]:
    """The first ordinary golden records.

    Args:
        count: Number of records to return.

    Returns:
        The first `count` records of the golden record file.
    """
    loaded = json.loads((GOLDEN / "records.json").read_text(encoding="utf-8"))
    return list(loaded["records"][:count])


def specify() -> dict[str, Any]:
    """The hand-written reference model's specification.

    Returns:
        The specification of the hand-written reference model.
    """
    return parity.goldens(GOLDEN).spec


def invert() -> dict[str, Any]:
    """The same model with every coefficient negated: a wrong model.

    Returns:
        A copy of the golden specification with negated coefficients and
        intercept, versioned ``golden-inverted``.
    """
    spec = copy.deepcopy(specify())
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
    packaging.scaffold(parity.handwrite(spec), folder, signing_key=key)
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
        self.servers: list[mocks.Stub] = []
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
        chosen = port or demo.unused()
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
        process = subprocess.Popen(  # noqa: S603 -- our interpreter, our file  # nosec B603
            [sys.executable, str(folder / "serve.py")],
            cwd=folder,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        instance = Instance(process, chosen, folder)
        self.instances.append(instance)
        return instance

    def serve(self, server: mocks.Stub) -> mocks.Stub:
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
        process = subprocess.Popen(  # noqa: S603 -- our interpreter, our code  # nosec B603
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
                # handled: counted as stray process, returned
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
        # handled: documented (0, {}) for unreachable
        return 0, {}


def boot(instance: Instance, timeout: float | None = None) -> float | None:
    """Seconds until ``/readyz`` is 200, or None (time-scaled default).

    Args:
        instance: The running instance.
        timeout: Seconds to wait; the time-scaled default if None.

    Returns:
        The seconds it took to become ready, or None if it did not within
        the timeout or the process exited first.
    """
    limit = recovery.scaled(READY_SECONDS) if timeout is None else timeout
    return demo.settle(instance.base, instance.process, limit)


def failing(instance: Instance, timeout: float | None = None) -> dict[str, Any]:
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
            status, body = demo.retrieve(f"{instance.base}/readyz", timeout=2.0)
        except OSError:
            # handled: expected while starting; retried until deadline
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
        seen["livez"] = demo.retrieve(f"{instance.base}/livez", timeout=2.0)[0]
    except OSError:
        # handled: recorded livez status 0
        seen["livez"] = 0
    seen["predict"] = predict(instance.base)[0]
    return seen


def corrupt(service: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
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


def digest(service: pathlib.Path) -> str:
    """SHA-256 of the model file's bytes (what tampering changes).

    Args:
        service: A service folder.

    Returns:
        The digest of the folder's model file.
    """
    path = service / "model" / bundle.MODEL_FILE
    return provenance.sha256_digest(path.read_bytes())


def provision(prefix: str) -> pathlib.Path:
    """A fresh scratch folder.

    Args:
        prefix: Name part to tell the folder apart.

    Returns:
        The path of a new temporary directory.
    """
    return pathlib.Path(tempfile.mkdtemp(prefix=f"scmle-recovery-{prefix}-"))


def interrupt(
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


@contextlib.contextmanager
def stage(
    prefix: str, fault: str, environment: str
) -> Iterator[tuple[Fleet, pathlib.Path, recovery.Outcome]]:
    """Gives a scenario its fleet, scratch folder and result, then cleans up.

    On leaving, the fleet is closed (the processes still alive afterwards are
    recorded in the result's detail) and the folder is removed, whether the
    scenario returned or raised.

    Args:
        prefix: Name part to tell the scratch folder apart.
        fault: The fault id of the result.
        environment: Where the scenario runs, as the result reports it.

    Yields:
        The fleet, the scratch folder and the result to fill in.
    """
    fleet = Fleet()
    base = provision(prefix)
    result = recovery.Outcome(fault, environment=environment)
    try:
        yield fleet, base, result
    finally:
        result.detail["processes_still_alive_after_close"] = fleet.close()
        shutil.rmtree(base, ignore_errors=True)


def unready(verdict: Mapping[str, Any]) -> bool:
    """Whether a service stayed alive but unready, as `failing` saw it.

    Args:
        verdict: What `failing` observed.

    Returns:
        True if ``/livez`` answered 200 while ``/readyz`` and ``/predict``
        answered 503.
    """
    return bool(
        verdict["livez"] == http.HTTPStatus.OK
        and verdict["readyz"] == http.HTTPStatus.SERVICE_UNAVAILABLE
        and verdict["predict"] == http.HTTPStatus.SERVICE_UNAVAILABLE
    )
