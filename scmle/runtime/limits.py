"""Resource limits: enforced by supervising each trial in its own process.

* Time budget: the run ends at its start plus the budget. Before training,
  the post-training work (final refit, diagnostics, packaging) is estimated
  from measured cost models and held back as a reserve, so trials run in a
  window that closes `reserve` seconds before the end (`Governor.until`).
  No trial starts after the window, and a running trial still going when it
  closes is killed and recorded as failed, whatever code it is executing.
  A run whose minimum (preparation, one cheapest trial and the reserve)
  cannot fit is refused before any work (`budget.Reserve.admit`).
* Memory: no new trial starts above the limit, and a trial whose resident
  memory exceeds the limit is killed and recorded as failed. Resident memory
  (the trial and every process it started) is sampled while it runs, so a
  very fast allocation spike can overshoot by one sampling interval.
* CPU: the worker count is capped at `cpu_limit`, each worker's math
  libraries are limited to its share of the CPUs, and the main process's
  math libraries to `cpu_limit` threads while the stages run.
* GPU: no built-in model uses a GPU; `gpu="required"` blocks the run.

Supervision applies when `Resources.enforcement` is `hard` (the default);
`cooperative` only stops new trials from starting.
"""

from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Mapping
import contextlib
import dataclasses
import importlib
import logging
import multiprocessing
from multiprocessing import connection
import multiprocessing.process
import os
import pathlib
import signal
import subprocess  # nosec B404
import sys
import threading
import time
from typing import Any
from typing import cast

import threadpoolctl

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.runtime import budget
from scmle.runtime import logs

LOGGER = logging.getLogger(__name__)

resource: Any  # the `resource` module, or None where it is missing
try:
    resource = importlib.import_module("resource")
except ImportError:  # Windows has no `resource` module.
    # handled: optional module absent; None checked by callers (Windows)
    resource = None
    logs.degrade(
        LOGGER,
        "no `resource` module: peak memory is read through psutil and "
        "excludes finished child processes",
    )

WINDOWS = sys.platform == "win32"
KILL_SIGNAL = signal.SIGTERM if WINDOWS else signal.SIGKILL  # type: ignore[attr-defined,unused-ignore]
BYTES_PER_GB = 1024**3
KILOBYTES = 1024
POLL_SECONDS = 0.1
# Enforcement slack: a trial past its ceiling is noticed at the next poll
# (at most POLL_SECONDS later) and then killed and joined. Kill plus join
# measured 0.028-0.032 s over 12 kills of a spinning trial (macOS arm64,
# 2026-11-17, `scratchpad/train/killtime.py`); KILL_SECONDS is about three
# times the largest, for the `ps` scan of descendants under load.
KILL_SECONDS = 0.1
SLACK_SECONDS = POLL_SECONDS + KILL_SECONDS
PHASES = ("trials", "finish")
PS_FIELDS = 3
PRELOAD_MODULES = ["numpy", "pandas", "sklearn"]
ACCELERATE_THREADS = "VECLIB_MAXIMUM_THREADS"


class TrialError(errors.ScmleError):
    """Raised when a supervised trial was stopped or lost."""


# A plain constant, so type checkers on Linux do not treat the macOS branch
# below as unreachable.
ON_MACOS = sys.platform == "darwin"
ENVIRONMENT_LOCK = threading.Lock()
ENVIRONMENT_USERS = [0]
"""How many threads are inside `preforked` (one-item list)."""
ENVIRONMENT_SAVED: list[str | None] = [None]
"""The variable's value before the first of them changed it."""


@contextlib.contextmanager
def preforked() -> Generator[None, None, None]:
    """Starts the forkserver with a fork-safe BLAS configuration.

    On macOS the Accelerate library parallelises large matrix products with
    Grand Central Dispatch, whose thread pool does not survive `fork()`: a
    trial process forked from the warmed-up server crashes on its first big
    product. Accelerate reads `VECLIB_MAXIMUM_THREADS` when it loads, so the
    server (which loads numpy) is started with it set to 1. Trials are
    already single-threaded; parallelism comes from running several. The
    variable is restored afterwards so this process is unaffected. Threads
    share one change: the first in saves the original value and the last
    out restores it, so overlapping uses cannot leave the "1" behind.
    """
    if not ON_MACOS:
        yield
        return
    with ENVIRONMENT_LOCK:
        if ENVIRONMENT_USERS[0] == 0:
            ENVIRONMENT_SAVED[0] = os.environ.get(ACCELERATE_THREADS)
            os.environ[ACCELERATE_THREADS] = "1"
        ENVIRONMENT_USERS[0] += 1
    try:
        yield
    finally:
        with ENVIRONMENT_LOCK:
            ENVIRONMENT_USERS[0] -= 1
            if ENVIRONMENT_USERS[0] == 0:
                previous = ENVIRONMENT_SAVED[0]
                if previous is None:
                    os.environ.pop(ACCELERATE_THREADS, None)
                else:
                    os.environ[ACCELERATE_THREADS] = previous


def peak_memory_gb() -> float:
    """Peak resident memory of this process and its finished children.

    Windows does not report finished children, so only this process counts.

    Returns:
        The larger of this process's and its finished children's peak resident
        memory, in gigabytes (the resident set on Windows, where children are
        not counted).
    """
    if resource is None:
        import psutil  # noqa: PLC0415 -- Windows-only dependency  # pyright: ignore[reportMissingModuleSource]  # psutil is a Windows-only dependency and ships no stub source here

        peak_wset: float = psutil.Process().memory_info().peak_wset
        return peak_wset / BYTES_PER_GB
    scale = 1 if sys.platform == "darwin" else KILOBYTES  # macOS reports bytes
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale
    peak: float = max(own, children) / BYTES_PER_GB
    return peak


def tabulate() -> dict[int, tuple[int, int]]:
    """Process table from `/proc` (Linux), without needing `ps`.

    Returns:
        A mapping from pid to (parent pid, resident bytes). Processes that
        vanish or cannot be read while scanning are omitted.
    """
    table: dict[int, tuple[int, int]] = {}
    page = resource.getpagesize()
    for entry in pathlib.Path("/proc").iterdir():
        name = entry.name
        if not name.isdigit():
            continue
        try:
            with (entry / "stat").open(encoding="ascii") as handle:
                after_name = handle.read().rsplit(")", 1)[1].split()
            with (entry / "statm").open(encoding="ascii") as handle:
                resident = int(handle.read().split()[1]) * page
        except (OSError, IndexError, ValueError):
            # handled: process vanished mid-scan; expected race, entry skipped
            continue
        table[int(name)] = (int(after_name[1]), resident)
    return table


def ps_table() -> dict[int, tuple[int, int]]:
    """Process table from `ps` (macOS and other Unix systems).

    Returns:
        A mapping from pid to (parent pid, resident bytes). Lines of the `ps`
        output that are not three integer fields are ignored.
    """
    completed = subprocess.run(  # nosec B603 B607
        ["ps", "-A", "-o", "pid=,ppid=,rss="],  # noqa: S607 -- `ps` is found on PATH by design
        capture_output=True,
        text=True,
        check=False,
    )
    table: dict[int, tuple[int, int]] = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) == PS_FIELDS and all(f.isdigit() for f in fields):
            table[int(fields[0])] = (int(fields[1]), int(fields[2]) * KILOBYTES)
    return table


def psutil_table() -> dict[int, tuple[int, int]]:
    """Process table from `psutil` (Windows).

    Returns:
        A mapping from pid to (parent pid, resident bytes). Processes whose
        memory information is unavailable are omitted.
    """
    import psutil  # noqa: PLC0415 -- Windows-only dependency  # pyright: ignore[reportMissingModuleSource]  # psutil is a Windows-only dependency and ships no stub source here

    table: dict[int, tuple[int, int]] = {}
    for process in psutil.process_iter(["pid", "ppid", "memory_info"]):
        info = process.info
        if info["memory_info"] is not None:
            table[info["pid"]] = (info["ppid"], info["memory_info"].rss)
    return table


def pids() -> dict[int, tuple[int, int]]:
    """Maps every live pid to its (parent pid, resident bytes).

    Returns:
        A mapping from pid to (parent pid, resident bytes), read from the
        source suited to the platform.
    """
    if WINDOWS:
        return psutil_table()
    if pathlib.Path("/proc/self").is_dir():
        return tabulate()
    return ps_table()


def footprint(root: int) -> dict[int, int]:
    """Resident bytes of a process and all its descendants, by pid.

    Args:
        root: Pid of the process whose subtree is measured.

    Returns:
        A mapping from pid to resident bytes for the root and every descendant
        found in the process table; a root missing from the table is left
        out, so a vanished root yields an empty mapping.
    """
    table = pids()
    members = {root}
    grew = True
    while grew:
        grew = False
        for pid, entry in table.items():
            if entry[0] in members and pid not in members:
                members.add(pid)
                grew = True
    return {pid: table[pid][1] for pid in members if pid in table}


def residency(pid: int) -> int:
    """Resident memory of a process and its descendants, 0 if it is gone.

    Args:
        pid: Pid of the root of the process subtree.

    Returns:
        The summed resident memory, in bytes.
    """
    return sum(footprint(pid).values())


def execute(
    channel: connection.Connection,
    function: Callable[[Any], Any],
    argument: Any,
) -> None:
    """Runs `function(argument)` in the supervised process.

    The outcome is sent over the channel as `(True, value)` on success or
    `(False, failure)` if the function raised any exception, where `failure`
    is `errors.Failure.dump()` with the child's traceback, and
    the channel is then closed.

    Args:
        channel: Sending end of the pipe back to the supervisor.
        function: Picklable module-level function to call.
        argument: Picklable argument passed to the function.
    """
    try:
        channel.send((True, function(argument)))
    except BaseException as error:  # noqa: BLE001 -- reported to the supervisor
        # handled: sent to the supervisor with its traceback (Failure)
        channel.send((False, dict(errors.Failure.of(error).dump())))
    finally:
        channel.close()


@dataclasses.dataclass(frozen=True)
class Supervisor:
    """Runs one call in a separate process under hard limits.

    Attributes:
        ceiling_seconds: Wall-clock limit (None = unlimited).
        memory_gb: Resident-memory limit in gigabytes (None = unlimited).
    """

    ceiling_seconds: float | None = None
    memory_gb: float | None = None

    def run(self, function: Callable[[Any], Any], argument: Any) -> Any:
        """Calls `function(argument)` in a child process and returns its value.

        If the child is still running when the call ends (a limit was
        exceeded, say), it and every process it started are killed.

        Args:
            function: Picklable module-level function.
            argument: Picklable argument.

        Returns:
            The value `function(argument)` returned in the child process.

        Raises:
            TrialError: If the child exceeded a limit, died without
                sending a result, or raised (the message is then
                `ErrorType: message` of the child's exception).
        """
        context: Any
        if WINDOWS:
            context = multiprocessing.get_context("spawn")
        else:
            context = multiprocessing.get_context("forkserver")
            context.set_forkserver_preload(PRELOAD_MODULES)
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=execute, args=(sender, function, argument)
        )
        started = time.monotonic()
        with preforked():
            process.start()
        sender.close()
        try:
            while not receiver.poll(POLL_SECONDS):
                reason = self.violation(process, time.monotonic() - started)
                if reason:
                    raise TrialError(reason)
                if not process.is_alive() and not receiver.poll(0):
                    raise TrialError(
                        f"trial process died (exit code {process.exitcode})"
                    )
            succeeded, value = receiver.recv()
        except EOFError as error:
            raise TrialError("trial process died unexpectedly") from error
        finally:
            if process.is_alive():
                if process.pid is not None:
                    self.reap(process.pid)
                process.kill()
            process.join()
            receiver.close()
        if not succeeded:
            if isinstance(value, Mapping):
                raise TrialError(
                    f"{value.get('kind')}: {value.get('message')}",
                    {"failure": dict(value)},
                )
            raise TrialError(str(value))
        return value

    @staticmethod
    def reap(root: int) -> None:
        """Kills processes the trial started, so none outlive it.

        Args:
            root: Pid of the trial process; it is not itself signalled.
        """
        for pid in footprint(root):
            if pid != root:
                try:
                    os.kill(pid, KILL_SIGNAL)
                except ProcessLookupError:
                    # handled: process already dead on kill
                    continue
                except PermissionError:
                    # Windows reports a process that has just exited (or one
                    # that is exiting) as access denied. A descendant we
                    # cannot signal because it is already gone is not an
                    # error; one that is alive and protected is left alone.
                    # handled: process already exiting on kill; documented in
                    # comment
                    continue

    def violation(
        self, process: multiprocessing.process.BaseProcess, elapsed: float
    ) -> str | None:
        """Why the running trial must be killed, or None.

        Args:
            process: The running trial process.
            elapsed: Seconds since the trial started.

        Returns:
            A message naming the exceeded limit (hard time ceiling or memory),
            or None if the trial is within its limits.
        """
        if self.ceiling_seconds is not None and elapsed > self.ceiling_seconds:
            return (
                f"trial exceeded its {self.ceiling_seconds:.1f}s hard time "
                "ceiling and was killed"
            )
        if self.memory_gb is not None and process.pid is not None:
            used = residency(process.pid) / BYTES_PER_GB
            if used > self.memory_gb:
                return (
                    f"trial used {used:.2f} GB, above the "
                    f"{self.memory_gb:.2f} GB memory limit, and was killed"
                )
        return None


@dataclasses.dataclass
class Governor:
    """Applies and reports the configured resource limits.

    Attributes:
        spec: The resource specification.
        started_at: Monotonic start time of the run.
        reserve: Seconds of the budget held back for post-training work
            (`budget.Reserve`); empty until its `provision`.
    """

    spec: tasks.Resources
    started_at: float = dataclasses.field(default_factory=time.monotonic)
    reserve: budget.Reserve = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Starts an empty reserve for the configured time budget."""
        self.reserve = budget.Reserve(self.spec.time_budget_seconds)

    def until(self, phase: str) -> float:
        """Returns the monotonic time by which a phase must end.

        Args:
            phase: `finish` (the whole run: start plus the budget) or
                `trials` (every trial, including strategy search, tuning and
                ensembles: the finish minus the reserve).

        Returns:
            The monotonic time.

        Raises:
            InputError: If the phase is not one of `PHASES`.
        """
        if phase not in PHASES:
            raise errors.InputError(
                f"unknown phase {phase!r}; expected one of {PHASES}"
            )
        finish = self.started_at + self.spec.time_budget_seconds
        return finish - self.reserve.total if phase == "trials" else finish

    def left(self, phase: str) -> float:
        """Returns the seconds until a phase must end (negative when late).

        Args:
            phase: `trials` or `finish` (see `until`).

        Returns:
            The seconds left.
        """
        return self.until(phase) - time.monotonic()

    def overdue(self, seconds: float, keep: float = 0.0) -> str | None:
        """Why post-training work of a given length must not start, or None.

        Args:
            seconds: Estimated length of the work.
            keep: Seconds to leave before the finish for later work.

        Returns:
            A message when the work would not end `keep` seconds before
            the run's end or memory is above the limit, else None.
        """
        left = self.left("finish") - keep
        if seconds > left:
            return (
                f"needs about {seconds:.1f} s but {max(left, 0.0):.1f} s "
                "remain before the run's end"
            )
        if self.exceeds():
            return "memory limit reached"
        return None

    @property
    def workers(self) -> int:
        """Worker count after applying the CPU cap."""
        cap = self.spec.cpu_limit
        return min(self.spec.workers, cap) if cap else self.spec.workers

    @property
    def threadcap(self) -> int | None:
        """Threads each worker's math libraries may use (None = default)."""
        cap = self.spec.cpu_limit
        if cap:
            return max(1, cap // max(self.workers, 1))
        return 1 if self.workers > 1 else None

    def cap(self) -> contextlib.AbstractContextManager[object]:
        """Limits this process's math-library threads to `cpu_limit`.

        Trial workers get their share of the CPUs through `threadcap`;
        everything else (profiling, diagnostics and evaluation fits) runs in
        the main process and would otherwise use every core whatever the
        limit says. With no `cpu_limit` nothing changes.

        Returns:
            A context manager that applies the limit while it is open.
        """
        return cast(
            "contextlib.AbstractContextManager[object]",
            threadpoolctl.threadpool_limits(limits=self.spec.cpu_limit),
        )

    def exceeds(self) -> bool:
        """Whether peak memory is above the configured limit.

        Returns:
            True if the larger of this process's and its finished children's
            peak resident memory exceeds `memory_limit_gb`.
        """
        return peak_memory_gb() > self.spec.memory_limit_gb

    @property
    def hard(self) -> bool:
        """Whether trials run under process supervision."""
        return self.spec.enforcement == "hard"

    def explain(self, hard_part: str) -> str:
        """Describes how a limit is applied under the current mode.

        Args:
            hard_part: Wording for what hard enforcement additionally does.

        Returns:
            The hard-mode description (stated limit plus `hard_part`) when
            enforcement is hard, else the cooperative-mode description.
        """
        if self.hard:
            return f"no trial starts past the limit; {hard_part}"
        return "cooperative: no trial starts past the limit"

    def deadline(self) -> float:
        """Hard time ceiling for a trial dispatched now.

        The trial window closes at `until("trials")`; there is no grace past
        it, so a trial is killed when the window closes (plus the
        enforcement slack, `SLACK_SECONDS`).

        Returns:
            The seconds until the trial window closes, never below zero.
        """
        return max(self.left("trials"), 0.0)

    def skip_reason(self, first: bool) -> str | None:
        """Why the next trial must not start, or None.

        Args:
            first: True for the first trial, which runs above the memory
                limit so that a result exists. Time applies to it too: the
                run was admitted only if one trial fits.

        Returns:
            A message saying why the next trial must not start (the trial
            window is too short to start one, or the memory limit is
            reached), or None if it may start.
        """
        left = self.left("trials")
        if left < budget.START_SECONDS:
            return (
                f"trial window closed: {max(left, 0.0):.1f} s left before "
                f"the {self.reserve.total:.1f} s reserved for the final "
                "refit, diagnostics and packaging"
            )
        if not first and self.exceeds():
            return "memory limit reached before this trial started"
        return None

    def report(self) -> dict[str, Any]:
        """Describes each limit, how it was applied and what was observed.

        Returns:
            A dict with `time_budget`, `cpu`, `memory` and `gpu` entries. The
            time and memory entries give the limit, the observed value, how
            it is enforced and `within_limit`; `cpu` gives the limit, worker
            count, threads per worker and enforcement; `gpu` gives the mode
            and enforcement.
        """
        elapsed = time.monotonic() - self.started_at
        peak = peak_memory_gb()
        return {
            "time_budget": {
                "limit_seconds": self.spec.time_budget_seconds,
                "elapsed_seconds": elapsed,
                "enforcement": self.explain(
                    "running trials are killed at the hard ceiling, the "
                    "end of the trial window (budget minus the reserve; "
                    f"enforcement slack {SLACK_SECONDS:.1f} s)"
                ),
                "reserve_seconds": dict(self.reserve.parts),
                "within_limit": elapsed <= self.spec.time_budget_seconds,
            },
            "cpu": {
                "limit": self.spec.cpu_limit,
                "workers": self.workers,
                "threads_per_worker": self.threadcap,
                "enforcement": "worker count and library threads capped"
                if self.spec.cpu_limit
                else "no cap configured",
            },
            "memory": {
                "limit_gb": self.spec.memory_limit_gb,
                "observed_peak_gb": peak,
                "enforcement": self.explain(
                    "running trials above the limit are killed "
                    "(resident memory sampled while they run)"
                ),
                "within_limit": peak <= self.spec.memory_limit_gb,
            },
            "gpu": {
                "mode": self.spec.gpu,
                "enforcement": "no built-in model uses a GPU",
            },
        }
