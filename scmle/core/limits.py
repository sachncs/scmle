"""Resource limits: enforced by supervising each trial in its own process.

* Time budget: no new trial starts after the budget, and a running trial that
  passes its hard ceiling (the remaining budget plus a grace share) is
  killed and recorded as failed, whatever code it is executing.
* Memory: no new trial starts above the limit, and a trial whose resident
  memory exceeds the limit is killed and recorded as failed. Resident memory
  (the trial and every process it started) is sampled while it runs, so a
  very fast allocation spike can overshoot by one sampling interval.
* CPU: the worker count is capped at `cpu_limit` and each worker's math
  libraries are limited to its share of the CPUs.
* GPU: no built-in model uses a GPU; `gpu="required"` blocks the run.

Supervision applies when `ResourceSpec.enforcement` is `hard` (the default);
`cooperative` only stops new trials from starting.
"""

from collections.abc import Callable
from collections.abc import Iterator
import contextlib
import dataclasses
import multiprocessing
from multiprocessing import connection
import os
import signal
import subprocess  # nosec B404
import sys
import time
from typing import Any

from scmle.contracts import errors
from scmle.contracts import tasks

try:
    import resource
except ImportError:  # Windows has no `resource` module.
    resource = None  # type: ignore[assignment]

WINDOWS = sys.platform == "win32"
KILL_SIGNAL = signal.SIGTERM if WINDOWS else signal.SIGKILL  # type: ignore[attr-defined,unused-ignore]
BYTES_PER_GB = 1024**3
KILOBYTES = 1024
GRACE_SHARE = 0.1
MIN_CEILING_SECONDS = 1.0
POLL_SECONDS = 0.1
PS_FIELDS = 3
PRELOAD_MODULES = ["numpy", "pandas", "sklearn"]
ACCELERATE_THREADS = "VECLIB_MAXIMUM_THREADS"


class TrialLimitError(errors.ScmleError):
    """Raised when a supervised trial was stopped or lost."""


@contextlib.contextmanager
def forkserver_environment() -> Iterator[None]:
    """Starts the forkserver with a fork-safe BLAS configuration.

    On macOS the Accelerate library parallelises large matrix products with
    Grand Central Dispatch, whose thread pool does not survive `fork()`: a
    trial process forked from the warmed-up server crashes on its first big
    product. Accelerate reads `VECLIB_MAXIMUM_THREADS` when it loads, so the
    server (which loads numpy) is started with it set to 1. Trials are
    already single-threaded; parallelism comes from running several. The
    variable is restored afterwards so this process is unaffected.
    """
    if sys.platform != "darwin":
        yield
        return
    previous = os.environ.get(ACCELERATE_THREADS)
    os.environ[ACCELERATE_THREADS] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(ACCELERATE_THREADS, None)
        else:
            os.environ[ACCELERATE_THREADS] = previous


def peak_memory_gb() -> float:
    """Peak resident memory of this process and its finished children.

    Windows does not report finished children, so only this process counts.
    """
    if resource is None:
        import psutil  # Windows-only dependency.

        return psutil.Process().memory_info().peak_wset / BYTES_PER_GB
    scale = 1 if sys.platform == "darwin" else KILOBYTES  # macOS reports bytes
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale
    return max(own, children) / BYTES_PER_GB


def proc_table() -> dict[int, tuple[int, int]]:
    """Process table from `/proc` (Linux), without needing `ps`."""
    table: dict[int, tuple[int, int]] = {}
    page = resource.getpagesize()
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", encoding="ascii") as handle:
                after_name = handle.read().rsplit(")", 1)[1].split()
            with open(f"/proc/{name}/statm", encoding="ascii") as handle:
                resident = int(handle.read().split()[1]) * page
        except (OSError, IndexError, ValueError):
            continue
        table[int(name)] = (int(after_name[1]), resident)
    return table


def ps_table() -> dict[int, tuple[int, int]]:
    """Process table from `ps` (macOS and other Unix systems)."""
    completed = subprocess.run(  # nosec B603 B607
        ["ps", "-A", "-o", "pid=,ppid=,rss="],
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
    """Process table from `psutil` (Windows)."""
    import psutil  # Windows-only dependency.

    table: dict[int, tuple[int, int]] = {}
    for process in psutil.process_iter(["pid", "ppid", "memory_info"]):
        info = process.info
        if info["memory_info"] is not None:
            table[info["pid"]] = (info["ppid"], info["memory_info"].rss)
    return table


def process_table() -> dict[int, tuple[int, int]]:
    """Maps every live pid to its (parent pid, resident bytes)."""
    if WINDOWS:
        return psutil_table()
    if os.path.isdir("/proc/self"):
        return proc_table()
    return ps_table()


def process_tree(root: int) -> dict[int, int]:
    """Resident bytes of a process and all its descendants, by pid."""
    table = process_table()
    members = {root}
    grew = True
    while grew:
        grew = False
        for pid, entry in table.items():
            if entry[0] in members and pid not in members:
                members.add(pid)
                grew = True
    return {pid: table[pid][1] for pid in members if pid in table}


def resident_bytes(pid: int) -> int:
    """Resident memory of a process and its descendants, 0 if it is gone."""
    return sum(process_tree(pid).values())


def child_main(
    channel: connection.Connection,
    function: Callable[[Any], Any],
    argument: Any,
) -> None:
    """Runs `function(argument)` in the supervised process."""
    try:
        channel.send((True, function(argument)))
    except BaseException as error:  # Reported to the supervisor.
        channel.send((False, f"{type(error).__name__}: {error}"))
    finally:
        channel.close()


@dataclasses.dataclass(frozen=True)
class TrialSupervisor:
    """Runs one call in a separate process under hard limits.

    Attributes:
        ceiling_seconds: Wall-clock limit (None = unlimited).
        memory_gb: Resident-memory limit in gigabytes (None = unlimited).
    """

    ceiling_seconds: float | None = None
    memory_gb: float | None = None

    def run(self, function: Callable[[Any], Any], argument: Any) -> Any:
        """Calls `function(argument)` in a child process and returns its value.

        Args:
            function: Picklable module-level function.
            argument: Picklable argument.

        Raises:
            TrialLimitError: If the child exceeded a limit, died, or raised.
        """
        context: Any
        if WINDOWS:
            context = multiprocessing.get_context("spawn")
        else:
            context = multiprocessing.get_context("forkserver")
            context.set_forkserver_preload(PRELOAD_MODULES)
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=child_main, args=(sender, function, argument)
        )
        started = time.monotonic()
        with forkserver_environment():
            process.start()
        sender.close()
        try:
            while not receiver.poll(POLL_SECONDS):
                reason = self.violation(process, time.monotonic() - started)
                if reason:
                    raise TrialLimitError(reason)
                if not process.is_alive() and not receiver.poll(0):
                    raise TrialLimitError(
                        f"trial process died (exit code {process.exitcode})"
                    )
            succeeded, value = receiver.recv()
        except EOFError as error:
            raise TrialLimitError("trial process died unexpectedly") from error
        finally:
            if process.is_alive():
                if process.pid is not None:
                    self.kill_descendants(process.pid)
                process.kill()
            process.join()
            receiver.close()
        if not succeeded:
            raise TrialLimitError(value)
        return value

    @staticmethod
    def kill_descendants(root: int) -> None:
        """Kills processes the trial started, so none outlive it."""
        for pid in process_tree(root):
            if pid != root:
                try:
                    os.kill(pid, KILL_SIGNAL)
                except ProcessLookupError:
                    continue

    def violation(
        self, process: multiprocessing.process.BaseProcess, elapsed: float
    ) -> str | None:
        """Why the running trial must be killed, or None."""
        if self.ceiling_seconds is not None and elapsed > self.ceiling_seconds:
            return (
                f"trial exceeded its {self.ceiling_seconds:.1f}s hard time "
                "ceiling and was killed"
            )
        if self.memory_gb is not None and process.pid is not None:
            used = resident_bytes(process.pid) / BYTES_PER_GB
            if used > self.memory_gb:
                return (
                    f"trial used {used:.2f} GB, above the "
                    f"{self.memory_gb:.2f} GB memory limit, and was killed"
                )
        return None


@dataclasses.dataclass(frozen=True)
class ResourceGovernor:
    """Applies and reports the configured resource limits.

    Attributes:
        spec: The resource specification.
        started_at: Monotonic start time of the run.
    """

    spec: tasks.ResourceSpec
    started_at: float = dataclasses.field(default_factory=time.monotonic)

    @property
    def workers(self) -> int:
        """Worker count after applying the CPU cap."""
        cap = self.spec.cpu_limit
        return min(self.spec.workers, cap) if cap else self.spec.workers

    @property
    def thread_limit(self) -> int | None:
        """Threads each worker's math libraries may use (None = default)."""
        cap = self.spec.cpu_limit
        if cap:
            return max(1, cap // max(self.workers, 1))
        return 1 if self.workers > 1 else None

    def memory_exceeded(self) -> bool:
        """Whether peak memory is above the configured limit."""
        return peak_memory_gb() > self.spec.memory_limit_gb

    @property
    def hard(self) -> bool:
        """Whether trials run under process supervision."""
        return self.spec.enforcement == "hard"

    def enforcement_text(self, hard_part: str) -> str:
        """Describes how a limit is applied under the current mode."""
        if self.hard:
            return f"no trial starts past the limit; {hard_part}"
        return "cooperative: no trial starts past the limit"

    def trial_ceiling_seconds(self) -> float:
        """Hard time ceiling for a trial dispatched now."""
        budget = self.spec.time_budget_seconds
        remaining = max(budget - (time.monotonic() - self.started_at), 0.0)
        return max(remaining + GRACE_SHARE * budget, MIN_CEILING_SECONDS)

    def skip_reason(self, first: bool) -> str | None:
        """Why the next trial must not start, or None.

        Args:
            first: True for the first trial, which always runs so that a
                result exists even under a tiny budget.
        """
        if first:
            return None
        if time.monotonic() - self.started_at >= self.spec.time_budget_seconds:
            return "time budget exhausted before this trial started"
        if self.memory_exceeded():
            return "memory limit reached before this trial started"
        return None

    def report(self) -> dict[str, Any]:
        """Describes each limit, how it was applied and what was observed."""
        elapsed = time.monotonic() - self.started_at
        peak = peak_memory_gb()
        return {
            "time_budget": {
                "limit_seconds": self.spec.time_budget_seconds,
                "elapsed_seconds": elapsed,
                "enforcement": self.enforcement_text(
                    "running trials are killed at the hard ceiling "
                    f"(remaining budget + {GRACE_SHARE:.0%} grace)"
                ),
                "within_limit": elapsed <= self.spec.time_budget_seconds,
            },
            "cpu": {
                "limit": self.spec.cpu_limit,
                "workers": self.workers,
                "threads_per_worker": self.thread_limit,
                "enforcement": "worker count and library threads capped"
                if self.spec.cpu_limit
                else "no cap configured",
            },
            "memory": {
                "limit_gb": self.spec.memory_limit_gb,
                "observed_peak_gb": peak,
                "enforcement": self.enforcement_text(
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
