"""Resource limits: enforce what can be enforced, report what cannot.

* Time budget: no new trial starts after the budget, and a running trial is
  interrupted by a hard ceiling (the remaining budget plus a grace share)
  delivered as a timer signal, so it fails instead of overrunning.
* CPU: the worker count is capped at `cpu_limit` and each worker's math
  libraries are limited to its share of the CPUs.
* Memory: no new trial starts above the limit, and where the operating
  system honours address-space limits (Linux) each trial runs under a hard
  cap, so an allocation beyond it fails the trial. macOS cannot enforce
  this; there the limit stays cooperative and the report says so.
* GPU: no built-in model uses a GPU; `gpu="required"` blocks the run.
"""

from collections.abc import Iterator
import contextlib
import dataclasses
import resource
import signal
import sys
import threading
import time
from types import FrameType
from typing import Any
from typing import Never

from mml.contracts import errors
from mml.contracts import tasks

BYTES_PER_GB = 1024**3
KILOBYTES = 1024
GRACE_SHARE = 0.1
MIN_CEILING_SECONDS = 1.0
HARD_MEMORY_SUPPORTED = sys.platform.startswith("linux")


class TrialLimitError(errors.MagicMLError):
    """Raised inside a trial that exceeded its hard time ceiling."""


def current_address_space() -> int:
    """Bytes of address space this process holds now (Linux only)."""
    with open("/proc/self/statm", encoding="ascii") as handle:
        pages = int(handle.read().split()[0])
    return pages * resource.getpagesize()


@contextlib.contextmanager
def hard_limits(
    ceiling_seconds: float | None, memory_gb: float | None
) -> Iterator[None]:
    """Applies a hard time ceiling and (on Linux) a memory cap to a block.

    The time ceiling uses a timer signal, so it only applies in the main
    thread; elsewhere the block runs unbounded and the governor's report
    states that. Previous handlers and limits are always restored.

    Args:
        ceiling_seconds: Seconds before `TrialLimitError` is raised.
        memory_gb: Address-space cap in gigabytes.

    Yields:
        None.
    """
    previous_handler = None
    previous_memory = None
    use_timer = threading.current_thread() is threading.main_thread()
    if ceiling_seconds is None or not use_timer:
        use_timer = False
    else:

        def interrupt(signum: int, frame: FrameType | None) -> Never:
            raise TrialLimitError(
                f"trial exceeded its {ceiling_seconds:.1f}s hard time ceiling"
            )

        previous_handler = signal.signal(signal.SIGALRM, interrupt)
        signal.setitimer(signal.ITIMER_REAL, ceiling_seconds)
    if memory_gb is not None and HARD_MEMORY_SUPPORTED:
        previous_memory = resource.getrlimit(resource.RLIMIT_AS)
        hard = previous_memory[1]
        wanted = current_address_space() + int(memory_gb * BYTES_PER_GB)
        if hard != resource.RLIM_INFINITY:
            wanted = min(wanted, hard)
        resource.setrlimit(resource.RLIMIT_AS, (wanted, hard))
    try:
        yield
    finally:
        if previous_memory is not None:
            resource.setrlimit(resource.RLIMIT_AS, previous_memory)
        if use_timer:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)


def peak_memory_gb() -> float:
    """Peak resident memory of this process and its finished children."""
    scale = 1 if sys.platform == "darwin" else KILOBYTES  # macOS reports bytes
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale
    return max(own, children) / BYTES_PER_GB


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
                "enforcement": "no trial starts after the budget; running "
                "trials are interrupted at the hard ceiling (remaining "
                f"budget + {GRACE_SHARE:.0%} grace) in a main thread",
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
                "enforcement": "no trial starts above the limit; each trial "
                "runs under a hard cap on address-space growth"
                if HARD_MEMORY_SUPPORTED
                else "cooperative only on this platform: no trial starts "
                "above the limit (the OS cannot cap address space)",
                "within_limit": peak <= self.spec.memory_limit_gb,
            },
            "gpu": {
                "mode": self.spec.gpu,
                "enforcement": "no built-in model uses a GPU",
            },
        }
