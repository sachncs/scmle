"""Resource limits: enforce what can be enforced, report what cannot.

* Time budget: cooperative, checked before each trial or chunk of trials
  (a running trial cannot be interrupted safely).
* CPU: the worker count is capped at `cpu_limit` and each worker's math
  libraries are limited to its share of the CPUs.
* Memory: cooperative. Python offers no portable hard cap, so the run stops
  starting new trials once this process's peak memory exceeds the limit;
  worker processes are not individually bounded.
* GPU: no built-in model uses a GPU; `gpu="required"` blocks the run.
"""

import dataclasses
import resource
import sys
import time
from typing import Any

from mml.contracts import tasks

BYTES_PER_GB = 1024**3
KILOBYTES = 1024


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
                "enforcement": "cooperative, checked between trials",
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
                "enforcement": "cooperative: no new trial starts above the "
                "limit; worker processes are not individually bounded",
                "within_limit": peak <= self.spec.memory_limit_gb,
            },
            "gpu": {
                "mode": self.spec.gpu,
                "enforcement": "no built-in model uses a GPU",
            },
        }
