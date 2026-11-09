"""Readiness and admission control shared by the HTTP serving adapters.

`Readiness` is the difference between "the process is up" (liveness) and "the
process may take traffic" (readiness): it becomes true only after the bundle
was verified, loaded and warmed up, in that order. `AdmissionControl` bounds
concurrent requests and the time one request may take, so an overloaded or
stuck model answers quickly instead of queueing without limit.
"""

from collections.abc import Callable
from collections.abc import Iterator
import concurrent.futures
import contextlib
import enum
import re
import threading
import time
from typing import Any
import uuid

from scmle.contracts import errors

REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


class Phase(enum.StrEnum):
    """Where a service is on the way to taking traffic."""

    STARTING = "starting"
    VERIFYING = "verifying"
    LOADING = "loading"
    WARMING = "warming"
    READY = "ready"
    FAILED = "failed"


STEPS = (Phase.VERIFYING, Phase.LOADING, Phase.WARMING)
STEP_COUNT = len(STEPS)


class NotReadyError(errors.ScmleError):
    """The service has not finished verify, load and warm-up (or failed)."""

    category = "not_ready"


class OverloadedError(errors.ScmleError):
    """The concurrency limit is reached; the caller should retry."""

    category = "overloaded"


class DeadlineError(errors.ScmleError):
    """The request exceeded its time limit."""

    category = "timeout"


class Readiness:
    """Tracks verify, load and warm-up; ready only when all three finished."""

    def __init__(self) -> None:
        """Starts in the `starting` phase with no step completed."""
        self.lock = threading.Lock()
        self.phase = Phase.STARTING
        self.done: dict[str, float] = {}
        self.notes: dict[str, str] = {}
        self.reason: str | None = None
        self.started = time.monotonic()

    def begin(self, step: Phase) -> None:
        """Enters `step`; steps must run in order and none may be skipped.

        The step must be the first one not yet completed, so a step that is
        already in progress may be entered again. A `step` that is not one of
        the three steps makes the lookup of its position raise `ValueError`.

        Args:
            step: The step to begin: verifying, loading or warming.

        Raises:
            ScmleError: If the service has failed, or `step` is not the next
                step after those already completed.
        """
        with self.lock:
            index = STEPS.index(step)
            expected = (
                STEPS[len(self.done)] if len(self.done) < STEP_COUNT else None
            )
            if self.phase is Phase.FAILED or step is not expected:
                raise errors.ScmleError(
                    f"cannot begin {step.value} from {self.phase.value}"
                )
            self.phase = STEPS[index]

    def complete(self, step: Phase, note: str = "") -> None:
        """Marks `step` finished; the last step makes the service ready.

        Records the seconds since construction at which the step finished.
        The phase returns to `starting` between steps.

        Args:
            step: The step that finished; it must be the one in progress.
            note: Optional detail recorded for the step.

        Raises:
            ScmleError: If `step` is not the phase currently in progress.
        """
        with self.lock:
            if self.phase is not step:
                raise errors.ScmleError(f"{step.value} is not in progress")
            self.done[step.value] = round(time.monotonic() - self.started, 4)
            if note:
                self.notes[step.value] = note
            self.phase = (
                Phase.READY if len(self.done) == len(STEPS) else Phase.STARTING
            )

    def fail(self, reason: str) -> None:
        """Records why the service will not become ready.

        Sets the phase to `failed` from any phase, including `ready`; later
        `begin` calls are refused.

        Args:
            reason: The failure description reported by `snapshot`.
        """
        with self.lock:
            self.phase = Phase.FAILED
            self.reason = reason

    @property
    def ready(self) -> bool:
        """Whether verify, load and warm-up all finished."""
        with self.lock:
            return self.phase is Phase.READY

    def snapshot(self) -> dict[str, Any]:
        """Reports the state for `/readyz`.

        Returns:
            A mapping with `ready`, the current `phase` name, the seconds
            since start at which each step `completed`, per-step `notes`,
            and the failure `reason` (None unless failed).
        """
        with self.lock:
            return {
                "ready": self.phase is Phase.READY,
                "phase": self.phase.value,
                "completed": dict(self.done),
                "notes": dict(self.notes),
                "reason": self.reason,
            }


class AdmissionControl:
    """Bounds concurrent requests and the time one request may take."""

    def __init__(self, max_concurrent: int, timeout_seconds: float) -> None:
        """Creates the concurrency slots and a worker pool of that size.

        Args:
            max_concurrent: Requests allowed in flight; more get a 429.
            timeout_seconds: Time a scoring call may take before the caller
                gets a 504. The computation itself cannot be cancelled, so a
                timed-out call keeps its worker until it returns.

        Raises:
            ConfigError: If a limit is not positive.
        """
        if max_concurrent < 1 or timeout_seconds <= 0:
            raise errors.ConfigError("limits must be positive")
        self.max_concurrent = max_concurrent
        self.timeout_seconds = timeout_seconds
        self.slots = threading.BoundedSemaphore(max_concurrent)
        self.pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent, thread_name_prefix="score"
        )

    @contextlib.contextmanager
    def admit(self) -> Iterator[None]:
        """Takes a slot without waiting, or raises `OverloadedError`.

        Yields:
            None, while the slot is held; it is released on exit.

        Raises:
            OverloadedError: If all slots are in use.
        """
        if not self.slots.acquire(blocking=False):
            raise OverloadedError(
                f"more than {self.max_concurrent} requests in flight"
            )
        try:
            yield
        finally:
            self.slots.release()

    def run(self, work: Callable[[], Any]) -> Any:
        """Runs `work` and waits at most the timeout for its result.

        An exception raised by `work` propagates to the caller, including a
        `TimeoutError` of its own, which is not a `DeadlineError`.

        Args:
            work: Zero-argument callable executed on a worker thread.

        Returns:
            Whatever `work` returns.

        Raises:
            DeadlineError: If `work` does not finish within the timeout. The
                call keeps running on its worker.
        """
        future = self.pool.submit(work)
        try:
            return future.result(timeout=self.timeout_seconds)
        except concurrent.futures.TimeoutError as error:
            if future.done():
                return future.result()
            raise DeadlineError(
                f"request exceeded {self.timeout_seconds} seconds"
            ) from error

    def close(self) -> None:
        """Releases the workers without waiting for stuck calls.

        Calls that have not started are cancelled.
        """
        self.pool.shutdown(wait=False, cancel_futures=True)


def is_valid_request_id(request_id: str) -> bool:
    """Checks whether a caller request id is safe to echo in a header.

    Args:
        request_id: The id supplied by the caller.

    Returns:
        True when it matches `REQUEST_ID_PATTERN` in full: 1 to 128
        characters, starting with a letter or digit, then letters, digits and
        `.`, `_`, `:` or `-`.
    """
    return REQUEST_ID_PATTERN.fullmatch(request_id) is not None


def new_request_id() -> str:
    """Returns an opaque server-side request id with no request data.

    Returns:
        A new random id of the form `srv-<32 hex digits>`.
    """
    return f"srv-{uuid.uuid4().hex}"
