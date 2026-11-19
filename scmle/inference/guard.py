"""Readiness and admission control shared by the HTTP serving adapters.

`Readiness` is the difference between "the process is up" (liveness) and "the
process may take traffic" (readiness): it becomes true only after the bundle
was verified, loaded and warmed up, in that order. `Admission` bounds
concurrent requests and the time one request may take, so an overloaded or
stuck model answers quickly instead of queueing without limit; a call that
timed out makes the replica not ready until it returns, and `STRIKES` of them
in a row make it not alive, so the platform restarts it.
"""

from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Mapping
import concurrent.futures
import contextlib
import contextvars
import enum
import functools
import re
import threading
import time
from typing import Any
import uuid

from scmle.contracts import errors

REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
# Consecutive timed-out calls after which the replica reports itself not
# alive, so the platform restarts it. Policy: one or two slow requests are
# noise; three in a row with no success between them means the model or a
# worker is wedged, and a computation cannot be cancelled in-process.
STRIKES = 3
CURRENT = threading.local()


def abandoned() -> bool:
    """Whether the caller of the work running on this thread gave up.

    Work run by `Admission.run` checks this before it mutates state,
    so a request answered with a 504 changes nothing (SRV-2).

    Returns:
        True once the call has timed out; False outside such work.
    """
    flag = getattr(CURRENT, "abandoned", None)
    return bool(flag is not None and flag.is_set())


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


class UnreadyError(errors.ScmleError):
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


class Admission:
    """Bounds concurrent requests and the time one request may take."""

    def __init__(
        self,
        max_concurrent: int,
        timeout_seconds: float,
        strikes: int = STRIKES,
    ) -> None:
        """Creates the concurrency slots and a worker pool of that size.

        Args:
            max_concurrent: Requests allowed in flight; more get a 429.
            timeout_seconds: Time a scoring call may take before the caller
                gets a 504. The computation itself cannot be cancelled, so a
                timed-out call keeps its worker until it returns; while it
                does the replica is not ready (SRV-1).
            strikes: Consecutive timeouts after which `alive` is false.

        Raises:
            ConfigError: If a limit is not positive.
        """
        if max_concurrent < 1 or timeout_seconds <= 0 or strikes < 1:
            raise errors.ConfigError("limits must be positive")
        self.max_concurrent = max_concurrent
        self.timeout_seconds = timeout_seconds
        self.strikes = strikes
        self.slots = threading.BoundedSemaphore(max_concurrent)
        self.pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent, thread_name_prefix="score"
        )
        self.lock = threading.Lock()
        self.busy = 0
        self.stuck = 0
        self.misses = 0

    @property
    def clear(self) -> bool:
        """Whether no timed-out call is still holding a worker."""
        with self.lock:
            return self.stuck == 0

    @property
    def alive(self) -> bool:
        """Whether fewer than `strikes` calls in a row have timed out."""
        with self.lock:
            return self.misses < self.strikes

    def state(self) -> dict[str, int]:
        """Counts for the probes: busy and stuck workers, timeouts in a row."""
        with self.lock:
            return {
                "busy": self.busy,
                "stuck": self.stuck,
                "misses": self.misses,
            }

    @contextlib.contextmanager
    def admit(self) -> Generator[None, None, None]:
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
            OverloadedError: If every worker is busy, for example with calls
                that timed out and are still running: the request is refused
                at once instead of queueing behind them (SRV-1).
            DeadlineError: If `work` does not finish within the timeout. The
                call keeps running on its worker, is marked abandoned (see
                `abandoned`), and counts as stuck until it returns.
        """
        flag = threading.Event()

        def guarded() -> Any:
            CURRENT.abandoned = flag
            try:
                return work()
            finally:
                CURRENT.abandoned = None

        with self.lock:
            if self.busy >= self.max_concurrent:
                raise OverloadedError(
                    f"all {self.max_concurrent} workers are busy"
                )
            self.busy += 1
        # The worker runs in the caller's context, so the request id bound
        # for logging and the current trace span follow the call.
        future = self.pool.submit(contextvars.copy_context().run, guarded)
        future.add_done_callback(functools.partial(self.release, flag))
        try:
            result = future.result(timeout=self.timeout_seconds)
        except concurrent.futures.TimeoutError as error:
            with self.lock:
                if not future.done():
                    flag.set()
                    self.stuck += 1
                    self.misses += 1
            if not flag.is_set():
                return future.result()
            raise DeadlineError(
                f"request exceeded {self.timeout_seconds} seconds"
            ) from error
        with self.lock:
            self.misses = 0
        return result

    def release(
        self, flag: threading.Event, done: concurrent.futures.Future[Any]
    ) -> None:
        """Frees the worker of a finished call, stuck or not.

        Args:
            flag: The call's abandoned flag.
            done: The finished call, as the executor passes it; whether it
                succeeded does not matter here.
        """
        del done
        with self.lock:
            self.busy -= 1
            if flag.is_set():
                self.stuck -= 1

    def close(self) -> None:
        """Releases the workers without waiting for stuck calls.

        Calls that have not started are cancelled.
        """
        self.pool.shutdown(wait=False, cancel_futures=True)


def assess(readiness: Readiness, admission: Admission) -> Mapping[str, Any]:
    """Readiness, which a timed-out call still running takes away.

    Args:
        readiness: The verify, load and warm-up tracker.
        admission: The admission control whose workers are counted.

    Returns:
        The readiness snapshot, with `ready` false and the reason set while
        any timed-out call holds a worker (SRV-1), and the admission counts
        under `workers`.
    """
    snapshot = readiness.snapshot()
    workers = admission.state()
    if snapshot["ready"] and workers["stuck"]:
        snapshot["ready"] = False
        snapshot["reason"] = (
            f"{workers['stuck']} timed-out call(s) still running"
        )
    return {**snapshot, "workers": workers}


def wellformed(request_id: str) -> bool:
    """Checks whether a caller request id is safe to echo in a header.

    Args:
        request_id: The id supplied by the caller.

    Returns:
        True when it matches `REQUEST_ID_PATTERN` in full: 1 to 128
        characters, starting with a letter or digit, then letters, digits and
        `.`, `_`, `:` or `-`.
    """
    return REQUEST_ID_PATTERN.fullmatch(request_id) is not None


def mint() -> str:
    """Returns an opaque server-side request id with no request data.

    Returns:
        A new random id of the form `srv-<32 hex digits>`.
    """
    return f"srv-{uuid.uuid4().hex}"
