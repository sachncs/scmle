"""The scoring workers of `Service`, and what hung calls leave of them.

A scoring call cannot be cancelled in-process: one that hangs keeps its
worker until it returns, which may be never. Two things follow, and this
module holds both:

* `Crew` runs the calls on daemon threads. The standard thread pool joins
  its workers when the interpreter exits, so a single call that never
  returned kept a drained server process alive after SIGTERM.
* `Capacity` decides readiness and liveness from the workers that are
  still free. One hung call takes one worker; the replica keeps answering
  on the others and stays in rotation. Only when no call can run (every
  worker holds a timed-out call, or the model serializes scoring and a
  timed-out call holds its lock) is it not ready, and once that lasts
  `patience` seconds it is not alive either, so the platform restarts it.
"""

from collections.abc import Callable
from collections.abc import Mapping
import concurrent.futures
import queue
import threading
import time
from typing import Any

from scmle.contracts import errors
from scmle.inference import guard

# One queued call: the future it settles and the work that settles it.
Job = tuple[concurrent.futures.Future[Any], Callable[[], Any]]


class Crew(concurrent.futures.ThreadPoolExecutor):
    """An executor whose workers are daemon threads, started on demand.

    Only `submit` and `shutdown` are used; both are its own, so none of the
    standard pool's workers is ever started or joined at exit.

    Attributes:
        size: Most threads it starts.
        threads: The threads started so far.
        jobs: Calls waiting for a thread; None tells a thread to end.
        closed: Whether `shutdown` was called.
    """

    def __init__(self, size: int, name: str = "score") -> None:
        """Creates the crew with no thread started.

        Args:
            size: Most threads it starts; at least one.
            name: Prefix of the thread names.

        Raises:
            ConfigError: If `size` is below one.
        """
        if size < 1:
            raise errors.ConfigError("a crew needs at least one thread")
        super().__init__(max_workers=size, thread_name_prefix=name)
        self.size = size
        self.name = name
        self.threads: list[threading.Thread] = []
        self.jobs: queue.SimpleQueue[Job | None] = queue.SimpleQueue()
        self.closed = False
        self.guard = threading.Lock()
        self.idle = 0

    def submit(
        self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> concurrent.futures.Future[Any]:
        """Queues one call and starts a thread if none is idle.

        Args:
            fn: The callable.
            *args: Its positional arguments.
            **kwargs: Its keyword arguments.

        Returns:
            The future of the call.

        Raises:
            OrderError: If the crew was shut down.
        """
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        with self.guard:
            if self.closed:
                raise errors.OrderError("the crew is shut down")
            self.jobs.put((future, lambda: fn(*args, **kwargs)))
            if self.idle == 0 and len(self.threads) < self.size:
                thread = threading.Thread(
                    target=self.work,
                    name=f"{self.name}-{len(self.threads)}",
                    daemon=True,
                )
                self.threads.append(thread)
                thread.start()
            else:
                self.idle -= min(self.idle, 1)
        return future

    def work(self) -> None:
        """Runs queued calls until told to end."""
        while True:
            job = self.jobs.get()
            if job is None:
                return
            future, call = job
            if future.set_running_or_notify_cancel():
                try:
                    result = call()
                except BaseException as error:  # noqa: BLE001 -- the future carries it
                    future.set_exception(error)
                else:
                    future.set_result(result)
            with self.guard:
                self.idle += 1

    def shutdown(
        self, wait: bool = True, *, cancel_futures: bool = False
    ) -> None:
        """Refuses new calls and ends every thread once it is free.

        Args:
            wait: Whether to wait for the threads; a hung call makes this
                wait as long as it hangs, so the server never waits.
            cancel_futures: Whether to cancel the calls not yet started.
        """
        with self.guard:
            self.closed = True
            waiting: list[Job] = []
            while cancel_futures:
                try:
                    job = self.jobs.get_nowait()
                except queue.Empty:
                    break
                if job is not None:
                    waiting.append(job)
            for job in waiting:
                job[0].cancel()
            for ignored in self.threads:
                self.jobs.put(None)
        if wait:
            for thread in self.threads:
                thread.join()


class Capacity:
    """Readiness and liveness from the workers hung calls leave free.

    Attributes:
        admission: The admission control whose workers are counted.
        serial: Tells whether the served model serializes scoring.
        patience: Seconds no call can run before the replica is not alive.
        since: Monotonic time from which no call could run, or None.
    """

    def __init__(
        self,
        admission: guard.Admission,
        serial: Callable[[], bool],
        patience: float | None = None,
    ) -> None:
        """Binds the judge to the admission control.

        Args:
            admission: The admission control whose workers are counted.
            serial: Returns whether the served model serializes scoring (a
                model with series features does).
            patience: Seconds no call can run before `alive` is false. By
                default `guard.STRIKES` times the scoring timeout: the time
                the strikes that fail liveness on their own would take,
                which a replica that cannot start a call never accumulates.

        Raises:
            ConfigError: If `patience` is not positive.
        """
        chosen = (
            guard.STRIKES * admission.timeout if patience is None else patience
        )
        if chosen <= 0:
            raise errors.ConfigError("the patience must be positive")
        self.admission = admission
        self.serial = serial
        self.patience = chosen
        self.since: float | None = None
        self.lock = threading.Lock()

    def blocked(self) -> str | None:
        """Why no call can run now, and since when that has been so.

        Returns:
            The reason, or None while a call can still run.
        """
        stuck = self.admission.state()["stuck"]
        reason = None
        if stuck >= self.admission.capacity:
            reason = f"{stuck} timed-out call(s) hold every worker"
        elif stuck and self.serial():
            reason = (
                f"{stuck} timed-out call(s) hold the lock that serializes "
                "scoring"
            )
        with self.lock:
            if reason is None:
                self.since = None
            elif self.since is None:
                self.since = time.monotonic()
        return reason

    def assess(self, readiness: guard.Readiness) -> Mapping[str, Any]:
        """Readiness, which a replica that cannot start a call loses.

        Args:
            readiness: The verify, load and warm-up tracker.

        Returns:
            The readiness snapshot, with `ready` false and the reason set
            while no call can run, and the admission counts under
            `workers`.
        """
        snapshot = readiness.snapshot()
        reason = self.blocked()
        if snapshot["ready"] and reason is not None:
            snapshot["ready"] = False
            snapshot["reason"] = reason
        return {**snapshot, "workers": self.admission.state()}

    def alive(self) -> bool:
        """Whether the replica can still serve without a restart.

        Returns:
            False after `guard.STRIKES` timeouts in a row, or once no call
            has been able to run for `patience` seconds; True otherwise.
        """
        if self.blocked() is not None:
            with self.lock:
                since = self.since
            if since is not None and time.monotonic() - since >= self.patience:
                return False
        return self.admission.alive


def enlist(
    workers: int, timeout: float, serial: Callable[[], bool]
) -> Capacity:
    """Admission control on a `Crew`, with the `Capacity` that judges it.

    Daemon workers: a call that never returns cannot hold the process at
    exit, and the replica keeps serving on the other workers.

    Args:
        workers: Requests allowed in flight, and threads that score them.
        timeout: Seconds one scoring call may take.
        serial: Tells whether the served model serializes scoring.

    Returns:
        The capacity; its `admission` is the server's admission control.
    """
    admission = guard.Admission(workers, timeout)
    admission.pool = Crew(workers)
    return Capacity(admission, serial)
