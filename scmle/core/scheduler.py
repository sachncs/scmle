"""Execution backends: where independent trials run.

The control plane only ever calls `Executor.map`, so moving from a
single process to a pool (or a Ray cluster) never touches orchestration.
"""

import abc
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any
from typing import ClassVar

import joblib

from scmle.contracts import errors


class Executor(abc.ABC):
    """Runs a function over independent work items.

    Attributes:
        name: Backend name.
        workers: Number of items that can run concurrently.
    """

    name: ClassVar[str]
    workers: int

    @abc.abstractmethod
    def map(
        self,
        function: Callable[[Any], Any],
        items: Sequence[Any],
        on_failure: Callable[[Any, Exception], Any] | None = None,
    ) -> list[Any]:
        """Applies `function` to every item.

        Args:
            function: Picklable function of one argument.
            items: Independent work items.
            on_failure: Turns an item whose worker was lost into a result,
                so one lost worker never ends the run (backends that can
                lose a worker use it; the others ignore it).

        Returns:
            Results in the order of `items`.
        """


class Serial(Executor):
    """Runs everything in the calling process, one item at a time."""

    name = "sequential"

    def __init__(self) -> None:
        """Initializes a single-worker backend."""
        self.workers = 1

    def map(
        self,
        function: Callable[[Any], Any],
        items: Sequence[Any],
        on_failure: Callable[[Any, Exception], Any] | None = None,
    ) -> list[Any]:
        """Applies `function` in order, in this process.

        Args:
            function: Called once per item.
            items: The inputs, in order.
            on_failure: Not used by this backend; an exception raised by
                `function` propagates to the caller.

        Returns:
            The results of `function`, in the order of `items`.
        """
        return [function(item) for item in items]


class Pool(Executor):
    """Runs items in a pool of worker processes (loky)."""

    name = "process"

    def __init__(self, workers: int) -> None:
        """Initializes the pool size.

        Args:
            workers: Number of worker processes (>= 2).

        Raises:
            ConfigError: If fewer than two workers are requested.
        """
        if workers < 2:
            raise errors.ConfigError("Pool needs at least 2 workers")
        self.workers = workers

    def map(
        self,
        function: Callable[[Any], Any],
        items: Sequence[Any],
        on_failure: Callable[[Any, Exception], Any] | None = None,
    ) -> list[Any]:
        """Applies `function` across worker processes, preserving order.

        Zero or one items run in this process without a pool; otherwise the
        pool uses at most the configured number of workers, and no more
        workers than items.

        Args:
            function: Called once per item; must be picklable.
            items: The inputs.
            on_failure: Not used by this backend; an exception raised by
                `function` propagates to the caller.

        Returns:
            The results of `function`, in the order of `items`.
        """
        if len(items) <= 1:
            return [function(item) for item in items]
        parallel = joblib.Parallel(
            n_jobs=min(self.workers, len(items)), backend="loky"
        )
        return list(parallel(joblib.delayed(function)(item) for item in items))


def create_backend(
    workers: int, executor: str = "process", address: str | None = None
) -> Executor:
    """Chooses the backend for a worker count and executor.

    Args:
        workers: Requested concurrency.
        executor: `process` (loky pool) or `ray`.
        address: With `ray`, the cluster to join (local runtime if None).

    Returns:
        Sequential for one worker with the process executor, otherwise a
        process pool or a Ray backend.

    Raises:
        ConfigError: For an unknown executor.
    """
    if executor == "ray":
        from scmle.integrations import ray_backend

        return ray_backend.RayBackend(workers, address)
    if executor != "process":
        raise errors.ConfigError(f"unknown executor '{executor}'")
    return Serial() if workers <= 1 else Pool(workers)
