"""Execution backends: where independent trials run.

The control plane only ever calls `ExecutionBackend.map`, so moving from a
single process to a pool (or, later, a cluster) never touches orchestration.
"""

import abc
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any
from typing import ClassVar

import joblib

from mml.contracts import errors


class ExecutionBackend(abc.ABC):
    """Runs a function over independent work items.

    Attributes:
        name: Backend name.
        workers: Number of items that can run concurrently.
    """

    name: ClassVar[str]
    workers: int

    @abc.abstractmethod
    def map(
        self, function: Callable[[Any], Any], items: Sequence[Any]
    ) -> list[Any]:
        """Applies `function` to every item.

        Args:
            function: Picklable function of one argument.
            items: Independent work items.

        Returns:
            Results in the order of `items`.
        """


class SequentialBackend(ExecutionBackend):
    """Runs everything in the calling process, one item at a time."""

    name = "sequential"

    def __init__(self) -> None:
        """Initializes a single-worker backend."""
        self.workers = 1

    def map(
        self, function: Callable[[Any], Any], items: Sequence[Any]
    ) -> list[Any]:
        """Applies `function` in order, in this process."""
        return [function(item) for item in items]


class ProcessBackend(ExecutionBackend):
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
            raise errors.ConfigError("ProcessBackend needs at least 2 workers")
        self.workers = workers

    def map(
        self, function: Callable[[Any], Any], items: Sequence[Any]
    ) -> list[Any]:
        """Applies `function` across worker processes, preserving order."""
        if len(items) <= 1:
            return [function(item) for item in items]
        parallel = joblib.Parallel(
            n_jobs=min(self.workers, len(items)), backend="loky"
        )
        return list(parallel(joblib.delayed(function)(item) for item in items))


def create_backend(workers: int) -> ExecutionBackend:
    """Chooses the backend for a worker count.

    Args:
        workers: Requested concurrency.

    Returns:
        Sequential for one worker, otherwise a process pool.
    """
    return SequentialBackend() if workers <= 1 else ProcessBackend(workers)
