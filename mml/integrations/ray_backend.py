"""Ray execution backend (`distributed` extra)."""

from collections.abc import Callable
from collections.abc import Sequence
import logging
from typing import Any

from mml.contracts import errors
from mml.core import scheduler


class RayBackend(scheduler.ExecutionBackend):
    """Runs items as Ray tasks (`distributed` extra), local or cluster."""

    name = "ray"

    def __init__(self, workers: int) -> None:
        """Starts (or joins) a Ray runtime.

        Args:
            workers: CPUs to claim when starting a local runtime.

        Raises:
            ConfigError: If Ray is not installed.
        """
        try:
            import ray
        except ImportError as error:
            raise errors.ConfigError(
                "RayBackend needs the 'distributed' extra (ray)"
            ) from error
        self.ray = ray
        self.workers = max(workers, 1)
        if not ray.is_initialized():
            ray.init(
                num_cpus=self.workers,
                ignore_reinit_error=True,
                include_dashboard=False,
                logging_level=logging.ERROR,
            )

    def map(
        self, function: Callable[[Any], Any], items: Sequence[Any]
    ) -> list[Any]:
        """Submits one Ray task per item and gathers results in order."""
        if not items:
            return []
        remote = self.ray.remote(function)
        return list(self.ray.get([remote.remote(item) for item in items]))
