"""Ray execution backend (`distributed` extra)."""

from collections.abc import Callable
from collections.abc import Sequence
import logging
from typing import Any

from scmle.contracts import errors
from scmle.core import scheduler

DEFAULT_RETRIES = 1


class RayBackend(scheduler.Executor):
    """Runs items as Ray tasks (`distributed` extra), local or cluster.

    A task whose worker process or node disappears is retried once; if it
    still fails, `map` hands the item to `on_failure` so the run records a
    failed trial and continues instead of ending.

    Attributes:
        retries: How many times Ray reruns a task whose worker was lost.
    """

    name = "ray"

    def __init__(
        self,
        workers: int,
        address: str | None = None,
        retries: int = DEFAULT_RETRIES,
    ) -> None:
        """Starts a local Ray runtime, or joins the cluster at `address`.

        Args:
            workers: CPUs to claim when starting a local runtime.
            address: Cluster address (for example `auto` or `host:6379`).
            retries: Reruns of a task after its worker is lost.

        Raises:
            ConfigError: If Ray is not installed.
        """
        try:
            import ray  # noqa: PLC0415 -- optional dependency
        except ImportError as error:
            raise errors.ConfigError(
                "RayBackend needs the 'distributed' extra (ray)"
            ) from error
        self.ray = ray
        self.workers = max(workers, 1)
        self.retries = retries
        if not ray.is_initialized():
            settings: dict[str, Any] = {
                "ignore_reinit_error": True,
                "include_dashboard": False,
                "logging_level": logging.ERROR,
            }
            if address:
                settings["address"] = address
            else:
                settings["num_cpus"] = self.workers
            ray.init(**settings)

    def map(
        self,
        function: Callable[[Any], Any],
        items: Sequence[Any],
        on_failure: Callable[[Any, Exception], Any] | None = None,
    ) -> list[Any]:
        """Submits one Ray task per item and gathers results in order.

        Args:
            function: Picklable function of one argument.
            items: Independent work items.
            on_failure: Called with the item and the Ray error when its task
                still fails after the retries; its return value replaces the
                item's result. If None, the Ray error is re-raised.

        Returns:
            Results in the order of `items`; empty if `items` is empty.
        """
        if not items:
            return []
        remote = self.ray.remote(max_retries=self.retries)(function)
        references = [remote.remote(item) for item in items]
        outputs = []
        for item, reference in zip(items, references, strict=True):
            try:
                outputs.append(self.ray.get(reference))
            except self.ray.exceptions.RayError as error:
                if on_failure is None:
                    raise
                outputs.append(on_failure(item, error))
        return outputs
