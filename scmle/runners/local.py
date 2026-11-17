"""The local runner: the same job, in this process.

The workload is a callable that receives a `Context`. It saves progress
with `context.step(n, state)`, and on a restart asks `context.resume()` for
the last good checkpoint. `step` is also where this runner checks the cost
budget and the timeout, and where an injected interruption (a stand-in for a
preemption) strikes, right after the checkpoint is saved.
"""

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
import json
import logging
from typing import Any

from scmle.contracts import errors
from scmle.core import retry
from scmle.runners import base
from scmle.runners import checkpoint
from scmle.runners import roles

LOGGER = logging.getLogger(__name__)


class PreemptedError(errors.ScmleError):
    """The platform took the job away (preemption); retry from a checkpoint."""


class StoppedError(errors.ScmleError):
    """The supervisor stopped the job (budget or timeout)."""


@dataclasses.dataclass(slots=True)
class Context:
    """What a workload sees.

    Attributes:
        store: Durable storage, limited to the training role.
        run_id: The run.
        attempt: Attempt number, from 1.
        spec: The job spec.
        checkpoints: This run's checkpoints.
        hook: Called by `step` after saving (interruption, budget).
    """

    store: roles.RoleStore
    run_id: str
    attempt: int
    spec: base.Job
    checkpoints: checkpoint.CheckpointStore
    hook: Callable[[int], None] = lambda step: None  # noqa: ARG005 -- hook signature

    def resume(self) -> checkpoint.Checkpoint | None:
        """Finds where to resume.

        Corrupt checkpoints passed over on the way are recorded in durable
        storage (`CheckpointStore.record`), so a restart from an older step
        or from zero is on file (ERR-10).

        Returns:
            The newest checkpoint that verifies, or None at a fresh start.
        """
        found = self.checkpoints.latest()
        self.checkpoints.record(self.attempt, found)
        return found

    def step(self, number: int, state: Any) -> None:
        """Saves a checkpoint, then lets the supervisor act.

        The hook may then raise `PreemptedError` (the platform interrupts
        after this step) or `StoppedError` (the budget or timeout stops the
        job).

        Args:
            number: Step number.
            state: JSON-serializable state to save.
        """
        self.checkpoints.save(number, state)
        self.hook(number)

    def finish(self, result: Mapping[str, Any]) -> None:
        """Records the job's result (metrics, trial results, evidence).

        Args:
            result: JSON-serializable mapping, stored as the run's result.
        """
        body = json.dumps(dict(result), indent=2, sort_keys=True).encode()
        self.store.put(f"runs/{self.run_id}/result.json", body)


Workload = Callable[[Context], None]
Interruption = Callable[[int, int], bool]


@dataclasses.dataclass(slots=True)
class Outcome:
    """A finished local attempt.

    Attributes:
        status: How the attempt ended.
    """

    status: base.Status


class Runner(base.Runner):
    """Runs the workload inline, as the job.

    Attributes:
        workload: The training function.
        interruptions: Decides which steps are interrupted, or None.
        jobs: Outcome of each submitted attempt, by job id.
    """

    kind = "local"

    def __init__(
        self,
        store: Any,
        workload: Workload,
        *,
        serving_identity: str,
        interruptions: Interruption | None = None,
        **options: Any,
    ) -> None:
        """Initializes the runner.

        Args:
            store: Durable storage (any `Depot`).
            workload: The training function.
            serving_identity: The serving identity (see `Runner`).
            interruptions: `(attempt, step) -> bool`; true makes the job
                interrupted right after that step's checkpoint.
            **options: `clock` and `poll_seconds` (see `Runner`).
        """
        super().__init__(store, serving_identity=serving_identity, **options)
        self.workload = workload
        self.interruptions = interruptions
        self.jobs: dict[str, Outcome] = {}

    def submit(self, spec: base.Job, attempt: int) -> base.Ticket:
        """Runs the workload to completion and remembers how it ended.

        An interruption becomes `INTERRUPTED`, a supervisor stop `CANCELLED`, a
        transient error a retryable `FAILED`, and any other exception a
        non-retryable `FAILED` carrying its type and message (cut to 300
        characters).

        Args:
            spec: The job.
            attempt: Attempt number, from 1.

        Returns:
            The handle of the finished attempt.
        """
        handle = base.Ticket(f"local-{spec.run_id}-{attempt}", attempt)

        def hook(step: int) -> None:
            """Lets the supervisor act after a checkpoint.

            Args:
                step: The step just saved.

            Raises:
                StoppedError: If the supervisor wants the job stopped.
                PreemptedError: If the injected interruption strikes at
                    this attempt and step.
            """
            if self.watcher() is not None:
                raise StoppedError
            if self.interruptions and self.interruptions(attempt, step):
                raise PreemptedError

        context = Context(
            roles.RoleStore(self.store, roles.TRAINING_ROLE),
            spec.run_id,
            attempt,
            spec,
            checkpoint.CheckpointStore(self.store, spec.run_id),
            hook,
        )
        try:
            self.workload(context)
            status = base.Status(base.State.SUCCEEDED)
        except PreemptedError:
            # handled: interruption recorded as INTERRUPTED job status
            status = base.Status(base.State.INTERRUPTED, "interrupted")
        except StoppedError:
            # handled: supervisor stop recorded as CANCELLED job status
            status = base.Status(base.State.CANCELLED, "stopped")
        except retry.TransientError as error:
            # handled: transient error recorded as retryable FAILED status
            status = base.Status(
                base.State.FAILED, f"transient: {error}", retryable=True
            )
        except Exception as error:
            LOGGER.exception("job %s failed", handle.job_id)
            reason = errors.Failure.of(error).text[:300]
            status = base.Status(base.State.FAILED, reason)
        self.jobs[handle.job_id] = Outcome(status)
        return handle

    def poll(self, handle: base.Ticket) -> base.Status:
        """Reports the remembered outcome.

        Args:
            handle: The attempt.

        Returns:
            The status the attempt ended with.
        """
        return self.jobs[handle.job_id].status

    def cancel(self, handle: base.Ticket, reason: str) -> None:
        """Does nothing: the attempt already ended.

        Args:
            handle: The attempt. Unused.
            reason: Why it is stopped. Unused.
        """
