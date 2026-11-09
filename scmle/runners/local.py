"""The local runner: the same job, in this process.

The workload is a callable that receives a `JobContext`. It saves progress
with `context.step(n, state)`, and on a restart asks `context.resume()` for
the last good checkpoint. `step` is also where this runner checks the cost
budget and the timeout, and where an injected interruption (a stand-in for a
preemption) strikes, right after the checkpoint is saved.
"""

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
import json
from typing import Any

from scmle.contracts import errors
from scmle.core import retry
from scmle.runners import base
from scmle.runners import checkpoint
from scmle.runners import roles


class JobInterruptedError(errors.ScmleError):
    """The platform took the job away (preemption); retry from a checkpoint."""


class JobStoppedError(errors.ScmleError):
    """The supervisor stopped the job (budget or timeout)."""


@dataclasses.dataclass(slots=True)
class JobContext:
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

        Returns:
            The newest checkpoint that verifies, or None at a fresh start.
        """
        return self.checkpoints.latest()

    def step(self, number: int, state: Any) -> None:
        """Saves a checkpoint, then lets the supervisor act.

        The hook may then raise `JobInterruptedError` (the platform interrupts
        after this step) or `JobStoppedError` (the budget or timeout stops the
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


Workload = Callable[[JobContext], None]
Interruption = Callable[[int, int], bool]


@dataclasses.dataclass(slots=True)
class Outcome:
    """A finished local attempt.

    Attributes:
        status: How the attempt ended.
    """

    status: base.JobStatus


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
            store: Durable storage (any `ObjectStore`).
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

    def submit(self, spec: base.Job, attempt: int) -> base.JobHandle:
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
        handle = base.JobHandle(f"local-{spec.run_id}-{attempt}", attempt)

        def hook(step: int) -> None:
            """Lets the supervisor act after a checkpoint.

            Args:
                step: The step just saved.

            Raises:
                JobStoppedError: If the supervisor wants the job stopped.
                JobInterruptedError: If the injected interruption strikes at
                    this attempt and step.
            """
            if self.watcher() is not None:
                raise JobStoppedError
            if self.interruptions and self.interruptions(attempt, step):
                raise JobInterruptedError

        context = JobContext(
            roles.RoleStore(self.store, roles.TRAINING_ROLE),
            spec.run_id,
            attempt,
            spec,
            checkpoint.CheckpointStore(self.store, spec.run_id),
            hook,
        )
        try:
            self.workload(context)
            status = base.JobStatus(base.JobState.SUCCEEDED)
        except JobInterruptedError:
            status = base.JobStatus(base.JobState.INTERRUPTED, "interrupted")
        except JobStoppedError:
            status = base.JobStatus(base.JobState.CANCELLED, "stopped")
        except retry.TransientError as error:
            status = base.JobStatus(
                base.JobState.FAILED, f"transient: {error}", retryable=True
            )
        except Exception as error:  # noqa: BLE001 -- job boundary
            reason = f"{type(error).__name__}: {error}"[:300]
            status = base.JobStatus(base.JobState.FAILED, reason)
        self.jobs[handle.job_id] = Outcome(status)
        return handle

    def poll(self, handle: base.JobHandle) -> base.JobStatus:
        """Reports the remembered outcome.

        Args:
            handle: The attempt.

        Returns:
            The status the attempt ended with.
        """
        return self.jobs[handle.job_id].status

    def cancel(self, handle: base.JobHandle, reason: str) -> None:
        """Does nothing: the attempt already ended.

        Args:
            handle: The attempt. Unused.
            reason: Why it is stopped. Unused.
        """
