"""Handlers that connect workflow steps to the rest of the framework."""

from collections.abc import Callable
from collections.abc import Mapping
from typing import Any

from scmle.contracts import errors
from scmle.runners import base
from scmle.workflow import executor


def runner_handler(
    runner: base.Runner,
    make_spec: Callable[[executor.StepContext], base.JobSpec],
) -> executor.Handler:
    """A handler that runs a training job through a `Runner`.

    The job's run id should come from the step's idempotency key, so a retry
    or a repeated invocation of the step is the same job and resumes from its
    checkpoints instead of starting over. The job's estimated cost is charged
    to the step.

    Args:
        runner: Where the job runs.
        make_spec: Builds the `JobSpec` for a step execution.

    Returns:
        A handler. A job that succeeds returns its run id, manifest digest
        fields and cost; one that ended `retries_exhausted` raises a
        retryable `StepInterruptedError`; any other final state raises
        `ScmleError`.
    """

    def handler(context: executor.StepContext) -> Mapping[str, Any]:
        manifest = runner.run(make_spec(context))
        context.spend(manifest.total_cost)
        if manifest.state == base.RunState.SUCCEEDED:
            return {
                "run_id": manifest.run_id,
                "result_sha256": manifest.result_sha256,
                "attempts": len(manifest.attempts),
                "cost": manifest.total_cost,
            }
        if manifest.state == base.RunState.RETRIES_EXHAUSTED:
            raise executor.StepInterruptedError(
                f"job {manifest.run_id} was interrupted on every attempt"
            )
        raise errors.ScmleError(f"job {manifest.run_id} ended {manifest.state}")

    return handler
