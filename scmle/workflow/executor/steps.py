"""One step: its key, its record, its retries and what earlier runs saved."""

from collections.abc import Mapping
import functools
from typing import Any

from scmle.contracts import errors
from scmle.core import retry
from scmle.workflow import definition
from scmle.workflow.executor import context
from scmle.workflow.executor import leases
from scmle.workflow.executor import records
from scmle.workflow.executor import status

# Retries for writing a step's success record after the handler succeeded:
# store writes only, never the handler (ERR-7). Policy: three attempts over
# about three seconds ride out a throttled or briefly unreachable store.
WRITES = retry.Retries(3, 1.0, 2.0, 4.0)


class Steps(leases.Leases):
    """Runs one step with its retries and records what it did."""

    def idempotency_key(
        self,
        run_id: str,
        step: definition.Step,
        upstream: Mapping[str, Mapping[str, Any]],
    ) -> str:
        """Computes the step's key from its run, parameters and upstream.

        Args:
            run_id: The run.
            step: The step.
            upstream: Outputs of the steps it depends on, by step name.

        Returns:
            The hex digest of the run id, step name, parameters and the digests
            of the upstream outputs.
        """
        return records.distill(
            [
                run_id,
                step.name,
                dict(step.parameters),
                {k: records.distill(v) for k, v in sorted(upstream.items())},
            ]
        )

    def stash(
        self,
        run_id: str,
        step: definition.Step,
        state: status.Phase,
        **fields: Any,
    ) -> None:
        """Writes a step record.

        Args:
            run_id: The run.
            step: The step.
            state: The step's new state.
            **fields: Further fields stored in the record.
        """
        with self.guard:
            self.write(
                f"{self.base(run_id)}steps/{step.name}.json",
                {"step": step.name, "state": state.value, **fields},
            )

    def note(
        self,
        attempts: list[Mapping[str, Any]],
        attempt: int,
        error: Exception,
        state: str,
    ) -> str:
        """Records a failed attempt.

        A framework error keeps its whole text in a failed attempt; any other
        is cut to 300 characters. ERR-20: an InputError was a bare ValueError
        (or TypeError, RuntimeError) and is still cut like one. A retryable
        attempt keeps its whole text.

        Args:
            attempts: The step's attempts so far, appended to in place.
            attempt: The attempt number.
            error: What the handler raised.
            state: `retryable` or `failed`.

        Returns:
            The error text recorded.
        """
        failure = errors.Failure.of(error)
        text = failure.text
        if state == "failed" and (
            not isinstance(error, errors.ScmleError)
            or isinstance(error, errors.InputError)
        ):
            text = text[:300]
        attempts.append(
            {
                "attempt": attempt,
                "state": state,
                "error": text,
                "failure": dict(failure.dump()),
            }
        )
        return text

    def begin(
        self,
        run_id: str,
        step: definition.Step,
        attempt: int,
        key: str,
        upstream: Mapping[str, Mapping[str, Any]],
        run_spent: float,
    ) -> context.StepContext:
        """Builds what the handler sees for one attempt, timed from now.

        Args:
            run_id: The run.
            step: The step.
            attempt: The attempt number, from 1.
            key: The step's idempotency key.
            upstream: Outputs of the steps it depends on, by step name.
            run_spent: Cost of the run before this attempt.

        Returns:
            The context of the attempt.
        """
        return context.StepContext(
            run_id,
            step,
            attempt,
            key,
            upstream,
            self.store,
            self.timer(),
            self.timer,
            run_spent=run_spent,
            run_budget=self.workflow.cost_budget,
        )

    def keep(
        self,
        run_id: str,
        step: definition.Step,
        key: str,
        attempts: list[Mapping[str, Any]],
        result: Mapping[str, Any],
        total: float,
    ) -> None:
        """Writes the record of a step that succeeded.

        ERR-7: the success record is written outside the handler's retry,
        with its own retries for the store; a write that still fails
        propagates, so a handler that succeeded is neither run again nor
        recorded as failed.

        Args:
            run_id: The run.
            step: The step.
            key: The step's idempotency key.
            attempts: The step's attempts, the last one succeeded.
            result: The handler's outputs.
            total: The cost of every attempt.
        """
        save = functools.partial(
            self.stash,
            run_id,
            step,
            status.Phase.SUCCEEDED,
            idempotency_key=key,
            attempts=attempts,
            outputs=result,
            cost=round(total, 6),
        )
        retry.call(save, WRITES, sleep=self.sleep)

    def run_step(
        self,
        run_id: str,
        step: definition.Step,
        upstream: Mapping[str, Mapping[str, Any]],
        run_spent: float,
    ) -> context.Outcome:
        """Runs one step with its retries.

        A step that already succeeded with the same idempotency key is not run
        again. A transient error is retried after the policy's delay; any other
        error fails the step.

        Args:
            run_id: The run.
            step: The step.
            upstream: Outputs of the steps it depends on, by step name.
            run_spent: Cost of the run before this step.

        Returns:
            The final step state value, the handler's outputs (empty on failure)
            and the cost incurred.
        """
        key = self.idempotency_key(run_id, step, upstream)
        saved = self.step_record(run_id, step.name)
        if (
            saved is not None
            and saved["state"] == status.Phase.SUCCEEDED
            and saved["idempotency_key"] == key
        ):
            return status.Phase.SUCCEEDED.value, saved["outputs"], 0.0
        handler = self.handlers[step.name]
        attempts: list[Mapping[str, Any]] = []
        total = 0.0
        last_error = ""
        for attempt in range(1, step.retries.max_attempts + 1):
            current = self.begin(
                run_id, step, attempt, key, upstream, run_spent + total
            )
            self.stash(
                run_id,
                step,
                status.Phase.RUNNING,
                idempotency_key=key,
                attempts=attempts,
            )
            try:
                result = dict(handler(current) or {})
                current.check()
            except retry.TransientError as error:
                # handled: transient error saved in attempts and retried; FAILED
                # step record
                total += current.spent
                last_error = self.note(attempts, attempt, error, "retryable")
                if attempt < step.retries.max_attempts:
                    self.sleep(step.retries.delay(attempt))
            except Exception as error:  # noqa: BLE001 -- handler boundary
                # handled: handler exception saved in attempts; step recorded
                # FAILED with error, its text cut by `note`.
                total += current.spent
                last_error = self.note(attempts, attempt, error, "failed")
                break
            else:
                total += current.spent
                attempts.append({"attempt": attempt, "state": "succeeded"})
                self.keep(run_id, step, key, attempts, result, total)
                return status.Phase.SUCCEEDED.value, result, total
        self.stash(
            run_id,
            step,
            status.Phase.FAILED,
            idempotency_key=key,
            attempts=attempts,
            error=last_error,
            cost=round(total, 6),
        )
        return status.Phase.FAILED.value, {}, total
