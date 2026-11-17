"""The executor: starts and resumes runs and drives them under the lease."""

from collections.abc import Callable
from collections.abc import Mapping
import threading
import time
from typing import Any

from scmle.contracts import errors
from scmle.storage import objectstore
from scmle.workflow import definition
from scmle.workflow.executor import context
from scmle.workflow.executor import failures
from scmle.workflow.executor import leases
from scmle.workflow.executor import loop
from scmle.workflow.executor import records
from scmle.workflow.executor import status


class Executor(loop.Loop):
    """Runs a `Workflow` against handlers, keeping durable state.

    Attributes:
        workflow: The definition.
        store: Durable state.
        handlers: Step name to handler.
        approver_keys: Approver identity to the key their decisions are signed
            with.
        clock: Returns epoch seconds.
        timer: Returns monotonic seconds, for how long a step has run, so a
            step of the wall clock neither fails nor extends a step (CON-20);
            `time.monotonic` unless replaced, as tests do.
        sleep: Waits between retries.
        lease_seconds: Validity of a run lease.
        owner: Name written into the lease.
        guard: Serializes writes of step records.
    """

    def __init__(
        self,
        workflow: definition.Workflow,
        store: objectstore.Depot,
        handlers: Mapping[str, context.Handler],
        *,
        approver_keys: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        lease_seconds: float = leases.LEASE_SECONDS,
        owner: str = "executor",
    ) -> None:
        """Initializes the executor.

        Args:
            workflow: The definition.
            store: Durable state.
            handlers: Step name to handler. Every non-approval step needs one.
            approver_keys: Approver identity to the HMAC key their decisions
                are signed with (held by the approver, not by the system
                under approval).
            clock: Callable returning the current time in epoch seconds,
                for records and lease expiry, which other processes read.
            sleep: Callable taking a delay in seconds, called between retries.
            lease_seconds: Validity of a run lease.
            owner: Name written into the lease.

        Raises:
            ConfigError: If a step has no handler.
        """
        missing = [
            s.name
            for s in workflow.steps
            if s.kind is not definition.StepKind.APPROVAL
            and s.name not in handlers
        ]
        if missing:
            raise errors.ConfigError(f"no handler for steps {missing}")
        self.workflow = workflow
        self.store = store
        self.handlers = dict(handlers)
        self.approver_keys = dict(approver_keys or {})
        self.clock = clock
        self.timer = time.monotonic
        self.sleep = sleep
        self.lease_seconds = lease_seconds
        self.owner = owner
        self.guard = threading.Lock()
        self.fences = {}

    def start(
        self, trigger_key: str, requested_by: str = "scheduler"
    ) -> dict[str, Any]:
        """Starts a run for a trigger, or continues the one it already is.

        Invoking the same trigger again returns the finished record without
        running anything, or continues an unfinished run after its lease
        expired. Driving a run can raise `ConcurrencyLimitError` (the workflow
        has its maximum of runs in flight), `BusyError` (another
        executor holds the run's lease) and ConfigError (the run was started
        with a different workflow definition).

        Args:
            trigger_key: Identifies the trigger (a schedule slot, a request).
            requested_by: Who asked; they cannot approve their own run.

        Returns:
            The run record when this call stops (finished or waiting).
        """
        run_id = records.run_id_for(self.workflow, trigger_key)
        existing = self.run_record(run_id)
        if existing is not None and existing["state"] in {
            s.value for s in status.FINISHED
        }:
            return existing
        return self.drive(run_id, trigger_key, requested_by)

    def resume(self, run_id: str) -> dict[str, Any]:
        """Continues a run (after an approval, a crash or a failure).

        A failed run is retried from its failed steps; succeeded steps are kept.
        Runs that were rejected, expired, budget-stopped or succeeded are final
        and returned as they are. Calling `start` again with the same trigger
        never retries a failed run: that is an explicit choice made here.

        Args:
            run_id: The run.

        Returns:
            The run record when this call stops.

        Raises:
            ConfigError: If the run does not exist.
        """
        record = self.run_record(run_id)
        if record is None:
            raise errors.ConfigError(f"no run {run_id}")
        settled = {s.value for s in status.FINISHED} - {
            status.Progress.FAILED.value
        }
        if record["state"] in settled:
            return record
        return self.drive(run_id, record["trigger_key"], record["requested_by"])

    def drive(
        self, run_id: str, trigger_key: str, requested_by: str
    ) -> dict[str, Any]:
        """Runs the steps under the lease.

        Args:
            run_id: The run.
            trigger_key: Identifies the trigger; stored on a new run.
            requested_by: Who asked; stored on a new run.

        Returns:
            The run record when the run stops. Another executor holding the
            lease raises `BusyError` (from `acquire`).

        Raises:
            ConcurrencyLimitError: If the workflow already has its maximum
                number of runs in flight.
            ConfigError: If the run was started with a different workflow
                definition.
        """
        others = [r for r in self.scan() if r != run_id]
        if len(others) >= self.workflow.max_concurrent_runs:
            raise failures.ConcurrencyLimitError(
                f"{len(others)} runs of {self.workflow.name} are in flight"
            )
        self.acquire(run_id)
        try:
            if len(self.scan()) > self.workflow.max_concurrent_runs:
                raise failures.ConcurrencyLimitError(
                    f"too many runs of {self.workflow.name} in flight"
                )
            record = self.run_record(run_id) or {
                "run_id": run_id,
                "workflow": self.workflow.name,
                "workflow_version": self.workflow.version,
                "workflow_digest": self.workflow.digest(),
                "trigger_key": trigger_key,
                "requested_by": requested_by,
                "state": status.Progress.RUNNING.value,
                "cost": 0.0,
                "started_at": self.clock(),
            }
            if record["workflow_digest"] != self.workflow.digest():
                raise errors.ConfigError(
                    "the run was started with a different workflow definition"
                )
            record["state"] = status.Progress.RUNNING.value
            self.write(self.base(run_id) + "run.json", record)
            record = self.execute(run_id, record)
            self.write(self.base(run_id) + "run.json", record)
            return record
        finally:
            self.relinquish(run_id)
