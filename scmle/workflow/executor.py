"""The local executor: the same definition, run with durable state.

Layout under ``workflows/<name>/runs/<run id>/`` in an `ObjectStore`:
``run.json`` (state, trigger, requester, cost), ``steps/<step>.json`` (state,
attempts, outputs, idempotency key, cost), ``approvals/<step>.json`` and
``lock`` (a lease). Everything is written atomically, so a crashed executor
leaves a consistent record and a new one continues from it.

* **Durable run id.** The run id is derived from the workflow name and the
  caller's `trigger_key` (a schedule slot, a request id), so invoking the
  same trigger twice is the same run.
* **Idempotency keys.** A step's key hashes the run id, the step, its
  parameters and the digests of its upstream outputs. A step that already
  succeeded with the same key is not run again; the key is also handed to the
  handler so that external effects can be deduplicated.
* **Partial failure.** A step that fails (after its retries) stops its
  dependents, which are recorded as `skipped`; independent steps still finish.
  `resume` runs only what has not succeeded.
* **Concurrency.** `max_concurrent_runs` is enforced with run leases;
  `max_parallel` and per-group limits bound the steps in flight.
* **Timeouts** are cooperative here: the handler calls `context.check()`;
  a hosting engine enforces its own.
* **Cost.** Handlers report cost with `context.spend`; a step over its budget
  fails, and a run over its budget stops the remaining steps
  (`budget_stopped`).
* **Approval.** An approval step waits (`waiting_approval`) until a signed
  decision from a listed approver, who must not be the requester, is recorded;
  a rejection or expiry ends the run without the approved actions.
* **Leases.** A run holds a lease it renews as it works. A second executor
  gets `RunInFlightError` while it is valid and may take over after it
  expires (the first one is presumed dead). Take-over is a check, delete and
  conditional create; two executors racing for an expired lease cannot both
  win, but a first executor that is merely slow can still write after losing
  its lease, so handlers must be idempotent.
"""

from collections.abc import Callable
from collections.abc import Mapping
import concurrent.futures
import dataclasses
import enum
import hashlib
import hmac
import json
import threading
import time
from typing import Any

from scmle.contracts import errors
from scmle.contracts import signing
from scmle.core import retry
from scmle.data.store import objectstore
from scmle.workflow import definition

LEASE_SECONDS = 300.0


class RunState(enum.StrEnum):
    """Where a run is.

    Attributes:
        RUNNING: Steps are being executed.
        WAITING_APPROVAL: Stopped at an approval step with no decision yet.
        SUCCEEDED: Every step succeeded.
        FAILED: A step failed.
        BUDGET_STOPPED: The run's cost budget was passed.
        REJECTED: An approver rejected the run.
        EXPIRED: An approval was not decided in time.
    """

    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BUDGET_STOPPED = "budget_stopped"
    REJECTED = "rejected"
    EXPIRED = "expired"


FINISHED = frozenset(
    {
        RunState.SUCCEEDED,
        RunState.FAILED,
        RunState.BUDGET_STOPPED,
        RunState.REJECTED,
        RunState.EXPIRED,
    }
)


class StepState(enum.StrEnum):
    """Where a step is.

    Attributes:
        PENDING: Not started.
        RUNNING: Executing.
        SUCCEEDED: Ended successfully.
        FAILED: Ended with an error after its retries.
        SKIPPED: Not run because an upstream step did not succeed or the run's
            budget was passed.
        WAITING: An approval step waiting for a decision.
        REJECTED: An approval step that was rejected.
        EXPIRED: An approval step that was not decided in time.
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING = "waiting"
    REJECTED = "rejected"
    EXPIRED = "expired"


class RunInFlightError(errors.ScmleError):
    """Another executor holds a valid lease on the run."""

    category = "run_in_flight"


class ConcurrencyLimitError(errors.ScmleError):
    """The workflow already has its maximum number of runs in flight."""

    category = "concurrency_limit"


class StepTimeoutError(errors.ScmleError):
    """A step passed its timeout."""

    category = "timeout"


class StepBudgetError(errors.ScmleError):
    """A step or the run passed its cost budget."""

    category = "budget"


class StepInterruptedError(retry.TransientError):
    """The step's compute was taken away; the step is retried."""

    category = "interrupted"


class ApprovalError(errors.ScmleError):
    """An approval was refused (unknown approver, bad signature, requester)."""

    category = "approval"


@dataclasses.dataclass(slots=True)
class StepContext:
    """What a handler sees.

    Attributes:
        run_id: The durable run id.
        step: The step definition.
        attempt: Attempt number, from 1.
        idempotency_key: Stable across retries and re-invocations.
        upstream: Outputs of the steps it depends on, by step name.
        store: The executor's durable store.
        started: Clock reading when the attempt started.
        clock: Returns the current time in epoch seconds.
        spent: Cost recorded by this attempt so far.
        run_spent: Cost of the run before this step.
        run_budget: Ceiling for the run's cost, or None.
    """

    run_id: str
    step: definition.Step
    attempt: int
    idempotency_key: str
    upstream: Mapping[str, Mapping[str, Any]]
    store: objectstore.ObjectStore
    started: float
    clock: Callable[[], float]
    spent: float = 0.0
    run_spent: float = 0.0
    run_budget: float | None = None

    def check(self) -> None:
        """Raises if the step is past its timeout.

        Raises:
            StepTimeoutError: If more than the step's timeout has passed since
                the attempt started.
        """
        if self.clock() - self.started > self.step.timeout_seconds:
            raise StepTimeoutError(
                f"{self.step.name} passed {self.step.timeout_seconds}s"
            )

    def spend(self, amount: float) -> None:
        """Records cost; raises if a budget is passed.

        Args:
            amount: Cost to add.

        Raises:
            StepBudgetError: If the step's or the run's budget is exceeded.
        """
        self.spent += amount
        limit = self.step.cost_budget
        if limit is not None and self.spent > limit:
            raise StepBudgetError(f"{self.step.name} cost passed {limit}")
        if (
            self.run_budget is not None
            and self.run_spent + self.spent > self.run_budget
        ):
            raise StepBudgetError("the run's cost budget was passed")


Handler = Callable[[StepContext], Mapping[str, Any] | None]
StepResult = tuple[str, Mapping[str, Any], float]


def digest_of(value: Any) -> str:
    """Digests a value as canonical JSON.

    Args:
        value: A JSON-serializable value; other types are converted with `str`.

    Returns:
        The hex SHA-256 of the key-sorted JSON.
    """
    text = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def run_id_for(workflow: definition.Workflow, trigger_key: str) -> str:
    """Derives the durable run id of a trigger.

    Args:
        workflow: The workflow.
        trigger_key: Identifies the trigger (a schedule slot, a request).

    Returns:
        The workflow name and a 16-character digest of its name, version and the
        trigger key, truncated to 64 characters.
    """
    short = digest_of([workflow.name, workflow.version, trigger_key])[:16]
    return f"{workflow.name}-{short}"[:64]


class LocalExecutor:
    """Runs a `Workflow` against handlers, keeping durable state.

    Attributes:
        workflow: The definition.
        store: Durable state.
        handlers: Step name to handler.
        approver_keys: Approver identity to the key their decisions are signed
            with.
        clock: Returns epoch seconds.
        sleep: Waits between retries.
        lease_seconds: Validity of a run lease.
        owner: Name written into the lease.
        guard: Serializes writes of step records.
    """

    def __init__(
        self,
        workflow: definition.Workflow,
        store: objectstore.ObjectStore,
        handlers: Mapping[str, Handler],
        *,
        approver_keys: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        lease_seconds: float = LEASE_SECONDS,
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
            clock: Callable returning the current time in epoch seconds.
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
        self.sleep = sleep
        self.lease_seconds = lease_seconds
        self.owner = owner
        self.guard = threading.Lock()

    # ---- keys and records -------------------------------------------------

    def base(self, run_id: str) -> str:
        """Builds the key prefix of a run.

        Args:
            run_id: The run.

        Returns:
            The prefix under which the run's records are stored.
        """
        return f"workflows/{self.workflow.name}/runs/{run_id}/"

    def read(self, key: str) -> dict[str, Any] | None:
        """Reads a stored JSON record.

        Args:
            key: Object key.

        Returns:
            The record, or None if there is no such object.
        """
        try:
            data, ignored = self.store.get(key)
        except objectstore.NotFoundError:
            return None
        record: dict[str, Any] = json.loads(data)
        return record

    def write(self, key: str, record: Mapping[str, Any]) -> None:
        """Stores a JSON record atomically.

        Args:
            key: Object key.
            record: The record to store.
        """
        body = json.dumps(record, indent=2, sort_keys=True).encode()
        self.store.put(key, body)

    def run_record(self, run_id: str) -> dict[str, Any] | None:
        """Reads the stored run record.

        Args:
            run_id: The run.

        Returns:
            The run record, or None if the run is unknown.
        """
        return self.read(self.base(run_id) + "run.json")

    def step_record(self, run_id: str, name: str) -> dict[str, Any] | None:
        """Reads the stored record of one step.

        Args:
            run_id: The run.
            name: The step name.

        Returns:
            The step record, or None if the step has not been recorded.
        """
        return self.read(f"{self.base(run_id)}steps/{name}.json")

    # ---- leases and concurrency ------------------------------------------

    def take_lease(self, run_id: str) -> None:
        """Acquires the run's lease, or raises `RunInFlightError`.

        An expired lease is taken over; the take-over is a delete followed by a
        conditional create.

        Args:
            run_id: The run.

        Raises:
            RunInFlightError: If another executor holds a valid lease, or wins a
                race for an expired one.
        """
        key = self.base(run_id) + "lock"
        body = json.dumps(
            {"owner": self.owner, "expires": self.clock() + self.lease_seconds}
        ).encode()
        try:
            self.store.put(key, body, if_absent=True)
            return
        except objectstore.AlreadyExistsError:
            pass
        held = self.read(key) or {"expires": 0.0}
        if held["expires"] > self.clock():
            raise RunInFlightError(f"run {run_id} is held by {held['owner']}")
        self.store.delete(key)
        try:
            self.store.put(key, body, if_absent=True)
        except objectstore.AlreadyExistsError as error:
            raise RunInFlightError(f"run {run_id} was just taken") from error

    def renew_lease(self, run_id: str) -> None:
        """Extends the lease.

        Args:
            run_id: The run.
        """
        self.write(
            self.base(run_id) + "lock",
            {
                "owner": self.owner,
                "expires": self.clock() + self.lease_seconds,
            },
        )

    def release_lease(self, run_id: str) -> None:
        """Gives the lease up.

        Args:
            run_id: The run.
        """
        self.store.delete(self.base(run_id) + "lock")

    def active_runs(self) -> list[str]:
        """Lists the runs of this workflow whose lease is valid.

        Returns:
            The run ids.
        """
        prefix = f"workflows/{self.workflow.name}/runs/"
        active = []
        for meta in self.store.list_objects(prefix):
            if not meta.key.endswith("/lock"):
                continue
            held = self.read(meta.key)
            if held and held["expires"] > self.clock():
                active.append(meta.key.split("/")[3])
        return active

    # ---- starting and resuming -------------------------------------------

    def start(
        self, trigger_key: str, requested_by: str = "scheduler"
    ) -> dict[str, Any]:
        """Starts a run for a trigger, or continues the one it already is.

        Invoking the same trigger again returns the finished record without
        running anything, or continues an unfinished run after its lease
        expired. Driving a run can raise `ConcurrencyLimitError` (the workflow
        has its maximum of runs in flight), `RunInFlightError` (another
        executor holds the run's lease) and ConfigError (the run was started
        with a different workflow definition).

        Args:
            trigger_key: Identifies the trigger (a schedule slot, a request).
            requested_by: Who asked; they cannot approve their own run.

        Returns:
            The run record when this call stops (finished or waiting).
        """
        run_id = run_id_for(self.workflow, trigger_key)
        existing = self.run_record(run_id)
        if existing is not None and existing["state"] in {
            s.value for s in FINISHED
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
        settled = {s.value for s in FINISHED} - {RunState.FAILED.value}
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
            lease raises `RunInFlightError` (from `take_lease`).

        Raises:
            ConcurrencyLimitError: If the workflow already has its maximum
                number of runs in flight.
            ConfigError: If the run was started with a different workflow
                definition.
        """
        others = [r for r in self.active_runs() if r != run_id]
        if len(others) >= self.workflow.max_concurrent_runs:
            raise ConcurrencyLimitError(
                f"{len(others)} runs of {self.workflow.name} are in flight"
            )
        self.take_lease(run_id)
        try:
            if len(self.active_runs()) > self.workflow.max_concurrent_runs:
                raise ConcurrencyLimitError(
                    f"too many runs of {self.workflow.name} in flight"
                )
            record = self.run_record(run_id) or {
                "run_id": run_id,
                "workflow": self.workflow.name,
                "workflow_version": self.workflow.version,
                "workflow_digest": self.workflow.digest(),
                "trigger_key": trigger_key,
                "requested_by": requested_by,
                "state": RunState.RUNNING.value,
                "cost": 0.0,
                "started_at": self.clock(),
            }
            if record["workflow_digest"] != self.workflow.digest():
                raise errors.ConfigError(
                    "the run was started with a different workflow definition"
                )
            record["state"] = RunState.RUNNING.value
            self.write(self.base(run_id) + "run.json", record)
            record = self.execute(run_id, record)
            self.write(self.base(run_id) + "run.json", record)
            return record
        finally:
            self.release_lease(run_id)

    # ---- the run loop ----------------------------------------------------

    def execute(self, run_id: str, record: dict[str, Any]) -> dict[str, Any]:
        """Runs ready steps until nothing more can run.

        Args:
            run_id: The run.
            record: The run record, updated in place.

        Returns:
            The run record with its final state, step states, cost and end time.
        """
        states: dict[str, str] = {}
        outputs: dict[str, Mapping[str, Any]] = {}
        for step in self.workflow.steps:
            saved = self.step_record(run_id, step.name)
            if saved is not None and saved["state"] == StepState.SUCCEEDED:
                states[step.name] = StepState.SUCCEEDED.value
                outputs[step.name] = saved["outputs"]
            else:
                states[step.name] = StepState.PENDING.value
        record["cost"] = self.recorded_cost(run_id)
        budget_hit = False
        group_use: dict[str, int] = {}
        in_flight: dict[
            concurrent.futures.Future[StepResult], definition.Step
        ] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.workflow.max_parallel
        ) as pool:
            while True:
                changed = False
                self.renew_lease(run_id)
                budget = self.workflow.cost_budget
                if budget is not None and record["cost"] > budget:
                    budget_hit = True
                for name in self.workflow.order():
                    step = self.workflow.step(name)
                    if states[name] != StepState.PENDING.value or budget_hit:
                        continue
                    parents = [states[d] for d in step.depends_on]
                    if any(
                        p
                        in (
                            StepState.FAILED,
                            StepState.SKIPPED,
                            StepState.REJECTED,
                            StepState.EXPIRED,
                        )
                        for p in parents
                    ):
                        states[name] = StepState.SKIPPED.value
                        self.save_step(
                            run_id, step, StepState.SKIPPED, reason="upstream"
                        )
                        changed = True
                        continue
                    if any(p != StepState.SUCCEEDED for p in parents):
                        continue
                    group = step.concurrency_group
                    if group is not None and group_use.get(
                        group, 0
                    ) >= self.workflow.group_limit(group):
                        continue
                    if step.kind is definition.StepKind.APPROVAL:
                        settled = self.settle_approval(run_id, step, record)
                        states[name] = settled.value
                        if settled is StepState.SUCCEEDED:
                            saved = self.step_record(run_id, name) or {}
                            outputs[name] = saved.get("outputs", {})
                        changed = changed or settled is not StepState.WAITING
                        continue
                    if len(in_flight) >= self.workflow.max_parallel:
                        continue
                    states[name] = StepState.RUNNING.value
                    if group is not None:
                        group_use[group] = group_use.get(group, 0) + 1
                    future = pool.submit(
                        self.run_step,
                        run_id,
                        step,
                        {d: outputs[d] for d in step.depends_on},
                        record["cost"],
                    )
                    in_flight[future] = step
                if not in_flight and not changed:
                    break
                if not in_flight:
                    continue
                done, ignored = concurrent.futures.wait(
                    in_flight, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    step = in_flight.pop(future)
                    state, result, cost = future.result()
                    states[step.name] = state
                    record["cost"] = round(record["cost"] + cost, 6)
                    if state == StepState.SUCCEEDED.value:
                        outputs[step.name] = result
                    if step.concurrency_group is not None:
                        group_use[step.concurrency_group] -= 1
        if budget_hit or (
            self.workflow.cost_budget is not None
            and record["cost"] > self.workflow.cost_budget
        ):
            for name, state in states.items():
                if state == StepState.PENDING.value:
                    self.save_step(
                        run_id,
                        self.workflow.step(name),
                        StepState.SKIPPED,
                        reason="run budget",
                    )
            record["state"] = RunState.BUDGET_STOPPED.value
        else:
            record["state"] = self.final_state(states).value
        record["steps"] = states
        record["ended_at"] = self.clock()
        return record

    def final_state(self, states: Mapping[str, str]) -> RunState:
        """Derives the run's state from its steps' states.

        Args:
            states: Step state value by step name.

        Returns:
            `WAITING_APPROVAL`, `REJECTED` or `EXPIRED` if any step is in the
            corresponding state (in that order of precedence), `SUCCEEDED` if
            all succeeded, otherwise `FAILED`.
        """
        values = set(states.values())
        if StepState.WAITING.value in values:
            return RunState.WAITING_APPROVAL
        if StepState.REJECTED.value in values:
            return RunState.REJECTED
        if StepState.EXPIRED.value in values:
            return RunState.EXPIRED
        if values == {StepState.SUCCEEDED.value}:
            return RunState.SUCCEEDED
        return RunState.FAILED

    def recorded_cost(self, run_id: str) -> float:
        """Sums the cost of steps already recorded as succeeded for this run.

        Args:
            run_id: The run.

        Returns:
            The total cost, rounded to six decimals.
        """
        total = 0.0
        for step in self.workflow.steps:
            saved = self.step_record(run_id, step.name)
            if saved is not None and saved["state"] == StepState.SUCCEEDED:
                total += saved["cost"]
        return round(total, 6)

    # ---- one step --------------------------------------------------------

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
        return digest_of(
            [
                run_id,
                step.name,
                dict(step.parameters),
                {k: digest_of(v) for k, v in sorted(upstream.items())},
            ]
        )

    def save_step(
        self,
        run_id: str,
        step: definition.Step,
        state: StepState,
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

    def run_step(
        self,
        run_id: str,
        step: definition.Step,
        upstream: Mapping[str, Mapping[str, Any]],
        run_spent: float,
    ) -> StepResult:
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
            and saved["state"] == StepState.SUCCEEDED
            and saved["idempotency_key"] == key
        ):
            return StepState.SUCCEEDED.value, saved["outputs"], 0.0
        handler = self.handlers[step.name]
        attempts: list[dict[str, Any]] = []
        total = 0.0
        last_error = ""
        for attempt in range(1, step.retries.max_attempts + 1):
            context = StepContext(
                run_id,
                step,
                attempt,
                key,
                upstream,
                self.store,
                self.clock(),
                self.clock,
                run_spent=run_spent + total,
                run_budget=self.workflow.cost_budget,
            )
            self.save_step(
                run_id,
                step,
                StepState.RUNNING,
                idempotency_key=key,
                attempts=attempts,
            )
            try:
                result = dict(handler(context) or {})
                context.check()
                total += context.spent
                attempts.append({"attempt": attempt, "state": "succeeded"})
                self.save_step(
                    run_id,
                    step,
                    StepState.SUCCEEDED,
                    idempotency_key=key,
                    attempts=attempts,
                    outputs=result,
                    cost=round(total, 6),
                )
                return StepState.SUCCEEDED.value, result, total
            except retry.TransientError as error:
                total += context.spent
                last_error = f"{type(error).__name__}: {error}"
                attempts.append(
                    {
                        "attempt": attempt,
                        "state": "retryable",
                        "error": last_error,
                    }
                )
                if attempt < step.retries.max_attempts:
                    self.sleep(step.retries.delay_after(attempt))
            except errors.ScmleError as error:
                total += context.spent
                last_error = f"{type(error).__name__}: {error}"
                attempts.append(
                    {"attempt": attempt, "state": "failed", "error": last_error}
                )
                break
            except Exception as error:
                total += context.spent
                last_error = f"{type(error).__name__}: {error}"[:300]
                attempts.append(
                    {"attempt": attempt, "state": "failed", "error": last_error}
                )
                break
        self.save_step(
            run_id,
            step,
            StepState.FAILED,
            idempotency_key=key,
            attempts=attempts,
            error=last_error,
            cost=round(total, 6),
        )
        return StepState.FAILED.value, {}, total

    # ---- approval --------------------------------------------------------

    def approval_payload(
        self, run_id: str, step: str, approver: str, decision: str, at: float
    ) -> dict[str, Any]:
        """Builds the content an approver signs.

        Args:
            run_id: The run.
            step: The approval step name.
            approver: The approver's identity.
            decision: `approve` or `reject`.
            at: Decision time in epoch seconds.

        Returns:
            A mapping that also names the workflow digest.
        """
        return {
            "run_id": run_id,
            "workflow_digest": self.workflow.digest(),
            "step": step,
            "approver": approver,
            "decision": decision,
            "at": at,
        }

    def sign_approval(
        self,
        run_id: str,
        step: str,
        approver: str,
        decision: str,
        key: str,
        at: float | None = None,
    ) -> dict[str, Any]:
        """Builds a signed decision (what an approver's tool does).

        Args:
            run_id: The run.
            step: The approval step name.
            approver: The approver's identity.
            decision: `approve` or `reject`.
            key: The approver's key.
            at: Decision time in epoch seconds; the current time if None.

        Returns:
            The decision payload (run, workflow digest, step, approver,
            decision and time) with an added `signature` entry.
        """
        moment = self.clock() if at is None else at
        payload = self.approval_payload(
            run_id, step, approver, decision, moment
        )
        return {**payload, "signature": signing.sign_payload(payload, key)}

    def submit_approval(self, decision: Mapping[str, Any]) -> None:
        """Records a signed decision after checking it.

        Args:
            decision: A decision produced by `sign_approval`. A decision that
                names a step the workflow does not have raises `ConfigError`
                (from `Workflow.step`).

        Raises:
            ApprovalError: If the approver is unknown or not listed for the
                step, is the person who requested the run, the signature does
                not verify, the decision is not approve or reject, a decision is
                already recorded, or the run does not exist.
        """
        run_id, name = str(decision["run_id"]), str(decision["step"])
        record = self.run_record(run_id)
        if record is None:
            raise ApprovalError("no such run")
        step = self.workflow.step(name)
        approver = str(decision["approver"])
        if (
            step.kind is not definition.StepKind.APPROVAL
            or approver not in step.approvers
        ):
            raise ApprovalError("approver is not listed for this step")
        if approver == record["requested_by"]:
            raise ApprovalError("the requester cannot approve their own run")
        key = self.approver_keys.get(approver)
        payload = self.approval_payload(
            run_id,
            name,
            approver,
            str(decision["decision"]),
            float(decision["at"]),
        )
        if key is None or not signing_matches(payload, key, decision):
            raise ApprovalError("the approval signature does not verify")
        if decision["decision"] not in ("approve", "reject"):
            raise ApprovalError("decision must be approve or reject")
        if self.read(f"{self.base(run_id)}approvals/{name}.json"):
            raise ApprovalError("a decision is already recorded")
        self.write(f"{self.base(run_id)}approvals/{name}.json", dict(decision))

    def settle_approval(
        self, run_id: str, step: definition.Step, record: Mapping[str, Any]
    ) -> StepState:
        """Determines the approval step's state now.

        A recorded decision settles the step to succeeded (approve) or rejected;
        otherwise the step waits, and expires once it has waited longer than its
        expiry.

        Args:
            run_id: The run.
            step: The approval step.
            record: The run record. Unused.

        Returns:
            The step's new state.
        """
        decision = self.read(f"{self.base(run_id)}approvals/{step.name}.json")
        if decision is not None:
            if decision["decision"] == "approve":
                self.save_step(
                    run_id,
                    step,
                    StepState.SUCCEEDED,
                    outputs={"approver": decision["approver"]},
                    cost=0.0,
                    idempotency_key="approval",
                )
                return StepState.SUCCEEDED
            self.save_step(
                run_id, step, StepState.REJECTED, approver=decision["approver"]
            )
            return StepState.REJECTED
        waiting = self.step_record(run_id, step.name)
        since = (
            waiting["waiting_since"]
            if waiting and waiting["state"] == StepState.WAITING
            else self.clock()
        )
        if self.clock() - since > step.expires_seconds:
            self.save_step(run_id, step, StepState.EXPIRED, waiting_since=since)
            return StepState.EXPIRED
        self.save_step(run_id, step, StepState.WAITING, waiting_since=since)
        return StepState.WAITING


def signing_matches(
    payload: Mapping[str, Any], key: str, decision: Mapping[str, Any]
) -> bool:
    """Checks that the decision's signature is the payload's signature.

    Args:
        payload: The content that was signed.
        key: The approver's key.
        decision: The submitted decision, with its `signature`.

    Returns:
        True if the signature equals the HMAC of the payload under the key.
    """
    expected = signing.sign_payload(payload, key)
    return hmac.compare_digest(expected, str(decision.get("signature", "")))
