"""The local executor: the same definition, run with durable state.

Layout under ``workflows/<name>/runs/<run id>/`` in a `Depot`:
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
* **Leases.** A run holds a lease with an owner and a fencing token, renewed
  by a heartbeat while steps run. A second executor gets `BusyError`
  while it is valid and may take over after it expires (the first one is
  presumed dead) with one compare-and-swap that raises the token, so two
  executors racing for it cannot both win. Renewal and release check owner
  and token, and every run record is written only after that check
  (`EvictionError` otherwise), so a slow first executor stops instead of
  overwriting the new holder's state (CON-1). The check and the write are
  two store calls, so handlers must still be idempotent.
"""

from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Mapping
import concurrent.futures
import contextlib
import dataclasses
import enum
import functools
import hashlib
import hmac
import json
import logging
import threading
import time
from typing import Any

from scmle.contracts import errors
from scmle.contracts import signing
from scmle.core import retry
from scmle.data.store import objectstore
from scmle.workflow import definition

LEASE_SECONDS = 300.0
# Tries at taking a lease that changes hands under us (deleted between the
# failed create and the read); more means someone else keeps winning.
LEASE_ATTEMPTS = 3
LOGGER = logging.getLogger(__name__)
# A conditional write on a lease that changed or vanished under us.
CHANGED = (objectstore.ConflictError, objectstore.AbsentError)
# Retries for writing a step's success record after the handler succeeded:
# store writes only, never the handler (ERR-7). Policy: three attempts over
# about three seconds ride out a throttled or briefly unreachable store.
WRITES = retry.Retries(3, 1.0, 2.0, 4.0)


def parse(key: str, data: bytes) -> dict[str, Any]:
    """Decodes a stored run, step, approval or lease record.

    Args:
        key: The object key, named in the error.
        data: The stored bytes.

    Returns:
        The record.

    Raises:
        ArtifactError: If the bytes are not a JSON object; a raw
            `JSONDecodeError` used to escape with no key (ERR-17).
    """
    try:
        record = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise errors.ArtifactError(
            f"workflow record {key} is corrupt: {error}", {"key": key}
        ) from error
    if not isinstance(record, dict):
        raise errors.ArtifactError(
            f"workflow record {key} is not a JSON object", {"key": key}
        )
    return record


class Progress(enum.StrEnum):
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
        Progress.SUCCEEDED,
        Progress.FAILED,
        Progress.BUDGET_STOPPED,
        Progress.REJECTED,
        Progress.EXPIRED,
    }
)


class Phase(enum.StrEnum):
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


class BusyError(errors.ScmleError):
    """Another executor holds a valid lease on the run."""

    category = "run_in_flight"


class EvictionError(BusyError):
    """This executor no longer holds the run's lease and must stop writing."""

    category = "lease_lost"


class ConcurrencyLimitError(errors.ScmleError):
    """The workflow already has its maximum number of runs in flight."""

    category = "concurrency_limit"


class DeadlineError(errors.ScmleError):
    """A step passed its timeout."""

    category = "timeout"


class StepBudgetError(errors.ScmleError):
    """A step or the run passed its cost budget."""

    category = "budget"


class PreemptedError(retry.TransientError):
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
    store: objectstore.Depot
    started: float
    clock: Callable[[], float]
    spent: float = 0.0
    run_spent: float = 0.0
    run_budget: float | None = None

    def check(self) -> None:
        """Raises if the step is past its timeout.

        Raises:
            DeadlineError: If more than the step's timeout has passed since
                the attempt started.
        """
        if self.clock() - self.started > self.step.timeout_seconds:
            raise DeadlineError(
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
Outcome = tuple[str, Mapping[str, Any], float]


def distill(value: Any) -> str:
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
    short = distill([workflow.name, workflow.version, trigger_key])[:16]
    return f"{workflow.name}-{short}"[:64]


class Executor:
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
        store: objectstore.Depot,
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
        self.fences: dict[str, int] = {}

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

        Raises:
            ArtifactError: If the object is not a JSON object (see `parse`).
        """
        try:
            data = self.store.get(key)[0]
        except objectstore.AbsentError:
            # handled: missing object returned as None, documented
            return None
        return parse(key, data)

    def write(self, key: str, record: Mapping[str, Any]) -> None:
        """Stores a JSON record atomically, fenced by the run's lease.

        A record under a run this executor leases is written only after
        `hold` confirms the lease is still ours, so an executor that lost
        its lease stops instead of overwriting the new holder's state.

        Args:
            key: Object key.
            record: The record to store.

        Raises:
            EvictionError: If the run's lease is no longer ours.
        """
        prefix = f"workflows/{self.workflow.name}/runs/"
        if key.startswith(prefix) and not key.endswith("/lock"):
            run_id = key[len(prefix) :].split("/", 1)[0]
            if run_id in self.fences:
                self.hold(run_id)
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

    def lease(self, run_id: str, fence: int) -> bytes:
        """The body of this executor's lease on a run.

        Args:
            run_id: The run.
            fence: The fencing token: one more than the lease it replaced.

        Returns:
            The JSON bytes to store.
        """
        del run_id
        return json.dumps(
            {
                "owner": self.owner,
                "expires": self.clock() + self.lease_seconds,
                "fence": fence,
            }
        ).encode()

    def acquire(self, run_id: str) -> None:
        """Acquires the run's lease, or raises `BusyError`.

        A free run is taken with a conditional create. An expired lease is
        taken over with one compare-and-swap from the version just read, so
        of two executors racing for it exactly one wins (CON-1); the winner's
        fencing token is one more than the expired lease's.

        Args:
            run_id: The run.

        Raises:
            BusyError: If another executor holds a valid lease, or wins
                the race for an expired one.
        """
        key = self.base(run_id) + "lock"
        for ignored in range(LEASE_ATTEMPTS):
            try:
                self.store.put(key, self.lease(run_id, 1), if_absent=True)
            except objectstore.ExistsError:
                # handled: existing lease is expected; falls through to read and
                # check it
                pass
            else:
                self.fences[run_id] = 1
                return
            try:
                data, meta = self.store.get(key)
            except objectstore.AbsentError:
                # handled: lease vanished mid-read; bounded retry then
                # BusyError
                continue
            held = parse(key, data)
            if held["expires"] > self.clock():
                raise BusyError(f"run {run_id} is held by {held['owner']}")
            fence = int(held.get("fence", 0)) + 1
            try:
                self.store.put(
                    key, self.lease(run_id, fence), if_version=meta.version
                )
            except (
                objectstore.ConflictError,
                objectstore.AbsentError,
            ) as error:
                raise BusyError(f"run {run_id} was just taken") from error
            self.fences[run_id] = fence
            return
        raise BusyError(f"run {run_id} is changing hands; retry")

    def hold(self, run_id: str) -> objectstore.ObjectMeta:
        """Checks that this executor still holds the run's lease.

        Args:
            run_id: The run.

        Returns:
            The lease object's metadata, whose version a renewal or release
            must match.

        Raises:
            EvictionError: If the lease is gone, or another executor or a
                later fencing token holds it.
        """
        fence = self.fences.get(run_id)
        try:
            data, meta = self.store.get(self.base(run_id) + "lock")
        except objectstore.AbsentError:
            raise EvictionError(f"the lease on run {run_id} is gone") from None
        held = parse(self.base(run_id) + "lock", data)
        if (
            fence is None
            or held.get("owner") != self.owner
            or held.get("fence") != fence
        ):
            raise EvictionError(
                f"run {run_id} is now held by {held.get('owner')} "
                f"(fence {held.get('fence')}, ours {fence})"
            )
        return meta

    def renew_lease(self, run_id: str) -> None:
        """Extends the lease, only if this executor still holds it.

        Args:
            run_id: The run.

        Raises:
            EvictionError: If the lease was taken over or changed between
                the check and the renewal.
        """
        meta = self.hold(run_id)
        try:
            self.store.put(
                self.base(run_id) + "lock",
                self.lease(run_id, self.fences[run_id]),
                if_version=meta.version,
            )
        except CHANGED as error:
            raise EvictionError(
                f"the lease on run {run_id} changed while renewing"
            ) from error

    def relinquish(self, run_id: str) -> None:
        """Gives the lease up, never one another executor now holds.

        Args:
            run_id: The run.
        """
        try:
            meta = self.hold(run_id)
            self.store.delete(
                self.base(run_id) + "lock", if_version=meta.version
            )
        except (EvictionError, objectstore.ConflictError):
            LOGGER.warning(
                "WARNING run %s: the lease was no longer ours at release",
                run_id,
            )
        finally:
            self.fences.pop(run_id, None)

    @contextlib.contextmanager
    def pulse(self, run_id: str) -> Generator[None, None, None]:
        """Keeps the run's lease renewed by a heartbeat while steps run.

        Args:
            run_id: The run.

        Yields:
            None, while the heartbeat runs; it stops on exit.
        """
        stop = threading.Event()
        heartbeat = threading.Thread(
            target=self.beat, args=(run_id, stop), daemon=True
        )
        heartbeat.start()
        try:
            yield
        finally:
            stop.set()
            heartbeat.join()

    def beat(self, run_id: str, stop: threading.Event) -> None:
        """Renews the lease every third of its validity until `stop` is set.

        A failed renewal is logged and ends the heartbeat; the next write
        under the run then refuses (`hold`), so a lost lease never writes.

        Args:
            run_id: The run.
            stop: Set when the run's steps are done.
        """
        while not stop.wait(self.lease_seconds / 3):
            try:
                self.renew_lease(run_id)
            except EvictionError:
                LOGGER.warning("WARNING run %s: the lease was lost", run_id)
                return

    def scan(self) -> list[str]:
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
        has its maximum of runs in flight), `BusyError` (another
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
        settled = {s.value for s in FINISHED} - {Progress.FAILED.value}
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
            raise ConcurrencyLimitError(
                f"{len(others)} runs of {self.workflow.name} are in flight"
            )
        self.acquire(run_id)
        try:
            if len(self.scan()) > self.workflow.max_concurrent_runs:
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
                "state": Progress.RUNNING.value,
                "cost": 0.0,
                "started_at": self.clock(),
            }
            if record["workflow_digest"] != self.workflow.digest():
                raise errors.ConfigError(
                    "the run was started with a different workflow definition"
                )
            record["state"] = Progress.RUNNING.value
            self.write(self.base(run_id) + "run.json", record)
            record = self.execute(run_id, record)
            self.write(self.base(run_id) + "run.json", record)
            return record
        finally:
            self.relinquish(run_id)

    # ---- the run loop ----------------------------------------------------

    def execute(self, run_id: str, record: dict[str, Any]) -> dict[str, Any]:
        """Runs ready steps until nothing more can run.

        Args:
            run_id: The run.
            record: The run record, updated in place.

        Returns:
            The run record with its final state, step states, cost and end time.
        """
        states, outputs = self.restore(run_id)
        record["cost"] = self.tally(run_id)
        group_use: dict[str, int] = {}
        in_flight: dict[
            concurrent.futures.Future[Outcome], definition.Step
        ] = {}
        budget_hit = False
        with (
            self.pulse(run_id),
            concurrent.futures.ThreadPoolExecutor(
                max_workers=self.workflow.max_parallel
            ) as pool,
        ):
            while True:
                self.renew_lease(run_id)
                budget = self.workflow.cost_budget
                if budget is not None and record["cost"] > budget:
                    budget_hit = True
                changed = self.dispatch(
                    run_id,
                    record,
                    states,
                    outputs,
                    group_use,
                    in_flight,
                    pool,
                    budget_hit=budget_hit,
                )
                if not in_flight and not changed:
                    break
                if not in_flight:
                    continue
                self.collect(record, states, outputs, group_use, in_flight)
        return self.close(run_id, record, states, budget_hit=budget_hit)

    def restore(
        self, run_id: str
    ) -> tuple[dict[str, str], dict[str, Mapping[str, Any]]]:
        """Reads which steps already succeeded in an earlier attempt.

        Args:
            run_id: The run.

        Returns:
            The state value of every step (succeeded or pending), and the
            saved outputs of the steps that succeeded.
        """
        states: dict[str, str] = {}
        outputs: dict[str, Mapping[str, Any]] = {}
        for step in self.workflow.steps:
            saved = self.step_record(run_id, step.name)
            if saved is not None and saved["state"] == Phase.SUCCEEDED:
                states[step.name] = Phase.SUCCEEDED.value
                outputs[step.name] = saved["outputs"]
            else:
                states[step.name] = Phase.PENDING.value
        return states, outputs

    def skip(
        self, run_id: str, step: definition.Step, states: dict[str, str]
    ) -> bool:
        """Skips a step when one of the steps it depends on did not succeed.

        Args:
            run_id: The run.
            step: The pending step.
            states: Step state value by step name, updated in place.

        Returns:
            True when the step was skipped.
        """
        parents = [states[d] for d in step.depends_on]
        if not any(
            p
            in (
                Phase.FAILED,
                Phase.SKIPPED,
                Phase.REJECTED,
                Phase.EXPIRED,
            )
            for p in parents
        ):
            return False
        states[step.name] = Phase.SKIPPED.value
        self.stash(run_id, step, Phase.SKIPPED, reason="upstream")
        return True

    def admit(
        self,
        step: definition.Step,
        states: Mapping[str, str],
        group_use: Mapping[str, int],
    ) -> bool:
        """Checks that the step's parents succeeded and its group has room.

        Args:
            step: The pending step.
            states: Step state value by step name.
            group_use: Running steps per concurrency group.

        Returns:
            True when every step it depends on succeeded and its
            concurrency group (if any) is below its limit.
        """
        if any(states[d] != Phase.SUCCEEDED for d in step.depends_on):
            return False
        group = step.concurrency_group
        return group is None or group_use.get(group, 0) < self.workflow.quota(
            group
        )

    def adjudicate(
        self,
        run_id: str,
        step: definition.Step,
        states: dict[str, str],
        outputs: dict[str, Mapping[str, Any]],
    ) -> bool:
        """Settles an approval step and records its outputs when approved.

        Args:
            run_id: The run.
            step: The approval step.
            states: Step state value by step name, updated in place.
            outputs: Step outputs by step name, updated in place.

        Returns:
            True when the step's state changed from waiting.
        """
        settled = self.settle(run_id, step)
        states[step.name] = settled.value
        if settled is Phase.SUCCEEDED:
            saved = self.step_record(run_id, step.name) or {}
            outputs[step.name] = saved.get("outputs", {})
        return settled is not Phase.WAITING

    def dispatch(
        self,
        run_id: str,
        record: Mapping[str, Any],
        states: dict[str, str],
        outputs: dict[str, Mapping[str, Any]],
        group_use: dict[str, int],
        in_flight: dict[concurrent.futures.Future[Outcome], definition.Step],
        pool: concurrent.futures.Executor,
        *,
        budget_hit: bool,
    ) -> bool:
        """Starts, skips or settles every pending step that is ready.

        Args:
            run_id: The run.
            record: The run record; its cost is the cost so far.
            states: Step state value by step name, updated in place.
            outputs: Step outputs by step name, updated in place.
            group_use: Running steps per concurrency group, updated in place.
            in_flight: Running steps by their future, updated in place.
            pool: Executor that runs the steps.
            budget_hit: Whether the run budget is already exceeded, in which
                case nothing new starts.

        Returns:
            True when a step was skipped or an approval step left waiting.
        """
        changed = False
        for name in self.workflow.order():
            step = self.workflow.step(name)
            if states[name] != Phase.PENDING.value or budget_hit:
                continue
            if self.skip(run_id, step, states):
                changed = True
                continue
            if not self.admit(step, states, group_use):
                continue
            if step.kind is definition.StepKind.APPROVAL:
                changed = (
                    self.adjudicate(run_id, step, states, outputs) or changed
                )
                continue
            if len(in_flight) >= self.workflow.max_parallel:
                continue
            states[name] = Phase.RUNNING.value
            group = step.concurrency_group
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
        return changed

    def collect(
        self,
        record: dict[str, Any],
        states: dict[str, str],
        outputs: dict[str, Mapping[str, Any]],
        group_use: dict[str, int],
        in_flight: dict[concurrent.futures.Future[Outcome], definition.Step],
    ) -> None:
        """Waits for at least one running step and records the finished ones.

        Args:
            record: The run record; its cost is increased in place.
            states: Step state value by step name, updated in place.
            outputs: Step outputs by step name, updated in place.
            group_use: Running steps per concurrency group, updated in place.
            in_flight: Running steps by their future; finished ones are
                removed in place.
        """
        done = concurrent.futures.wait(
            in_flight, return_when=concurrent.futures.FIRST_COMPLETED
        ).done
        for future in done:
            step = in_flight.pop(future)
            state, result, cost = future.result()
            states[step.name] = state
            record["cost"] = round(record["cost"] + cost, 6)
            if state == Phase.SUCCEEDED.value:
                outputs[step.name] = result
            if step.concurrency_group is not None:
                group_use[step.concurrency_group] -= 1

    def close(
        self,
        run_id: str,
        record: dict[str, Any],
        states: dict[str, str],
        *,
        budget_hit: bool,
    ) -> dict[str, Any]:
        """Settles the run's final state once no step can run any more.

        Args:
            run_id: The run.
            record: The run record, updated in place.
            states: Step state value by step name.
            budget_hit: Whether the run budget was exceeded during the run.

        Returns:
            The run record with its final state, step states and end time.
        """
        if budget_hit or (
            self.workflow.cost_budget is not None
            and record["cost"] > self.workflow.cost_budget
        ):
            for name, state in states.items():
                if state == Phase.PENDING.value:
                    self.stash(
                        run_id,
                        self.workflow.step(name),
                        Phase.SKIPPED,
                        reason="run budget",
                    )
            record["state"] = Progress.BUDGET_STOPPED.value
        else:
            record["state"] = self.derive(states).value
        record["steps"] = states
        record["ended_at"] = self.clock()
        return record

    def derive(self, states: Mapping[str, str]) -> Progress:
        """Derives the run's state from its steps' states.

        Args:
            states: Step state value by step name.

        Returns:
            `WAITING_APPROVAL`, `REJECTED` or `EXPIRED` if any step is in the
            corresponding state (in that order of precedence), `SUCCEEDED` if
            all succeeded, otherwise `FAILED`.
        """
        values = set(states.values())
        if Phase.WAITING.value in values:
            return Progress.WAITING_APPROVAL
        if Phase.REJECTED.value in values:
            return Progress.REJECTED
        if Phase.EXPIRED.value in values:
            return Progress.EXPIRED
        if values == {Phase.SUCCEEDED.value}:
            return Progress.SUCCEEDED
        return Progress.FAILED

    def tally(self, run_id: str) -> float:
        """Sums the cost of steps already recorded as succeeded for this run.

        Args:
            run_id: The run.

        Returns:
            The total cost, rounded to six decimals.
        """
        total = 0.0
        for step in self.workflow.steps:
            saved = self.step_record(run_id, step.name)
            if saved is not None and saved["state"] == Phase.SUCCEEDED:
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
        return distill(
            [
                run_id,
                step.name,
                dict(step.parameters),
                {k: distill(v) for k, v in sorted(upstream.items())},
            ]
        )

    def stash(
        self,
        run_id: str,
        step: definition.Step,
        state: Phase,
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
    ) -> Outcome:
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
            and saved["state"] == Phase.SUCCEEDED
            and saved["idempotency_key"] == key
        ):
            return Phase.SUCCEEDED.value, saved["outputs"], 0.0
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
            self.stash(
                run_id,
                step,
                Phase.RUNNING,
                idempotency_key=key,
                attempts=attempts,
            )
            try:
                result = dict(handler(context) or {})
                context.check()
            except retry.TransientError as error:
                # handled: transient error saved in attempts and retried; FAILED
                # step record
                total += context.spent
                failure = errors.Failure.of(error)
                last_error = failure.text
                attempts.append(
                    {
                        "attempt": attempt,
                        "state": "retryable",
                        "error": last_error,
                        "failure": dict(failure.dump()),
                    }
                )
                if attempt < step.retries.max_attempts:
                    self.sleep(step.retries.delay(attempt))
            except errors.ScmleError as error:
                # handled: scmle error saved in attempts; step recorded FAILED
                total += context.spent
                failure = errors.Failure.of(error)
                last_error = failure.text
                attempts.append(
                    {
                        "attempt": attempt,
                        "state": "failed",
                        "error": last_error,
                        "failure": dict(failure.dump()),
                    }
                )
                break
            except Exception as error:  # noqa: BLE001 -- handler boundary
                # handled: handler exception saved in attempts; step recorded
                # FAILED with error
                total += context.spent
                failure = errors.Failure.of(error)
                last_error = failure.text[:300]
                attempts.append(
                    {
                        "attempt": attempt,
                        "state": "failed",
                        "error": last_error,
                        "failure": dict(failure.dump()),
                    }
                )
                break
            else:
                # ERR-7: the success record is written outside the handler's
                # retry, with its own retries for the store; a write that
                # still fails propagates, so a handler that succeeded is
                # neither run again nor recorded as failed.
                total += context.spent
                attempts.append({"attempt": attempt, "state": "succeeded"})
                save = functools.partial(
                    self.stash,
                    run_id,
                    step,
                    Phase.SUCCEEDED,
                    idempotency_key=key,
                    attempts=attempts,
                    outputs=result,
                    cost=round(total, 6),
                )
                retry.call(save, WRITES, sleep=self.sleep)
                return Phase.SUCCEEDED.value, result, total
        self.stash(
            run_id,
            step,
            Phase.FAILED,
            idempotency_key=key,
            attempts=attempts,
            error=last_error,
            cost=round(total, 6),
        )
        return Phase.FAILED.value, {}, total

    # ---- approval --------------------------------------------------------

    def compose(
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

    def decide(
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
        payload = self.compose(run_id, step, approver, decision, moment)
        return {**payload, "signature": signing.mac(payload, key)}

    def ratify(self, decision: Mapping[str, Any]) -> None:
        """Records a signed decision after checking it.

        Args:
            decision: A decision produced by `decide`. A decision that
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
        payload = self.compose(
            run_id,
            name,
            approver,
            str(decision["decision"]),
            float(decision["at"]),
        )
        if key is None or not hmac.compare_digest(
            signing.mac(payload, key),
            str(decision.get("signature", "")),
        ):
            raise ApprovalError("the approval signature does not verify")
        if decision["decision"] not in ("approve", "reject"):
            raise ApprovalError("decision must be approve or reject")
        if self.read(f"{self.base(run_id)}approvals/{name}.json"):
            raise ApprovalError("a decision is already recorded")
        self.write(f"{self.base(run_id)}approvals/{name}.json", dict(decision))

    def settle(self, run_id: str, step: definition.Step) -> Phase:
        """Determines the approval step's state now.

        A recorded decision settles the step to succeeded (approve) or rejected;
        otherwise the step waits, and expires once it has waited longer than its
        expiry.

        Args:
            run_id: The run.
            step: The approval step.

        Returns:
            The step's new state.
        """
        decision = self.read(f"{self.base(run_id)}approvals/{step.name}.json")
        if decision is not None:
            if decision["decision"] == "approve":
                self.stash(
                    run_id,
                    step,
                    Phase.SUCCEEDED,
                    outputs={"approver": decision["approver"]},
                    cost=0.0,
                    idempotency_key="approval",
                )
                return Phase.SUCCEEDED
            self.stash(
                run_id, step, Phase.REJECTED, approver=decision["approver"]
            )
            return Phase.REJECTED
        waiting = self.step_record(run_id, step.name)
        since = (
            waiting["waiting_since"]
            if waiting and waiting["state"] == Phase.WAITING
            else self.clock()
        )
        if self.clock() - since > step.expires_seconds:
            self.stash(run_id, step, Phase.EXPIRED, waiting_since=since)
            return Phase.EXPIRED
        self.stash(run_id, step, Phase.WAITING, waiting_since=since)
        return Phase.WAITING
