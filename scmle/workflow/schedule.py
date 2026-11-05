"""Which scheduled runs are due, including after the scheduler was away.

Slots are ``start_epoch + n * interval``. The scheduler keeps the newest slot
it has dealt with in the store and, on each `tick`, finds the slots that came
due since. What it does with slots that were missed (it was not running, or
a run was still in flight) is the schedule's declared `catch_up` policy:

``run_all``
    Run every due slot in order, up to ``max_catch_up``; the older ones
    beyond that are recorded as missed.
``latest_only``
    Run the newest due slot; record the older ones as missed.
``skip``
    Run only a slot that is no older than the grace period; record every
    other due slot as missed and run nothing for it.

Each slot's run id comes from the slot time, so running a slot twice (a
scheduler that lost its state, two schedulers) is the same run: the executor
returns the finished record instead of repeating the work. On its very first
tick the scheduler does not backfill history: only the newest slot counts.
A slot whose run could not start because another run holds the workflow is
`deferred` and stays due for the next tick.
"""

import dataclasses
import json
from typing import Any

from scmle.contracts import errors
from scmle.data.store import objectstore
from scmle.workflow import definition
from scmle.workflow import executor as executor_module

GRACE_FRACTION = 0.1


@dataclasses.dataclass(frozen=True, slots=True)
class TickReport:
    """What one tick did.

    Attributes:
        ran: Slots that were run, with their run states.
        missed: Slots recorded as missed.
        deferred: Slots that could not start and stay due.
    """

    ran: tuple[tuple[float, str], ...]
    missed: tuple[float, ...]
    deferred: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "ran": [[slot, state] for slot, state in self.ran],
            "missed": list(self.missed),
            "deferred": list(self.deferred),
        }


class Scheduler:
    """Drives a scheduled workflow through an executor."""

    def __init__(
        self,
        workflow: definition.Workflow,
        store: objectstore.ObjectStore,
    ) -> None:
        """Initializes the scheduler.

        Args:
            workflow: A workflow with a schedule.
            store: Where the scheduler keeps its position.

        Raises:
            ConfigError: If the workflow has no schedule.
        """
        if workflow.schedule is None:
            raise errors.ConfigError("the workflow has no schedule")
        self.workflow = workflow
        self.schedule: definition.Schedule = workflow.schedule
        self.store = store
        self.key = f"workflows/{workflow.name}/schedule/state.json"

    def last_slot(self) -> float | None:
        """The newest slot already dealt with."""
        try:
            data, ignored = self.store.get(self.key)
        except objectstore.NotFoundError:
            return None
        value = json.loads(data)["last_slot"]
        return None if value is None else float(value)

    def save(self, slot: float, report: dict[str, Any]) -> None:
        """Records progress."""
        body = json.dumps({"last_slot": slot, "last": report}, sort_keys=True)
        self.store.put(self.key, body.encode())

    def due(self, now: float) -> list[float]:
        """Slots that have come due since the last one dealt with."""
        schedule = self.schedule
        if now < schedule.start_epoch:
            return []
        newest = schedule.start_epoch + schedule.interval_seconds * int(
            (now - schedule.start_epoch) // schedule.interval_seconds
        )
        last = self.last_slot()
        if last is None:
            return [newest]
        count = round((newest - last) / schedule.interval_seconds)
        return [
            last + schedule.interval_seconds * n for n in range(1, count + 1)
        ]

    def tick(self, executor: executor_module.LocalExecutor) -> TickReport:
        """Runs what is due under the catch-up policy.

        Args:
            executor: Runs the slots (its clock is the scheduler's `now`).
        """
        now = executor.clock()
        slots = self.due(now)
        policy = self.schedule.catch_up
        grace = self.schedule.interval_seconds * GRACE_FRACTION
        if policy is definition.CatchUp.RUN_ALL:
            chosen = slots[: self.schedule.max_catch_up]
        elif policy is definition.CatchUp.LATEST_ONLY:
            chosen = slots[-1:]
        else:
            chosen = [s for s in slots[-1:] if now - s <= grace]
        missed = [s for s in slots if s not in chosen]
        ran: list[tuple[float, str]] = []
        deferred: list[float] = []
        for slot in chosen:
            try:
                record = executor.start(f"slot-{int(slot)}", "scheduler")
            except (
                executor_module.RunInFlightError,
                executor_module.ConcurrencyLimitError,
            ):
                deferred.append(slot)
                break
            ran.append((slot, record["state"]))
        progressed = [s for s, ignored in ran]
        if slots and not deferred:
            self.save(slots[-1], {"ran": progressed, "missed": missed})
        elif progressed or missed:
            settled = [s for s in slots if s < deferred[0]]
            if settled:
                self.save(settled[-1], {"ran": progressed, "missed": missed})
        return TickReport(tuple(ran), tuple(missed), tuple(deferred))
