"""The run loop: start ready steps, collect finished ones, settle the run."""

from collections.abc import Mapping
import concurrent.futures
from typing import Any
from typing import TYPE_CHECKING

from scmle.runtime import engine
from scmle.runtime import state
from scmle.workflow import definition
from scmle.workflow.executor import approvals
from scmle.workflow.executor import context
from scmle.workflow.executor import status

if TYPE_CHECKING:
    import scmle.pipeline.engine


class Loop(approvals.Approvals):
    """Runs ready steps until nothing more can run, then settles the run."""

    def execute(
        self,
        run_id: str,
        record: dict[str, Any],
        machine: state.Machine[status.Progress],
    ) -> dict[str, Any]:
        """Runs ready steps until nothing more can run.

        Args:
            run_id: The run.
            record: The run record, updated in place.
            machine: The run's state, `RUNNING`; it ends where the run ends.

        Returns:
            The run record with its final state, step states, cost and end time.
        """
        states, outputs = self.restore(run_id)
        # STG-12: a resume keeps what earlier attempts spent, failed ones too:
        # the stored run cost covers every step the last drive settled, the
        # step records cover any it recorded after its last run write.
        record["cost"] = max(record.get("cost", 0.0), self.tally(run_id))
        group_use: dict[str, int] = {}
        in_flight: dict[
            concurrent.futures.Future[context.Outcome], definition.Step
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
        return self.close(
            run_id, record, states, machine, budget_hit=budget_hit
        )

    def dispatch(
        self,
        run_id: str,
        record: Mapping[str, Any],
        states: dict[str, str],
        outputs: dict[str, Mapping[str, Any]],
        group_use: dict[str, int],
        in_flight: dict[
            concurrent.futures.Future[context.Outcome], definition.Step
        ],
        pool: concurrent.futures.Executor,
        *,
        budget_hit: bool,
    ) -> bool:
        """Starts, skips or settles every pending step that is ready.

        The engine's ready set (`scmle.runtime.engine.ready`, by step name)
        names the pending steps whose every dependency succeeded, in the
        workflow's order; a step with a dependency that did not succeed is
        skipped first.

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
        if budget_hit:
            return False
        changed = False
        steps = [self.workflow.step(name) for name in self.workflow.order()]
        for step in steps:
            if states[step.name] == status.Phase.PENDING.value:
                changed = self.skip(run_id, step, states) or changed
        succeeded = [
            step
            for step in steps
            if states[step.name] == status.Phase.SUCCEEDED.value
        ]
        pending = [
            step
            for step in steps
            if states[step.name] == status.Phase.PENDING.value
        ]
        for step in engine.ready(succeeded, pending, engine.NAMES):
            if not self.admit(step, states, group_use):
                continue
            if step.kind is definition.StepKind.APPROVAL:
                changed = (
                    self.adjudicate(run_id, step, states, outputs) or changed
                )
                continue
            if len(in_flight) >= self.workflow.max_parallel:
                continue
            states[step.name] = status.Phase.RUNNING.value
            group = step.concurrency_group
            if group is not None:
                group_use[group] = group_use.get(group, 0) + 1
            # STG-12: a step sees the spend so far plus the declared cost
            # of the steps still running, so steps started together cannot
            # pass the run's budget between them.
            reserved = sum(s.cost_budget or 0.0 for s in in_flight.values())
            future = pool.submit(
                self.run_step,
                run_id,
                step,
                {d: outputs[d] for d in step.depends_on},
                record["cost"] + reserved,
            )
            in_flight[future] = step
        return changed

    def collect(
        self,
        record: dict[str, Any],
        states: dict[str, str],
        outputs: dict[str, Mapping[str, Any]],
        group_use: dict[str, int],
        in_flight: dict[
            concurrent.futures.Future[context.Outcome], definition.Step
        ],
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
            if state == status.Phase.SUCCEEDED.value:
                outputs[step.name] = result
            if step.concurrency_group is not None:
                group_use[step.concurrency_group] -= 1

    def close(
        self,
        run_id: str,
        record: dict[str, Any],
        states: dict[str, str],
        machine: state.Machine[status.Progress],
        *,
        budget_hit: bool,
    ) -> dict[str, Any]:
        """Settles the run's final state once no step can run any more.

        Args:
            run_id: The run.
            record: The run record, updated in place.
            states: Step state value by step name.
            machine: The run's state, moved to where the run ends.
            budget_hit: Whether the run budget was exceeded during the run.

        Returns:
            The run record with its final state, step states and end time.
        """
        if budget_hit or (
            self.workflow.cost_budget is not None
            and record["cost"] > self.workflow.cost_budget
        ):
            for name, state in states.items():
                if state == status.Phase.PENDING.value:
                    self.stash(
                        run_id,
                        self.workflow.step(name),
                        status.Phase.SKIPPED,
                        reason="run budget",
                    )
            machine.transition(status.Progress.BUDGET_STOPPED)
        else:
            machine.transition(self.derive(states))
        record["state"] = machine.state.value
        record["steps"] = states
        record["ended_at"] = self.clock()
        return record

    def derive(self, states: Mapping[str, str]) -> status.Progress:
        """Derives the run's state from its steps' states.

        Args:
            states: Step state value by step name.

        Returns:
            `WAITING_APPROVAL`, `REJECTED` or `EXPIRED` if any step is in the
            corresponding state (in that order of precedence), `SUCCEEDED` if
            all succeeded, otherwise `FAILED`.
        """
        values = set(states.values())
        if status.Phase.WAITING.value in values:
            return status.Progress.WAITING_APPROVAL
        if status.Phase.REJECTED.value in values:
            return status.Progress.REJECTED
        if status.Phase.EXPIRED.value in values:
            return status.Progress.EXPIRED
        if values == {status.Phase.SUCCEEDED.value}:
            return status.Progress.SUCCEEDED
        return status.Progress.FAILED

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
        archive: scmle.pipeline.engine.Archive[Mapping[str, Any]] = self.bucket(
            run_id
        )
        for index, step in enumerate(self.workflow.steps):
            found = archive.restore(index, step.name, None, {})
            if found is not None:
                states[step.name] = status.Phase.SUCCEEDED.value
                outputs[step.name] = found[0]
            else:
                states[step.name] = status.Phase.PENDING.value
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
        if not any(states[d] in status.UNMET for d in step.depends_on):
            return False
        states[step.name] = status.Phase.SKIPPED.value
        self.stash(run_id, step, status.Phase.SKIPPED, reason="upstream")
        return True

    def tally(self, run_id: str) -> float:
        """Sums the cost recorded by every step of this run, whatever its state.

        A failed step's spend is as real as a succeeded one's (STG-12);
        records without a cost (running, skipped) count nothing.

        Args:
            run_id: The run.

        Returns:
            The total cost, rounded to six decimals.
        """
        found = (self.step_record(run_id, s.name) for s in self.workflow.steps)
        return round(sum(float(r.get("cost", 0.0)) for r in found if r), 6)
