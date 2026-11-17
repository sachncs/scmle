"""What a step handler sees, and what a step returns."""

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
from typing import Any

from scmle.data.store import objectstore
from scmle.workflow import definition
from scmle.workflow.executor import failures


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
        started: Timer reading when the attempt started.
        clock: Returns the timer reading in seconds: monotonic, so a step
            of the wall clock neither fails nor extends a step (CON-20).
        spent: Cost recorded by this attempt so far.
        run_spent: Cost of the run before this step.
        run_budget: Ceiling for the run's cost, or None.
    """

    # The public path: `Executor`'s signature names `Handler`, and the API
    # snapshot pins its text, which spells this class with its module.
    __module__ = "scmle.workflow.executor"

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
            raise failures.DeadlineError(
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
            raise failures.StepBudgetError(
                f"{self.step.name} cost passed {limit}"
            )
        if (
            self.run_budget is not None
            and self.run_spent + self.spent > self.run_budget
        ):
            raise failures.StepBudgetError("the run's cost budget was passed")


Handler = Callable[[StepContext], Mapping[str, Any] | None]
Outcome = tuple[str, Mapping[str, Any], float]
