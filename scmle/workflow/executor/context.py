"""What a step handler sees, and what a step returns."""

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
from typing import Any

from scmle.storage import objectstore
from scmle.workflow import definition
from scmle.workflow.executor import failures


@dataclasses.dataclass(slots=True)
class Context:
    """What a handler sees.

    Attributes:
        run: The durable run id.
        step: The step definition.
        attempt: Attempt number, from 1.
        key: Stable across retries and re-invocations.
        upstream: Outputs of the steps it depends on, by step name.
        store: The executor's durable store.
        started: Timer reading when the attempt started.
        clock: Returns the timer reading in seconds: monotonic, so a step
            of the wall clock neither fails nor extends a step (CON-20).
        spent: Cost recorded by this attempt so far.
        accrued: Cost of the run before this step.
        budget: Ceiling for the run's cost, or None.
    """

    # The public path: `Executor`'s signature names `Handler`, and the API
    # snapshot pins its text, which spells this class with its module.
    __module__ = "scmle.workflow.executor"

    run: str
    step: definition.Step
    attempt: int
    key: str
    upstream: Mapping[str, Mapping[str, Any]]
    store: objectstore.Depot
    started: float
    clock: Callable[[], float]
    spent: float = 0.0
    accrued: float = 0.0
    budget: float | None = None

    def check(self) -> None:
        """Raises if the step is past its timeout.

        Raises:
            DeadlineError: If more than the step's timeout has passed since
                the attempt started.
        """
        if self.clock() - self.started > self.step.timeout:
            raise failures.DeadlineError(
                f"{self.step.name} passed {self.step.timeout}s"
            )

    def spend(self, amount: float) -> None:
        """Records cost; raises if a budget is passed.

        Args:
            amount: Cost to add.

        Raises:
            StepBudgetError: If the step's or the run's budget is exceeded.
        """
        self.spent += amount
        limit = self.step.budget
        if limit is not None and self.spent > limit:
            raise failures.BudgetError(f"{self.step.name} cost passed {limit}")
        if self.budget is not None and self.accrued + self.spent > self.budget:
            raise failures.BudgetError("the run's cost budget was passed")


Handler = Callable[[Context], Mapping[str, Any] | None]
Outcome = tuple[str, Mapping[str, Any], float]
