"""The reservation stage: the post-training reserve and the minimum run."""

import dataclasses
from typing import ClassVar

from scmle.pipeline import products
from scmle.pipeline import profiler
import scmle.pipeline.context
from scmle.pipeline.stages import cost
from scmle.pipeline.stages import ingestion
from scmle.pipeline.stages import interface
from scmle.runtime import limits
import scmle.runtime.state


@dataclasses.dataclass(frozen=True)
class Allowance(products.Derived):
    """The time held back for the final steps; held again on resume.

    Attributes:
        governor: The run's governor, whose reserve the stage provisioned.
    """

    governor: limits.Governor | None

    def summary(self) -> object:
        """The table shape and the seconds held back by part.

        Returns:
            The reserve, which depends only on the shape and the budget.
        """
        if self.governor is None:
            return None
        reserve = self.governor.reserve
        return {"shape": reserve.shape, "parts": reserve.parts}


class Reservation(interface.Stage):
    """Holds the post-training reserve and refuses a run that cannot fit.

    Runs right after ingestion, in the same state, before any profiling or
    training: the reserve for the final refit, diagnostics and packaging is
    estimated from the table's shape and held back from the trial window
    (`Governor.until`).
    """

    requires: ClassVar[tuple[type, ...]] = (ingestion.IngestStage,)
    makes: ClassVar[type[products.Product] | None] = Allowance
    state: ClassVar[scmle.runtime.state.Progress] = (
        scmle.runtime.state.Progress.VALIDATING
    )

    def describe(
        self,
        context: scmle.pipeline.context.Context,  # noqa: ARG002 -- Stage.describe
    ) -> str:
        """Names the stage.

        Args:
            context: The run context; not consulted.

        Returns:
            The text "reserve time for the final steps".
        """
        return "reserve time for the final steps"

    def execute(self, context: scmle.pipeline.context.Context) -> None:
        """Provisions the reserve, records it and admits the run.

        Args:
            context: The run context, holding the ingested frame.

        Raises:
            BlockedError: If preparation, one cheapest trial and the reserve
                cannot fit in the time budget (`budget.Reserve.admit`).
        """
        reserve = context.services.governor.reserve
        parts = reserve.provision(*cost.dimensions(context), profiler.estimate)
        context.journal.emit(
            "budget.reserved",
            f"{reserve.total:.1f} s of the "
            f"{context.services.spec.resources.time_budget_seconds} s budget "
            "held for the final refit, diagnostics and packaging",
            data={"reserve_seconds": parts},
        )
        reserve.admit()
