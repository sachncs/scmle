"""The reservation stage: the post-training reserve and the minimum run."""

from typing import ClassVar

from scmle.pipeline import profiler
import scmle.pipeline.context
from scmle.pipeline.stages import cost
from scmle.pipeline.stages import ingestion
from scmle.pipeline.stages import interface
import scmle.runtime.state


class Reservation(interface.Stage):
    """Holds the post-training reserve and refuses a run that cannot fit.

    Runs right after ingestion, in the same state, before any profiling or
    training: the reserve for the final refit, diagnostics and packaging is
    estimated from the table's shape and held back from the trial window
    (`Governor.until`).
    """

    requires: ClassVar[tuple[type, ...]] = (ingestion.IngestStage,)
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
        reserve = context.regulator.reserve
        parts = reserve.provision(*cost.dimensions(context), profiler.estimate)
        context.emit(
            "budget.reserved",
            f"{reserve.total:.1f} s of the "
            f"{context.spec.resources.time_budget_seconds} s budget held for "
            "the final refit, diagnostics and packaging",
            data={"reserve_seconds": parts},
        )
        reserve.admit()
