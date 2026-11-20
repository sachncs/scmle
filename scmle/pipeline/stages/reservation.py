"""The reservation stage: the post-training reserve and the minimum run."""

import dataclasses
from typing import ClassVar

from scmle.contracts import errors
from scmle.pipeline import engine
from scmle.pipeline import products
from scmle.pipeline.stages import cost
from scmle.pipeline.stages import ingestion
from scmle.pipeline.stages import interface
from scmle.runtime import limits
import scmle.runtime.state


@dataclasses.dataclass(frozen=True)
class Reserve(products.Derived):
    """The time held back for the final steps; held again on resume.

    Attributes:
        governor: The run's governor, whose reserve (`budget.Reserve`) the
            stage provisioned; applying the product sets it on the run's
            services.
    """

    governor: limits.Governor

    def apply(self, board: object) -> None:
        """Sets the governor on the run's services and holds the reserve.

        Args:
            board: The run's board.

        Raises:
            KindError: If `board` is not a run's board.
        """
        if not isinstance(board, engine.Board):
            raise errors.KindError("a reserve applies to a run's board")
        board.services = dataclasses.replace(
            board.services, governor=self.governor
        )
        super().apply(board)

    def summary(self) -> object:
        """The table shape and the seconds held back by part.

        Returns:
            The reserve, which depends only on the shape and the budget.
        """
        reserve = self.governor.reserve
        return {"shape": reserve.shape, "parts": reserve.parts}


class Reservation(interface.Stage):
    """Holds the post-training reserve and refuses a run that cannot fit.

    Runs right after ingestion, in the same state, before any profiling or
    training: the reserve for the final refit, diagnostics and packaging is
    estimated from the table's shape and held back from the trial window
    (`Governor.until`).
    """

    requires: ClassVar[tuple[type, ...]] = (ingestion.Ingest,)
    makes: ClassVar[type[products.Product] | None] = Reserve
    state: ClassVar[scmle.runtime.state.Progress] = (
        scmle.runtime.state.Progress.VALIDATING
    )

    def describe(
        self,
        board: engine.Board,  # noqa: ARG002 -- Stage.describe
    ) -> str:
        """Names the stage.

        Args:
            board: The run's board; not consulted.

        Returns:
            The text "reserve time for the final steps".
        """
        return "reserve time for the final steps"

    def execute(self, board: engine.Board) -> Reserve:
        """Provisions the reserve, records it and admits the run.

        Args:
            board: The run's board, holding the ingested frame.

        Returns:
            The reserve held for the final refit, diagnostics and packaging.

        Raises:
            BlockedError: If preparation, one cheapest trial and the reserve
                cannot fit in the time budget (`budget.Reserve.admit`).
        """
        reserve = board.services.governor.reserve
        intake = board.get(ingestion.Intake)
        parts = reserve.provision(
            *cost.dimensions(board.services.spec, intake.frame, None),
            intake.kind.cost,
        )
        board.journal.emit(
            "budget.reserved",
            f"{reserve.total:.1f} s of the "
            f"{board.services.spec.resources.time_budget_seconds} s budget "
            "held for the final refit, diagnostics and packaging",
            data={"reserve_seconds": parts},
        )
        reserve.admit()
        return Reserve(board.services.governor)
