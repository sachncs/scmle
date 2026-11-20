"""The training stage."""

import dataclasses
from typing import ClassVar
from typing import Self

from scmle.contracts import errors
from scmle.contracts import experiments
from scmle.pipeline import engine
from scmle.pipeline import ensemble_search
from scmle.pipeline import products
from scmle.pipeline import strategy_search
from scmle.pipeline import trials
from scmle.pipeline import tuning
import scmle.pipeline.journal
from scmle.pipeline.stages import cost
from scmle.pipeline.stages import ingestion
from scmle.pipeline.stages import interface
from scmle.pipeline.stages import planning
from scmle.pipeline.stages import profiling
from scmle.pipeline.stages import reservation
import scmle.runtime.state


@dataclasses.dataclass(frozen=True)
class Search(products.Kept):
    """The trials of the candidate search and what was decided on the way.

    Attributes:
        trials: The trial results, in order.
        outcomes: The evidence-based decisions so far.
        counter: The trial identifiers handed out.
        skipped: The work reduced or skipped for the budget so far.
    """

    trials: list[experiments.Trial]
    outcomes: list[experiments.Decision]
    counter: int
    skipped: list[dict[str, str]]

    def apply(self, board: object) -> None:
        """Sets the trials and decisions on the run's journal; holds them.

        Args:
            board: The run's board.

        Raises:
            KindError: If `board` is not a run's board.
        """
        if not isinstance(board, engine.Board):
            raise errors.KindError("a search applies to a run's board")
        with board.journal.lock:
            board.journal.trials = self.trials
            board.journal.outcomes = self.outcomes
            board.journal.counter = self.counter
            board.journal.skipped = self.skipped
        super().apply(board)

    @classmethod
    def copy(cls, journal: scmle.pipeline.journal.Journal) -> Self:
        """Copies the search out of the run's journal, under its lock.

        Args:
            journal: The journal the trials were transcribed to.

        Returns:
            The search; later stages' entries do not change it.
        """
        with journal.lock:
            return cls(
                list(journal.trials),
                list(journal.outcomes),
                journal.counter,
                list(journal.skipped),
            )


class Training(interface.Stage):
    """Cross-validates every candidate within the time budget."""

    recorded: ClassVar[str] = "TrainStage"
    requires: ClassVar[tuple[type, ...]] = (
        planning.Planning,
        reservation.Reservation,
    )
    makes: ClassVar[type[products.Product] | None] = Search
    state: ClassVar[scmle.runtime.state.Progress] = (
        scmle.runtime.state.Progress.TRAINING
    )

    def describe(self, board: engine.Board) -> str:
        """Reports the number of candidates.

        Args:
            board: The run's board, holding the execution plan.

        Returns:
            The candidate count as text.
        """
        plan = products.require(board.get(planning.Blueprint).plan, "plan")
        return f"{len(plan.candidates)} candidates"

    def execute(self, board: engine.Board) -> Search:
        """Runs every candidate, then searches strategies, tunes and ensembles.

        Candidates run in chunks of `backend.workers`. The trial window and
        the memory limit are checked before every chunk, and a chunk that
        must not start is recorded as skipped. Results are recorded in
        candidate order regardless of the backend.

        Args:
            board: The run's board, holding the execution plan; its journal
                transcribes the trials as they end.

        Returns:
            The search: a copy of what the journal holds once it is done.

        Raises:
            BlockedError: If preparation left no trial window for the first
                trial; the message names the smallest budget that fits.
        """
        governor = board.services.governor
        partition = board.get(profiling.Partition)
        if governor.reserve.shape is None:
            governor.reserve.provision(
                *cost.dimensions(
                    board.services.spec, partition.frame, partition.task
                ),
                board.get(ingestion.Intake).kind.cost,
            )
        runner = trials.Runner(
            board, partition, board.get(ingestion.Intake).kind.featurise
        )
        plan = products.require(board.get(planning.Blueprint).plan, "plan")
        names = plan.candidates
        for start in range(0, len(names), runner.workers):
            board.services.check()
            chunk = names[start : start + runner.workers]
            ids = board.journal.allocate(len(chunk))
            jobs = [
                trials.Job(trial_id, name)
                for trial_id, name in zip(ids, chunk, strict=True)
            ]
            outcomes = runner.dispatch(jobs, first=start == 0)
            if start == 0 and all(
                o.status is experiments.Termination.SKIPPED for o in outcomes
            ):
                prepared = board.services.elapsed()
                smallest = governor.reserve.smallest(prepared)
                raise errors.BlockedError(
                    f"preparation took {prepared:.1f} s and left no time "
                    f"for a first trial ({outcomes[0].error}); the smallest "
                    f"budget that fits is about {smallest} s",
                    {
                        "prepared_seconds": round(prepared, 2),
                        "smallest_budget_seconds": smallest,
                    },
                )
            for outcome in outcomes:
                board.journal.transcribe(outcome)
        strategy_search.Strategist(board, runner).search(
            list(board.journal.trials)
        )
        tuning.Tuner(board, runner).tune(list(board.journal.trials))
        ensemble_search.Stacker(board, runner).search(
            list(board.journal.trials)
        )
        return Search.copy(board.journal)
