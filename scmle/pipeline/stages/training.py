"""The training stage."""

import time
from typing import ClassVar

from scmle.contracts import errors
from scmle.contracts import experiments
from scmle.data.profiling import profiler
from scmle.pipeline import ensemble_search
from scmle.pipeline import strategy_search
from scmle.pipeline import trials
from scmle.pipeline import tuning
import scmle.pipeline.context
from scmle.pipeline.stages import cost
from scmle.pipeline.stages import interface
from scmle.pipeline.stages import planning
import scmle.runtime.state


class TrainStage(interface.Stage):
    """Cross-validates every candidate within the time budget."""

    requires: ClassVar[tuple[type, ...]] = (planning.Planning,)
    state: ClassVar[scmle.runtime.state.Progress] = (
        scmle.runtime.state.Progress.TRAINING
    )

    def describe(self, context: scmle.pipeline.context.Context) -> str:
        """Reports the number of candidates.

        Args:
            context: The run context, holding the execution plan.

        Returns:
            The candidate count as text.
        """
        plan = scmle.pipeline.context.require(context.plan, "plan")
        return f"{len(plan.candidate_models)} candidates"

    def execute(self, context: scmle.pipeline.context.Context) -> None:
        """Runs every candidate, then searches strategies, tunes and ensembles.

        Candidates run in chunks of `backend.workers`. The trial window and
        the memory limit are checked before every chunk, and a chunk that
        must not start is recorded as skipped. Results are recorded in
        candidate order regardless of the backend.

        Args:
            context: The run context, holding the execution plan; receives the
                trials.

        Raises:
            BlockedError: If preparation left no trial window for the first
                trial; the message names the smallest budget that fits.
        """
        governor = context.regulator
        if governor.reserve.shape is None:
            governor.reserve.provision(
                *cost.dimensions(context), profiler.estimate
            )
        runner = trials.Runner(context)
        plan = scmle.pipeline.context.require(context.plan, "plan")
        names = plan.candidate_models
        for start in range(0, len(names), runner.workers):
            context.check_cancelled()
            chunk = names[start : start + runner.workers]
            ids = context.allocate(len(chunk))
            jobs = [
                trials.Job(trial_id, name)
                for trial_id, name in zip(ids, chunk, strict=True)
            ]
            outcomes = runner.run_many(jobs, first=start == 0)
            if start == 0 and all(
                o.status is experiments.Termination.SKIPPED for o in outcomes
            ):
                prepared = time.monotonic() - context.started_at
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
                context.transcribe(outcome)
        strategy_search.Strategist(context, runner).search(list(context.trials))
        tuning.Tuner(context, runner).tune(list(context.trials))
        ensemble_search.Stacker(context, runner).search(list(context.trials))
