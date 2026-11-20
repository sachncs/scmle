"""Hyperband tuning: halving brackets that start at different fold costs.

`Hyperband` is the bracket schedule of `tuning.Tuner`: early brackets try
many configurations on few folds, the last a few on every fold, and each
keeps the best `1/eta` at every rung.
"""

import itertools
import math
import operator

from scmle.contracts import experiments
from scmle.contracts import metrics
from scmle.contracts import tasks
from scmle.optimization import optimizers
from scmle.pipeline import engine
from scmle.pipeline import trials

MIN_BATCH = 2
MIN_FOLDS = 2


class Hyperband:
    """The Hyperband schedule of a tuner, and the recording of its trials.

    Attributes:
        board: Run board.
        runner: Trial runner used for every evaluation.
        search: The effective search specification.
        metric: The metric being optimized.
    """

    board: engine.Board
    runner: trials.Runner
    search: tasks.Search
    metric: metrics.Metric

    def ladder(self) -> list[list[int | None]]:
        """Computes the fold budgets of each Hyperband bracket, cheapest first.

        With `F` folds and reduction factor `eta`, rungs are the distinct fold
        counts `int(F / eta^k)` for `k` from 0 to 3 (at least one fold, fewer
        than `F`), in ascending order, followed by the full evaluation
        (`None`). Bracket `s` starts at rung `s`: early brackets try many
        configurations cheaply, the last one a few at full cost.

        Returns:
            One list of rungs per bracket, each ending in `None` (all folds); a
            single full-evaluation rung if there are fewer than two folds.
        """
        folds = len(self.runner.folds)
        if folds < MIN_FOLDS:
            return [[None]]
        eta = self.search.eta
        counts = sorted({max(int(folds / eta**power), 1) for power in range(4)})
        rungs: list[int | None] = [c for c in counts if c < folds]
        rungs.append(None)
        return [rungs[start:] for start in range(len(rungs))]

    def hyperband(
        self,
        name: str,
        strategy: str,
        optimizer: optimizers.Optimizer,
    ) -> None:
        """Runs Hyperband: halving brackets with different start costs.

        The configuration budget is shared between brackets, weighted toward the
        cheap ones (many tries need cheap screening); each bracket keeps the
        best `1/eta` at every rung and finishes the survivors on every fold.

        Args:
            name: Model being tuned.
            strategy: Data strategy the model is tuned under.
            optimizer: Source of configurations; told every score.
        """
        brackets = self.ladder()
        weights = [len(brackets) - index for index in range(len(brackets))]
        total = sum(weights)
        remaining = self.search.configs_per_model
        for index, rungs in enumerate(brackets):
            if remaining <= 0:
                return
            reason = self.board.services.governor.skip(first=False)
            if reason:
                self.board.journal.forgo(
                    f"tuning {name}: {remaining} configurations", reason
                )
                return
            self.board.services.check()
            share = max(
                MIN_BATCH,
                round(self.search.configs_per_model * weights[index] / total),
            )
            size = min(share, remaining)
            remaining -= size
            self.ascend(name, strategy, optimizer, size, rungs)

    def ascend(
        self,
        name: str,
        strategy: str,
        optimizer: optimizers.Optimizer,
        size: int,
        rungs: list[int | None],
    ) -> None:
        """Evaluates `size` configurations through the rungs of one bracket.

        Args:
            name: Model being tuned.
            strategy: Data strategy the model is tuned under.
            optimizer: Source of configurations; told every score.
            size: Number of configurations to propose.
            rungs: Fold counts of the rungs, ending in `None` for all folds.
        """
        proposals = list(
            map(operator.call, itertools.repeat(optimizer.ask, size))
        )
        ids = self.board.journal.allocate(len(proposals))
        current = self.runner.dispatch(
            [
                trials.Job(trial_id, name, params, rungs[0], strategy)
                for trial_id, params in zip(ids, proposals, strict=True)
            ]
        )
        sign = 1 if self.metric.maximize else -1
        for rung in rungs[1:]:
            alive = [
                t for t in current if t.status is experiments.Termination.OK
            ]
            keep = max(1, math.ceil(len(alive) / self.search.eta))
            winners = sorted(alive, key=lambda t: -sign * t.score)[:keep]
            finished = {
                t.identifier: t
                for t in self.runner.dispatch(
                    [
                        trials.Job(
                            t.identifier, t.model, t.params, rung, t.strategy
                        )
                        for t in winners
                    ]
                )
            }
            for trial in current:
                if (
                    trial.status is experiments.Termination.OK
                    and trial.identifier not in finished
                ):
                    trial.status = experiments.Termination.PRUNED
                    trial.error = f"pruned before the {rung or 'full'} rung"
                    self.complete(trial, optimizer)
            current = [
                finished[t.identifier]
                for t in winners
                if t.identifier in finished
            ]
        for trial in current:
            if trial.status is experiments.Termination.OK:
                self.complete(trial, optimizer)
            else:
                self.board.journal.transcribe(trial)

    def complete(
        self, trial: experiments.Trial, optimizer: optimizers.Optimizer
    ) -> None:
        """Tells the optimizer the trial's score (if any) and records it.

        Successful and pruned trials are reported to the optimizer.

        Args:
            trial: The trial result.
            optimizer: The optimizer that proposed the configuration.
        """
        if trial.status in (
            experiments.Termination.OK,
            experiments.Termination.PRUNED,
        ):
            optimizer.tell(trial.params, trial.score)
        self.board.journal.transcribe(trial)
