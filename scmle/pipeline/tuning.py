"""Budget-aware hyperparameter tuning with successive-halving pruning.

Screened models are ranked, the most promising tunable ones get a search, and
each batch of proposed configurations is first scored on a cheap rung (the
first cross-validation fold). Only the best fraction advances to full
cross-validation; the rest are recorded as pruned.
"""

import itertools
import math
import operator

from scmle.contracts import experiments
from scmle.optimization import optimizers
from scmle.pipeline import engine
from scmle.pipeline import hyperband
from scmle.pipeline import products
from scmle.pipeline import trials


class Tuner(hyperband.Hyperband):
    """Tunes the top screened models within the remaining time budget.

    The successive-halving schedule is here; the Hyperband schedule is
    `hyperband.Hyperband`'s.

    Attributes:
        board: Run board.
        runner: Trial runner used for every evaluation.
        search: The effective search specification.
        metric: The metric being optimized.
    """

    def __init__(self, board: engine.Board, runner: trials.Runner) -> None:
        """Initializes the tuner.

        Args:
            board: Run board (spec, registry, deadline).
            runner: Trial runner used for every evaluation, holding the
                metric.
        """
        self.board = board
        self.runner = runner
        self.search = board.services.spec.budgeted()
        self.metric = runner.metric

    def shortlist(
        self, screened: list[experiments.Trial]
    ) -> list[tuple[str, str]]:
        """Picks the best (model, strategy) per model, best models first.

        Only successful default-hyperparameter trials compete, under the
        default strategy or one the strategy search adopted on evidence
        (`strategy_search.Strategist.judge`), and only tunable models are
        kept.

        Args:
            screened: Trials from screening and strategy search.

        Returns:
            Up to `search.top_models` distinct (model, strategy) pairs.
        """
        sign = 1 if self.metric.greater_is_better else -1
        adopted = {
            d.technique
            for d in self.board.journal.outcomes
            if d.action is experiments.Edict.INCLUDE
        }
        ok = [
            t
            for t in screened
            if t.status is experiments.Termination.OK
            and not t.params
            and (
                t.strategy == experiments.DEFAULT_STRATEGY
                or f"{t.strategy}@{t.model_name}" in adopted
            )
        ]
        ranked = sorted(ok, key=lambda t: -sign * t.score_mean)
        registry = self.board.services.registry
        chosen: dict[str, str] = {}
        for trial in ranked:
            tunable = registry.get(trial.model_name).search_space is not None
            if tunable and trial.model_name not in chosen:
                chosen[trial.model_name] = trial.strategy
        return list(chosen.items())[: self.search.top_models]

    def tune(self, screened: list[experiments.Trial]) -> None:
        """Searches the most promising candidates; results go to the board.

        Args:
            screened: Default-configuration trials from earlier steps.
        """
        if self.search.top_models == 0:
            return
        for index, (name, strategy) in enumerate(self.shortlist(screened)):
            self.optimize(name, strategy, index)

    def optimize(self, name: str, strategy: str, index: int) -> None:
        """Runs batched, pruned search for one model under one strategy.

        Args:
            name: Model to tune.
            strategy: Data strategy the model is tuned under.
            index: Position of the model among those tuned; offsets the
                optimizer seed.
        """
        board, search = self.board, self.search
        space = products.require(
            board.services.registry.get(name).search_space,
            f"{name}.search_space",
        )
        optimizer = optimizers.instantiate(
            search.optimizer,
            space,
            board.services.spec.validation.seed + index,
            self.metric.greater_is_better,
        )
        if search.pruning == "hyperband":
            self.hyperband(name, strategy, optimizer)
            return
        batch_size = max(
            hyperband.MIN_BATCH, math.ceil(search.configs_per_model / 3)
        )
        remaining = search.configs_per_model
        while remaining > 0:
            reason = board.services.governor.skip(first=False)
            if reason:
                board.journal.forgo(
                    f"tuning {name}: {remaining} configurations", reason
                )
                return
            board.services.check()
            size = min(batch_size, remaining)
            remaining -= size
            self.batch(name, strategy, optimizer, size)

    def batch(
        self,
        name: str,
        strategy: str,
        optimizer: optimizers.Optimizer,
        size: int,
    ) -> None:
        """Scores a batch on the cheap rung, prunes, finishes the survivors.

        Both rungs are dispatched through the execution backend, so a batch
        runs concurrently when workers are available.

        Args:
            name: Model being tuned.
            strategy: Data strategy the model is tuned under.
            optimizer: Source of configurations; told every score.
            size: Number of configurations in the batch.
        """
        multi_fold = len(self.runner.folds) > 1
        rung_folds = 1 if multi_fold else None
        proposals = list(
            map(operator.call, itertools.repeat(optimizer.ask, size))
        )
        ids = self.board.journal.allocate(len(proposals))
        first_rung = self.runner.dispatch(
            [
                trials.Job(trial_id, name, params, rung_folds, strategy)
                for trial_id, params in zip(ids, proposals, strict=True)
            ]
        )
        survivors = self.promote(first_rung) if multi_fold else None
        finished = {}
        if survivors:
            winners = [t for t in first_rung if t.trial_id in survivors]
            finished = {
                t.trial_id: t
                for t in self.runner.dispatch(
                    [
                        trials.Job(
                            t.trial_id, t.model_name, t.params, None, t.strategy
                        )
                        for t in winners
                    ]
                )
            }
        for trial in first_rung:
            if trial.status is not experiments.Termination.OK:
                self.board.journal.transcribe(trial)
            elif survivors is None:
                self.complete(trial, optimizer)
            elif trial.trial_id in finished:
                self.complete(finished[trial.trial_id], optimizer)
            else:
                trial.status = experiments.Termination.PRUNED
                trial.error = "pruned after the first rung"
                self.complete(trial, optimizer)

    def promote(self, first_rung: list[experiments.Trial]) -> set[str]:
        """Selects the first-rung trials that advance.

        Args:
            first_rung: Results of the first rung.

        Returns:
            The ids of the best successful trials, keeping at least one and
            dropping the configured prune fraction.
        """
        ok = [t for t in first_rung if t.status is experiments.Termination.OK]
        keep = max(1, math.ceil(len(ok) * (1 - self.search.prune_fraction)))
        sign = 1 if self.metric.greater_is_better else -1
        ranked = sorted(ok, key=lambda t: -sign * t.score_mean)
        return {t.trial_id for t in ranked[:keep]}
