"""Budget-aware hyperparameter tuning with successive-halving pruning.

Screened models are ranked, the most promising tunable ones get a search, and
each batch of proposed configurations is first scored on a cheap rung (the
first cross-validation fold). Only the best fraction advances to full
cross-validation; the rest are recorded as pruned.
"""

import math

from mml.contracts import experiments
from mml.core import context as context_module
from mml.core import trials
from mml.optimization import optimizers

MIN_BATCH = 2
Status = experiments.TrialStatus


class Tuner:
    """Tunes the top screened models within the remaining time budget."""

    def __init__(
        self, context: context_module.RunContext, runner: trials.TrialRunner
    ) -> None:
        """Initializes the tuner.

        Args:
            context: Run context (spec, registry, metric, deadline).
            runner: Trial runner used for every evaluation.
        """
        self.context = context
        self.runner = runner
        self.search = context.spec.effective_search()
        self.metric = context_module.require(context.metric, "metric")

    def promising_candidates(
        self, screened: list[experiments.TrialResult]
    ) -> list[tuple[str, str]]:
        """Picks the best (model, strategy) per model, best models first.

        Only successful default-hyperparameter trials compete, across all
        strategies, and only tunable models are kept.

        Args:
            screened: Trials from screening and strategy search.

        Returns:
            Up to `search.top_models` distinct (model, strategy) pairs.
        """
        sign = 1 if self.metric.greater_is_better else -1
        ok = [t for t in screened if t.status is Status.OK and not t.params]
        ranked = sorted(ok, key=lambda t: -sign * t.score_mean)
        registry = self.context.registry
        chosen: dict[str, str] = {}
        for trial in ranked:
            tunable = registry.get(trial.model_name).search_space is not None
            if tunable and trial.model_name not in chosen:
                chosen[trial.model_name] = trial.strategy
        return list(chosen.items())[: self.search.top_models]

    def tune(self, screened: list[experiments.TrialResult]) -> None:
        """Searches the most promising candidates; results go to the context.

        Args:
            screened: Default-configuration trials from earlier phases.
        """
        if self.search.top_models == 0:
            return
        for index, (name, strategy) in enumerate(
            self.promising_candidates(screened)
        ):
            self.tune_model(name, strategy, index)

    def tune_model(self, name: str, strategy: str, index: int) -> None:
        """Runs batched, pruned search for one model under one strategy."""
        context, search = self.context, self.search
        space = context_module.require(
            context.registry.get(name).search_space, f"{name}.search_space"
        )
        optimizer = optimizers.create_optimizer(
            search.optimizer,
            space,
            context.spec.validation.seed + index,
            self.metric.greater_is_better,
        )
        batch_size = max(MIN_BATCH, math.ceil(search.configs_per_model / 3))
        remaining = search.configs_per_model
        while (
            remaining > 0
            and context.resource_governor.skip_reason(first=False) is None
        ):
            context.check_cancelled()
            size = min(batch_size, remaining)
            remaining -= size
            self.run_batch(name, strategy, optimizer, size)

    def run_batch(
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
        proposals = [optimizer.ask() for count in range(size)]
        ids = self.context.allocate_trial_ids(len(proposals))
        first_rung = self.runner.run_many(
            [
                trials.TrialJob(trial_id, name, params, rung_folds, strategy)
                for trial_id, params in zip(ids, proposals, strict=True)
            ]
        )
        survivors = self.survivor_ids(first_rung) if multi_fold else None
        finished = {}
        if survivors:
            winners = [t for t in first_rung if t.trial_id in survivors]
            finished = {
                t.trial_id: t
                for t in self.runner.run_many(
                    [
                        trials.TrialJob(
                            t.trial_id, t.model_name, t.params, None, t.strategy
                        )
                        for t in winners
                    ]
                )
            }
        for trial in first_rung:
            if trial.status is not Status.OK:
                self.context.record_trial(trial)
            elif survivors is None:
                self.complete(trial, optimizer)
            elif trial.trial_id in finished:
                self.complete(finished[trial.trial_id], optimizer)
            else:
                trial.status = Status.PRUNED
                trial.error = "pruned after the first rung"
                self.complete(trial, optimizer)

    def survivor_ids(
        self, first_rung: list[experiments.TrialResult]
    ) -> set[str]:
        """Returns the ids of first-rung trials that advance."""
        ok = [t for t in first_rung if t.status is Status.OK]
        keep = max(1, math.ceil(len(ok) * (1 - self.search.prune_fraction)))
        sign = 1 if self.metric.greater_is_better else -1
        ranked = sorted(ok, key=lambda t: -sign * t.score_mean)
        return {t.trial_id for t in ranked[:keep]}

    def complete(
        self, trial: experiments.TrialResult, optimizer: optimizers.Optimizer
    ) -> None:
        """Tells the optimizer the trial's score (if any) and records it."""
        if trial.status in (Status.OK, Status.PRUNED):
            optimizer.tell(trial.params, trial.score_mean)
        self.context.record_trial(trial)
