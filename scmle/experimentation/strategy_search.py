"""Evidence-based data-strategy search.

After screening, the top models are each combined with every applicable data
strategy (class weighting, resampling, feature engineering) and cross-validated
on the *original* validation folds. A strategy is adopted only if it beats the
same model's default pipeline; every verdict is written to the decision log.
"""

from scmle.contracts import experiments
from scmle.core import context as context_module
from scmle.experimentation import trials
from scmle.sampling import strategies

Status = experiments.TrialStatus
Action = experiments.DecisionAction


def best_default_trials(
    screened: list[experiments.TrialResult],
    metric_greater_is_better: bool,
    limit: int,
    baseline_family_names: frozenset[str],
) -> list[experiments.TrialResult]:
    """Ranks successful default-pipeline trials best-first.

    Args:
        screened: Trials from screening.
        metric_greater_is_better: Orientation of the metric.
        limit: Number of models to keep.
        baseline_family_names: Models excluded from strategy search.

    Returns:
        Up to `limit` default-pipeline trials, best first.
    """
    sign = 1 if metric_greater_is_better else -1
    ok = [
        t
        for t in screened
        if t.status is Status.OK
        and not t.params
        and t.strategy == experiments.DEFAULT_STRATEGY
        and t.model_name not in baseline_family_names
    ]
    ranked = sorted(ok, key=lambda t: -sign * t.score_mean)
    return ranked[:limit]


class StrategySearch:
    """Evaluates data strategies for the most promising screened models.

    Attributes:
        context: Run context.
        runner: Trial runner used for every evaluation.
        metric: The metric being optimized.
        task: The resolved task.
        profile: The data profile.
    """

    def __init__(
        self, context: context_module.RunContext, runner: trials.TrialRunner
    ) -> None:
        """Initializes the search.

        Args:
            context: Run context (spec, registry, profile, deadline).
            runner: Trial runner used for every evaluation.
        """
        self.context = context
        self.runner = runner
        self.metric = context_module.require(context.metric, "metric")
        self.task = context_module.require(context.task, "task")
        self.profile = context_module.require(context.profile, "profile")

    def search(self, screened: list[experiments.TrialResult]) -> None:
        """Runs the search; results and verdicts go to the context.

        Args:
            screened: Default-configuration trials from the screening step.
        """
        spec = self.context.spec.effective_strategies()
        if not spec.names or spec.top_models == 0:
            return
        registry = self.context.registry
        baselines = frozenset(
            name
            for name, adapter in registry.adapters.items()
            if adapter.family == "baseline"
        )
        leaders = best_default_trials(
            screened, self.metric.greater_is_better, spec.top_models, baselines
        )
        for leader in leaders:
            if self.context.resource_governor.skip_reason(first=False):
                break
            self.context.check_cancelled()
            self.search_model(leader, spec.names)

    def search_model(
        self, leader: experiments.TrialResult, names: list[str]
    ) -> None:
        """Tries every applicable strategy for one model.

        Inapplicable strategies are rejected with the reason; the others are
        evaluated as trials and judged against the leader.

        Args:
            leader: The model's default-pipeline trial.
            names: Names of the strategies to try.
        """
        adapter = self.context.registry.get(leader.model_name)
        chosen: list[str] = []
        for name in names:
            strategy = strategies.get_strategy(name)
            reason = strategy.applicability(self.profile, self.task, adapter)
            if reason is None:
                chosen.append(name)
            else:
                self.context.add_outcome(
                    f"{name}@{leader.model_name}",
                    Action.REJECT,
                    f"not applicable: {reason}",
                )
        ids = self.context.allocate_trial_ids(len(chosen))
        jobs = [
            trials.TrialJob(trial_id, leader.model_name, {}, None, name)
            for trial_id, name in zip(ids, chosen, strict=True)
        ]
        for outcome in self.runner.run_many(jobs):
            self.context.record_trial(outcome)
            self.judge(leader, outcome)

    def judge(
        self, leader: experiments.TrialResult, outcome: experiments.TrialResult
    ) -> None:
        """Compares a strategy trial with its model's default pipeline.

        The strategy is included if it beats the default in the metric's favour,
        otherwise rejected; a trial that did not succeed is rejected.

        Args:
            leader: The model's default-pipeline trial.
            outcome: The strategy trial's result.
        """
        technique = f"{outcome.strategy}@{outcome.model_name}"
        if outcome.status is not Status.OK:
            self.context.add_outcome(
                technique,
                Action.REJECT,
                f"trial {outcome.status}: {outcome.error}",
            )
            return
        sign = 1 if self.metric.greater_is_better else -1
        gain = sign * (outcome.score_mean - leader.score_mean)
        verdict = Action.INCLUDE if gain > 0 else Action.REJECT
        self.context.add_outcome(
            technique,
            verdict,
            f"cv {self.metric.name} {outcome.score_mean:.4f} vs default "
            f"{leader.score_mean:.4f} ({gain:+.4f} in the metric's favour)",
        )
