"""Evidence-based data-strategy search.

After screening, the top models are each combined with every applicable data
strategy (class weighting, resampling, feature engineering) and cross-validated
on the *original* validation folds. A strategy is adopted only if it beats the
same model's default pipeline; every verdict is written to the decision log.
"""

from scmle.contracts import experiments
from scmle.pipeline import trials
import scmle.pipeline.context
from scmle.preparation import strategies


class Strategist:
    """Evaluates data strategies for the most promising screened models.

    Attributes:
        context: Run context.
        runner: Trial runner used for every evaluation.
        metric: The metric being optimized.
        task: The resolved task.
        profile: The data profile.
    """

    def __init__(
        self, context: scmle.pipeline.context.Context, runner: trials.Runner
    ) -> None:
        """Initializes the search.

        Args:
            context: Run context (spec, registry, profile, deadline).
            runner: Trial runner used for every evaluation.
        """
        self.context = context
        self.runner = runner
        self.metric = scmle.pipeline.context.require(context.metric, "metric")
        self.task = scmle.pipeline.context.require(context.task, "task")
        self.profile = scmle.pipeline.context.require(
            context.profile, "profile"
        )

    def search(self, screened: list[experiments.Trial]) -> None:
        """Runs the search; results and verdicts go to the context.

        Args:
            screened: Default-configuration trials from the screening step.
        """
        spec = self.context.services.spec.effective()
        if not spec.names or spec.top_models == 0:
            return
        registry = self.context.services.registry
        baselines = frozenset(
            name
            for name, adapter in registry.adapters.items()
            if adapter.family == "baseline"
        )
        sign = 1 if self.metric.greater_is_better else -1
        default_trials = [
            t
            for t in screened
            if t.status is experiments.Termination.OK
            and not t.params
            and t.strategy == experiments.DEFAULT_STRATEGY
            and t.model_name not in baselines
        ]
        leaders = sorted(default_trials, key=lambda t: -sign * t.score_mean)[
            : spec.top_models
        ]
        for leader in leaders:
            reason = self.context.services.governor.skip_reason(first=False)
            if reason:
                self.context.journal.forgo(
                    f"strategy search for {leader.model_name}", reason
                )
                continue
            self.context.services.check()
            self.tune(leader, spec.names)

    def tune(self, leader: experiments.Trial, names: list[str]) -> None:
        """Tries every applicable strategy for one model.

        Inapplicable strategies are rejected with the reason; the others are
        evaluated as trials and judged against the leader. Each trial gets
        its ceiling when it is dispatched, and one that must not start is
        recorded as skipped (`trials.Runner.run_many`).

        Args:
            leader: The model's default-pipeline trial.
            names: Names of the strategies to try.
        """
        adapter = self.context.services.registry.get(leader.model_name)
        chosen: list[str] = []
        for name in names:
            strategy = strategies.lookup(name)
            reason = strategy.applicability(self.profile, self.task, adapter)
            if reason is None:
                chosen.append(name)
            else:
                self.context.journal.enter(
                    f"{name}@{leader.model_name}",
                    experiments.Edict.REJECT,
                    f"not applicable: {reason}",
                )
        ids = self.context.journal.allocate(len(chosen))
        jobs = [
            trials.Job(trial_id, leader.model_name, {}, None, name)
            for trial_id, name in zip(ids, chosen, strict=True)
        ]
        for outcome in self.runner.run_many(jobs):
            self.context.journal.transcribe(outcome)
            self.judge(leader, outcome)

    def judge(
        self, leader: experiments.Trial, outcome: experiments.Trial
    ) -> None:
        """Compares a strategy trial with its model's default pipeline.

        The strategy is included if it beats the default in the metric's favour,
        otherwise rejected; a trial that did not succeed is rejected.

        Args:
            leader: The model's default-pipeline trial.
            outcome: The strategy trial's result.
        """
        technique = f"{outcome.strategy}@{outcome.model_name}"
        if outcome.status is not experiments.Termination.OK:
            self.context.journal.enter(
                technique,
                experiments.Edict.REJECT,
                f"trial {outcome.status}: {outcome.error}",
            )
            return
        sign = 1 if self.metric.greater_is_better else -1
        gain = sign * (outcome.score_mean - leader.score_mean)
        verdict = (
            experiments.Edict.INCLUDE if gain > 0 else experiments.Edict.REJECT
        )
        self.context.journal.enter(
            technique,
            verdict,
            f"cv {self.metric.name} {outcome.score_mean:.4f} vs default "
            f"{leader.score_mean:.4f} ({gain:+.4f} in the metric's favour)",
        )
