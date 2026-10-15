"""Stacking-ensemble search.

After screening, strategy search and tuning, the best configuration of each
distinct model is combined into a stack and cross-validated like any other
candidate. The ensemble is adopted only if it beats its best member.
"""


from mml.contracts import experiments
from mml.core import context as context_module
from mml.core import trials
from mml.models import ensembles

Status = experiments.TrialStatus
Action = experiments.DecisionAction
STACK = ensembles.StackingAdapter.name


class EnsembleSearch:
    """Assembles and evaluates a stacking candidate."""

    def __init__(
        self, context: context_module.RunContext, runner: trials.TrialRunner
    ) -> None:
        """Initializes the search.

        Args:
            context: Run context (spec, registry, metric, deadline).
            runner: Trial runner used for the evaluation.
        """
        self.context = context
        self.runner = runner
        self.metric = context_module.require(context.metric, "metric")

    def members(
        self, evaluated: list[experiments.TrialResult], size: int
    ) -> list[experiments.TrialResult]:
        """Picks the best default-strategy trial of each distinct model.

        Args:
            evaluated: All trials so far.
            size: Maximum number of members.

        Returns:
            Up to `size` trials of distinct, non-baseline, non-ensemble
            models, best first.
        """
        sign = 1 if self.metric.greater_is_better else -1
        eligible = [
            t
            for t in evaluated
            if t.status is Status.OK
            and t.strategy == experiments.DEFAULT_STRATEGY
            and t.model_name != STACK
            and self.context.registry.get(t.model_name).family != "baseline"
        ]
        best: dict[str, experiments.TrialResult] = {}
        for trial in sorted(eligible, key=lambda t: -sign * t.score_mean):
            best.setdefault(trial.model_name, trial)
        return list(best.values())[:size]

    def search(self, evaluated: list[experiments.TrialResult]) -> None:
        """Evaluates one stack and records the verdict.

        Args:
            evaluated: All trials produced before the ensemble phase.
        """
        size = self.context.spec.ensemble.size
        if (
            size < ensembles.MIN_MEMBERS
            or self.context.resource_governor.skip_reason(first=False)
        ):
            return
        members = self.members(evaluated, size)
        if len(members) < ensembles.MIN_MEMBERS:
            self.context.add_outcome(
                STACK, Action.REJECT, "fewer than two eligible distinct models"
            )
            return
        self.context.check_cancelled()
        params = {
            "members": [
                {"model": t.model_name, "params": dict(t.params)}
                for t in members
            ]
        }
        trial_id = self.context.allocate_trial_ids(1)[0]
        outcome = self.runner.run(trial_id, STACK, params)
        self.context.record_trial(outcome)
        self.judge(members[0], outcome, [t.model_name for t in members])

    def judge(
        self,
        leader: experiments.TrialResult,
        outcome: experiments.TrialResult,
        names: list[str],
    ) -> None:
        """Compares the stack with its best member."""
        if outcome.status is not Status.OK:
            self.context.add_outcome(
                STACK, Action.REJECT, f"trial {outcome.status}: {outcome.error}"
            )
            return
        sign = 1 if self.metric.greater_is_better else -1
        gain = sign * (outcome.score_mean - leader.score_mean)
        self.context.add_outcome(
            STACK,
            Action.INCLUDE if gain > 0 else Action.REJECT,
            f"stack of {', '.join(names)}: cv {self.metric.name} "
            f"{outcome.score_mean:.4f} vs best member {leader.score_mean:.4f} "
            f"({gain:+.4f} in the metric's favour)",
        )
