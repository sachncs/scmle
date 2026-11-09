"""Ensemble search: stacking and greedy blending.

After screening, strategy search and tuning, the best configuration of each
distinct model is combined (stacked, and blended with greedy weights) and
cross-validated like any other candidate. An ensemble is adopted only if it
beats its best member.
"""

from scmle.contracts import experiments
import scmle.core.context
from scmle.experimentation import trials
from scmle.models import ensembles

COMBINERS = (ensembles.Stacking.name, ensembles.Blending.name)


class EnsembleSearch:
    """Assembles and evaluates a stacking candidate.

    Attributes:
        context: Run context.
        runner: Trial runner used for the evaluation.
        metric: The metric being optimized.
    """

    def __init__(
        self, context: scmle.core.context.Context, runner: trials.TrialRunner
    ) -> None:
        """Initializes the search.

        Args:
            context: Run context (spec, registry, metric, deadline).
            runner: Trial runner used for the evaluation.
        """
        self.context = context
        self.runner = runner
        self.metric = scmle.core.context.require(context.metric, "metric")

    def members(
        self, evaluated: list[experiments.Trial], size: int
    ) -> list[experiments.Trial]:
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
            if t.status is experiments.TrialStatus.OK
            and t.strategy == experiments.DEFAULT_STRATEGY
            and t.model_name not in COMBINERS
            and self.context.registry.get(t.model_name).family != "baseline"
        ]
        best: dict[str, experiments.Trial] = {}
        for trial in sorted(eligible, key=lambda t: -sign * t.score_mean):
            best.setdefault(trial.model_name, trial)
        return list(best.values())[:size]

    def search(self, evaluated: list[experiments.Trial]) -> None:
        """Evaluates one stack and one blend and records the verdicts.

        Nothing is done if the configured ensemble size is below
        `ensembles.MIN_MEMBERS` or the resource governor asks to skip. With
        fewer than `ensembles.MIN_MEMBERS` eligible models, each combiner is
        rejected with that reason.

        Args:
            evaluated: All trials produced before the ensemble step.
        """
        size = self.context.spec.ensemble.size
        if (
            size < ensembles.MIN_MEMBERS
            or self.context.resource_governor.skip_reason(first=False)
        ):
            return
        members = self.members(evaluated, size)
        if len(members) < ensembles.MIN_MEMBERS:
            for combiner in COMBINERS:
                self.context.add_outcome(
                    combiner,
                    experiments.DecisionAction.REJECT,
                    "fewer than two eligible distinct models",
                )
            return
        params = {
            "members": [
                {"model": t.model_name, "params": dict(t.params)}
                for t in members
            ]
        }
        for combiner in COMBINERS:
            if self.context.resource_governor.skip_reason(first=False):
                return
            self.context.check_cancelled()
            trial_id = self.context.allocate_trial_ids(1)[0]
            outcome = self.runner.run(trial_id, combiner, params)
            self.context.record_trial(outcome)
            self.judge(
                combiner,
                members[0],
                outcome,
                [t.model_name for t in members],
            )

    def judge(
        self,
        combiner: str,
        leader: experiments.Trial,
        outcome: experiments.Trial,
        names: list[str],
    ) -> None:
        """Compares an ensemble with its best member and records the verdict.

        The ensemble is included if it beats the leader in the metric's favour,
        otherwise rejected; a trial that did not succeed is rejected.

        Args:
            combiner: Name of the ensemble kind.
            leader: The best member trial.
            outcome: The ensemble's trial result.
            names: Model names of the members, for the explanation.
        """
        if outcome.status is not experiments.TrialStatus.OK:
            self.context.add_outcome(
                combiner,
                experiments.DecisionAction.REJECT,
                f"trial {outcome.status}: {outcome.error}",
            )
            return
        sign = 1 if self.metric.greater_is_better else -1
        gain = sign * (outcome.score_mean - leader.score_mean)
        self.context.add_outcome(
            combiner,
            experiments.DecisionAction.INCLUDE
            if gain > 0
            else experiments.DecisionAction.REJECT,
            f"{combiner} of {', '.join(names)}: cv {self.metric.name} "
            f"{outcome.score_mean:.4f} vs best member {leader.score_mean:.4f} "
            f"({gain:+.4f} in the metric's favour)",
        )
