"""Evidence-based data-strategy search.

After screening, the top models are each combined with every applicable data
strategy (class weighting, resampling, feature engineering) and cross-validated
on the *original* validation folds. A strategy is adopted only if the shared
folds make it credibly better than the same model's default pipeline
(`odds`, `CONFIDENCE`); every verdict is written to the decision log.
"""

import math

from scmle.contracts import experiments
from scmle.evaluation import comparison
from scmle.pipeline import engine
from scmle.pipeline import trials
from scmle.preparation import strategies

# A variant of a model's default pipeline (a data strategy, tuned
# hyperparameters, an ensemble) is adopted only when the correlated Bayesian
# t-test on the shared folds (`comparison.pair`) puts the probability that it
# is better at `CONFIDENCE` or more: a one-sided test at ERROR, the 0.05 of
# the claims' one-sided tests, per variant. Adopting on any positive mean
# difference let the best of about 15 variants win on fold noise (C4,
# Is-this-a-good-customer, 1,148 rows: an undersampled booster at +0.010 in
# cross-validation had probability 0.62 of being better, and its pick
# changed from run to run with the trials that fitted in the budget). A
# Bonferroni share ERROR / m over the m variants of a run (0.995 at m = 10)
# was measured too: on five folds it also rejected the log transform of
# skewed columns on Marketing_Campaign (+0.013, probability 0.971), whose
# test ROC AUC it raised from 0.890 to 0.902, so it is not used.
ERROR = 0.05
CONFIDENCE = 1.0 - ERROR


def plain(trial: experiments.Trial) -> bool:
    """Whether a trial is its model's default pipeline.

    Args:
        trial: A trial.

    Returns:
        True for the default strategy with no hyperparameter overrides.
    """
    return trial.strategy == experiments.DEFAULT_STRATEGY and not trial.params


def odds(
    variant: experiments.Trial, reference: experiments.Trial, sign: int
) -> float:
    """The probability that a variant beats its reference.

    Args:
        variant: The variant's trial.
        reference: The trial it must beat, scored on the same folds.
        sign: 1 when larger scores are better, -1 otherwise.

    Returns:
        `comparison.pair`'s posterior probability that the variant is
        better; 0 when the two were not scored on the same, enough folds or
        a score is not finite.
    """
    first, second = variant.fold_scores, reference.fold_scores
    if len(first) != len(second) or len(first) < comparison.MIN_FOLDS:
        return 0.0
    if not all(math.isfinite(s) for s in [*first, *second]):
        return 0.0
    verdict = comparison.pair(
        [sign * s for s in first], [sign * s for s in second]
    )
    return verdict.superior


class Strategist:
    """Evaluates data strategies for the most promising screened models.

    Attributes:
        board: Run board.
        runner: Trial runner used for every evaluation.
        metric: The metric being optimized.
        task: The resolved task.
        profile: The data profile.
    """

    def __init__(self, board: engine.Board, runner: trials.Runner) -> None:
        """Initializes the search.

        Args:
            board: Run board (spec, registry, deadline).
            runner: Trial runner used for every evaluation, holding the
                metric, task and profile.
        """
        self.board = board
        self.runner = runner
        self.metric = runner.metric
        self.task = runner.task
        self.profile = runner.profile

    def search(self, screened: list[experiments.Trial]) -> None:
        """Runs the search; results and verdicts go to the board.

        Args:
            screened: Default-configuration trials from the screening step.
        """
        spec = self.board.services.spec.effective()
        if not spec.names or spec.top_models == 0:
            return
        registry = self.board.services.registry
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
            reason = self.board.services.governor.skip(first=False)
            if reason:
                self.board.journal.forgo(
                    f"strategy search for {leader.model_name}", reason
                )
                continue
            self.board.services.check()
            self.tune(leader, spec.names)

    def tune(self, leader: experiments.Trial, names: list[str]) -> None:
        """Tries every applicable strategy for one model.

        Inapplicable strategies are rejected with the reason; the others are
        evaluated as trials and judged against the leader. Each trial gets
        its ceiling when it is dispatched, and one that must not start is
        recorded as skipped (`trials.Runner.dispatch`).

        Args:
            leader: The model's default-pipeline trial.
            names: Names of the strategies to try.
        """
        adapter = self.board.services.registry.get(leader.model_name)
        chosen: list[str] = []
        for name in names:
            strategy = strategies.lookup(name)
            reason = strategy.applicability(self.profile, self.task, adapter)
            if reason is None:
                chosen.append(name)
            else:
                self.board.journal.enter(
                    f"{name}@{leader.model_name}",
                    experiments.Edict.REJECT,
                    f"not applicable: {reason}",
                )
        ids = self.board.journal.allocate(len(chosen))
        jobs = [
            trials.Job(trial_id, leader.model_name, {}, None, name)
            for trial_id, name in zip(ids, chosen, strict=True)
        ]
        for outcome in self.runner.dispatch(jobs):
            self.board.journal.transcribe(outcome)
            self.judge(leader, outcome)

    def judge(
        self, leader: experiments.Trial, outcome: experiments.Trial
    ) -> None:
        """Compares a strategy trial with its model's default pipeline.

        The strategy is included if the shared folds make it credibly better
        than the default (`odds` at least `CONFIDENCE`), otherwise rejected;
        a trial that did not succeed is rejected.

        Args:
            leader: The model's default-pipeline trial.
            outcome: The strategy trial's result.
        """
        technique = f"{outcome.strategy}@{outcome.model_name}"
        if outcome.status is not experiments.Termination.OK:
            self.board.journal.enter(
                technique,
                experiments.Edict.REJECT,
                f"trial {outcome.status}: {outcome.error}",
            )
            return
        sign = 1 if self.metric.greater_is_better else -1
        gain = sign * (outcome.score_mean - leader.score_mean)
        chance = odds(outcome, leader, sign)
        verdict = (
            experiments.Edict.INCLUDE
            if chance >= CONFIDENCE
            else experiments.Edict.REJECT
        )
        self.board.journal.enter(
            technique,
            verdict,
            f"cv {self.metric.name} {outcome.score_mean:.4f} vs default "
            f"{leader.score_mean:.4f} ({gain:+.4f} in the metric's favour; "
            f"probability better {chance:.3f}, needed {CONFIDENCE:.3f})",
        )
