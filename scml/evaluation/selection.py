"""Model selection policies."""

import abc
import math

from scml.contracts import errors
from scml.contracts import experiments
from scml.contracts import metrics as metric_contracts
from scml.contracts import tasks
from scml.evaluation import comparison
from scml.models import registry as registry_module

SCORE_PRECISION = 12
BASELINE = "dummy"


class SelectionPolicy(abc.ABC):
    """Orders trials from best to worst and picks the winner.

    Attributes:
        description: Human-readable rule recorded in the evaluation report.
    """

    description: str

    @abc.abstractmethod
    def rank(
        self,
        trials: list[experiments.TrialResult],
        metric: metric_contracts.Metric,
        registry: registry_module.ModelRegistry,
    ) -> list[experiments.TrialResult]:
        """Orders the eligible trials best first.

        Args:
            trials: All trials, any status.
            metric: The optimized metric.
            registry: Registry used to look up model costs.

        Returns:
            Successful, eligible trials, best first.

        Raises:
            ScmlError: If no trial succeeded.
        """

    def select(
        self,
        trials: list[experiments.TrialResult],
        metric: metric_contracts.Metric,
        registry: registry_module.ModelRegistry,
    ) -> experiments.TrialResult:
        """Returns the best-ranked trial."""
        return self.rank(trials, metric, registry)[0]


class BestMeanPolicy(SelectionPolicy):
    """Best mean cross-validation score, with an explicit tie policy.

    Candidates whose score is indistinguishable from the best (within
    `tie_tolerance`, and with `uncertainty="one_standard_error"` within one
    standard error of the best mean) are ordered by the configured
    `tie_breaker`; everyone else follows by score. The dummy baseline is
    excluded whenever another model ran.

    Attributes:
        rule: The selection settings.
    """

    def __init__(self, rule: tasks.SelectionSpec | None = None) -> None:
        """Initializes the policy.

        Args:
            rule: Selection settings; defaults to the plain best-mean rule.
        """
        self.rule = rule or tasks.SelectionSpec()
        self.description = self.describe(self.rule)

    @staticmethod
    def describe(rule: tasks.SelectionSpec) -> str:
        """Human-readable rule recorded in the evaluation report."""
        tie = f"{rule.tie_tolerance:g}" if rule.tie_tolerance else "exact"
        noise = {
            "one_standard_error": (
                "; scores within one standard error of the best are ties"
            ),
            "bayesian": (
                "; candidates are ties unless the best is better with "
                f"probability >= {rule.bayesian_confidence:g}"
            ),
        }.get(rule.uncertainty, "")
        return (
            f"best mean CV score; {tie} ties -> {rule.tie_breaker}{noise}; "
            "dummy baseline excluded when other models ran"
        )

    def probably_equivalent(
        self,
        best: experiments.TrialResult,
        other: experiments.TrialResult,
        sign: int,
    ) -> bool:
        """Whether the best candidate is not credibly better than `other`.

        Only used with `uncertainty="bayesian"` and when both candidates were
        scored on the same number of folds (at least three).
        """
        if self.rule.uncertainty != "bayesian":
            return False
        if (
            len(best.fold_scores) != len(other.fold_scores)
            or len(best.fold_scores) < comparison.MIN_FOLDS
        ):
            return False
        verdict = comparison.compare_paired(
            [sign * s for s in best.fold_scores],
            [sign * s for s in other.fold_scores],
            rope=self.rule.rope,
        )
        return verdict.probability_first_better < self.rule.bayesian_confidence

    def rank(
        self,
        trials: list[experiments.TrialResult],
        metric: metric_contracts.Metric,
        registry: registry_module.ModelRegistry,
    ) -> list[experiments.TrialResult]:
        """Sorts by score, resolving ties with the configured tie-breaker."""
        ok = [t for t in trials if t.status is experiments.TrialStatus.OK]
        if not ok:
            raise errors.ScmlError(
                "all trials failed", {"errors": [t.error for t in trials]}
            )
        pool = [t for t in ok if t.model_name != BASELINE] or ok
        sign = 1 if metric.greater_is_better else -1

        def by_score(trial: experiments.TrialResult) -> tuple:
            return (-sign * round(trial.score_mean, SCORE_PRECISION),)

        def tie_key(trial: experiments.TrialResult) -> tuple:
            cost = registry.get(trial.model_name).capabilities.cost
            plain = trial.strategy == experiments.DEFAULT_STRATEGY
            ordering = (
                (round(trial.score_std, SCORE_PRECISION),)
                if self.rule.tie_breaker == "lower_variance"
                else ()
            )
            return (
                *ordering,
                cost,
                not plain,
                trial.model_name,
                trial.strategy,
                trial.trial_id,
            )

        ordered = sorted(pool, key=lambda t: (*by_score(t), *tie_key(t)))
        best = ordered[0]
        folds = max(len(best.fold_scores), 1)
        tolerance = self.rule.tie_tolerance
        if self.rule.uncertainty == "one_standard_error":
            tolerance = max(tolerance, best.score_std / math.sqrt(folds))
        tied = [
            t
            for t in ordered
            if sign * (best.score_mean - t.score_mean) <= tolerance + 1e-12
            or self.probably_equivalent(best, t, sign)
        ]
        rest = [t for t in ordered if t not in tied]
        return sorted(tied, key=tie_key) + rest
