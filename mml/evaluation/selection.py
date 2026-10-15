"""Model selection policies."""

import abc
from typing import ClassVar

from mml.contracts import errors
from mml.contracts import experiments
from mml.evaluation import metrics
from mml.models import registry as registry_module

SCORE_PRECISION = 12
BASELINE = "dummy"


class SelectionPolicy(abc.ABC):
    """Orders trials from best to worst and picks the winner.

    Attributes:
        description: Human-readable rule recorded in the evaluation report.
    """

    description: ClassVar[str]

    @abc.abstractmethod
    def rank(
        self,
        trials: list[experiments.TrialResult],
        metric: metrics.Metric,
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
            MagicMLError: If no trial succeeded.
        """

    def select(
        self,
        trials: list[experiments.TrialResult],
        metric: metrics.Metric,
        registry: registry_module.ModelRegistry,
    ) -> experiments.TrialResult:
        """Returns the best-ranked trial."""
        return self.rank(trials, metric, registry)[0]


class BestMeanPolicy(SelectionPolicy):
    """Best mean cross-validation score; ties go to the simpler candidate."""

    description = (
        "best mean CV score; ties -> cheaper model, then default strategy; "
        "dummy baseline excluded when other models ran"
    )

    def rank(
        self,
        trials: list[experiments.TrialResult],
        metric: metrics.Metric,
        registry: registry_module.ModelRegistry,
    ) -> list[experiments.TrialResult]:
        """Sorts successful trials by score, then simplicity."""
        ok = [t for t in trials if t.status is experiments.TrialStatus.OK]
        if not ok:
            raise errors.MagicMLError(
                "all trials failed", {"errors": [t.error for t in trials]}
            )
        pool = [t for t in ok if t.model_name != BASELINE] or ok
        sign = 1 if metric.greater_is_better else -1

        def order(trial: experiments.TrialResult) -> tuple:
            cost = registry.get(trial.model_name).capabilities.cost
            score = round(trial.score_mean, SCORE_PRECISION)
            simple = trial.strategy != experiments.DEFAULT_STRATEGY
            return (
                -sign * score,
                cost,
                simple,
                trial.model_name,
                trial.strategy,
                trial.trial_id,
            )

        return sorted(pool, key=order)
