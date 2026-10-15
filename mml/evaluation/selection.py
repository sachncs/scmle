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
    """Chooses the winning trial.

    Attributes:
        description: Human-readable rule recorded in the evaluation report.
    """

    description: ClassVar[str]

    @abc.abstractmethod
    def select(
        self,
        trials: list[experiments.TrialResult],
        metric: metrics.Metric,
        registry: registry_module.ModelRegistry,
    ) -> experiments.TrialResult:
        """Selects the best trial.

        Args:
            trials: All trials, any status.
            metric: The optimized metric.
            registry: Registry used to look up model costs.

        Returns:
            The winner.

        Raises:
            MagicMLError: If no trial succeeded.
        """


class BestMeanPolicy(SelectionPolicy):
    """Best mean cross-validation score; ties go to the cheaper model."""

    description = (
        "best mean CV score; ties -> cheaper model; dummy baseline "
        "excluded when other models ran"
    )

    def select(
        self,
        trials: list[experiments.TrialResult],
        metric: metrics.Metric,
        registry: registry_module.ModelRegistry,
    ) -> experiments.TrialResult:
        """Applies the best-mean rule over successful trials."""
        ok = [t for t in trials if t.status is experiments.TrialStatus.OK]
        if not ok:
            raise errors.MagicMLError(
                "all trials failed", {"errors": [t.error for t in trials]}
            )
        pool = [t for t in ok if t.model_name != BASELINE] or ok
        sign = 1 if metric.greater_is_better else -1

        def rank(trial: experiments.TrialResult) -> tuple:
            cost = registry.get(trial.model_name).capabilities.cost
            score = round(trial.score_mean, SCORE_PRECISION)
            return (-sign * score, cost, trial.model_name)

        return min(pool, key=rank)
