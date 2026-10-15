"""Trial execution: cross-validate one complete candidate pipeline."""

import time

import numpy as np
from sklearn import model_selection

from mml.contracts import experiments
from mml.core import context as context_module
from mml.data import preprocessing


class TrialRunner:
    """Evaluates candidate pipelines with failure isolation."""

    def __init__(self, context: context_module.RunContext) -> None:
        """Initializes the runner from a context holding a split and plan.

        Args:
            context: Run context with `split`, `profile`, `task` and `metric`.
        """
        self.context = context
        self.folds = [
            (list(train), list(val)) for train, val in context.split.manifest.folds
        ]

    def run(self, trial_id: str, model_name: str) -> experiments.TrialResult:
        """Cross-validates one model; never raises for model failures.

        Args:
            trial_id: Identifier for the result.
            model_name: Registered model to evaluate.

        Returns:
            An `OK` result with fold scores, or a `FAILED` one describing the
            error, so one broken candidate never ends the run.
        """
        context = self.context
        metric = context.metric
        started = time.monotonic()
        result = experiments.TrialResult(
            trial_id, model_name, experiments.TrialStatus.FAILED, metric.name
        )
        try:
            pipeline = preprocessing.build_pipeline(
                context.profile.feature_plan,
                context.registry.get(model_name),
                context.task,
                context.spec.validation.seed,
            )
            cv = model_selection.cross_validate(
                pipeline,
                context.split.dev_features,
                context.split.dev_target,
                cv=self.folds,
                scoring=metric.scorer_name,
                error_score="raise",
            )
            scores = [metric.to_natural(float(s)) for s in cv["test_score"]]
            result.fold_scores = scores
            result.score_mean = float(np.mean(scores))
            result.score_std = float(np.std(scores))
            result.status = experiments.TrialStatus.OK
        except Exception as error:  # Isolation: any model failure is data.
            result.error = f"{type(error).__name__}: {error}"
        result.fit_seconds = time.monotonic() - started
        return result

    def skip(self, trial_id: str, model_name: str, reason: str) -> experiments.TrialResult:
        """Creates a `SKIPPED` result."""
        return experiments.TrialResult(
            trial_id,
            model_name,
            experiments.TrialStatus.SKIPPED,
            self.context.metric.name,
            error=reason,
        )
