"""Trial execution: cross-validate one complete candidate pipeline."""

from collections.abc import Mapping
import time
from typing import Any

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
        self.split = context_module.require(context.split, "split")
        self.task = context_module.require(context.task, "task")
        self.metric = context_module.require(context.metric, "metric")
        self.feature_plan = context_module.require(
            context.profile, "profile"
        ).feature_plan
        self.folds = [
            (list(train), list(val)) for train, val in self.split.manifest.folds
        ]

    def run(
        self,
        trial_id: str,
        model_name: str,
        params: Mapping[str, Any] | None = None,
        fold_count: int | None = None,
    ) -> experiments.TrialResult:
        """Cross-validates one candidate; never raises for model failures.

        Args:
            trial_id: Identifier for the result.
            model_name: Registered model to evaluate.
            params: Hyperparameter overrides.
            fold_count: Evaluate only the first N folds (a cheaper rung for
                pruning); None evaluates all folds.

        Returns:
            An `OK` result with fold scores, or a `FAILED` one describing the
            error, so one broken candidate never ends the run.
        """
        context, metric = self.context, self.metric
        cached = self.reuse(trial_id, model_name, params, fold_count)
        if cached is not None:
            return cached
        folds = self.folds if fold_count is None else self.folds[:fold_count]
        started = time.monotonic()
        result = experiments.TrialResult(
            trial_id,
            model_name,
            experiments.TrialStatus.FAILED,
            metric.name,
            params=dict(params or {}),
        )
        try:
            pipeline = preprocessing.build_pipeline(
                self.feature_plan,
                context.registry.get(model_name),
                self.task,
                context.spec.validation.seed,
                params,
            )
            cv = model_selection.cross_validate(
                pipeline,
                self.split.dev_features,
                self.split.dev_target,
                cv=folds,
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

    def reuse(
        self,
        trial_id: str,
        model_name: str,
        params: Mapping[str, Any] | None,
        fold_count: int | None,
    ) -> experiments.TrialResult | None:
        """Returns a checkpointed result from an interrupted run, if valid.

        A checkpoint is reused only if the trial id, model and parameters
        match and it covers at least the requested folds; folds are
        deterministic, so a prefix of a full evaluation equals the cheaper
        rung that would have been computed.

        Args:
            trial_id: Identifier being evaluated.
            model_name: Model being evaluated.
            params: Hyperparameter overrides.
            fold_count: Requested folds, or None for all.

        Returns:
            A result view with exactly the requested folds, or None.
        """
        checkpoint = self.context.checkpoints.get(trial_id)
        usable = (experiments.TrialStatus.OK, experiments.TrialStatus.PRUNED)
        if (
            checkpoint is None
            or checkpoint.status not in usable
            or checkpoint.model_name != model_name
            or checkpoint.params != dict(params or {})
        ):
            return None
        wanted = len(self.folds) if fold_count is None else fold_count
        if len(checkpoint.fold_scores) < wanted:
            return None
        scores = checkpoint.fold_scores[:wanted]
        return experiments.TrialResult(
            trial_id,
            model_name,
            experiments.TrialStatus.OK,
            checkpoint.metric,
            float(np.mean(scores)),
            float(np.std(scores)),
            list(scores),
            checkpoint.fit_seconds,
            params=dict(checkpoint.params),
        )

    def skip(
        self, trial_id: str, model_name: str, reason: str
    ) -> experiments.TrialResult:
        """Creates a `SKIPPED` result."""
        return experiments.TrialResult(
            trial_id,
            model_name,
            experiments.TrialStatus.SKIPPED,
            self.metric.name,
            error=reason,
        )
