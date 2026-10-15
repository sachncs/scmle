"""Trial execution: cross-validate complete candidate pipelines.

A trial is described by a picklable `TrialPayload` and evaluated by the
module-level `evaluate_payload`, so the same code runs in-process or in a
worker process depending on the execution backend.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import hashlib
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn import base
import threadpoolctl

from mml.contracts import data as data_contracts
from mml.contracts import experiments
from mml.contracts import models
from mml.contracts import tasks
from mml.core import context as context_module
from mml.core import limits
from mml.data import strategies
from mml.evaluation import metrics


@dataclasses.dataclass(frozen=True, slots=True)
class TrialJob:
    """What to evaluate.

    Attributes:
        trial_id: Identifier of the resulting trial.
        model_name: Registered model to evaluate.
        params: Hyperparameter overrides.
        fold_count: Evaluate only the first N folds (a cheaper rung for
            pruning); None evaluates all folds.
        strategy: Name of the data strategy that builds the candidate.
    """

    trial_id: str
    model_name: str
    params: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    fold_count: int | None = None
    strategy: str = experiments.DEFAULT_STRATEGY


@dataclasses.dataclass(frozen=True, slots=True)
class TrialPayload:
    """Everything a worker needs to evaluate one job, with no shared state.

    Attributes:
        job: The job.
        adapter: Model adapter that builds the estimator.
        strategy: Data strategy that assembles the candidate pipeline.
        feature_plan: Feature columns by kind.
        task: Resolved task type.
        seed: Random seed.
        metric: Metric to optimize.
        positive_label: Positive class of a binary target, else None.
        features: Development features.
        target: Development labels.
        folds: Cross-validation folds (positions in the development data).
        thread_limit: Threads each worker may use (None = library default).
        ceiling_seconds: Hard time ceiling for the trial (None = none).
        memory_gb: Hard memory cap for the trial (None = none).
    """

    job: TrialJob
    adapter: models.ModelAdapter
    strategy: strategies.DataStrategy
    feature_plan: data_contracts.FeaturePlan
    task: tasks.TaskType
    seed: int
    metric: metrics.Metric
    positive_label: Any
    features: pd.DataFrame
    target: pd.Series
    folds: Sequence[tuple[list[int], list[int]]]
    thread_limit: int | None
    ceiling_seconds: float | None = None
    memory_gb: float | None = None


def access_record(row_ids: pd.Index) -> dict[str, Any]:
    """Row count and digest of the rows a fit or scoring call received."""
    ordered = sorted(map(str, row_ids))
    digest = hashlib.sha256("\x1f".join(ordered).encode()).hexdigest()
    return {"rows": len(ordered), "digest": digest[:16]}


def evaluate_payload(payload: TrialPayload) -> experiments.TrialResult:
    """Cross-validates one job; never raises for model failures.

    Args:
        payload: Self-contained description of the work.

    Returns:
        An `OK` result with fold scores, or a `FAILED` one describing the
        error, so one broken candidate never ends the run.
    """
    job, metric = payload.job, payload.metric
    folds = payload.folds[: job.fold_count]
    started = time.monotonic()
    result = experiments.TrialResult(
        job.trial_id,
        job.model_name,
        experiments.TrialStatus.FAILED,
        metric.name,
        params=dict(job.params),
        strategy=job.strategy,
    )
    try:
        pipeline = payload.strategy.build(
            payload.feature_plan,
            payload.adapter,
            payload.task,
            payload.seed,
            job.params,
        )
        scorer = metric.scorer(payload.positive_label)
        raw_scores = []
        with threadpoolctl.threadpool_limits(limits=payload.thread_limit):
            for fit_list, evaluate_list in folds:
                fit = np.asarray(fit_list)
                evaluate = np.asarray(evaluate_list)
                candidate = base.clone(pipeline)
                candidate.fit(
                    payload.features.iloc[fit], payload.target.iloc[fit]
                )
                raw_scores.append(
                    scorer(
                        candidate,
                        payload.features.iloc[evaluate],
                        payload.target.iloc[evaluate],
                    )
                )
                result.observed_access.append(
                    {
                        "fit": access_record(payload.features.index.take(fit)),
                        "evaluate": access_record(
                            payload.features.index.take(evaluate)
                        ),
                    }
                )
        scores = [metric.to_natural(float(s)) for s in raw_scores]
        result.fold_scores = scores
        result.score_mean = float(np.mean(scores))
        result.score_std = float(np.std(scores))
        result.status = experiments.TrialStatus.OK
    except Exception as error:  # Isolation: any model failure is data.
        result.error = f"{type(error).__name__}: {error}"
    result.fit_seconds = time.monotonic() - started
    return result


def guarded_evaluate(payload: TrialPayload) -> experiments.TrialResult:
    """Evaluates a job in a supervised process under hard limits.

    A trial killed for exceeding a limit, or lost with its process, becomes a
    `FAILED` result so the run continues.
    """
    supervisor = limits.TrialSupervisor(
        payload.ceiling_seconds, payload.memory_gb
    )
    try:
        return supervisor.run(evaluate_payload, payload)
    except limits.TrialLimitError as error:
        job = payload.job
        return experiments.TrialResult(
            job.trial_id,
            job.model_name,
            experiments.TrialStatus.FAILED,
            payload.metric.name,
            params=dict(job.params),
            strategy=job.strategy,
            error=f"TrialLimitError: {error}",
        )


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

    @property
    def workers(self) -> int:
        """Concurrency of the execution backend."""
        return self.context.backend.workers

    def payload(self, job: TrialJob) -> TrialPayload:
        """Builds the self-contained payload for a job."""
        context = self.context
        governor = context.resource_governor
        return TrialPayload(
            job=job,
            adapter=context.registry.get(job.model_name),
            strategy=strategies.get_strategy(job.strategy),
            feature_plan=self.feature_plan,
            task=self.task,
            seed=context.spec.validation.seed,
            metric=self.metric,
            positive_label=metrics.positive_label(
                self.split.dev_target, self.task
            ),
            features=self.split.dev_features,
            target=self.split.dev_target,
            folds=self.folds,
            thread_limit=governor.thread_limit,
            ceiling_seconds=governor.trial_ceiling_seconds(),
            memory_gb=context.spec.resources.memory_limit_gb,
        )

    def run(
        self,
        trial_id: str,
        model_name: str,
        params: Mapping[str, Any] | None = None,
        fold_count: int | None = None,
    ) -> experiments.TrialResult:
        """Evaluates a single candidate.

        Args:
            trial_id: Identifier for the result.
            model_name: Registered model to evaluate.
            params: Hyperparameter overrides.
            fold_count: Evaluate only the first N folds; None for all.

        Returns:
            The trial result.
        """
        job = TrialJob(trial_id, model_name, dict(params or {}), fold_count)
        return self.run_many([job])[0]

    def run_many(
        self, jobs: Sequence[TrialJob]
    ) -> list[experiments.TrialResult]:
        """Evaluates independent jobs, concurrently if the backend allows.

        Checkpointed results from an interrupted run are reused; only the
        remaining jobs are dispatched.

        Args:
            jobs: Independent jobs.

        Returns:
            Results in the order of `jobs`.
        """
        results: list[experiments.TrialResult | None] = [None] * len(jobs)
        slots: list[int] = []
        payloads: list[TrialPayload] = []
        for index, job in enumerate(jobs):
            cached = self.reuse(job)
            if cached is not None:
                results[index] = cached
            else:
                slots.append(index)
                payloads.append(self.payload(job))
        governor = self.context.resource_governor
        function = guarded_evaluate if governor.hard else evaluate_payload
        outputs = self.context.backend.map(function, payloads)
        for index, output in zip(slots, outputs, strict=True):
            results[index] = output
        return [r for r in results if r is not None]

    def reuse(self, job: TrialJob) -> experiments.TrialResult | None:
        """Returns a checkpointed result from an interrupted run, if valid.

        A checkpoint is reused only if the trial id, model and parameters
        match and it covers at least the requested folds; folds are
        deterministic, so a prefix of a full evaluation equals the cheaper
        rung that would have been computed.

        Args:
            job: The job about to run.

        Returns:
            A result view with exactly the requested folds, or None.
        """
        checkpoint = self.context.checkpoints.get(job.trial_id)
        usable = (experiments.TrialStatus.OK, experiments.TrialStatus.PRUNED)
        if (
            checkpoint is None
            or checkpoint.status not in usable
            or checkpoint.model_name != job.model_name
            or checkpoint.params != dict(job.params)
            or checkpoint.strategy != job.strategy
        ):
            return None
        wanted = len(self.folds) if job.fold_count is None else job.fold_count
        if len(checkpoint.fold_scores) < wanted:
            return None
        scores = checkpoint.fold_scores[:wanted]
        return experiments.TrialResult(
            job.trial_id,
            job.model_name,
            experiments.TrialStatus.OK,
            checkpoint.metric,
            float(np.mean(scores)),
            float(np.std(scores)),
            list(scores),
            checkpoint.fit_seconds,
            params=dict(checkpoint.params),
            strategy=checkpoint.strategy,
            observed_access=list(checkpoint.observed_access[:wanted]),
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
