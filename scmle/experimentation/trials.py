"""Trial execution: cross-validate complete candidate pipelines.

A trial is described by a picklable `TrialPayload` and evaluated by the
module-level `evaluate_payload`, so the same code runs in-process or in a
worker process depending on the execution backend.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import hashlib
import importlib.metadata
import platform
import time
from typing import Any

import numpy
import pandas
from sklearn import base
import threadpoolctl

from scmle.contracts import data
from scmle.contracts import experiments
from scmle.contracts import models
from scmle.contracts import tasks
import scmle.contracts.metrics
from scmle.core import limits
import scmle.core.context
from scmle.evaluation import metrics
from scmle.sampling import strategies


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
        weighted: Score folds with the feature plan's weight column.
    """

    job: TrialJob
    adapter: models.Model
    strategy: strategies.Strategy
    feature_plan: data.FeaturePlan
    task: tasks.Task
    seed: int
    metric: scmle.contracts.metrics.Metric
    positive_label: Any
    features: pandas.DataFrame
    target: pandas.Series
    folds: Sequence[tuple[list[int], list[int]]]
    thread_limit: int | None
    ceiling_seconds: float | None = None
    memory_gb: float | None = None
    weighted: bool = False


TRAIN_SCORE_ROWS = 2000


def access_record(row_ids: pandas.Index) -> dict[str, Any]:
    """Summarizes the rows a fit or scoring call received.

    Args:
        row_ids: Index of the rows.

    Returns:
        A dict with the row count and a 16-character digest of the sorted row
        ids.
    """
    ordered = sorted(map(str, row_ids))
    digest = hashlib.sha256("\x1f".join(ordered).encode()).hexdigest()
    return {"rows": len(ordered), "digest": digest[:16]}


def runtime_requirements(adapter: models.Model) -> dict[str, Any]:
    """Lists the Python version and the library versions a model needs.

    Args:
        adapter: The model adapter, whose `requires` names the libraries.

    Returns:
        A dict with `python` and `libraries` (name to installed version, or
        `unknown` if not installed).
    """
    needed = {}
    for module in adapter.requires:
        try:
            needed[module] = importlib.metadata.version(module)
        except importlib.metadata.PackageNotFoundError:
            needed[module] = "unknown"
    return {"python": platform.python_version(), "libraries": needed}


def fold_weights(
    payload: TrialPayload, positions: numpy.ndarray
) -> dict[str, numpy.ndarray]:
    """Builds scorer keyword arguments carrying the rows' weights.

    Args:
        payload: The trial payload.
        positions: Row positions in the development data.

    Returns:
        `{"sample_weight": ...}` with the weight column's values at those
        positions, or an empty dict when the trial is not weighted.
    """
    if not payload.weighted:
        return {}
    column = payload.feature_plan.weight_column
    values = payload.features[column].iloc[positions].to_numpy(dtype=float)
    return {"sample_weight": values}


def evaluate_payload(payload: TrialPayload) -> experiments.Trial:
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
    result = experiments.Trial(
        job.trial_id,
        job.model_name,
        experiments.TrialStatus.FAILED,
        metric.name,
        params=dict(job.params),
        strategy=job.strategy,
        evaluation_config={
            "metric": metric.name,
            "folds": len(folds),
            "seed": payload.seed,
            "weighted": payload.weighted,
        },
        runtime=runtime_requirements(payload.adapter),
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
        raw_train_scores = []
        with threadpoolctl.threadpool_limits(limits=payload.thread_limit):
            for fit_list, evaluate_list in folds:
                fit = numpy.asarray(fit_list)
                evaluate = numpy.asarray(evaluate_list)
                candidate = base.clone(pipeline)
                candidate.fit(
                    payload.features.iloc[fit], payload.target.iloc[fit]
                )
                sample = fit[:TRAIN_SCORE_ROWS]
                raw_scores.append(
                    scorer(
                        candidate,
                        payload.features.iloc[evaluate],
                        payload.target.iloc[evaluate],
                        **fold_weights(payload, evaluate),
                    )
                )
                raw_train_scores.append(
                    scorer(
                        candidate,
                        payload.features.iloc[sample],
                        payload.target.iloc[sample],
                        **fold_weights(payload, sample),
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
        result.train_scores = [
            metric.to_natural(float(s)) for s in raw_train_scores
        ]
        result.score_mean = float(numpy.mean(scores))
        result.score_std = float(numpy.std(scores))
        result.status = experiments.TrialStatus.OK
    except Exception as error:  # noqa: BLE001 -- isolation: any model failure is data
        result.error = f"{type(error).__name__}: {error}"
    result.fit_seconds = time.monotonic() - started
    return result


def lost_trial(payload: TrialPayload, error: Exception) -> experiments.Trial:
    """Builds a failed result for a trial whose worker was lost.

    Args:
        payload: The payload that was being evaluated.
        error: The error that reported the loss.

    Returns:
        A `FAILED` result whose error starts with `WorkerLost`.
    """
    job = payload.job
    return experiments.Trial(
        job.trial_id,
        job.model_name,
        experiments.TrialStatus.FAILED,
        payload.metric.name,
        params=dict(job.params),
        strategy=job.strategy,
        error=f"WorkerLost: {type(error).__name__}: {error}",
    )


def guarded_evaluate(payload: TrialPayload) -> experiments.Trial:
    """Evaluates a job in a supervised process under hard limits.

    A trial killed for exceeding a limit, or lost with its process, becomes a
    `FAILED` result so the run continues.

    Args:
        payload: Self-contained description of the work.

    Returns:
        The trial result, `FAILED` with a `TrialLimitError` message if a limit
        was exceeded.
    """
    supervisor = limits.Supervisor(payload.ceiling_seconds, payload.memory_gb)
    try:
        return supervisor.run(evaluate_payload, payload)
    except limits.TrialLimitError as error:
        job = payload.job
        return experiments.Trial(
            job.trial_id,
            job.model_name,
            experiments.TrialStatus.FAILED,
            payload.metric.name,
            params=dict(job.params),
            strategy=job.strategy,
            error=f"TrialLimitError: {error}",
        )


class TrialRunner:
    """Evaluates candidate pipelines with failure isolation.

    Attributes:
        context: Run context.
        split: The data split.
        task: The resolved task.
        metric: The metric being optimized.
        feature_plan: Feature columns by kind.
        folds: Cross-validation folds as lists of positions.
    """

    def __init__(self, context: scmle.core.context.Context) -> None:
        """Initializes the runner from a context holding a split and plan.

        Args:
            context: Run context with `split`, `profile`, `task` and `metric`.
        """
        self.context = context
        self.split = scmle.core.context.require(context.split, "split")
        self.task = scmle.core.context.require(context.task, "task")
        self.metric = scmle.core.context.require(context.metric, "metric")
        self.feature_plan = scmle.core.context.require(
            context.profile, "profile"
        ).feature_plan
        self.folds = [
            (list(train), list(val)) for train, val in self.split.manifest.folds
        ]

    @property
    def workers(self) -> int:
        """The concurrency of the execution backend."""
        return self.context.backend.workers

    def payload(self, job: TrialJob) -> TrialPayload:
        """Builds the self-contained payload for a job.

        Args:
            job: The job.

        Returns:
            The payload, with scoring weighted only if the selection spec asks
            for it, a weight column exists and the metric accepts weights.
        """
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
            weighted=bool(
                context.spec.selection.weighted
                and self.feature_plan.weight_column
                and self.metric.accepts_weights()
            ),
        )

    def run(
        self,
        trial_id: str,
        model_name: str,
        params: Mapping[str, Any] | None = None,
        fold_count: int | None = None,
    ) -> experiments.Trial:
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

    def run_many(self, jobs: Sequence[TrialJob]) -> list[experiments.Trial]:
        """Evaluates independent jobs, concurrently if the backend allows.

        Checkpointed results from an interrupted run are reused; only the
        remaining jobs are dispatched.

        Args:
            jobs: Independent jobs.

        Returns:
            Results in the order of `jobs`.
        """
        results: list[experiments.Trial | None] = [None] * len(jobs)
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
        outputs = self.context.backend.map(function, payloads, lost_trial)
        for index, output in zip(slots, outputs, strict=True):
            results[index] = output
        return [r for r in results if r is not None]

    def reuse(self, job: TrialJob) -> experiments.Trial | None:
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
        return experiments.Trial(
            job.trial_id,
            job.model_name,
            experiments.TrialStatus.OK,
            checkpoint.metric,
            float(numpy.mean(scores)),
            float(numpy.std(scores)),
            list(scores),
            checkpoint.fit_seconds,
            params=dict(checkpoint.params),
            strategy=checkpoint.strategy,
            observed_access=list(checkpoint.observed_access[:wanted]),
            evaluation_config={
                **checkpoint.evaluation_config,
                "folds": wanted,
            },
            runtime=dict(checkpoint.runtime),
            train_scores=list(checkpoint.train_scores[:wanted]),
        )

    def skip(
        self, trial_id: str, model_name: str, reason: str
    ) -> experiments.Trial:
        """Creates a `SKIPPED` result.

        Args:
            trial_id: Identifier for the result.
            model_name: Registered model that was not evaluated.
            reason: Why it was skipped.

        Returns:
            The result, with the reason as its error text.
        """
        return experiments.Trial(
            trial_id,
            model_name,
            experiments.TrialStatus.SKIPPED,
            self.metric.name,
            error=reason,
        )
