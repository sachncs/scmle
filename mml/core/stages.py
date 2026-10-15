"""Workflow stages.

Each `Stage` subclass owns one state of the run lifecycle and implements
`execute`; the orchestrator treats them uniformly. Because stages only read
and write a `RunContext`, the ingest/profile prefix is reused for the
side-effect-free plan preview.
"""

import abc
import time
from typing import Any
from typing import ClassVar

import pandas as pd

from mml.api import results
from mml.artifacts import bundle as bundle_module
from mml.artifacts import report
from mml.contracts import errors
from mml.contracts import experiments
from mml.contracts import tasks
from mml.core import context as context_module
from mml.core import ensemble_search
from mml.core import state as run_state
from mml.core import strategy_search
from mml.core import trials
from mml.core import tuning
from mml.data import profiling
from mml.data import readers
from mml.data import splitting
from mml.data import strategies
from mml.evaluation import diagnostics
from mml.evaluation import resources
from mml.intelligence import knowledge
from mml.intelligence import planner as planner_module
from mml.models import thresholds
from mml.monitoring import reference as reference_module

LEAKAGE_MARKER = "leakage"
BASELINE = "dummy"
require = context_module.require


class Stage(abc.ABC):
    """One step of the workflow.

    Attributes:
        state: Lifecycle state the run is in while the stage executes.
    """

    state: ClassVar[run_state.RunState]

    @abc.abstractmethod
    def describe(self, context: context_module.RunContext) -> str:
        """Returns the reason recorded on entering the stage."""

    @abc.abstractmethod
    def execute(self, context: context_module.RunContext) -> None:
        """Performs the stage, reading and writing the context."""


class IngestStage(Stage):
    """Reads the dataset and fingerprints it."""

    state = run_state.RunState.VALIDATING

    def describe(self, context: context_module.RunContext) -> str:
        """Names the stage."""
        return "ingest dataset"

    def execute(self, context: context_module.RunContext) -> None:
        """Loads data (copying in-memory input) and builds the manifest."""
        data = context.spec.data
        if context.input_frame is not None:
            frame = context.input_frame.copy()
        else:
            frame = readers.read_table(data.uri)
        context.frame = frame
        context.manifest = readers.build_manifest(frame, data.target, data.uri)
        context.write_json("dataset_manifest.json", context.manifest.to_dict())
        context.track(
            dataset_id=context.manifest.dataset_id,
            data_hash=context.manifest.data_hash,
        )


class ProfileStage(Stage):
    """Resolves the task and profiles the data."""

    state = run_state.RunState.PROFILING

    def describe(self, context: context_module.RunContext) -> str:
        """Names the stage."""
        return "profile dataset"

    def execute(self, context: context_module.RunContext) -> None:
        """Resolves task/metric, profiles, and enforces the leakage policy.

        Raises:
            BlockedError: When `leakage_policy` is `block` and a leakage
                hypothesis was raised.
        """
        spec = context.spec
        frame = require(context.frame, "frame")
        task = profiling.TaskResolver(spec).resolve(frame[spec.data.target])
        context.task = task
        context.metric = context.catalog.resolve(spec.metric, task)
        profile = profiling.DatasetProfiler(spec).profile(
            frame, require(context.manifest, "manifest"), task
        )
        context.profile = profile
        context.write_json("dataset_profile.json", profile.to_dict())
        context.track(
            fingerprint=knowledge.fingerprint(profile, task), task=task.value
        )
        if spec.leakage_policy == "block":
            leaks = [r for r in profile.risk_findings if LEAKAGE_MARKER in r]
            if leaks:
                raise errors.BlockedError(
                    "unresolved leakage risk: " + "; ".join(leaks)
                )


def build_plan(
    context: context_module.RunContext,
) -> experiments.ExperimentPlan:
    """Creates the experiment plan from a profiled context."""
    approved = context.approved_plan
    planner = planner_module.Planner(context.registry, context.catalog)
    return planner.plan(
        planner_module.PlanningContext(
            context.spec,
            require(context.profile, "profile"),
            require(context.task, "task"),
            approved.candidate_models if approved is not None else None,
            context.history,
        )
    )


class PlanStage(Stage):
    """Establishes the split manifest and the experiment plan."""

    state = run_state.RunState.PLANNING

    def describe(self, context: context_module.RunContext) -> str:
        """Names the stage."""
        return "build split manifest and execution plan"

    def execute(self, context: context_module.RunContext) -> None:
        """Splits the data and plans candidates."""
        manager = splitting.SplitManager(
            context.spec, require(context.task, "task")
        )
        split = manager.split(require(context.frame, "frame"))
        context.split = split
        context.write_json("validation_manifest.json", split.manifest.to_dict())
        context.track(split_fingerprint=split.manifest.fingerprint)
        plan = build_plan(context)
        context.plan = plan
        context.write_json("execution_plan.json", plan.to_dict())
        context.write_json(
            "diagnostics/decision_log.json",
            {"plan": [d.to_dict() for d in plan.decisions], "outcomes": []},
        )


class TrainStage(Stage):
    """Cross-validates every candidate within the time budget."""

    state = run_state.RunState.TRAINING

    def describe(self, context: context_module.RunContext) -> str:
        """Reports the number of candidates."""
        plan = require(context.plan, "plan")
        return f"{len(plan.candidate_models)} candidates"

    def execute(self, context: context_module.RunContext) -> None:
        """Screens candidates, then tunes the most promising ones.

        Candidates run in chunks of `backend.workers`; the budget is checked
        before every chunk but the first, and results are recorded in
        candidate order regardless of the backend.
        """
        runner = trials.TrialRunner(context)
        plan = require(context.plan, "plan")
        names = plan.candidate_models
        for start in range(0, len(names), runner.workers):
            context.check_cancelled()
            chunk = names[start : start + runner.workers]
            ids = context.allocate_trial_ids(len(chunk))
            if start > 0 and time.monotonic() >= context.deadline:
                outcomes = [
                    runner.skip(
                        trial_id,
                        name,
                        "time budget exhausted before this trial started",
                    )
                    for trial_id, name in zip(ids, chunk, strict=True)
                ]
            else:
                jobs = [
                    trials.TrialJob(trial_id, name)
                    for trial_id, name in zip(ids, chunk, strict=True)
                ]
                outcomes = runner.run_many(jobs)
            for outcome in outcomes:
                context.record_trial(outcome)
        strategy_search.StrategySearch(context, runner).search(
            list(context.trials)
        )
        tuning.Tuner(context, runner).tune(list(context.trials))
        ensemble_search.EnsembleSearch(context, runner).search(
            list(context.trials)
        )


class SelectStage(Stage):
    """Applies the selection policy and the serving constraints."""

    state = run_state.RunState.SELECTING

    def describe(self, context: context_module.RunContext) -> str:
        """Names the policy."""
        return context.policy.description

    def limits(
        self, context: context_module.RunContext
    ) -> resources.ResourceLimits:
        """Builds the serving limits from the specification."""
        spec = context.spec.selection
        return resources.ResourceLimits(
            spec.max_model_mb, spec.max_p95_latency_ms
        )

    def execute(self, context: context_module.RunContext) -> None:
        """Chooses the best trial that satisfies the serving constraints.

        Candidates are walked in policy order; each is refit and measured on
        this machine, and the first one within every limit wins.

        Raises:
            BlockedError: If no checked candidate satisfies the constraints.
        """
        ranked = context.policy.rank(
            context.trials, require(context.metric, "metric"), context.registry
        )
        limits = self.limits(context)
        if not limits.active:
            context.selected = ranked[0]
            return
        split = require(context.split, "split")
        task = require(context.task, "task")
        plan = require(context.profile, "profile").feature_plan
        checked = ranked[: context.spec.selection.max_candidates_checked]
        for trial in checked:
            candidate = strategies.get_strategy(trial.strategy).build(
                plan,
                context.registry.get(trial.model_name),
                task,
                context.spec.validation.seed,
                trial.params,
            )
            candidate.fit(split.dev_features, split.dev_target)
            usage = resources.measure(candidate, split.dev_features)
            broken = limits.violations(usage)
            label = f"{trial.model_name}/{trial.strategy}"
            if broken:
                context.add_outcome(
                    label, experiments.DecisionAction.REJECT, "; ".join(broken)
                )
                continue
            context.add_outcome(
                label,
                experiments.DecisionAction.INCLUDE,
                f"within serving limits ({usage.model_mb:.2f} MB, p95 "
                f"{usage.p95_latency_ms:.1f} ms on {usage.machine})",
            )
            context.refits[trial.trial_id] = candidate
            context.selected = trial
            return
        raise errors.BlockedError(
            f"none of the top {len(checked)} candidates satisfies the "
            "serving constraints; relax the limits or allow more candidates"
        )


def reference_scores(
    fitted: Any, features: pd.DataFrame, task: tasks.TaskType
) -> pd.Series:
    """Model outputs on held-out rows, used as the output-drift reference.

    Args:
        fitted: The final candidate.
        features: Held-out features.
        task: Resolved task type.

    Returns:
        Positive-class probabilities for binary tasks, predicted labels for
        multiclass tasks and predictions for regression.
    """
    if task is tasks.TaskType.BINARY:
        return pd.Series(fitted.predict_proba(features)[:, 1], name="score")
    return pd.Series(fitted.predict(features), name="prediction")


def choose_decision_threshold(
    context: context_module.RunContext, fitted: Any
) -> tuple[Any, dict[str, Any] | None]:
    """Optimizes the binary decision threshold on out-of-fold predictions.

    Only development data is used: each validation row is scored by a model
    that never saw it, so the locked test set stays untouched.

    Args:
        context: Run context holding the split and the selected trial.
        fitted: The selected candidate, already refit on development data.

    Returns:
        The estimator to evaluate (wrapped with the threshold when one was
        optimized) and a description of the decision, or None when disabled
        or not applicable.
    """
    decision = context.spec.decision
    task = require(context.task, "task")
    if decision.threshold_objective == "none" or task is not (
        tasks.TaskType.BINARY
    ):
        return fitted, None
    split = require(context.split, "split")
    selected = require(context.selected, "selected")
    strategy = strategies.get_strategy(selected.strategy)
    adapter = context.registry.get(selected.model_name)
    plan = require(context.profile, "profile").feature_plan
    seed = context.spec.validation.seed
    probabilities, positions = thresholds.out_of_fold_probabilities(
        lambda: strategy.build(plan, adapter, task, seed, selected.params),
        split.dev_features,
        split.dev_target,
        split.manifest.folds,
    )
    positive = fitted.classes_[1]
    actual = (split.dev_target.iloc[positions] == positive).to_numpy()
    kind = thresholds.OBJECTIVES[decision.threshold_objective]
    objective = (
        kind(decision.false_positive_cost, decision.false_negative_cost)
        if decision.threshold_objective == "cost"
        else kind()
    )
    threshold, score = thresholds.optimize_threshold(
        probabilities, actual, objective
    )
    description = {
        "objective": decision.threshold_objective,
        "threshold": threshold,
        "validation_score": score,
        "validation_rows": len(positions),
    }
    return thresholds.ThresholdClassifier.around_fitted(fitted, threshold), (
        description
    )


class FinalEvaluationStage(Stage):
    """Refits the frozen candidate and opens the locked test set once."""

    state = run_state.RunState.TESTING

    def describe(self, context: context_module.RunContext) -> str:
        """Names the selected model."""
        selected = require(context.selected, "selected")
        return f"final one-time test evaluation of {selected.model_name}"

    def execute(self, context: context_module.RunContext) -> None:
        """Refits on all development data and evaluates on the test set."""
        context.check_cancelled()
        split = require(context.split, "split")
        selected = require(context.selected, "selected")
        task = require(context.task, "task")
        metric = require(context.metric, "metric")
        pipeline = context.refits.get(selected.trial_id)
        if pipeline is None:
            pipeline = strategies.get_strategy(selected.strategy).build(
                require(context.profile, "profile").feature_plan,
                context.registry.get(selected.model_name),
                task,
                context.spec.validation.seed,
                selected.params,
            )
            pipeline.fit(split.dev_features, split.dev_target)
        pipeline, decision = choose_decision_threshold(context, pipeline)
        locked = split.locked_test
        test_features, test_target = locked.open(locked.issue_token())
        baseline = next(
            (
                t
                for t in context.trials
                if t.model_name == BASELINE
                and t.status is experiments.TrialStatus.OK
            ),
            None,
        )
        context.final_pipeline = pipeline
        usage = resources.measure(pipeline, split.dev_features)
        test_metrics = context.catalog.evaluate(
            pipeline, test_features, test_target, task
        )
        context.track(
            selected_strategy=selected.strategy,
            selected_model=selected.model_name,
            metric=metric.name,
            cv_score=selected.score_mean,
            test_metrics=test_metrics,
        )
        context.reference_scores = reference_scores(
            pipeline, test_features, task
        )
        context.diagnostics = diagnostics.run_suite(
            diagnostics.DiagnosticInput(
                pipeline,
                test_features,
                test_target,
                task,
                metric,
                require(context.profile, "profile").feature_plan,
                context.spec.validation.seed,
            )
        )
        for name, report_body in context.diagnostics.items():
            context.write_json(f"diagnostics/{name}.json", report_body)
        context.evaluation = {
            "run_id": context.store.run_id if context.store else "",
            "task": task.value,
            "metric": metric.name,
            "metric_greater_is_better": metric.greater_is_better,
            "selection_policy": context.policy.description,
            "selected_model": selected.model_name,
            "selected_trial_id": selected.trial_id,
            "selected_params": selected.params,
            "selected_strategy": selected.strategy,
            "selected_cv_score": selected.score_mean,
            "selected_cv_std": selected.score_std,
            "baseline_cv_score": baseline.score_mean if baseline else None,
            "test_metrics": test_metrics,
            "n_dev": len(split.dev_features),
            "n_test": len(test_features),
            "split_fingerprint": split.manifest.fingerprint,
            "split_strategy": split.manifest.strategy,
            "decision": decision,
            "resources": usage.to_dict(),
            "test_set_opened": locked.opened,
            "diagnostics": sorted(context.diagnostics),
            "elapsed_seconds": time.monotonic() - context.started_at,
            "reproducibility": (
                "same seed/data/environment replays; threaded backends may "
                "differ"
            ),
        }


class PackageStage(Stage):
    """Writes the model bundle, leaderboard and report."""

    state = run_state.RunState.PACKAGING

    def describe(self, context: context_module.RunContext) -> str:
        """Names the stage."""
        return "write bundle and reports"

    def execute(self, context: context_module.RunContext) -> None:
        """Packages artifacts and assembles the `RunResult`."""
        store = context.run_store
        split = require(context.split, "split")
        selected = require(context.selected, "selected")
        evaluation = require(context.evaluation, "evaluation")
        profile = require(context.profile, "profile")
        bundle = bundle_module.ModelBundle.create(
            context.final_pipeline,
            profile.feature_plan,
            split.dev_features,
            context.spec.data.target,
            require(context.task, "task").value,
            require(context.metric, "metric").name,
            selected.model_name,
            model_params=selected.params,
            strategy=selected.strategy,
            decision_threshold=(evaluation["decision"] or {}).get("threshold"),
            reference=reference_module.ReferenceProfile.build(
                split.dev_features,
                profile.feature_plan,
                require(context.reference_scores, "reference_scores"),
                str(require(context.reference_scores, "reference_scores").name),
                evaluation["test_metrics"],
            ),
        )
        bundle.save(store.path("model"))
        store.write_json(
            "diagnostics/decision_log.json",
            {
                "plan": [
                    d.to_dict() for d in require(context.plan, "plan").decisions
                ],
                "outcomes": [d.to_dict() for d in context.outcomes],
                "selection": {
                    "model": selected.model_name,
                    "strategy": selected.strategy,
                    "params": selected.params,
                    "policy": context.policy.description,
                },
            },
        )
        store.write_json("evaluation.json", evaluation)
        result = results.RunResult(
            run_id=store.run_id,
            run_dir=store.directory,
            state=run_state.RunState.COMPLETED.value,
            plan=require(context.plan, "plan"),
            profile=profile,
            trials=context.trials,
            selected_model=selected.model_name,
            evaluation=evaluation,
            bundle=bundle,
            diagnostics=context.diagnostics,
        )
        board = result.leaderboard()
        board.to_parquet(store.path("leaderboard.parquet"), index=False)
        store.write_text(
            "report.html", report.HtmlReport().render(result, board)
        )
        context.bundle, context.result = bundle, result


def default_stages() -> tuple[Stage, ...]:
    """Returns the standard stage sequence, one per lifecycle state."""
    return (
        IngestStage(),
        ProfileStage(),
        PlanStage(),
        TrainStage(),
        SelectStage(),
        FinalEvaluationStage(),
        PackageStage(),
    )
