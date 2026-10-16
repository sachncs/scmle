"""Workflow stages.

Each `Stage` subclass owns one state of the run lifecycle and implements
`execute`; the orchestrator treats them uniformly. Because stages only read
and write a `RunContext`, the ingest/profile prefix is reused for the
side-effect-free plan preview.
"""

import abc
import dataclasses
import json
import time
from typing import Any
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn import base

from mml.api import results
from mml.artifacts import bundle as bundle_module
from mml.artifacts import report
from mml.contracts import data as data_contracts
from mml.contracts import errors
from mml.contracts import experiments
from mml.contracts import tasks
from mml.core import compiler
from mml.core import context as context_module
from mml.core import state as run_state
from mml.data.profiling import profiler
from mml.data.readers import tabular
from mml.data.splitting import splitters
from mml.data.validation import target as target_module
from mml.evaluation import confidence
from mml.evaluation import conformal
from mml.evaluation import diagnostics
from mml.evaluation import fit_health
from mml.evaluation import resources
from mml.evaluation import suite
from mml.experimentation import ensemble_search
from mml.experimentation import strategy_search
from mml.experimentation import trials
from mml.experimentation import tuning
from mml.features import feature_graph
from mml.governance import ledger
from mml.intelligence import knowledge
from mml.intelligence import planner as planner_module
from mml.intelligence import recommendations
from mml.models import thresholds
from mml.monitoring import reference as reference_module
from mml.sampling import strategies

LEAKAGE_MARKER = "leakage"
BASELINE = "dummy"
require = context_module.require


class Stage(abc.ABC):
    """One step of the workflow.

    Attributes:
        state: Lifecycle state the run is in while the stage executes.
        requires: Stages that must run before this one (the workflow is
            ordered by `compiler.compile_workflow`).
    """

    state: ClassVar[run_state.RunState]
    requires: ClassVar[tuple[type, ...]] = ()

    @abc.abstractmethod
    def describe(self, context: context_module.RunContext) -> str:
        """Returns the reason recorded on entering the stage."""

    @abc.abstractmethod
    def execute(self, context: context_module.RunContext) -> None:
        """Performs the stage, reading and writing the context."""


PARITY_ROWS = 200


class IngestStage(Stage):
    """Reads the dataset and fingerprints it."""

    state = run_state.RunState.VALIDATING

    def describe(self, context: context_module.RunContext) -> str:
        """Names the stage."""
        return "ingest dataset"

    def execute(self, context: context_module.RunContext) -> None:
        """Loads data (copying in-memory input) and builds the manifest."""
        data = context.spec.data
        started = time.monotonic()
        if context.input_frame is not None:
            frame = context.input_frame.copy()
        else:
            frame = tabular.read_table(data.uri)
        context.read_seconds = time.monotonic() - started
        context.manifest = tabular.build_manifest(frame, data.target, data.uri)
        frame, infinite = tabular.replace_infinities(frame)
        context.cleaning_notes = [
            f"column '{name}': {count} infinite value(s) treated as missing"
            for name, count in infinite.items()
        ]
        graph = context.spec.feature_graph
        if graph.nodes:
            feature_graph.validate_graph(graph, set(frame.columns), data)
            ordered = feature_graph.split_nodes(graph)[0]
            frame = feature_graph.SeriesFeatureBuilder(
                ordered, data.group_column, data.time_column
            ).apply(frame)
            context.emit(
                "features.graph",
                f"feature graph v{graph.version} {graph.fingerprint()}",
                data={
                    "fingerprint": graph.fingerprint(),
                    "nodes": [n.model_dump(mode="json") for n in graph.nodes],
                },
            )
        context.frame = frame
        self.refuse_stale_resume(context)
        context.write_json("dataset_manifest.json", context.manifest.to_dict())
        context.track(
            dataset_id=context.manifest.dataset_id,
            data_hash=context.manifest.data_hash,
        )

    def refuse_stale_resume(self, context: context_module.RunContext) -> None:
        """Rejects resuming trials computed on different data.

        Raises:
            ConfigError: If the run has checkpoints and the data hash differs
                from the one recorded when they were computed.
        """
        if context.store is None or not context.checkpoints:
            return
        saved = context.store.path("dataset_manifest.json")
        if not saved.exists():
            return
        previous = json.loads(saved.read_text()).get("data_hash")
        current = require(context.manifest, "manifest").data_hash
        if previous != current:
            raise errors.ConfigError(
                "cannot resume: the data changed since the checkpointed "
                "trials were computed",
                {"saved": previous, "current": current},
            )


class ProfileStage(Stage):
    """Resolves the task, splits the data, and profiles the development rows.

    Only the target is looked at to resolve the task and to split (class
    counts for stratification, groups and time for the other protocols). Every
    statistic that drives a modelling decision (column statistics, feature
    selection, leakage hypotheses, the fingerprint) is then computed on the
    development partition alone, so the locked test rows never influence
    them.
    """

    requires = (IngestStage,)
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
        target = frame[spec.data.target]
        if target.isna().any():
            raise errors.BlockedError(
                f"target '{target.name}' has {int(target.isna().sum())} "
                "missing values"
            )
        task = target_module.TaskResolver(spec).resolve(target)
        approved = context.approved_plan
        if approved is not None and approved.task != task.value:
            raise errors.ConfigError(
                f"the approved plan is for {approved.task} but the data "
                f"resolves to {task.value}"
            )
        context.task = task
        context.metric = context.catalog.resolve(spec.primary_metric, task)
        split = self.make_split(context, task, frame)
        development = split.dev_features.assign(
            **{spec.data.target: split.dev_target}
        )
        profile = profiler.DatasetProfiler(spec).profile(
            development,
            require(context.manifest, "manifest"),
            task,
            context.read_seconds,
            total_rows=len(frame),
        )
        profile = attach_feature_graph(spec, profile)
        if context.cleaning_notes:
            profile = dataclasses.replace(
                profile,
                quality_findings=profile.quality_findings
                + tuple(context.cleaning_notes),
            )
        context.profile = profile
        context.write_json("dataset_profile.json", profile.to_dict())
        context.emit(
            "features.selected",
            "feature set and exclusions fixed from the profile",
            ("dataset_profile.json",),
            {
                "numeric": list(profile.feature_plan.numeric),
                "categorical": list(profile.feature_plan.categorical),
                "dropped": dict(profile.feature_plan.dropped),
            },
        )
        context.track(
            fingerprint=knowledge.fingerprint(profile, task), task=task.value
        )
        if spec.leakage_policy == "block":
            leaks = [r for r in profile.risk_findings if LEAKAGE_MARKER in r]
            if leaks:
                raise errors.BlockedError(
                    "unresolved leakage risk: " + "; ".join(leaks)
                )

    def make_split(
        self,
        context: context_module.RunContext,
        task: tasks.TaskType,
        frame: pd.DataFrame,
    ) -> splitters.SplitResult:
        """Splits the dataset and records the manifest."""
        split = splitters.SplitManager(context.spec, task).split(frame)
        context.split = split
        context.write_json("validation_manifest.json", split.manifest.to_dict())
        context.track(split_fingerprint=split.manifest.fingerprint)
        context.emit(
            "split.created",
            f"{split.manifest.strategy} split",
            ("validation_manifest.json",),
            {
                "fingerprint": split.manifest.fingerprint,
                "strategy": split.manifest.strategy,
                "development_rows": len(split.manifest.train_ids),
                "test_rows": len(split.manifest.test_ids),
                "notes": list(split.manifest.notes),
            },
        )
        return split


def attach_feature_graph(
    spec: tasks.ProjectSpec, profile: data_contracts.DatasetProfile
) -> data_contracts.DatasetProfile:
    """Adds the feature graph to the profile's feature plan.

    Row-wise nodes become numeric model features computed inside the
    pipeline; ordered nodes are already columns of the profiled frame. Raw
    columns the graph reads but the model does not use are recorded as
    `extra_inputs`, so inference knows what the caller must supply.

    Args:
        spec: Project specification.
        profile: Profile whose plan lacks the graph.

    Returns:
        A profile whose feature plan carries the graph.
    """
    graph = spec.feature_graph
    if not graph.nodes:
        return profile
    ordered, rowwise = feature_graph.split_nodes(graph)
    plan = profile.feature_plan
    computed = {n.name for n in graph.nodes}
    needed = feature_graph.required_raw_inputs(graph.nodes, computed)
    if ordered:
        needed += [
            c
            for c in (spec.data.group_column, spec.data.time_column)
            if c and c not in needed
        ]
    used = set(plan.features) | {n.name for n in rowwise}
    extra = tuple(c for c in needed if c not in used)
    new_plan = dataclasses.replace(
        plan,
        numeric=plan.numeric + tuple(n.name for n in rowwise),
        derived=tuple(rowwise),
        series=tuple(ordered),
        extra_inputs=extra,
    )
    return dataclasses.replace(profile, feature_plan=new_plan)


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


def advise(context: context_module.RunContext) -> tuple[Any, ...]:
    """Derives and stores the recommendations for the user.

    Args:
        context: A context whose profile is populated.

    Returns:
        The recommendations, also kept on the context.
    """
    advice = recommendations.recommend(
        require(context.profile, "profile"), context.spec
    )
    context.recommendations = advice
    return advice


class PlanStage(Stage):
    """Builds the experiment plan from the development-set profile."""

    requires = (ProfileStage,)
    state = run_state.RunState.PLANNING

    def describe(self, context: context_module.RunContext) -> str:
        """Names the stage."""
        return "build execution plan"

    def execute(self, context: context_module.RunContext) -> None:
        """Plans candidates."""
        plan = build_plan(context)
        context.plan = plan
        context.write_json("execution_plan.json", plan.to_dict())
        context.write_json(
            "diagnostics/decision_log.json",
            {"plan": [d.to_dict() for d in plan.decisions], "outcomes": []},
        )
        advice = advise(context)
        context.write_json(
            "diagnostics/recommendations.json",
            [item.to_dict() for item in advice],
        )
        context.emit(
            "plan.recommendations",
            f"{len(advice)} recommendation(s) for the user",
            ("diagnostics/recommendations.json",),
        )


class TrainStage(Stage):
    """Cross-validates every candidate within the time budget."""

    requires = (PlanStage,)
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
            reason = context.resource_governor.skip_reason(first=start == 0)
            if reason is not None:
                outcomes = [
                    runner.skip(trial_id, name, reason)
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

    requires = (TrainStage,)
    state = run_state.RunState.SELECTING

    def describe(self, context: context_module.RunContext) -> str:
        """Names the policy."""
        return context.policy.description

    def announce(self, context: context_module.RunContext) -> None:
        """Audits that the selection is frozen before the test set opens."""
        selected = require(context.selected, "selected")
        context.emit(
            "selection.frozen",
            f"{selected.model_name}/{selected.strategy}",
            (f"trials/{selected.trial_id}.json",),
            {
                "trial_id": selected.trial_id,
                "params": selected.params,
                "cv_score": selected.score_mean,
            },
        )

    def limits(
        self, context: context_module.RunContext
    ) -> resources.ResourceLimits:
        """Builds the serving limits from the specification."""
        spec = context.spec.selection
        return resources.ResourceLimits(
            spec.max_model_mb,
            spec.max_p95_latency_ms,
            spec.latency_safety_factor,
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
            self.announce(context)
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
            self.announce(context)
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

    requires = (SelectStage,)
    state = run_state.RunState.TESTING

    def describe(self, context: context_module.RunContext) -> str:
        """Names the selected model."""
        selected = require(context.selected, "selected")
        return f"final one-time test evaluation of {selected.model_name}"

    def selected_health(
        self,
        context: context_module.RunContext,
        selected: experiments.TrialResult,
    ) -> dict[str, Any] | None:
        """Fit-health record of the selected trial, if it can be judged."""
        health = fit_health.assess_trial(
            selected,
            context.trials,
            require(context.metric, "metric").greater_is_better,
        )
        return None if health is None else health.to_dict()

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
        context.emit(
            "test.opened",
            "locked test set released once, after selection was frozen",
            data={
                "rows": len(test_features),
                "fingerprint": split.manifest.fingerprint,
            },
        )
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
                groups=(
                    test_features[context.spec.data.group_column]
                    if split.manifest.strategy == "group"
                    and context.spec.data.group_column
                    else None
                ),
                temporal=split.manifest.strategy == "temporal",
                reference_features=split.dev_features,
            ),
            suite.full_suite(),
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
            "fit_health": self.selected_health(context, selected),
            "cv_confidence": confidence.cv_interval(
                selected.fold_scores,
                float(np.mean([len(fold[0]) for fold in split.manifest.folds])),
                float(np.mean([len(fold[1]) for fold in split.manifest.folds])),
                split.manifest.strategy,
            ),
            "baseline_cv_score": baseline.score_mean if baseline else None,
            "test_metrics": test_metrics,
            "n_dev": len(split.dev_features),
            "n_test": len(test_features),
            "split_fingerprint": split.manifest.fingerprint,
            "split_strategy": split.manifest.strategy,
            "decision": decision,
            "resources": usage.to_dict(),
            "resource_limits": context.resource_governor.report(),
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

    requires = (FinalEvaluationStage,)
    state = run_state.RunState.PACKAGING

    def describe(self, context: context_module.RunContext) -> str:
        """Names the stage."""
        return "write bundle and reports"

    def conformal_calibration(
        self,
        context: context_module.RunContext,
        selected: experiments.TrialResult,
        bundle: bundle_module.ModelBundle,
    ) -> conformal.ConformalCalibrator | None:
        """Calibrates intervals or sets from out-of-fold predictions.

        The selected candidate is refit on each cross-validation fold's
        training rows and scored on its held-out rows, so every calibration
        score comes from a model that never saw that row.

        Returns:
            The calibrator, or None when the time budget is spent or too few
            held-out rows exist.
        """
        if context.resource_governor.skip_reason(first=False):
            return None
        split = require(context.split, "split")
        task = require(context.task, "task")
        template = strategies.get_strategy(selected.strategy).build(
            require(context.profile, "profile").feature_plan,
            context.registry.get(selected.model_name),
            task,
            context.spec.validation.seed,
            selected.params,
        )
        held_labels: list[Any] = []
        held_scores: list[Any] = []
        classes = list(bundle.classes or ())
        for fit_positions, evaluate_positions in split.manifest.folds:
            fit = np.asarray(fit_positions)
            evaluate = np.asarray(evaluate_positions)
            candidate = base.clone(template).fit(
                split.dev_features.iloc[fit], split.dev_target.iloc[fit]
            )
            rows = split.dev_features.iloc[evaluate]
            held_labels.extend(split.dev_target.iloc[evaluate])
            if task is tasks.TaskType.REGRESSION:
                held_scores.extend(candidate.predict(rows))
                continue
            known = list(candidate.classes_)
            probabilities = candidate.predict_proba(rows)
            held_scores.extend(
                [
                    probabilities[i, known.index(c)] if c in known else 0.0
                    for c in classes
                ]
                for i in range(len(rows))
            )
        if len(held_labels) < conformal.MIN_CALIBRATION_ROWS:
            return None
        if task is tasks.TaskType.REGRESSION:
            return conformal.ConformalCalibrator.for_regression(
                held_labels, held_scores
            )
        return conformal.ConformalCalibrator.for_classification(
            held_labels, np.asarray(held_scores), classes
        )

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
            graph_fingerprint=context.spec.feature_graph.fingerprint()
            if context.spec.feature_graph.nodes
            else "",
            group_column=context.spec.data.group_column,
            time_column=context.spec.data.time_column,
            history_source=context.frame,
            reference=reference_module.ReferenceProfile.build(
                split.dev_features,
                profile.feature_plan,
                require(context.reference_scores, "reference_scores"),
                str(require(context.reference_scores, "reference_scores").name),
                evaluation["test_metrics"],
            ),
        )
        bundle.conformal = self.conformal_calibration(context, selected, bundle)
        bundle.save(store.path("model"))
        evaluation["artifact_verification"] = bundle.verify_round_trip(
            store.path("model"), split.dev_features.head(PARITY_ROWS)
        )
        store.write_json(
            "diagnostics/data_access.json",
            ledger.build_ledger(
                split,
                context.trials,
                int(split.locked_test.opened),
                evaluation["decision"] is not None,
            ),
        )
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
        context.track(artifacts=self.lineage(store))
        board = result.leaderboard()
        board.to_parquet(store.path("leaderboard.parquet"), index=False)
        store.write_text(
            "report.html", report.HtmlReport().render(result, board)
        )
        context.bundle, context.result = bundle, result

    def lineage(self, store: Any) -> dict[str, str]:
        """Checksums of the artifacts a run produced, for tracking lineage."""
        names = (
            "config.json",
            "dataset_manifest.json",
            "evaluation.json",
            f"model/{bundle_module.MANIFEST_FILE}",
            f"model/{bundle_module.MODEL_FILE}",
        )
        return {
            name: bundle_module.file_sha256(store.path(name))
            for name in names
            if store.path(name).exists()
        }


def default_stages() -> tuple[Stage, ...]:
    """Returns the standard stage sequence, one per lifecycle state."""
    return compiler.compile_workflow(
        (
            IngestStage(),
            ProfileStage(),
            PlanStage(),
            TrainStage(),
            SelectStage(),
            FinalEvaluationStage(),
            PackageStage(),
        )
    )
