"""Workflow stages.

Each `Stage` subclass owns one state of the run lifecycle and implements
`execute`; the orchestrator treats them uniformly. Because stages only read
and write a `RunContext`, the ingest/profile/plan prefix is reused for the
side-effect-free plan preview.
"""

import abc
import time
from typing import ClassVar

from mml.api import results
from mml.artifacts import bundle as bundle_module
from mml.artifacts import report
from mml.contracts import errors
from mml.contracts import experiments
from mml.core import context as context_module
from mml.core import state as run_state
from mml.core import trials
from mml.data import preprocessing
from mml.data import profiling
from mml.data import readers
from mml.data import splitting
from mml.intelligence import planner as planner_module

LEAKAGE_MARKER = "leakage"


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
            context.frame = context.input_frame.copy()
        else:
            context.frame = readers.read_table(data.uri)
        context.manifest = readers.build_manifest(
            context.frame, data.target, data.uri
        )
        context.write_json("dataset_manifest.json", context.manifest.to_dict())


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
        target = context.frame[spec.data.target]
        context.task = profiling.TaskResolver(spec).resolve(target)
        context.metric = context.catalog.resolve(spec.metric, context.task)
        context.profile = profiling.DatasetProfiler(spec).profile(
            context.frame, context.manifest, context.task
        )
        context.write_json("dataset_profile.json", context.profile.to_dict())
        if spec.leakage_policy == "block":
            leaks = [
                r for r in context.profile.risk_findings if LEAKAGE_MARKER in r
            ]
            if leaks:
                raise errors.BlockedError(
                    "unresolved leakage risk: " + "; ".join(leaks)
                )


def build_plan(context: context_module.RunContext) -> experiments.ExperimentPlan:
    """Creates the experiment plan from a profiled context."""
    approved = context.approved_plan
    planner = planner_module.Planner(context.registry, context.catalog)
    return planner.plan(
        planner_module.PlanningContext(
            context.spec,
            context.profile,
            context.task,
            approved.candidate_models if approved is not None else None,
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
        manager = splitting.SplitManager(context.spec, context.task)
        context.split = manager.split(context.frame)
        context.write_json("validation_manifest.json", context.split.manifest.to_dict())
        context.plan = build_plan(context)
        context.write_json("execution_plan.json", context.plan.to_dict())
        context.write_json(
            "diagnostics/decision_log.json",
            [d.to_dict() for d in context.plan.decisions],
        )


class TrainStage(Stage):
    """Cross-validates every candidate within the time budget."""

    state = run_state.RunState.TRAINING

    def describe(self, context: context_module.RunContext) -> str:
        """Reports the number of candidates."""
        return f"{len(context.plan.candidate_models)} candidates"

    def execute(self, context: context_module.RunContext) -> None:
        """Runs trials in order, checkpointing each result as it finishes."""
        runner = trials.TrialRunner(context)
        for number, name in enumerate(context.plan.candidate_models, 1):
            context.check_cancelled()
            trial_id = f"trial_{number:03d}"
            if number > 1 and time.monotonic() >= context.deadline:
                outcome = runner.skip(
                    trial_id, name, "time budget exhausted before this trial started"
                )
            else:
                outcome = runner.run(trial_id, name)
            context.trials.append(outcome)
            path = f"trials/{trial_id}.json"
            context.write_json(path, outcome.to_dict())
            context.emit(f"trial.{outcome.status}", name, (path,))


class SelectStage(Stage):
    """Applies the selection policy."""

    state = run_state.RunState.SELECTING

    def describe(self, context: context_module.RunContext) -> str:
        """Names the policy."""
        return context.policy.description

    def execute(self, context: context_module.RunContext) -> None:
        """Chooses the winning trial."""
        context.selected = context.policy.select(
            context.trials, context.metric, context.registry
        )


class FinalEvaluationStage(Stage):
    """Refits the frozen candidate and opens the locked test set once."""

    state = run_state.RunState.TESTING

    def describe(self, context: context_module.RunContext) -> str:
        """Names the selected model."""
        return f"final one-time test evaluation of {context.selected.model_name}"

    def execute(self, context: context_module.RunContext) -> None:
        """Refits on all development data and evaluates on the test set."""
        context.check_cancelled()
        split = context.split
        pipeline = preprocessing.build_pipeline(
            context.profile.feature_plan,
            context.registry.get(context.selected.model_name),
            context.task,
            context.spec.validation.seed,
        )
        pipeline.fit(split.dev_features, split.dev_target)
        test_features, test_target = split.locked_test.open(
            split.locked_test.issue_token()
        )
        baseline = next(
            (
                t
                for t in context.trials
                if t.model_name == "dummy"
                and t.status is experiments.TrialStatus.OK
            ),
            None,
        )
        selected = context.selected
        context.final_pipeline = pipeline
        context.evaluation = {
            "run_id": context.store.run_id if context.store else "",
            "task": context.task.value,
            "metric": context.metric.name,
            "metric_greater_is_better": context.metric.greater_is_better,
            "selection_policy": context.policy.description,
            "selected_model": selected.model_name,
            "selected_cv_score": selected.score_mean,
            "selected_cv_std": selected.score_std,
            "baseline_cv_score": baseline.score_mean if baseline else None,
            "test_metrics": context.catalog.evaluate(
                pipeline, test_features, test_target, context.task
            ),
            "n_dev": len(split.dev_features),
            "n_test": len(test_features),
            "split_fingerprint": split.manifest.fingerprint,
            "split_strategy": split.manifest.strategy,
            "test_set_opened": split.locked_test.opened,
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
        store = context.store
        bundle = bundle_module.ModelBundle.create(
            context.final_pipeline,
            context.profile.feature_plan,
            context.split.dev_features,
            context.spec.data.target,
            context.task.value,
            context.metric.name,
            context.selected.model_name,
        )
        bundle.save(store.path("model"))
        store.write_json("evaluation.json", context.evaluation)
        result = results.RunResult(
            run_id=store.run_id,
            run_dir=store.directory,
            state=run_state.RunState.COMPLETED.value,
            plan=context.plan,
            profile=context.profile,
            trials=context.trials,
            selected_model=context.selected.model_name,
            evaluation=context.evaluation,
            bundle=bundle,
        )
        board = result.leaderboard()
        board.to_parquet(store.path("leaderboard.parquet"), index=False)
        store.write_text("report.html", report.HtmlReport().render(result, board))
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
