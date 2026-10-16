"""Public API: `MLProject(...).plan()`, `.run()`, `result.predict()`."""

import pathlib
from typing import Any
from typing import Literal

import pandas as pd

from mml.api import results
from mml.artifacts import reproducibility
from mml.contracts import data as data_contracts
from mml.contracts import experiments
from mml.contracts import tasks
from mml.core import checkpoints as checkpoints_module
from mml.core import context as context_module
from mml.core import orchestrator as orchestrator_module
from mml.data.readers import interchange

ValidationStrategy = Literal[
    "auto", "stratified", "random", "group", "temporal"
]
LeakagePolicy = Literal["warn", "block"]


class MLProject:
    """A declarative ML project over a tabular dataset."""

    def __init__(
        self,
        data: Any,
        target: str,
        task: tasks.TaskRequest = "auto",
        metric: str | None = None,
        time_budget: int = 1800,
        *,
        workers: int = 1,
        enforcement: Literal["hard", "cooperative"] = "hard",
        id_columns: list[str] | None = None,
        group_column: str | None = None,
        time_column: str | None = None,
        excluded_columns: list[str] | None = None,
        unavailable_at_prediction: list[str] | None = None,
        embargo_rows: int = 0,
        validation: ValidationStrategy = "auto",
        folds: int = 5,
        test_fraction: float = 0.2,
        seed: int = 42,
        search: tasks.SearchSpec | None = None,
        strategies: tasks.StrategySpec | None = None,
        ensemble: tasks.EnsembleSpec | None = None,
        feature_graph: tasks.FeatureGraphSpec | None = None,
        selection: tasks.SelectionSpec | None = None,
        decision: tasks.DecisionSpec | None = None,
        leakage_policy: LeakagePolicy = "warn",
        checkpoints: tuple[checkpoints_module.PolicyCheckpoint, ...] = (),
        output_dir: str | pathlib.Path = "runs",
    ) -> None:
        """Defines a project.

        Args:
            data: CSV / Parquet / Feather path, or an in-memory table:
                pandas DataFrame, pyarrow Table, Polars frame, or anything
                implementing the Arrow stream protocol.
            target: Name of the target column.
            task: `auto`, `classification`, `regression`, or an explicit
                binary/multiclass task.
            metric: Primary metric; defaults per task.
            time_budget: Wall-clock budget in seconds.
            workers: Concurrent trial processes (1 = sequential).
            enforcement: `hard` supervises every trial in its own process and
                kills it past the time or memory limit; `cooperative` only
                stops new trials from starting (and allows in-process hooks).
            id_columns: Identifier columns excluded from features.
            group_column: Entity column that must not straddle splits.
            time_column: Column ordering rows in time.
            excluded_columns: Other columns to ignore.
            unavailable_at_prediction: Columns known only after the outcome
                (excluded; the feature-availability contract).
            embargo_rows: Rows dropped between training and validation
                windows in temporal validation.
            validation: Split strategy or `auto`.
            folds: Cross-validation folds.
            test_fraction: Share of rows locked away for the final test.
            seed: Random seed.
            search: Hyperparameter search settings; defaults to a small
                pruned random search. Use `SearchSpec(top_models=0)` to
                disable tuning.
            strategies: Which data strategies to evaluate; defaults to all.
                Use `StrategySpec(top_models=0)` to disable.
            ensemble: Stacking settings; `EnsembleSpec(size=0)` disables.
            feature_graph: Declarative derived features (row-wise, fitted and
                time-series operations), versioned with the configuration.
            selection: Serving constraints (max model size, p95 latency).
            decision: Threshold optimization for binary tasks (objective
                f1, accuracy, balanced_accuracy or an explicit cost).
            leakage_policy: `warn` or `block` on leakage hypotheses.
            checkpoints: Approval gates after planning and selection (the
                latter fires before the test set is opened).
            output_dir: Parent directory for run directories.

        Raises:
            pydantic.ValidationError: If the configuration is inconsistent.
        """
        frame = (
            interchange.to_pandas(data)
            if interchange.is_tabular(data)
            else None
        )
        self.spec = tasks.ProjectSpec(
            data=tasks.DataSpec(
                uri="<in-memory>" if frame is not None else str(data),
                target=target,
                id_columns=id_columns or [],
                group_column=group_column,
                time_column=time_column,
                excluded_columns=excluded_columns or [],
                unavailable_at_prediction=unavailable_at_prediction or [],
            ),
            task=task,
            metric=metric,
            validation=tasks.ValidationSpec(
                strategy=validation,
                embargo_rows=embargo_rows,
                folds=folds,
                test_fraction=test_fraction,
                seed=seed,
            ),
            resources=tasks.ResourceSpec(
                time_budget_seconds=time_budget,
                workers=workers,
                enforcement=enforcement,
            ),
            search=search or tasks.SearchSpec(),
            strategies=strategies or tasks.StrategySpec(),
            ensemble=ensemble or tasks.EnsembleSpec(),
            feature_graph=feature_graph or tasks.FeatureGraphSpec(),
            selection=selection or tasks.SelectionSpec(),
            decision=decision or tasks.DecisionSpec(),
            leakage_policy=leakage_policy,
        )
        self.output_dir = pathlib.Path(output_dir)
        self.orchestrator = orchestrator_module.Orchestrator(
            self.spec, frame, checkpoints=checkpoints
        )
        self.orchestrator.tracking_path = self.output_dir / "experiments.sqlite"
        self.profile: data_contracts.DatasetProfile | None = None

    def plan(self) -> experiments.ExperimentPlan:
        """Analyzes the data and proposes a workflow; trains nothing."""
        context = self.orchestrator.prepare()
        self.profile = context.profile
        return context_module.require(context.plan, "plan")

    def run(
        self, plan: experiments.ExperimentPlan | None = None
    ) -> results.RunResult:
        """Executes the workflow.

        Args:
            plan: An approved or modified plan restricting candidates.

        Returns:
            The completed run.
        """
        return self.orchestrator.run(self.output_dir, plan)

    @classmethod
    def resume(
        cls,
        run_directory: str | pathlib.Path,
        data: pd.DataFrame | None = None,
    ) -> results.RunResult:
        """Continues an interrupted run from its saved configuration.

        Completed trials are reused; the rest are computed.

        Args:
            run_directory: Directory of a cancelled, failed or crashed run.
            data: The original in-memory data, required only if the project
                was created from a DataFrame rather than a file.

        Returns:
            The completed run.
        """
        orchestrator = orchestrator_module.Orchestrator.for_run_directory(
            run_directory, frame=data
        )
        return orchestrator.resume(run_directory)

    @classmethod
    def verify_replay(
        cls,
        run_directory: str | pathlib.Path,
        data: pd.DataFrame | None = None,
        output_dir: str | pathlib.Path | None = None,
    ) -> reproducibility.ReplayVerdict:
        """Re-executes a finished run's configuration and compares the runs.

        Args:
            run_directory: A completed run directory.
            data: The original in-memory data, if the run used a DataFrame.
            output_dir: Where to write the rerun (default: next to the
                original).

        Returns:
            Whether the rerun is `exact`, `comparable` or `different`.
        """
        original = pathlib.Path(run_directory)
        orchestrator = orchestrator_module.Orchestrator.for_run_directory(
            original, frame=data
        )
        rerun = orchestrator.run(output_dir or original.parent)
        return reproducibility.compare_runs(
            reproducibility.load_run(original),
            reproducibility.load_run(rerun.run_dir),
        )

    def cancel(self) -> None:
        """Requests cancellation of a running workflow."""
        self.orchestrator.cancel()
