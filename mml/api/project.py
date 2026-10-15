"""Public API: `MLProject(...).plan()`, `.run()`, `result.predict()`."""

import pathlib
from typing import Literal

import pandas as pd

from mml.api import results
from mml.contracts import data as data_contracts
from mml.contracts import experiments
from mml.contracts import tasks
from mml.core import context as context_module
from mml.core import orchestrator as orchestrator_module

ValidationStrategy = Literal[
    "auto", "stratified", "random", "group", "temporal"
]
LeakagePolicy = Literal["warn", "block"]


class MLProject:
    """A declarative ML project over a tabular dataset."""

    def __init__(
        self,
        data: str | pathlib.Path | pd.DataFrame,
        target: str,
        task: tasks.TaskRequest = "auto",
        metric: str | None = None,
        time_budget: int = 1800,
        *,
        id_columns: list[str] | None = None,
        group_column: str | None = None,
        time_column: str | None = None,
        excluded_columns: list[str] | None = None,
        validation: ValidationStrategy = "auto",
        folds: int = 5,
        test_fraction: float = 0.2,
        seed: int = 42,
        search: tasks.SearchSpec | None = None,
        leakage_policy: LeakagePolicy = "warn",
        output_dir: str | pathlib.Path = "runs",
    ) -> None:
        """Defines a project.

        Args:
            data: CSV/Parquet path or an in-memory DataFrame.
            target: Name of the target column.
            task: `auto`, `classification`, `regression`, or an explicit
                binary/multiclass task.
            metric: Primary metric; defaults per task.
            time_budget: Wall-clock budget in seconds.
            id_columns: Identifier columns excluded from features.
            group_column: Entity column that must not straddle splits.
            time_column: Column ordering rows in time.
            excluded_columns: Other columns to ignore.
            validation: Split strategy or `auto`.
            folds: Cross-validation folds.
            test_fraction: Share of rows locked away for the final test.
            seed: Random seed.
            search: Hyperparameter search settings; defaults to a small
                pruned random search. Use `SearchSpec(top_models=0)` to
                disable tuning.
            leakage_policy: `warn` or `block` on leakage hypotheses.
            output_dir: Parent directory for run directories.

        Raises:
            pydantic.ValidationError: If the configuration is inconsistent.
        """
        frame = data.copy() if isinstance(data, pd.DataFrame) else None
        self.spec = tasks.ProjectSpec(
            data=tasks.DataSpec(
                uri="<in-memory>" if frame is not None else str(data),
                target=target,
                id_columns=id_columns or [],
                group_column=group_column,
                time_column=time_column,
                excluded_columns=excluded_columns or [],
            ),
            task=task,
            metric=metric,
            validation=tasks.ValidationSpec(
                strategy=validation,
                folds=folds,
                test_fraction=test_fraction,
                seed=seed,
            ),
            resources=tasks.ResourceSpec(time_budget_seconds=time_budget),
            search=search or tasks.SearchSpec(),
            leakage_policy=leakage_policy,
        )
        self.output_dir = pathlib.Path(output_dir)
        self.orchestrator = orchestrator_module.Orchestrator(self.spec, frame)
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

    def cancel(self) -> None:
        """Requests cancellation of a running workflow."""
        self.orchestrator.cancel()
