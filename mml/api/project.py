"""Public API: `MLProject(...).plan()`, `.run()`, `result.predict()`."""

import pathlib

import pandas as pd

from mml.api import results
from mml.contracts import data as data_contracts
from mml.contracts import experiments
from mml.contracts import tasks
from mml.core import orchestrator as orchestrator_module


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
        validation: str = "auto",
        folds: int = 5,
        test_fraction: float = 0.2,
        seed: int = 42,
        leakage_policy: str = "warn",
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
            leakage_policy: `warn` or `block` on leakage hypotheses.
            output_dir: Parent directory for run directories.

        Raises:
            pydantic.ValidationError: If the configuration is inconsistent.
        """
        in_memory = isinstance(data, pd.DataFrame)
        self.spec = tasks.ProjectSpec(
            data=tasks.DataSpec(
                uri="<in-memory>" if in_memory else str(data),
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
            leakage_policy=leakage_policy,
        )
        self.output_dir = pathlib.Path(output_dir)
        self.orchestrator = orchestrator_module.Orchestrator(
            self.spec, data.copy() if in_memory else None
        )
        self.profile: data_contracts.DatasetProfile | None = None

    def plan(self) -> experiments.ExperimentPlan:
        """Analyzes the data and proposes a workflow; trains nothing."""
        context = self.orchestrator.prepare()
        self.profile = context.profile
        return context.plan

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

    def cancel(self) -> None:
        """Requests cancellation of a running workflow."""
        self.orchestrator.cancel()
