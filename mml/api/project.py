"""Public API: MLProject(...).plan() / .run() / result.predict()."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mml.api.results import RunResult
from mml.contracts.data import DatasetProfile
from mml.contracts.experiments import ExperimentPlan
from mml.contracts.tasks import (
    DataSpec,
    ProjectSpec,
    ResourceSpec,
    TaskRequest,
    ValidationSpec,
)
from mml.core.orchestrator import Orchestrator


class MLProject:
    def __init__(
        self,
        data: str | Path | pd.DataFrame,
        target: str,
        task: TaskRequest = "auto",
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
        output_dir: str | Path = "runs",
    ) -> None:
        self._frame = data.copy() if isinstance(data, pd.DataFrame) else None
        self.spec = ProjectSpec(
            data=DataSpec(
                uri="<in-memory>" if self._frame is not None else str(data),
                target=target,
                id_columns=id_columns or [],
                group_column=group_column,
                time_column=time_column,
                excluded_columns=excluded_columns or [],
            ),
            task=task,
            metric=metric,
            validation=ValidationSpec(
                strategy=validation, folds=folds, test_fraction=test_fraction, seed=seed
            ),
            resources=ResourceSpec(time_budget_seconds=time_budget),
            leakage_policy=leakage_policy,
        )
        self.output_dir = Path(output_dir)
        self._orch = Orchestrator(self.spec, self._frame)
        self.profile: DatasetProfile | None = None

    def plan(self) -> ExperimentPlan:
        """Analyze the data and propose a workflow. No training happens."""
        prepared = self._orch.prepare()
        self.profile = prepared.profile
        return prepared.plan

    def run(self, plan: ExperimentPlan | None = None) -> RunResult:
        """Execute the workflow; pass an approved/modified plan to govern what runs."""
        return self._orch.run(self.output_dir, plan)

    def cancel(self) -> None:
        self._orch.cancel()
