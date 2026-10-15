"""Mutable state shared by the stages of one run."""

import dataclasses
import threading
import time
from typing import Any

import pandas as pd

from mml.api import results
from mml.artifacts import bundle as bundle_module
from mml.artifacts import store as store_module
from mml.contracts import data as data_contracts
from mml.contracts import errors
from mml.contracts import events
from mml.contracts import experiments
from mml.contracts import tasks
from mml.data import splitting
from mml.evaluation import metrics
from mml.evaluation import selection
from mml.models import registry as registry_module


@dataclasses.dataclass
class RunContext:
    """Inputs, services and progressively produced artifacts of a run.

    Services and inputs are set at construction; the remaining fields are
    filled by stages in order. When `store` is None (plan preview) writes and
    events are skipped.

    Attributes:
        spec: Project specification.
        registry: Model registry.
        catalog: Metric catalog.
        policy: Model selection policy.
        store: Run directory, or None for previews.
        log: Event log, or None for previews.
        cancel_event: Set to request cancellation.
        input_frame: In-memory data, if the project was built from one.
        approved_plan: Human-approved plan restricting candidates.
        frame: Ingested dataset.
        manifest: Dataset identity.
        task: Resolved task type.
        profile: Dataset profile.
        split: Split result with development data and locked test.
        plan: Experiment plan.
        metric: Primary metric.
        trials: Trial results.
        selected: Winning trial.
        final_pipeline: Selected pipeline refit on development data.
        evaluation: Final evaluation record.
        bundle: Packaged model.
        result: Final run result.
        deadline: Monotonic deadline of the time budget.
    """

    spec: tasks.ProjectSpec
    registry: registry_module.ModelRegistry
    catalog: metrics.MetricCatalog
    policy: selection.SelectionPolicy
    store: store_module.RunStore | None = None
    log: events.EventLog | None = None
    cancel_event: threading.Event = dataclasses.field(
        default_factory=threading.Event
    )
    input_frame: pd.DataFrame | None = None
    approved_plan: experiments.ExperimentPlan | None = None
    frame: pd.DataFrame | None = None
    manifest: data_contracts.DatasetManifest | None = None
    task: tasks.TaskType | None = None
    profile: data_contracts.DatasetProfile | None = None
    split: splitting.SplitResult | None = None
    plan: experiments.ExperimentPlan | None = None
    metric: metrics.Metric | None = None
    trials: list[experiments.TrialResult] = dataclasses.field(default_factory=list)
    selected: experiments.TrialResult | None = None
    final_pipeline: Any = None
    evaluation: dict[str, Any] | None = None
    bundle: bundle_module.ModelBundle | None = None
    result: results.RunResult | None = None
    started_at: float = dataclasses.field(default_factory=time.monotonic)

    @property
    def deadline(self) -> float:
        """Monotonic time at which the budget is exhausted."""
        return self.started_at + self.spec.resources.time_budget_seconds

    @property
    def run_directory(self) -> str:
        """Run directory as a string, empty for previews."""
        return str(self.store.directory) if self.store else ""

    def write_json(self, relative: str, payload: Any) -> None:
        """Writes a JSON artifact if the run is persisted."""
        if self.store is not None:
            self.store.write_json(relative, payload)

    def emit(
        self, kind: str, reason: str = "", artifacts: tuple[str, ...] = ()
    ) -> None:
        """Records an event if the run is persisted."""
        if self.log is not None and self.store is not None:
            self.log.emit(
                events.Event(
                    self.store.run_id,
                    kind,
                    reason,
                    self.spec.config_hash(),
                    artifacts,
                )
            )

    def check_cancelled(self) -> None:
        """Raises if cancellation was requested.

        Raises:
            CancelledError: Carrying the run directory in its details.
        """
        if self.cancel_event.is_set():
            raise errors.CancelledError(
                "run cancelled", {"run_dir": self.run_directory}
            )
