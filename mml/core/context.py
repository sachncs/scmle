"""Mutable state shared by the stages of one run."""

import dataclasses
import threading
import time
from typing import Any
from typing import TypeVar

import pandas as pd

from mml.api import results
from mml.artifacts import bundle as bundle_module
from mml.artifacts import store as store_module
from mml.contracts import data as data_contracts
from mml.contracts import errors
from mml.contracts import events
from mml.contracts import experiments
from mml.contracts import tasks
from mml.core import limits
from mml.core import scheduler
from mml.data.splitting import splitters
from mml.evaluation import metrics
from mml.evaluation import selection
from mml.experimentation import stores as tracking_stores
from mml.models import registry as registry_module

T = TypeVar("T")


def require(value: T | None, name: str) -> T:
    """Returns a stage product, failing loudly if its stage has not run.

    Args:
        value: The possibly unset context field.
        name: Field name used in the error message.

    Returns:
        `value`, narrowed to non-None.

    Raises:
        MagicMLError: If the producing stage has not executed.
    """
    if value is None:
        raise errors.MagicMLError(
            f"'{name}' is unavailable: its stage has not run yet"
        )
    return value


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
        tracker: Experiment tracking backend (no-op by default).
        checkpoints: Completed trials from an interrupted run, by trial id,
            reused instead of recomputed when resuming.
        backend: Where independent trials run (sequential by default).
        governor: Applies and reports the resource limits.
        cancel_event: Set to request cancellation.
        input_frame: In-memory data, if the project was built from one.
        approved_plan: Human-approved plan restricting candidates.
        frame: Ingested dataset.
        read_seconds: Time spent reading the dataset.
        cleaning_notes: Data repairs made at ingestion (reported as quality
            findings).
        manifest: Dataset identity.
        task: Resolved task type.
        profile: Dataset profile.
        split: Split result with development data and locked test.
        plan: Experiment plan.
        metric: Primary metric.
        trials: Trial results.
        history: Past runs available for meta-learning.
        outcomes: Evidence-based decisions made while running (which
            strategies helped), appended to the planner's decision log.
        trial_counter: Number of trial identifiers handed out so far.
        selected: Winning trial.
        diagnostics: Post-selection diagnostic reports by name.
        reference_scores: Model scores / predicted classes on the held-out
            rows, the reference for output-drift monitoring.
        refits: Candidates already refit on development data during
            constraint checks, by trial id (reused by the final evaluation).
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
    tracker: tracking_stores.ExperimentStore = dataclasses.field(
        default_factory=tracking_stores.NullExperimentStore
    )
    checkpoints: dict[str, experiments.TrialResult] = dataclasses.field(
        default_factory=dict
    )
    backend: scheduler.ExecutionBackend = dataclasses.field(
        default_factory=scheduler.SequentialBackend
    )
    governor: limits.ResourceGovernor | None = None
    cancel_event: threading.Event = dataclasses.field(
        default_factory=threading.Event
    )
    input_frame: pd.DataFrame | None = None
    approved_plan: experiments.ExperimentPlan | None = None
    frame: pd.DataFrame | None = None
    read_seconds: float = 0.0
    cleaning_notes: list[str] = dataclasses.field(default_factory=list)
    manifest: data_contracts.DatasetManifest | None = None
    task: tasks.TaskType | None = None
    profile: data_contracts.DatasetProfile | None = None
    split: splitters.SplitResult | None = None
    plan: experiments.ExperimentPlan | None = None
    metric: metrics.Metric | None = None
    trials: list[experiments.TrialResult] = dataclasses.field(
        default_factory=list
    )
    history: tuple[experiments.HistoricalRun, ...] = ()
    outcomes: list[experiments.Decision] = dataclasses.field(
        default_factory=list
    )
    trial_counter: int = 0
    selected: experiments.TrialResult | None = None
    diagnostics: dict[str, dict[str, Any]] = dataclasses.field(
        default_factory=dict
    )
    reference_scores: pd.Series | None = None
    refits: dict[str, Any] = dataclasses.field(default_factory=dict)
    final_pipeline: Any = None
    evaluation: dict[str, Any] | None = None
    bundle: bundle_module.ModelBundle | None = None
    result: results.RunResult | None = None
    started_at: float = dataclasses.field(default_factory=time.monotonic)

    @property
    def resource_governor(self) -> limits.ResourceGovernor:
        """The governor, created on first use from the specification."""
        if self.governor is None:
            self.governor = limits.ResourceGovernor(
                self.spec.resources, self.started_at
            )
        return self.governor

    @property
    def deadline(self) -> float:
        """Monotonic time at which the budget is exhausted."""
        return self.started_at + self.spec.resources.time_budget_seconds

    @property
    def run_store(self) -> store_module.RunStore:
        """The run store; raises for previews that have none."""
        return require(self.store, "store")

    @property
    def run_directory(self) -> str:
        """Run directory as a string, empty for previews."""
        return str(self.store.directory) if self.store else ""

    def write_json(self, relative: str, payload: Any) -> None:
        """Writes a JSON artifact if the run is persisted."""
        if self.store is not None:
            self.store.write_json(relative, payload)

    def emit(
        self,
        kind: str,
        reason: str = "",
        artifacts: tuple[str, ...] = (),
        data: dict[str, Any] | None = None,
    ) -> None:
        """Records an event if the run is persisted.

        Args:
            kind: Dotted event type.
            reason: Why it happened.
            artifacts: Run-relative artifact paths it refers to.
            data: Structured payload (for example hashes and counts).
        """
        if self.log is not None and self.store is not None:
            self.log.emit(
                events.Event(
                    self.store.run_id,
                    kind,
                    reason,
                    self.spec.config_hash(),
                    artifacts,
                    data or {},
                )
            )

    def allocate_trial_ids(self, count: int) -> list[str]:
        """Hands out the next `count` sequential trial identifiers."""
        start = self.trial_counter
        self.trial_counter += count
        return [f"trial_{start + offset + 1:03d}" for offset in range(count)]

    def add_outcome(
        self,
        technique: str,
        action: experiments.DecisionAction,
        reason: str,
    ) -> None:
        """Records an evidence-based decision for the decision log."""
        self.outcomes.append(experiments.Decision(technique, action, reason))

    def record_trial(self, outcome: experiments.TrialResult) -> None:
        """Appends a trial and persists, announces and tracks it.

        Raises:
            MagicMLError: If a different trial already used the identifier,
                which would silently overwrite its checkpoint.
        """
        if any(t.trial_id == outcome.trial_id for t in self.trials):
            raise errors.MagicMLError(
                f"duplicate trial id {outcome.trial_id}",
                {"trial_id": outcome.trial_id},
            )
        self.trials.append(outcome)
        path = f"trials/{outcome.trial_id}.json"
        self.write_json(path, outcome.to_dict())
        self.emit(f"trial.{outcome.status}", outcome.model_name, (path,))
        if self.store is not None:
            self.tracker.record_trial(self.store.run_id, outcome)

    def track(self, **fields: Any) -> None:
        """Updates the tracked run row if the run is persisted."""
        if self.store is not None:
            self.tracker.update_run(self.store.run_id, **fields)

    def check_cancelled(self) -> None:
        """Raises if cancellation was requested.

        Raises:
            CancelledError: Carrying the run directory in its details.
        """
        if self.cancel_event.is_set():
            raise errors.CancelledError(
                "run cancelled", {"run_dir": self.run_directory}
            )
