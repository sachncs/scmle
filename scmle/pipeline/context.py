"""Mutable state shared by the stages of one run."""

import dataclasses
import logging
import threading
import time
from typing import Any
from typing import TypeVar

import pandas

from scmle.artifacts import results
import scmle.artifacts.bundle
import scmle.artifacts.store
from scmle.assurance.custody import pipeline
from scmle.assurance.custody import sealed
from scmle.contracts import capabilities
from scmle.contracts import errors
from scmle.contracts import events
from scmle.contracts import experiments
from scmle.contracts import lineage
from scmle.contracts import tasks
import scmle.contracts.data
import scmle.contracts.metrics
from scmle.data.readers import streaming
from scmle.data.splitting import splitters
from scmle.evaluation import metrics
from scmle.evaluation import selection
import scmle.models.registry
from scmle.pipeline import journal
from scmle.pipeline import roster
import scmle.pipeline.recommendations
from scmle.runtime import limits
from scmle.runtime import resource_observation
from scmle.runtime import scheduler
from scmle.storage import stores

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


def require(value: T | None, name: str) -> T:
    """Returns a stage product, failing loudly if its stage has not run.

    Args:
        value: The possibly unset context field.
        name: Field name used in the error message.

    Returns:
        `value`, narrowed to non-None.

    Raises:
        ScmleError: If the producing stage has not executed.
    """
    if value is None:
        raise errors.ScmleError(
            f"'{name}' is unavailable: its stage has not run yet"
        )
    return value


@dataclasses.dataclass
class Context(journal.Journal):
    """Inputs, services and progressively produced artifacts of a run.

    Services and inputs are set at construction; the remaining fields are
    filled by stages in order. When `store` is None (plan preview) writes and
    events are skipped. The records a run keeps (files, events, trials and
    decisions) are written through the `journal.Journal` methods.

    Attributes:
        spec: Project specification.
        registry: Model registry.
        catalog: Metric catalog.
        policy: Model selection policy.
        capability_registry: Declared capabilities that planning checks the
            run against (the tabular beta registry by default).
        store: Run directory, or None for previews.
        log: Event log, or None for previews.
        capability_observation: Runtime resource facts captured before the
            capability gate.
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
        evidence_graph: Graph of the run's evidence, replaced by the one
            built when the run is packaged.
        task: Resolved task type.
        profile: Dataset profile.
        cluster_column: Entity column whose rows are not independent in the
            final evaluation (set for `known_entity_future`, where entities
            are not split units but their rows are correlated).
        curation: Curation report as first written, kept so that later
            stages can amend it when they change the feature plan.
        split: Split result with development data and locked test.
        custody: Three-way split outcome when a charter governs the run.
        custody_vault: Holds the sealed confirmation set of a charter run.
        plan: Experiment plan.
        recommendations: Advice for the user derived from the profile.
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
        coverage_claim_sha256: Hash of the coverage claim written for the
            run, or None when no claim was made.
        pending_ledger: The exploration of a charter run whose opening of the
            confirmation set waits for the model selection, or None.
        refits: Candidates already refit on development data during
            constraint checks, by trial id (reused by the final evaluation).
        final_pipeline: Selected pipeline refit on development data.
        test_rows: The released locked rows and labels, kept for reports
            written after the test was opened.
        stream: The chunked source and totals when the file was streamed.
        pareto_front: Score / latency / size of measured candidates.
        seasonal_baseline: Seasonal-naive forecasts for the locked rows,
            computed at split time (forecasting).
        propensity: How well the covariates predict treatment (uplift).
        fairness_mitigation: Evidence for group thresholds, applied or not.
        ordinal_labels: For ordinal tasks, the labels from lowest to highest
            (the data's target holds their integer ranks).
        unlabeled: Rows without a target, set aside by the `unlabeled` policy.
        ingest: Result of declared-source ingestion, or None when the project
            declared none.
        semi_supervised: Record of the self-training judgement, if any.
        evaluation: Final evaluation record.
        bundle: Packaged model.
        result: Final run result.
        started_at: Monotonic time at which the context was created; the
            time budget (and the `deadline` property) counts from it.
        skipped: Work reduced or skipped to keep the time budget, each as
            `{"work": ..., "reason": ...}`; also in the events and the
            evaluation.
    """

    spec: tasks.Config
    registry: scmle.models.registry.Catalog
    catalog: metrics.Repertoire
    policy: selection.Selector
    capability_registry: capabilities.Registry = dataclasses.field(
        default_factory=roster.enrol
    )
    capability_observation: resource_observation.Observation | None = None
    store: scmle.artifacts.store.Workspace | None = None
    log: events.Log | None = None
    tracker: stores.Store = dataclasses.field(default_factory=stores.Null)
    checkpoints: dict[str, experiments.Trial] = dataclasses.field(
        default_factory=dict
    )
    backend: scheduler.Executor = dataclasses.field(
        default_factory=scheduler.Serial
    )
    governor: limits.Governor | None = None
    cancel_event: threading.Event = dataclasses.field(
        default_factory=threading.Event
    )
    input_frame: pandas.DataFrame | None = None
    approved_plan: experiments.Plan | None = None
    frame: pandas.DataFrame | None = None
    read_seconds: float = 0.0
    cleaning_notes: list[str] = dataclasses.field(default_factory=list)
    manifest: scmle.contracts.data.DatasetManifest | None = None
    evidence_graph: lineage.Graph = dataclasses.field(
        default_factory=lineage.Graph
    )
    task: tasks.Task | None = None
    profile: scmle.contracts.data.Portrait | None = None
    curation: scmle.contracts.data.Curation | None = None
    cluster_column: str | None = None
    split: splitters.Split | None = None
    custody: pipeline.Outcome | None = None
    custody_vault: sealed.Vault | None = None
    plan: experiments.Plan | None = None
    recommendations: tuple[
        scmle.pipeline.recommendations.Recommendation, ...
    ] = ()
    metric: scmle.contracts.metrics.Metric | None = None
    trials: list[experiments.Trial] = dataclasses.field(default_factory=list)
    history: tuple[experiments.Precedent, ...] = ()
    outcomes: list[experiments.Decision] = dataclasses.field(
        default_factory=list
    )
    trial_counter: int = 0
    selected: experiments.Trial | None = None
    diagnostics: dict[str, dict[str, Any]] = dataclasses.field(
        default_factory=dict
    )
    reference_scores: pandas.Series | None = None
    refits: dict[str, Any] = dataclasses.field(default_factory=dict)
    pending_ledger: Any = None
    coverage_claim_sha256: str | None = None
    final_pipeline: Any = None
    test_rows: tuple[pandas.DataFrame, pandas.Series] | None = None
    stream: tuple[streaming.Feed, streaming.Totals] | None = None
    pareto_front: list[dict[str, Any]] | None = None
    seasonal_baseline: Any = None
    propensity: dict[str, Any] | None = None
    fairness_mitigation: dict[str, Any] | None = None
    ordinal_labels: tuple[Any, ...] | None = None
    unlabeled: pandas.DataFrame | None = None
    ingest: Any = None
    semi_supervised: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    bundle: scmle.artifacts.bundle.Bundle | None = None
    result: results.Run | None = None
    started_at: float = dataclasses.field(default_factory=time.monotonic)
    skipped: list[dict[str, str]] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        """Binds the captured facts to the registry used by planning."""
        if self.capability_observation is not None:
            self.capability_registry.bind(
                self.capability_observation.environment
            )

    @property
    def regulator(self) -> limits.Governor:
        """The governor, created on first use from the specification."""
        if self.governor is None:
            self.governor = limits.Governor(
                self.spec.resources, self.started_at
            )
        return self.governor

    @property
    def deadline(self) -> float:
        """Monotonic time at which the budget is exhausted."""
        return self.started_at + self.spec.resources.time_budget_seconds

    @property
    def workspace(self) -> scmle.artifacts.store.Workspace:
        """The run store; raises for previews that have none."""
        return require(self.store, "store")

    @property
    def directory(self) -> str:
        """Runs directory as a string, empty for previews."""
        return str(self.store.directory) if self.store else ""

    def close(self) -> None:
        """Releases resources owned by a prepared or executing context.

        A preview still creates a locked test-set boundary during profiling.
        The context owns that boundary until the caller has opened it or the
        preview is finished, so every public path can close it deterministically
        instead of relying on process garbage collection.
        """
        if self.split is not None:
            self.split.locked_test.close()
        if self.custody_vault is not None:
            self.custody_vault.close()

    def observe(self) -> None:
        """Persists the runtime facts immediately before capability planning."""
        observation = require(
            self.capability_observation, "capability_observation"
        )
        path = "evidence/capability_resources.json"
        self.write_json(path, observation.to_dict())
        self.emit(
            "capability.environment.observed",
            "runtime facts captured for the capability gate",
            (path,),
            observation.to_dict()["environment"],
        )

    def check_cancelled(self) -> None:
        """Raises if cancellation was requested.

        Raises:
            CancelledError: Carrying the run directory in its details.
        """
        if self.cancel_event.is_set():
            raise errors.CancelledError(
                "run cancelled", {"run_dir": self.directory}
            )
