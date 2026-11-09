"""Mutable state shared by the stages of one run."""

import dataclasses
import threading
import time
from typing import Any
from typing import TypeVar

import pandas

from scmle.api import results
import scmle.artifacts.bundle
import scmle.artifacts.store
from scmle.contracts import capabilities
from scmle.contracts import errors
from scmle.contracts import events
from scmle.contracts import experiments
from scmle.contracts import lineage
from scmle.contracts import tasks
import scmle.contracts.data
import scmle.contracts.metrics
from scmle.core import limits
from scmle.core import resource_observation
from scmle.core import scheduler
from scmle.data.custody import pipeline
from scmle.data.custody import sealed
from scmle.data.readers import streaming
from scmle.data.splitting import splitters
from scmle.evaluation import metrics
from scmle.evaluation import selection
from scmle.experimentation import stores
import scmle.intelligence.recommendations
import scmle.models.registry

T = TypeVar("T")


TABULAR_BETA_TASK_FAMILIES = (
    (
        tasks.Task.BINARY,
        (
            "baseline",
            "linear",
            "tree",
            "boosting",
            "kernel",
            "neighbors",
            "neural",
            "bagging",
            "deep_learning",
        ),
    ),
    (
        tasks.Task.MULTICLASS,
        (
            "baseline",
            "linear",
            "tree",
            "boosting",
            "kernel",
            "neighbors",
            "neural",
            "bagging",
            "deep_learning",
        ),
    ),
    (
        tasks.Task.REGRESSION,
        (
            "baseline",
            "linear",
            "tree",
            "boosting",
            "kernel",
            "neighbors",
            "neural",
            "bagging",
            "deep_learning",
        ),
    ),
    (tasks.Task.SURVIVAL, ("baseline", "linear", "boosting")),
    (tasks.Task.ORDINAL, ("baseline", "linear", "tree", "boosting")),
    (tasks.Task.RANKING, ("baseline", "boosting")),
    (
        tasks.Task.MULTILABEL,
        ("baseline", "linear", "tree", "boosting"),
    ),
    (
        tasks.Task.MULTIOUTPUT,
        ("baseline", "linear", "tree", "boosting"),
    ),
    (tasks.Task.FORECAST, ("baseline", "linear", "tree", "boosting")),
    (tasks.Task.UPLIFT, ("baseline", "causal")),
)
"""Explicit tabular model-family paths covered by the current beta."""


TABULAR_BETA_EVALUATIONS = {
    tasks.Task.BINARY: capabilities.Evaluation.STANDARD,
    tasks.Task.MULTICLASS: capabilities.Evaluation.STANDARD,
    tasks.Task.REGRESSION: capabilities.Evaluation.STANDARD,
    tasks.Task.SURVIVAL: capabilities.Evaluation.SURVIVAL,
    tasks.Task.ORDINAL: capabilities.Evaluation.STANDARD,
    tasks.Task.RANKING: capabilities.Evaluation.RANKING,
    tasks.Task.MULTILABEL: capabilities.Evaluation.MULTILABEL,
    tasks.Task.MULTIOUTPUT: capabilities.Evaluation.MULTIOUTPUT,
    tasks.Task.FORECAST: capabilities.Evaluation.FORECASTING,
    tasks.Task.UPLIFT: capabilities.Evaluation.UPLIFT,
}
"""Canonical evaluation contract for each tabular task path."""


def tabular_beta_capability_registry() -> capabilities.Registry:
    """Builds the explicit capability registry for the tabular beta.

    The registry intentionally declares only the library inference path with
    no portable explanation or predictive-uncertainty promise. Modalities,
    task/model paths, and contract dimensions not listed here are rejected
    before fitting rather than being inferred from an adapter's existence.

    Returns:
        A registry of the supported and experimental tabular capabilities,
        each with its failure modes, evidence links and limitations.
    """
    records: list[capabilities.Capability] = []
    for task, families in TABULAR_BETA_TASK_FAMILIES:
        for model_family in families:
            for fairness in (
                (capabilities.Fairness.NONE,)
                if task not in (tasks.Task.BINARY, tasks.Task.MULTICLASS)
                else (
                    capabilities.Fairness.NONE,
                    capabilities.Fairness.MITIGATION,
                )
            ):
                records.append(
                    capabilities.Capability(
                        key=capabilities.Coordinate(
                            modality=capabilities.Modality.TABULAR,
                            task=task,
                            model_family=model_family,
                            evaluation=TABULAR_BETA_EVALUATIONS[task],
                            uncertainty=capabilities.Uncertainty.NONE,
                            fairness=fairness,
                            explainability=capabilities.Explainability.NONE,
                            inference=capabilities.Inference.LIBRARY,
                        ),
                        status=capabilities.Status.SUPPORTED,
                        required_extras=(
                            ("deeplearning",)
                            if model_family == "deep_learning"
                            else ("core",)
                        ),
                        resources=capabilities.ResourceAssumptions(
                            memory_gb=1.0
                        ),
                        failure_modes=(
                            "malformed or schema-incompatible input",
                            "model-family dependency unavailable",
                            "resource limits exceeded",
                        ),
                        evidence_links=(
                            "tests/integration/test_capability_gate.py",
                        ),
                        limitations=(
                            "tabular modality only",
                            "library inference only",
                            "no portable explanation contract",
                            "no predictive uncertainty contract",
                        ),
                    )
                )
    for task in (
        tasks.Task.BINARY,
        tasks.Task.MULTICLASS,
        tasks.Task.REGRESSION,
    ):
        for fairness in (
            (
                capabilities.Fairness.NONE,
                capabilities.Fairness.MITIGATION,
            )
            if task in (tasks.Task.BINARY, tasks.Task.MULTICLASS)
            else (capabilities.Fairness.NONE,)
        ):
            records.append(
                capabilities.Capability(
                    key=capabilities.Coordinate(
                        modality=capabilities.Modality.TABULAR,
                        task=task,
                        model_family=capabilities.ModelFamily.TRANSFORMER.value,
                        evaluation=capabilities.Evaluation.STANDARD,
                        uncertainty=capabilities.Uncertainty.NONE,
                        fairness=fairness,
                        explainability=capabilities.Explainability.NONE,
                        inference=capabilities.Inference.LIBRARY,
                    ),
                    status=capabilities.Status.EXPERIMENTAL,
                    required_extras=("deeplearning",),
                    resources=capabilities.ResourceAssumptions(
                        memory_gb=4.0,
                        max_rows=100_000,
                    ),
                    failure_modes=(
                        "torch runtime or tabular transformer adapter is "
                        "unavailable",
                        "numeric feature-token training or evaluation fails",
                        "resource limits or experimental evidence are "
                        "incomplete",
                    ),
                    evidence_links=(
                        "tests/integration/test_capability_gate.py",
                        "tests/unit/test_deeplearning.py",
                    ),
                    limitations=(
                        "experimental tabular FT-Transformer only",
                        "requires explicit models.experimental_models opt-in",
                        "not a text, image, audio, video, or multimodal "
                        "transformer",
                        "no production support guarantee",
                    ),
                )
            )
    for task in (
        tasks.Task.BINARY,
        tasks.Task.MULTICLASS,
        tasks.Task.REGRESSION,
    ):
        for fairness in (
            (
                capabilities.Fairness.NONE,
                capabilities.Fairness.MITIGATION,
            )
            if task in (tasks.Task.BINARY, tasks.Task.MULTICLASS)
            else (capabilities.Fairness.NONE,)
        ):
            records.append(
                capabilities.Capability(
                    key=capabilities.Coordinate(
                        modality=capabilities.Modality.TABULAR,
                        task=task,
                        model_family=capabilities.ModelFamily.JEPA.value,
                        evaluation=capabilities.Evaluation.STANDARD,
                        uncertainty=capabilities.Uncertainty.NONE,
                        fairness=fairness,
                        explainability=capabilities.Explainability.NONE,
                        inference=capabilities.Inference.LIBRARY,
                    ),
                    status=capabilities.Status.EXPERIMENTAL,
                    required_extras=("deeplearning",),
                    resources=capabilities.ResourceAssumptions(
                        memory_gb=4.0,
                        max_rows=100_000,
                    ),
                    failure_modes=(
                        "torch runtime or tabular JEPA adapter is unavailable",
                        "numeric representation training or downstream "
                        "head evaluation fails",
                        "resource limits or experimental evidence are "
                        "incomplete",
                    ),
                    evidence_links=(
                        "tests/integration/test_capability_gate.py",
                        "tests/integration/test_experimental_models.py",
                        "tests/unit/test_jepa_contract.py",
                    ),
                    limitations=(
                        "experimental tabular numeric representation plus "
                        "supervised head only",
                        "requires explicit models.experimental_models opt-in",
                        "not a text, image, audio, video, graph, or "
                        "multimodal JEPA",
                        "no production support guarantee",
                    ),
                )
            )
    return capabilities.Registry.with_declared_model_families(records)


def extension_capability(
    key: capabilities.Coordinate,
) -> capabilities.Capability:
    """Describes an explicitly supplied adapter outside the beta guarantee.

    User-registered and test adapters are allowed to exercise the extension
    boundary, but they must never inherit the supported status of a built-in
    family. The plan records this distinction before the adapter is fitted.

    Args:
        key: The capability key of the supplied adapter.

    Returns:
        A capability with extension-only status, no required extras and the
        limitations that the built-in guarantee does not cover it.
    """
    return capabilities.Capability(
        key=key,
        status=capabilities.Status.EXTENSION_ONLY,
        required_extras=(),
        resources=capabilities.ResourceAssumptions(memory_gb=1.0),
        failure_modes=(
            "adapter conformance or dependency failure",
            "resource limits exceeded",
            "missing end-to-end lifecycle evidence",
        ),
        evidence_links=("explicit adapter registration",),
        limitations=(
            "not covered by the built-in production-beta guarantee",
            "model-specific evaluation and recovery evidence is caller-owned",
        ),
    )


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
class Context:
    """Inputs, services and progressively produced artifacts of a run.

    Services and inputs are set at construction; the remaining fields are
    filled by stages in order. When `store` is None (plan preview) writes and
    events are skipped.

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
    """

    spec: tasks.Config
    registry: scmle.models.registry.Catalog
    catalog: metrics.MetricCatalog
    policy: selection.Selector
    capability_registry: capabilities.Registry = dataclasses.field(
        default_factory=tabular_beta_capability_registry
    )
    capability_observation: resource_observation.Observation | None = None
    store: scmle.artifacts.store.RunStore | None = None
    log: events.Log | None = None
    tracker: stores.ExperimentStore = dataclasses.field(
        default_factory=stores.NullExperimentStore
    )
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
    profile: scmle.contracts.data.DatasetProfile | None = None
    curation: scmle.contracts.data.Curation | None = None
    cluster_column: str | None = None
    split: splitters.Split | None = None
    custody: pipeline.Outcome | None = None
    custody_vault: sealed.Vault | None = None
    plan: experiments.Plan | None = None
    recommendations: tuple[
        scmle.intelligence.recommendations.Recommendation, ...
    ] = ()
    metric: scmle.contracts.metrics.Metric | None = None
    trials: list[experiments.Trial] = dataclasses.field(default_factory=list)
    history: tuple[experiments.HistoricalRun, ...] = ()
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
    stream: tuple[streaming.ChunkSource, streaming.Totals] | None = None
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

    def __post_init__(self) -> None:
        """Binds the captured facts to the registry used by planning."""
        if self.capability_observation is not None:
            self.capability_registry.bind_runtime_environment(
                self.capability_observation.environment
            )

    @property
    def resource_governor(self) -> limits.Governor:
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
    def run_store(self) -> scmle.artifacts.store.RunStore:
        """The run store; raises for previews that have none."""
        return require(self.store, "store")

    @property
    def run_directory(self) -> str:
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

    def write_json(self, relative: str, payload: Any) -> None:
        """Writes a JSON artifact if the run is persisted.

        Args:
            relative: Artifact path relative to the run directory.
            payload: JSON-serializable content.
        """
        if self.store is not None:
            self.store.write_json(relative, payload)

    def write_text(self, relative: str, text: str) -> None:
        """Writes a text artifact if the run is persisted.

        Args:
            relative: Artifact path relative to the run directory.
            text: The content to write.
        """
        if self.store is not None:
            self.store.write_text(relative, text)

    def persist_capability_observation(self) -> None:
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
        """Hands out the next `count` sequential trial identifiers.

        Args:
            count: How many identifiers to reserve.

        Returns:
            Identifiers of the form `trial_001`, continuing from the trials
            already allocated.
        """
        start = self.trial_counter
        self.trial_counter += count
        return [f"trial_{start + offset + 1:03d}" for offset in range(count)]

    def add_outcome(
        self,
        technique: str,
        action: experiments.DecisionAction,
        reason: str,
    ) -> None:
        """Records an evidence-based decision for the decision log.

        Args:
            technique: The technique the decision concerns.
            action: What was decided (the decision action).
            reason: The evidence behind the decision.
        """
        self.outcomes.append(experiments.Decision(technique, action, reason))

    def record_trial(self, outcome: experiments.Trial) -> None:
        """Appends a trial and persists, announces and tracks it.

        Args:
            outcome: The finished trial to record.

        Raises:
            ScmleError: If a recorded trial already has the same identifier,
                which would overwrite its checkpoint.
        """
        if any(t.trial_id == outcome.trial_id for t in self.trials):
            raise errors.ScmleError(
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
        """Updates the tracked run row if the run is persisted.

        Args:
            **fields: Run-row fields to update.
        """
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
