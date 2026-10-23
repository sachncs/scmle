"""Mutable state shared by the stages of one run."""

import dataclasses
import threading
import time
from typing import Any
from typing import TypeVar

import pandas as pd

from scml.api import results
from scml.artifacts import bundle as bundle_module
from scml.artifacts import store as store_module
from scml.contracts import capabilities
from scml.contracts import data as data_contracts
from scml.contracts import errors
from scml.contracts import events
from scml.contracts import experiments
from scml.contracts import lineage
from scml.contracts import metrics as metric_contracts
from scml.contracts import tasks
from scml.core import limits
from scml.core import resource_observation
from scml.core import scheduler
from scml.data.readers import streaming
from scml.data.splitting import splitters
from scml.evaluation import metrics
from scml.evaluation import selection
from scml.experimentation import stores as tracking_stores
from scml.intelligence import recommendations as recommendations_module
from scml.models import registry as registry_module

T = TypeVar("T")


TABULAR_BETA_TASK_FAMILIES = (
    (
        tasks.TaskType.BINARY,
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
        tasks.TaskType.MULTICLASS,
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
        tasks.TaskType.REGRESSION,
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
    (tasks.TaskType.SURVIVAL, ("baseline", "linear", "boosting")),
    (tasks.TaskType.ORDINAL, ("baseline", "linear", "tree", "boosting")),
    (tasks.TaskType.RANKING, ("baseline", "boosting")),
    (
        tasks.TaskType.MULTILABEL,
        ("baseline", "linear", "tree", "boosting"),
    ),
    (
        tasks.TaskType.MULTIOUTPUT,
        ("baseline", "linear", "tree", "boosting"),
    ),
    (tasks.TaskType.FORECAST, ("baseline", "linear", "tree", "boosting")),
    (tasks.TaskType.UPLIFT, ("baseline", "causal")),
)
"""Explicit tabular model-family paths covered by the current beta."""


TABULAR_BETA_EVALUATIONS = {
    tasks.TaskType.BINARY: capabilities.Evaluation.STANDARD,
    tasks.TaskType.MULTICLASS: capabilities.Evaluation.STANDARD,
    tasks.TaskType.REGRESSION: capabilities.Evaluation.STANDARD,
    tasks.TaskType.SURVIVAL: capabilities.Evaluation.SURVIVAL,
    tasks.TaskType.ORDINAL: capabilities.Evaluation.STANDARD,
    tasks.TaskType.RANKING: capabilities.Evaluation.RANKING,
    tasks.TaskType.MULTILABEL: capabilities.Evaluation.MULTILABEL,
    tasks.TaskType.MULTIOUTPUT: capabilities.Evaluation.MULTIOUTPUT,
    tasks.TaskType.FORECAST: capabilities.Evaluation.FORECASTING,
    tasks.TaskType.UPLIFT: capabilities.Evaluation.UPLIFT,
}
"""Canonical evaluation contract for each tabular task path."""


def tabular_beta_capability_registry() -> capabilities.CapabilityRegistry:
    """Builds the explicit capability registry for the tabular beta.

    The registry intentionally declares only the library inference path with
    no portable explanation or predictive-uncertainty promise. Modalities,
    task/model paths, and contract dimensions not listed here are rejected
    before fitting rather than being inferred from an adapter's existence.
    """
    records: list[capabilities.Capability] = []
    for task, families in TABULAR_BETA_TASK_FAMILIES:
        for model_family in families:
            for fairness in (
                (capabilities.Fairness.NONE,)
                if task
                not in (tasks.TaskType.BINARY, tasks.TaskType.MULTICLASS)
                else (
                    capabilities.Fairness.NONE,
                    capabilities.Fairness.MITIGATION,
                )
            ):
                records.append(
                    capabilities.Capability(
                        key=capabilities.CapabilityKey(
                            modality=capabilities.Modality.TABULAR,
                            task=task,
                            model_family=model_family,
                            evaluation=TABULAR_BETA_EVALUATIONS[task],
                            uncertainty=capabilities.Uncertainty.NONE,
                            fairness=fairness,
                            explainability=capabilities.Explainability.NONE,
                            inference=capabilities.Inference.LIBRARY,
                        ),
                        status=capabilities.CapabilityStatus.SUPPORTED,
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
        tasks.TaskType.BINARY,
        tasks.TaskType.MULTICLASS,
        tasks.TaskType.REGRESSION,
    ):
        for fairness in (
            (
                capabilities.Fairness.NONE,
                capabilities.Fairness.MITIGATION,
            )
            if task in (tasks.TaskType.BINARY, tasks.TaskType.MULTICLASS)
            else (capabilities.Fairness.NONE,)
        ):
            records.append(
                capabilities.Capability(
                    key=capabilities.CapabilityKey(
                        modality=capabilities.Modality.TABULAR,
                        task=task,
                        model_family=capabilities.ModelFamily.TRANSFORMER.value,
                        evaluation=capabilities.Evaluation.STANDARD,
                        uncertainty=capabilities.Uncertainty.NONE,
                        fairness=fairness,
                        explainability=capabilities.Explainability.NONE,
                        inference=capabilities.Inference.LIBRARY,
                    ),
                    status=capabilities.CapabilityStatus.EXPERIMENTAL,
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
        tasks.TaskType.BINARY,
        tasks.TaskType.MULTICLASS,
        tasks.TaskType.REGRESSION,
    ):
        for fairness in (
            (
                capabilities.Fairness.NONE,
                capabilities.Fairness.MITIGATION,
            )
            if task in (tasks.TaskType.BINARY, tasks.TaskType.MULTICLASS)
            else (capabilities.Fairness.NONE,)
        ):
            records.append(
                capabilities.Capability(
                    key=capabilities.CapabilityKey(
                        modality=capabilities.Modality.TABULAR,
                        task=task,
                        model_family=capabilities.ModelFamily.JEPA.value,
                        evaluation=capabilities.Evaluation.STANDARD,
                        uncertainty=capabilities.Uncertainty.NONE,
                        fairness=fairness,
                        explainability=capabilities.Explainability.NONE,
                        inference=capabilities.Inference.LIBRARY,
                    ),
                    status=capabilities.CapabilityStatus.EXPERIMENTAL,
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
    return capabilities.CapabilityRegistry.with_declared_model_families(records)


def extension_capability(
    key: capabilities.CapabilityKey,
) -> capabilities.Capability:
    """Describe an explicitly supplied adapter outside the beta guarantee.

    User-registered and test adapters are allowed to exercise the extension
    boundary, but they must never inherit the supported status of a built-in
    family. The plan records this distinction before the adapter is fitted.
    """
    return capabilities.Capability(
        key=key,
        status=capabilities.CapabilityStatus.EXTENSION_ONLY,
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
        ScmlError: If the producing stage has not executed.
    """
    if value is None:
        raise errors.ScmlError(
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
        task: Resolved task type.
        profile: Dataset profile.
        split: Split result with development data and locked test.
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
        semi_supervised: Record of the self-training judgement, if any.
        evaluation: Final evaluation record.
        bundle: Packaged model.
        result: Final run result.
        deadline: Monotonic deadline of the time budget.
    """

    spec: tasks.ProjectSpec
    registry: registry_module.ModelRegistry
    catalog: metrics.MetricCatalog
    policy: selection.SelectionPolicy
    capability_registry: capabilities.CapabilityRegistry = dataclasses.field(
        default_factory=tabular_beta_capability_registry
    )
    capability_observation: (
        resource_observation.RuntimeResourceObservation | None
    ) = None
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
    evidence_graph: lineage.EvidenceGraph = dataclasses.field(
        default_factory=lineage.EvidenceGraph
    )
    task: tasks.TaskType | None = None
    profile: data_contracts.DatasetProfile | None = None
    split: splitters.SplitResult | None = None
    plan: experiments.ExperimentPlan | None = None
    recommendations: tuple[recommendations_module.Recommendation, ...] = ()
    metric: metric_contracts.Metric | None = None
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
    test_rows: tuple[pd.DataFrame, pd.Series] | None = None
    stream: tuple[streaming.ChunkSource, streaming.Totals] | None = None
    pareto_front: list[dict[str, Any]] | None = None
    seasonal_baseline: Any = None
    propensity: dict[str, Any] | None = None
    fairness_mitigation: dict[str, Any] | None = None
    ordinal_labels: tuple[Any, ...] | None = None
    unlabeled: pd.DataFrame | None = None
    semi_supervised: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    bundle: bundle_module.ModelBundle | None = None
    result: results.RunResult | None = None
    started_at: float = dataclasses.field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        """Bind the captured facts to the registry used by planning."""
        if self.capability_observation is not None:
            self.capability_registry.bind_runtime_environment(
                self.capability_observation.environment
            )

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

    def close(self) -> None:
        """Release resources owned by a prepared or executing context.

        A preview still creates a locked test-set boundary during profiling.
        The context owns that boundary until the caller has opened it or the
        preview is finished, so every public path can close it deterministically
        instead of relying on process garbage collection.
        """
        if self.split is not None:
            self.split.locked_test.close()

    def write_json(self, relative: str, payload: Any) -> None:
        """Writes a JSON artifact if the run is persisted."""
        if self.store is not None:
            self.store.write_json(relative, payload)

    def persist_capability_observation(self) -> None:
        """Persist the runtime facts immediately before capability planning."""
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
            ScmlError: If a different trial already used the identifier,
                which would silently overwrite its checkpoint.
        """
        if any(t.trial_id == outcome.trial_id for t in self.trials):
            raise errors.ScmlError(
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
