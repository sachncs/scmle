"""Public API: `Project(...).plan()`, `.run()`, `result.predict()`."""

import dataclasses
import pathlib
from typing import Any
from typing import Literal

import pandas

from scmle.api.curator import Curator
from scmle.api.estimation import declare
from scmle.api.estimation import estimate
from scmle.api.estimation import persist
from scmle.api.estimation import REFUSALS
from scmle.api.estimation import refuse
from scmle.api.estimation import restore
from scmle.api.non_tabular import vet
from scmle.artifacts import reproducibility
from scmle.artifacts import results
from scmle.contracts import errors
from scmle.contracts import experiments
from scmle.contracts import modalities
from scmle.contracts import performance
from scmle.contracts import tasks
import scmle.contracts.data
from scmle.data.readers import interchange
from scmle.evaluation import parameter_estimation
from scmle.pipeline import modality_lifecycle
from scmle.pipeline import policies
from scmle.pipeline import products
import scmle.pipeline.orchestrator
import scmle.pipeline.recommendations
from scmle.pipeline.stages import planning
from scmle.pipeline.stages import profiling

__all__ = [
    "Curator",
    "LeakagePolicy",
    "Modality",
    "Project",
    "ValidationStrategy",
]

ValidationStrategy = Literal[
    "auto",
    "stratified",
    "stratified_events",
    "multilabel_stratified",
    "treatment_stratified",
    "random",
    "group",
    "temporal",
    "rolling",
]
LeakagePolicy = Literal["warn", "block"]


@dataclasses.dataclass(eq=False, repr=False)
class Definition:
    """What a project declares: its data, task, controls and outputs.

    `Project` takes these as its constructor arguments. An inconsistent
    configuration raises pydantic's `ValidationError` when the project builds
    its specification.

    Attributes:
        data: Supported tabular path (CSV/TSV, JSON/JSONL, Excel, Parquet or
            Feather), or an in-memory table: pandas DataFrame, pyarrow
            Table, Polars frame, or anything implementing the Arrow stream
            protocol.
        target: Name of the target column, or for `multilabel` and
            `multioutput` tasks the list of all target columns.
        task: `auto`, `classification` (resolved from the target),
            `regression`, an explicit `binary_classification` or
            `multiclass_classification`, or one of `survival`, `ordinal`,
            `ranking`, `multilabel`, `multioutput`, `forecast` and
            `uplift`.
        metric: Primary metric; defaults per task.
        budget: Wall-clock budget in seconds.
        modality: Dataset family; a file needs a registered `kinds.Kind`.
        workers: Concurrent trial processes (1 = sequential).
        enforcement: `hard` supervises every trial in its own process and
            kills it past the time or memory limit; `cooperative` only stops
            new trials from starting (and allows in-process hooks).
        executor: `process` (default) or `ray`.
        address: Ray cluster to join (a local runtime if None).
        memory_limit_gb: Hard memory cap per trial.
        identifiers: Identifier columns excluded from features.
        group: Entity column that must not straddle splits.
        timestamp: Column ordering rows in time.
        cutoff: The prediction time, ISO 8601 text such as `2020-01-01`:
            a date column with values after it was not known when the
            prediction is made, and the leakage screen blocks it.
        known: Column to when it becomes known: `prediction` (by the time
            the prediction is made, as an unlisted column is taken to be) or
            `outcome` (only once the outcome is). The leakage screen blocks
            a feature declared `outcome`.
        time_as_feature: Also let the model use the time column as a feature
            (off by default; models must extrapolate in time).
        excluded: Other columns to ignore.
        unavailable: Columns known only after the outcome
            (excluded; the feature-availability contract).
        delays: Columns whose value is known only that many rows
            after the row it describes; the model sees the value from that
            many rows earlier (needs `timestamp`).
        sensitive: Columns describing groups of people; the fairness
            report compares outcomes across them.
        weight: Numeric column of per-row training weights (never a
            feature; not needed at prediction time; test metrics are also
            reported weighted).
        event: For `task="survival"`, the 0/1 column saying whether
            the event was observed (1) or the row was censored (0); the
            target is the time to the event or to censoring.
        query: For `task="ranking"`, the column identifying each
            row's query; the target is the relevance grade.
        memory_budget_gb: Largest in-memory size of the dataset; a bigger
            Parquet / CSV file is streamed (see `Dataset`).
        training: `sample` or `stream` for a streamed file.
        treatment: For `task="uplift"`, the 0/1 column saying whether
            a row was treated; the target is the outcome.
        horizon: For `task="forecast"`, how many steps ahead to predict
            (needs `timestamp`; `target` is the series value).
        period: Rows per season of a forecast series.
        lags: Lags of the value used as features (default 1, 2, 3
            and 7).
        windows: Rolling-window sizes (default 3 and 7).
        calendar: Date parts of the time column to add.
        encoding: `tfidf` (default) or `embedding` for free-text
            columns (needs the `embeddings` extra; falls back to TF-IDF with
            a warning when unavailable).
        model: Sentence-embedding model for `embedding`.
        order: For `task="ordinal"`, the target's labels from lowest
            to highest.
        unlabeled: What to do with rows whose target is missing: `block`
            (default), `ignore`, or `pseudo_label` (classification; used
            only if it wins in cross-validation).
        reference: Path of a recent sample of the population the model will
            serve; the development rows are compared with it.
        confidence: Smallest probability trusted as a label.
        constraints: Numeric feature to +1 / -1: predictions must
            never fall / rise as the feature grows (binary classification
            and regression).
        rows: Rows dropped between training and validation windows
            in temporal validation.
        span: Rolling validation: rows in each validation window.
        window: Rolling validation: rows in each training window (all
            earlier rows when omitted).
        entities: Temporal and rolling validation with `group`:
            `known` (default) forecasts the entities trained on, so an
            entity's past trains and its future is held out; `unseen`
            predicts new entities and keeps each on one side of every split.
        validation: Split strategy or `auto`.
        folds: Cross-validation folds.
        holdout: Share of rows locked away for the final test.
        seed: Random seed.
        search: Hyperparameter search settings; defaults to a small pruned
            random search. Use `Search(top_models=0)` to disable tuning.
        strategies: Which data strategies to evaluate; defaults to all. Use
            `Strategies(top_models=0)` to disable.
        models: Model names and families to exclude.
        ensemble: Stacking settings; `Stacking(size=0)` disables.
        graph: Declarative derived features (row-wise, fitted and
            time-series operations), versioned with the configuration.
        selection: Serving constraints (max model size, p95 latency).
        decision: Threshold optimization for binary tasks (objective f1,
            accuracy, balanced_accuracy or an explicit cost).
        fairness: Optional group-threshold mitigation of outcome gaps across
            a sensitive column (binary tasks).
        problem: The decision, the population and the scores the model must
            reach on the locked rows (checked after the one test release).
        estimand: Explicit population quantity for a separate parameter
            estimation evidence track. It does not affect model selection.
        policy: Explicit likelihood, covariance, missingness, and
            uncertainty policy for parameter estimation.
        leakage: `warn` or `block` on leakage hypotheses.
        charter: Path of an analysis charter (`scmle charter init`). When
            given, the development rows are split into exploration rows and
            a sealed confirmation set the exploration steps cannot read; the
            charter is checked before any profiling.
        attestations: JSON file of per-column attestations that a
            strong predictor flagged by the leakage screen is legitimate.
            Used only with `charter`; each counts only for that charter and
            while valid, and is recorded in the ledger at tier `attested`.
        baseline: Used with `charter` on a binary task whose charter
            has a cost model: the 0/1 decisions of the current process. It
            is never a feature. A rule is frozen on the exploration rows and
            the claim that it beats the current process over the cost range
            is registered in the ledger and tested once on the confirmation
            set (`baseline_report.json`).
        subject: `frozen_rule` (default) or `selected_model`; the
            latter defers the opening of the confirmation set until the
            model selection is frozen and tests the selected model against
            the current process instead of the logistic rule.
        draft: The `framing_draft.json` written by `scmle frame` for
            the charter; a claim that depends on an LLM-filled field of an
            unsigned charter stays `unverified`.
        propensity: With `action`, the recorded
            probability of action 1; never a feature. It enables the
            off-policy value of the frozen rule (`policy_value.json`).
        action: With a charter decision block, the 0/1 action
            the current process logged; never a feature.
        ingest: Path of an ingest plan (JSON); needs a charter with an
            `ingest` block. Features are synthesized through the as-of views
            of the ingest plan's database before the data are split.
        checkpoints: Approval gates after planning and selection (the latter
            fires before the test set is opened).
        directory: Parent directory for run directories.
    """

    data: Any
    target: str | list[str]
    task: tasks.TaskRequest = "auto"
    metric: str | None = None
    budget: int = 1800
    marker: dataclasses.KW_ONLY
    modality: tasks.DatasetModality = "tabular"
    workers: int = 1
    enforcement: Literal["hard", "cooperative"] = "hard"
    executor: Literal["process", "ray"] = "process"
    address: str | None = None
    memory_limit_gb: float = 8.0
    identifiers: list[str] | None = None
    group: str | None = None
    timestamp: str | None = None
    cutoff: str | None = None
    known: dict[str, Literal["prediction", "outcome"]] | None = None
    time_as_feature: bool = False
    excluded: list[str] | None = None
    unavailable: list[str] | None = None
    delays: dict[str, int] | None = None
    sensitive: list[str] | None = None
    weight: str | None = None
    event: str | None = None
    query: str | None = None
    memory_budget_gb: float | None = None
    training: Literal["sample", "stream"] = "sample"
    treatment: str | None = None
    horizon: int | None = None
    period: int | None = None
    lags: list[int] | None = None
    windows: list[int] | None = None
    calendar: list[str] | None = None
    encoding: Literal["tfidf", "embedding"] = "tfidf"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    order: list[str | int] | None = None
    unlabeled: Literal["block", "ignore", "pseudo_label"] = "block"
    reference: str | None = None
    confidence: float = 0.9
    constraints: dict[str, int] | None = None
    rows: int = 0
    span: int | None = None
    window: int | None = None
    entities: Literal["known", "unseen"] = "known"
    validation: ValidationStrategy = "auto"
    folds: int = 5
    holdout: float = 0.2
    seed: int = 42
    search: tasks.Search | None = None
    strategies: tasks.Strategies | None = None
    models: tasks.Candidates | None = None
    ensemble: tasks.Stacking | None = None
    graph: tasks.Derivation | None = None
    selection: tasks.Selection | None = None
    decision: tasks.Threshold | None = None
    fairness: tasks.Parity | None = None
    problem: tasks.Problem | None = None
    estimand: parameter_estimation.Estimand | None = None
    policy: parameter_estimation.Config | None = None
    leakage: LeakagePolicy = "warn"
    charter: str | pathlib.Path | None = None
    attestations: str | pathlib.Path | None = None
    baseline: str | None = None
    subject: Literal["frozen_rule", "selected_model"] = "frozen_rule"
    draft: str | pathlib.Path | None = None
    propensity: str | None = None
    action: str | None = None
    ingest: str | pathlib.Path | None = None
    checkpoints: tuple[policies.Reviewer, ...] = ()
    directory: str | pathlib.Path = "runs"


def describe(
    definition: Definition, frame: pandas.DataFrame | None
) -> tasks.Dataset:
    """Builds the dataset section of a project's specification.

    Args:
        definition: The project's declaration.
        frame: The declared data as a pandas frame, or None for a path.

    Returns:
        The dataset declaration, with omitted lists and mappings empty.
    """
    return tasks.Dataset(
        uri="<in-memory>" if frame is not None else str(definition.data),
        target=definition.target
        if isinstance(definition.target, str)
        else definition.target[0],
        modality=definition.modality,
        targets=[]
        if isinstance(definition.target, str)
        else list(definition.target),
        id_columns=definition.identifiers or [],
        group_column=definition.group,
        time_column=definition.timestamp,
        cutoff=definition.cutoff,
        known=definition.known or {},
        time_as_feature=definition.time_as_feature,
        excluded_columns=definition.excluded or [],
        unavailable_at_prediction=definition.unavailable or [],
        feature_delays=definition.delays or {},
        sensitive_columns=definition.sensitive or [],
        weight_column=definition.weight,
        event_column=definition.event,
        query_column=definition.query,
        memory_budget_gb=definition.memory_budget_gb,
        training=definition.training,
        treatment_column=definition.treatment,
        horizon=definition.horizon,
        seasonal_period=definition.period,
        forecast_lags=definition.lags or [],
        forecast_windows=definition.windows or [],
        forecast_calendar=definition.calendar or [],
        text_encoding=definition.encoding,
        embedding_model=definition.model,
        ordinal_order=definition.order or [],
        unlabeled=definition.unlabeled,
        reference_uri=definition.reference,
        pseudo_label_confidence=definition.confidence,
        monotonic_constraints=definition.constraints or {},
        baseline_column=definition.baseline,
        baseline_subject=definition.subject,
        propensity_column=definition.propensity,
        logged_action_column=definition.action,
        ingest_plan=None
        if definition.ingest is None
        else str(definition.ingest),
        framing_draft=(
            None if definition.draft is None else str(definition.draft)
        ),
    )


def configure(
    definition: Definition, frame: pandas.DataFrame | None
) -> tasks.Config:
    """Builds a project's full specification from its declaration.

    Args:
        definition: The project's declaration.
        frame: The declared data as a pandas frame, or None for a path.

    Returns:
        The specification, with each omitted settings block at its default.
    """
    return tasks.Config(
        data=describe(definition, frame),
        task=definition.task,
        metric=definition.metric,
        validation=tasks.Validation(
            strategy=definition.validation,
            embargo_rows=definition.rows,
            horizon_rows=definition.span,
            window_rows=definition.window,
            entities=definition.entities,
            folds=definition.folds,
            test_fraction=definition.holdout,
            seed=definition.seed,
        ),
        resources=tasks.Resources(
            time_budget_seconds=definition.budget,
            workers=definition.workers,
            enforcement=definition.enforcement,
            executor=definition.executor,
            cluster_address=definition.address,
            memory_limit_gb=definition.memory_limit_gb,
        ),
        search=definition.search or tasks.Search(),
        strategies=definition.strategies or tasks.Strategies(),
        models=definition.models or tasks.Candidates(),
        ensemble=definition.ensemble or tasks.Stacking(),
        feature_graph=definition.graph or tasks.Derivation(),
        selection=definition.selection or tasks.Selection(),
        decision=definition.decision or tasks.Threshold(),
        fairness=definition.fairness or tasks.Parity(),
        problem=definition.problem or tasks.Problem(),
        leakage_policy=definition.leakage,
        charter_path=None
        if definition.charter is None
        else str(definition.charter),
        leakage_attestations=(
            None
            if definition.attestations is None
            else str(definition.attestations)
        ),
    )


@dataclasses.dataclass(eq=False, repr=False)
class Project(Definition):
    """A declarative ML project over a tabular dataset.

    The constructor takes the attributes of `Definition` as its arguments.

    Attributes:
        spec: The full project specification.
        orchestrator: The orchestrator that executes the project.
        profile: The dataset profile, once `plan` has run.
    """

    def __post_init__(self) -> None:
        """Builds the specification and the orchestrator.

        Raises:
            ConfigError: If only one of `estimand` and `estimation_policy` is
                given.
        """
        frame = (
            interchange.frame(self.data)
            if interchange.tablelike(self.data)
            else None
        )
        self.spec = configure(self, frame)
        self.directory = pathlib.Path(self.directory)
        if (self.estimand is None) != (self.policy is None):
            raise errors.ConfigError(
                "parameter estimation requires both an explicit estimand "
                "and an estimation policy"
            )
        self.orchestrator = scmle.pipeline.orchestrator.Orchestrator(
            self.spec, frame, checkpoints=self.checkpoints
        )
        self.orchestrator.tracking = self.directory / "experiments.sqlite"
        self.profile: scmle.contracts.data.Portrait | None = None

    def plan(self) -> experiments.Plan:
        """Analyzes the data and proposes a workflow; trains nothing.

        Returns:
            The proposed experiment plan. The dataset profile is kept as
            `profile`.
        """
        board = self.orchestrator.prepare()
        try:
            self.profile = board.get(profiling.Partition).profile
            made = board.get(planning.Blueprint)
        finally:
            board.close()
        return products.require(made.plan, "plan")

    def recommendations(
        self,
    ) -> tuple[scmle.pipeline.recommendations.Recommendation, ...]:
        """Advice about the data and specification, from the profile.

        Returns:
            Recommendations for the user (never applied automatically).
        """
        board = self.orchestrator.prepare()
        try:
            return board.get(planning.Blueprint).recommendations
        finally:
            board.close()

    def run(self, plan: experiments.Plan | None = None) -> results.Run:
        """Executes the workflow.

        If an estimand and policy were declared, the parameter estimation runs
        before the workflow and its evidence is bound to the result after it.
        A refused estimate stops the run before anything is trained, and
        why it was refused is written to
        `output_dir/parameter_estimation_refusal.json`.

        Args:
            plan: An approved or modified plan restricting candidates.

        Returns:
            The completed run.

        Raises:
            ConfigError: If the declared estimate is refused for its
                configuration (see `estimate`).
            DataError: If it is refused for the data.
        """
        estimation = None
        if self.estimand is not None and self.policy is not None:
            try:
                estimation = self.estimate()
            except REFUSALS as error:
                refuse(
                    self.directory,
                    self.estimand,
                    self.policy,
                    error,
                )
                raise
        declaration = declare(self.estimand, self.policy)
        result = self.orchestrator.run(
            self.directory,
            plan,
            parameter_estimation_declaration=declaration,
        )
        if estimation is not None:
            persist(result, estimation)
        return result

    def estimate(
        self, artifact_directory: str | pathlib.Path | None = None
    ) -> parameter_estimation.Estimation:
        """Estimates declared parameters without changing model selection.

        The estimation track is intentionally separate from the model selection
        track. It only runs when both scientific declarations are present, and
        it fails before fitting when the dataset or declaration is incompatible.

        Args:
            artifact_directory: Optional directory in which to write
                ``parameter_estimation.json``, or, when the estimate is
                refused, ``parameter_estimation_refusal.json`` with why.

        Returns:
            The declared parameter estimates and uncertainty evidence.

        Raises:
            ConfigError: If the estimand or policy is undeclared. A modality or
                treatment-column mismatch raises it from
                `estimate`, and missing columns or
                values raise `DataError` from there.
        """
        if self.estimand is None or self.policy is None:
            raise errors.ConfigError(
                "parameter estimation is disabled: declare both estimand "
                "and estimation_policy explicitly"
            )
        try:
            return estimate(
                self.orchestrator,
                self.estimand,
                self.policy,
                artifact_directory,
            )
        except REFUSALS as error:
            if artifact_directory is not None:
                refuse(
                    artifact_directory,
                    self.estimand,
                    self.policy,
                    error,
                )
            raise

    @classmethod
    def resume(
        cls,
        run_directory: str | pathlib.Path,
        data: pandas.DataFrame | None = None,
        checkpoints: tuple[policies.Reviewer, ...] = (),
    ) -> results.Run:
        """Continues an interrupted run from its saved configuration.

        Completed trials are reused; the rest are computed. The saved
        configuration and parameter-estimation declaration must be intact and
        match (`ArtifactError` or `ConfigError` otherwise), and a run that
        already completed cannot be resumed. A run whose test set was opened,
        or that has no intact custody journal beside its directory, is
        refused (`BlockedError`) whatever its files say, and a gate a
        checkpoint rejected stays rejected until one of `checkpoints`
        approves it.

        Args:
            run_directory: Directory of a cancelled, failed or crashed run.
            data: The original in-memory data, required only if the project was
                created from a DataFrame rather than a file.
            checkpoints: Approval gates consulted after the stages in their
                own `gates`, as for `Project`.

        Returns:
            The completed run.
        """
        run_path = pathlib.Path(run_directory)
        orchestrator = scmle.pipeline.orchestrator.Orchestrator.reload(
            run_path, frame=data, checkpoints=checkpoints
        )
        estimand, policy = restore(
            run_path,
            orchestrator.spec.config_hash(),
        )
        estimation = None
        if estimand is not None and policy is not None:
            estimation = estimate(orchestrator, estimand, policy)
        result = orchestrator.resume(run_path)
        if estimation is not None:
            persist(result, estimation)
        return result

    @classmethod
    def verify(
        cls,
        run_directory: str | pathlib.Path,
        data: pandas.DataFrame | None = None,
        output_dir: str | pathlib.Path | None = None,
    ) -> reproducibility.Comparison:
        """Re-executes a finished run's configuration and compares the runs.

        Args:
            run_directory: A completed run directory.
            data: The original in-memory data, if the run used a DataFrame.
            output_dir: Parent directory for the rerun's new run directory
                (default: the original run's parent directory).

        Returns:
            A verdict whose level is `exact`, `comparable` or `different`.
        """
        original = pathlib.Path(run_directory)
        orchestrator = scmle.pipeline.orchestrator.Orchestrator.reload(
            original, frame=data
        )
        estimand, policy = restore(
            original,
            orchestrator.spec.config_hash(),
        )
        estimation = None
        declaration = declare(estimand, policy)
        if estimand is not None and policy is not None:
            estimation = estimate(orchestrator, estimand, policy)
        rerun = orchestrator.run(
            output_dir or original.parent,
            parameter_estimation_declaration=declaration,
        )
        if estimation is not None:
            persist(rerun, estimation)
        return reproducibility.judge(
            reproducibility.reread(original),
            reproducibility.reread(rerun.directory),
        )

    def cancel(self) -> None:
        """Requests cancellation of a running workflow."""
        self.orchestrator.cancel()


class Modality:
    """Runs a typed non-tabular dataset through the shared lifecycle.

    ``Project`` remains the supported tabular beta surface. This project
    surface accepts a concrete ``Preparer`` for experimental text,
    document, image, audio, video, graph, geospatial-vector, or raster data. It
    gives those datasets the same curation, explainability, packaging,
    inference, monitoring, documentation, and recovery evidence without
    pretending that a tabular estimator can decode arbitrary raw bytes.

    Attributes:
        adapter: The adapter holding the samples.
        signing_key: Key that authenticates the package.
        output_dir: Directory for the package and the lifecycle report.
        seed: Seed of the split and the model.
        observed_adapter: Shifted dataset for monitoring, if any.
        performance_budget: Declared performance budget, if any.
        performance_baseline: Comparable-run baseline, if any.
        manifest: Identity and governance contract, if any.
        require_independent_ground_truth: Whether an independent explanation
            reference suite is required before packaging.
        checkpoint_path: Recovery checkpoint path, if any.
        resume: Whether to resume from the checkpoint.
        cancellation: Cooperative cancellation signal.
        lifecycle: The lifecycle that executes the run.
        report: The report of the latest run, or None.
    """

    def __init__(
        self,
        adapter: modality_lifecycle.Preparer,
        key: str | bytes,
        *,
        directory: str | pathlib.Path = "runs",
        seed: int = 42,
        observed: modality_lifecycle.Preparer | None = None,
        budget: performance.Budget | None = None,
        baseline: performance.Baseline | None = None,
        manifest: scmle.contracts.data.Passport | None = None,
        independent: bool = False,
        cancellation: modality_lifecycle.Cancellation | None = None,
        checkpoint: str | pathlib.Path | None = None,
        resume: bool = False,
    ) -> None:
        """Defines an experimental lifecycle run.

        Args:
            adapter: Typed adapter that owns modality validation and features.
            key: Non-empty key used to authenticate the package.
            directory: Directory for the package (in its `model` subdirectory,
                which must not already exist) and the lifecycle report
                (`modality_lifecycle.json`).
            seed: Deterministic split and model seed.
            observed: Optional shifted dataset for monitoring.
            budget: Optional declared performance budget.
            baseline: Optional comparable-run baseline.
            manifest: Optional identity and governance contract.
            independent: Require a complete independent
                explanation reference suite before packaging.
            cancellation: Optional thread-safe signal for cooperative
                cancellation at lifecycle checkpoints.
            checkpoint: Optional signed checkpoint file for durable
                modality recovery. It must differ from the report path the
                project uses (`modality_lifecycle.json` in `directory`).
            resume: Resume from ``checkpoint`` after verifying its
                signature, identity, and deterministic replay of the completed
                work.

        Raises:
            TypeError: If an object is not a modality adapter, a flag or the
                seed has the wrong type, or the cancellation is not a
                cancellation signal.
            ValueError: If the signing key is empty.
        """
        if not isinstance(adapter, modality_lifecycle.Preparer):
            raise errors.KindError("adapter must be a Preparer")
        vet(key, seed, independent, resume)
        if cancellation is not None and not isinstance(
            cancellation, modality_lifecycle.Cancellation
        ):
            raise errors.KindError("cancellation must be a Cancellation")
        if observed is not None and not isinstance(
            observed, modality_lifecycle.Preparer
        ):
            raise errors.KindError("observed_adapter must be a Preparer")
        self.adapter = adapter
        self.key = key
        self.directory = pathlib.Path(directory)
        self.seed = seed
        self.observed = observed
        self.budget = budget
        self.baseline = baseline
        self.manifest = manifest
        self.independent = independent
        self.checkpoint = (
            pathlib.Path(checkpoint) if checkpoint is not None else None
        )
        self.resume = resume
        self.cancellation = cancellation or modality_lifecycle.Cancellation()
        self.lifecycle = modality_lifecycle.Lifecycle()
        self.report: modality_lifecycle.Record | None = None

    def run(self) -> modality_lifecycle.Record:
        """Runs the full lifecycle and writes its package and signed report.

        A cancelled run raises `AbortedError`; other lifecycle
        failures propagate unchanged.

        Returns:
            The lifecycle report, also kept as `report`.

        Raises:
            ValueError: If `directory` exists and is not a directory.
        """
        if self.directory.exists() and not self.directory.is_dir():
            raise errors.InputError("output_dir must be a directory")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.report = self.lifecycle.run(
            self.adapter,
            self.directory / "model",
            self.key,
            seed=self.seed,
            observed_adapter=self.observed,
            report_path=self.directory / "modality_lifecycle.json",
            performance_budget=self.budget,
            performance_baseline=self.baseline,
            manifest=self.manifest,
            require_independent_ground_truth=self.independent,
            cancellation_check=self.cancellation.is_cancelled,
            checkpoint_path=self.checkpoint,
            resume=self.resume,
        )
        return self.report

    def cancel(self) -> None:
        """Requests cooperative cancellation of the active lifecycle run."""
        self.cancellation.cancel()

    def declaration(self) -> modalities.Declaration:
        """Returns the experimental capability declaration after a run.

        Returns:
            The declaration derived from the latest report.

        Raises:
            RuntimeError: If `run` has not completed.
        """
        if self.report is None:
            raise errors.OrderError("run must complete before declaration")
        return self.lifecycle.declaration(self.report)
