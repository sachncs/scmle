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
    "ModalityProject",
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
        time_budget: Wall-clock budget in seconds.
        modality: Dataset family; a file needs a registered `kinds.Kind`.
        workers: Concurrent trial processes (1 = sequential).
        enforcement: `hard` supervises every trial in its own process and
            kills it past the time or memory limit; `cooperative` only stops
            new trials from starting (and allows in-process hooks).
        executor: `process` (default) or `ray`.
        cluster_address: Ray cluster to join (a local runtime if None).
        memory_limit_gb: Hard memory cap per trial.
        id_columns: Identifier columns excluded from features.
        group_column: Entity column that must not straddle splits.
        time_column: Column ordering rows in time.
        cutoff: The prediction time, ISO 8601 text such as `2020-01-01`:
            a date column with values after it was not known when the
            prediction is made, and the leakage screen blocks it.
        known: Column to when it becomes known: `prediction` (by the time
            the prediction is made, as an unlisted column is taken to be) or
            `outcome` (only once the outcome is). The leakage screen blocks
            a feature declared `outcome`.
        time_as_feature: Also let the model use the time column as a feature
            (off by default; models must extrapolate in time).
        excluded_columns: Other columns to ignore.
        unavailable_at_prediction: Columns known only after the outcome
            (excluded; the feature-availability contract).
        feature_delays: Columns whose value is known only that many rows
            after the row it describes; the model sees the value from that
            many rows earlier (needs `time_column`).
        sensitive_columns: Columns describing groups of people; the fairness
            report compares outcomes across them.
        weight_column: Numeric column of per-row training weights (never a
            feature; not needed at prediction time; test metrics are also
            reported weighted).
        event_column: For `task="survival"`, the 0/1 column saying whether
            the event was observed (1) or the row was censored (0); the
            target is the time to the event or to censoring.
        query_column: For `task="ranking"`, the column identifying each
            row's query; the target is the relevance grade.
        memory_budget_gb: Largest in-memory size of the dataset; a bigger
            Parquet / CSV file is streamed (see `Dataset`).
        training: `sample` or `stream` for a streamed file.
        treatment_column: For `task="uplift"`, the 0/1 column saying whether
            a row was treated; the target is the outcome.
        horizon: For `task="forecast"`, how many steps ahead to predict
            (needs `time_column`; `target` is the series value).
        seasonal_period: Rows per season of a forecast series.
        forecast_lags: Lags of the value used as features (default 1, 2, 3
            and 7).
        forecast_windows: Rolling-window sizes (default 3 and 7).
        forecast_calendar: Date parts of the time column to add.
        text_encoding: `tfidf` (default) or `embedding` for free-text
            columns (needs the `embeddings` extra; falls back to TF-IDF with
            a warning when unavailable).
        embedding_model: Sentence-embedding model for `embedding`.
        ordinal_order: For `task="ordinal"`, the target's labels from lowest
            to highest.
        unlabeled: What to do with rows whose target is missing: `block`
            (default), `ignore`, or `pseudo_label` (classification; used
            only if it wins in cross-validation).
        reference: Path of a recent sample of the population the model will
            serve; the development rows are compared with it.
        pseudo_label_confidence: Smallest probability trusted as a label.
        monotonic_constraints: Numeric feature to +1 / -1: predictions must
            never fall / rise as the feature grows (binary classification
            and regression).
        embargo_rows: Rows dropped between training and validation windows
            in temporal validation.
        horizon_rows: Rolling validation: rows in each validation window.
        window_rows: Rolling validation: rows in each training window (all
            earlier rows when omitted).
        entities: Temporal and rolling validation with `group_column`:
            `known` (default) forecasts the entities trained on, so an
            entity's past trains and its future is held out; `unseen`
            predicts new entities and keeps each on one side of every split.
        validation: Split strategy or `auto`.
        folds: Cross-validation folds.
        test_fraction: Share of rows locked away for the final test.
        seed: Random seed.
        search: Hyperparameter search settings; defaults to a small pruned
            random search. Use `Search(top_models=0)` to disable tuning.
        strategies: Which data strategies to evaluate; defaults to all. Use
            `Strategies(top_models=0)` to disable.
        models: Model names and families to exclude.
        ensemble: Stacking settings; `Stacking(size=0)` disables.
        feature_graph: Declarative derived features (row-wise, fitted and
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
        estimation_policy: Explicit likelihood, covariance, missingness, and
            uncertainty policy for parameter estimation.
        leakage_policy: `warn` or `block` on leakage hypotheses.
        charter: Path of an analysis charter (`scmle charter init`). When
            given, the development rows are split into exploration rows and
            a sealed confirmation set the exploration steps cannot read; the
            charter is checked before any profiling.
        leakage_attestations: JSON file of per-column attestations that a
            strong predictor flagged by the leakage screen is legitimate.
            Used only with `charter`; each counts only for that charter and
            while valid, and is recorded in the ledger at tier `attested`.
        baseline_column: Used with `charter` on a binary task whose charter
            has a cost model: the 0/1 decisions of the current process. It
            is never a feature. A rule is frozen on the exploration rows and
            the claim that it beats the current process over the cost range
            is registered in the ledger and tested once on the confirmation
            set (`baseline_report.json`).
        baseline_subject: `frozen_rule` (default) or `selected_model`; the
            latter defers the opening of the confirmation set until the
            model selection is frozen and tests the selected model against
            the current process instead of the logistic rule.
        framing_draft: The `framing_draft.json` written by `scmle frame` for
            the charter; a claim that depends on an LLM-filled field of an
            unsigned charter stays `unverified`.
        propensity_column: With `logged_action_column`, the recorded
            probability of action 1; never a feature. It enables the
            off-policy value of the frozen rule (`policy_value.json`).
        logged_action_column: With a charter decision block, the 0/1 action
            the current process logged; never a feature.
        ingest_plan: Path of an ingest plan (JSON); needs a charter with an
            `ingest` block. Features are synthesized through the as-of views
            of the ingest plan's database before the data are split.
        checkpoints: Approval gates after planning and selection (the latter
            fires before the test set is opened).
        output_dir: Parent directory for run directories.
    """

    data: Any
    target: str | list[str]
    task: tasks.TaskRequest = "auto"
    metric: str | None = None
    time_budget: int = 1800
    marker: dataclasses.KW_ONLY
    modality: tasks.DatasetModality = "tabular"
    workers: int = 1
    enforcement: Literal["hard", "cooperative"] = "hard"
    executor: Literal["process", "ray"] = "process"
    cluster_address: str | None = None
    memory_limit_gb: float = 8.0
    id_columns: list[str] | None = None
    group_column: str | None = None
    time_column: str | None = None
    cutoff: str | None = None
    known: dict[str, Literal["prediction", "outcome"]] | None = None
    time_as_feature: bool = False
    excluded_columns: list[str] | None = None
    unavailable_at_prediction: list[str] | None = None
    feature_delays: dict[str, int] | None = None
    sensitive_columns: list[str] | None = None
    weight_column: str | None = None
    event_column: str | None = None
    query_column: str | None = None
    memory_budget_gb: float | None = None
    training: Literal["sample", "stream"] = "sample"
    treatment_column: str | None = None
    horizon: int | None = None
    seasonal_period: int | None = None
    forecast_lags: list[int] | None = None
    forecast_windows: list[int] | None = None
    forecast_calendar: list[str] | None = None
    text_encoding: Literal["tfidf", "embedding"] = "tfidf"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ordinal_order: list[str | int] | None = None
    unlabeled: Literal["block", "ignore", "pseudo_label"] = "block"
    reference: str | None = None
    pseudo_label_confidence: float = 0.9
    monotonic_constraints: dict[str, int] | None = None
    embargo_rows: int = 0
    horizon_rows: int | None = None
    window_rows: int | None = None
    entities: Literal["known", "unseen"] = "known"
    validation: ValidationStrategy = "auto"
    folds: int = 5
    test_fraction: float = 0.2
    seed: int = 42
    search: tasks.Search | None = None
    strategies: tasks.Strategies | None = None
    models: tasks.Candidates | None = None
    ensemble: tasks.Stacking | None = None
    feature_graph: tasks.Derivation | None = None
    selection: tasks.Selection | None = None
    decision: tasks.Threshold | None = None
    fairness: tasks.Parity | None = None
    problem: tasks.Problem | None = None
    estimand: parameter_estimation.Estimand | None = None
    estimation_policy: parameter_estimation.Config | None = None
    leakage_policy: LeakagePolicy = "warn"
    charter: str | pathlib.Path | None = None
    leakage_attestations: str | pathlib.Path | None = None
    baseline_column: str | None = None
    baseline_subject: Literal["frozen_rule", "selected_model"] = "frozen_rule"
    framing_draft: str | pathlib.Path | None = None
    propensity_column: str | None = None
    logged_action_column: str | None = None
    ingest_plan: str | pathlib.Path | None = None
    checkpoints: tuple[policies.Reviewer, ...] = ()
    output_dir: str | pathlib.Path = "runs"


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
        id_columns=definition.id_columns or [],
        group_column=definition.group_column,
        time_column=definition.time_column,
        cutoff=definition.cutoff,
        known=definition.known or {},
        time_as_feature=definition.time_as_feature,
        excluded_columns=definition.excluded_columns or [],
        unavailable_at_prediction=definition.unavailable_at_prediction or [],
        feature_delays=definition.feature_delays or {},
        sensitive_columns=definition.sensitive_columns or [],
        weight_column=definition.weight_column,
        event_column=definition.event_column,
        query_column=definition.query_column,
        memory_budget_gb=definition.memory_budget_gb,
        training=definition.training,
        treatment_column=definition.treatment_column,
        horizon=definition.horizon,
        seasonal_period=definition.seasonal_period,
        forecast_lags=definition.forecast_lags or [],
        forecast_windows=definition.forecast_windows or [],
        forecast_calendar=definition.forecast_calendar or [],
        text_encoding=definition.text_encoding,
        embedding_model=definition.embedding_model,
        ordinal_order=definition.ordinal_order or [],
        unlabeled=definition.unlabeled,
        reference_uri=definition.reference,
        pseudo_label_confidence=definition.pseudo_label_confidence,
        monotonic_constraints=definition.monotonic_constraints or {},
        baseline_column=definition.baseline_column,
        baseline_subject=definition.baseline_subject,
        propensity_column=definition.propensity_column,
        logged_action_column=definition.logged_action_column,
        ingest_plan=None
        if definition.ingest_plan is None
        else str(definition.ingest_plan),
        framing_draft=(
            None
            if definition.framing_draft is None
            else str(definition.framing_draft)
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
            embargo_rows=definition.embargo_rows,
            horizon_rows=definition.horizon_rows,
            window_rows=definition.window_rows,
            entities=definition.entities,
            folds=definition.folds,
            test_fraction=definition.test_fraction,
            seed=definition.seed,
        ),
        resources=tasks.Resources(
            time_budget_seconds=definition.time_budget,
            workers=definition.workers,
            enforcement=definition.enforcement,
            executor=definition.executor,
            cluster_address=definition.cluster_address,
            memory_limit_gb=definition.memory_limit_gb,
        ),
        search=definition.search or tasks.Search(),
        strategies=definition.strategies or tasks.Strategies(),
        models=definition.models or tasks.Candidates(),
        ensemble=definition.ensemble or tasks.Stacking(),
        feature_graph=definition.feature_graph or tasks.Derivation(),
        selection=definition.selection or tasks.Selection(),
        decision=definition.decision or tasks.Threshold(),
        fairness=definition.fairness or tasks.Parity(),
        problem=definition.problem or tasks.Problem(),
        leakage_policy=definition.leakage_policy,
        charter_path=None
        if definition.charter is None
        else str(definition.charter),
        leakage_attestations=(
            None
            if definition.leakage_attestations is None
            else str(definition.leakage_attestations)
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
        self.output_dir = pathlib.Path(self.output_dir)
        if (self.estimand is None) != (self.estimation_policy is None):
            raise errors.ConfigError(
                "parameter estimation requires both an explicit estimand "
                "and an estimation policy"
            )
        self.orchestrator = scmle.pipeline.orchestrator.Orchestrator(
            self.spec, frame, checkpoints=self.checkpoints
        )
        self.orchestrator.tracking = self.output_dir / "experiments.sqlite"
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
                configuration (see `estimate_parameters`).
            DataError: If it is refused for the data.
        """
        estimation = None
        if self.estimand is not None and self.estimation_policy is not None:
            try:
                estimation = self.estimate_parameters()
            except REFUSALS as error:
                refuse(
                    self.output_dir,
                    self.estimand,
                    self.estimation_policy,
                    error,
                )
                raise
        declaration = declare(self.estimand, self.estimation_policy)
        result = self.orchestrator.run(
            self.output_dir,
            plan,
            parameter_estimation_declaration=declaration,
        )
        if estimation is not None:
            persist(result, estimation)
        return result

    def estimate_parameters(
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
        if self.estimand is None or self.estimation_policy is None:
            raise errors.ConfigError(
                "parameter estimation is disabled: declare both estimand "
                "and estimation_policy explicitly"
            )
        try:
            return estimate(
                self.orchestrator,
                self.estimand,
                self.estimation_policy,
                artifact_directory,
            )
        except REFUSALS as error:
            if artifact_directory is not None:
                refuse(
                    artifact_directory,
                    self.estimand,
                    self.estimation_policy,
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
    def verify_replay(
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
            reproducibility.reread(rerun.run_dir),
        )

    def cancel(self) -> None:
        """Requests cancellation of a running workflow."""
        self.orchestrator.cancel()


class ModalityProject:
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
        signing_key: str | bytes,
        *,
        output_dir: str | pathlib.Path = "runs",
        seed: int = 42,
        observed_adapter: modality_lifecycle.Preparer | None = None,
        performance_budget: performance.Budget | None = None,
        performance_baseline: performance.Baseline | None = None,
        manifest: scmle.contracts.data.Passport | None = None,
        require_independent_ground_truth: bool = False,
        cancellation: modality_lifecycle.Cancellation | None = None,
        checkpoint_path: str | pathlib.Path | None = None,
        resume: bool = False,
    ) -> None:
        """Defines an experimental lifecycle run.

        Args:
            adapter: Typed adapter that owns modality validation and features.
            signing_key: Non-empty key used to authenticate the package.
            output_dir: Directory for the package (in its `model` subdirectory,
                which must not already exist) and the lifecycle report
                (`modality_lifecycle.json`).
            seed: Deterministic split and model seed.
            observed_adapter: Optional shifted dataset for monitoring.
            performance_budget: Optional declared performance budget.
            performance_baseline: Optional comparable-run baseline.
            manifest: Optional identity and governance contract.
            require_independent_ground_truth: Require a complete independent
                explanation reference suite before packaging.
            cancellation: Optional thread-safe signal for cooperative
                cancellation at lifecycle checkpoints.
            checkpoint_path: Optional signed checkpoint file for durable
                modality recovery. It must differ from the report path the
                project uses (`modality_lifecycle.json` in `output_dir`).
            resume: Resume from ``checkpoint_path`` after verifying its
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
        vet(signing_key, seed, require_independent_ground_truth, resume)
        if cancellation is not None and not isinstance(
            cancellation, modality_lifecycle.Cancellation
        ):
            raise errors.KindError("cancellation must be a Cancellation")
        if observed_adapter is not None and not isinstance(
            observed_adapter, modality_lifecycle.Preparer
        ):
            raise errors.KindError("observed_adapter must be a Preparer")
        self.adapter = adapter
        self.signing_key = signing_key
        self.output_dir = pathlib.Path(output_dir)
        self.seed = seed
        self.observed_adapter = observed_adapter
        self.performance_budget = performance_budget
        self.performance_baseline = performance_baseline
        self.manifest = manifest
        self.require_independent_ground_truth = require_independent_ground_truth
        self.checkpoint_path = (
            pathlib.Path(checkpoint_path)
            if checkpoint_path is not None
            else None
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
            ValueError: If `output_dir` exists and is not a directory.
        """
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise errors.InputError("output_dir must be a directory")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report = self.lifecycle.run(
            self.adapter,
            self.output_dir / "model",
            self.signing_key,
            seed=self.seed,
            observed_adapter=self.observed_adapter,
            report_path=self.output_dir / "modality_lifecycle.json",
            performance_budget=self.performance_budget,
            performance_baseline=self.performance_baseline,
            manifest=self.manifest,
            require_independent_ground_truth=self.require_independent_ground_truth,
            cancellation_check=self.cancellation.is_cancelled,
            checkpoint_path=self.checkpoint_path,
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
