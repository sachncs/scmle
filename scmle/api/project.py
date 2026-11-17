"""Public API: `Project(...).plan()`, `.run()`, `result.predict()`."""

import pathlib
from typing import Any
from typing import Literal

import pandas

from scmle.api.curator import Curator
from scmle.api.estimation import declare
from scmle.api.estimation import estimate
from scmle.api.estimation import persist
from scmle.api.estimation import restore
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
import scmle.pipeline.context
import scmle.pipeline.orchestrator
import scmle.pipeline.recommendations

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


class Project:
    """A declarative ML project over a tabular dataset.

    Attributes:
        spec: The full project specification.
        output_dir: Parent directory for run directories.
        estimand: The declared population quantity for parameter estimation, or
            None.
        estimation_policy: The declared estimation policy, or None.
        orchestrator: The orchestrator that executes the project.
        profile: The dataset profile, once `plan` has run.
    """

    def __init__(
        self,
        data: Any,
        target: str | list[str],
        task: tasks.TaskRequest = "auto",
        metric: str | None = None,
        time_budget: int = 1800,
        *,
        modality: tasks.DatasetModality = "tabular",
        workers: int = 1,
        enforcement: Literal["hard", "cooperative"] = "hard",
        executor: Literal["process", "ray"] = "process",
        cluster_address: str | None = None,
        memory_limit_gb: float = 8.0,
        id_columns: list[str] | None = None,
        group_column: str | None = None,
        time_column: str | None = None,
        time_as_feature: bool = False,
        excluded_columns: list[str] | None = None,
        unavailable_at_prediction: list[str] | None = None,
        feature_delays: dict[str, int] | None = None,
        sensitive_columns: list[str] | None = None,
        weight_column: str | None = None,
        event_column: str | None = None,
        query_column: str | None = None,
        memory_budget_gb: float | None = None,
        training: Literal["sample", "stream"] = "sample",
        treatment_column: str | None = None,
        horizon: int | None = None,
        seasonal_period: int | None = None,
        forecast_lags: list[int] | None = None,
        forecast_windows: list[int] | None = None,
        forecast_calendar: list[str] | None = None,
        text_encoding: Literal["tfidf", "embedding"] = "tfidf",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        ordinal_order: list[str | int] | None = None,
        unlabeled: Literal["block", "ignore", "pseudo_label"] = "block",
        reference: str | None = None,
        pseudo_label_confidence: float = 0.9,
        monotonic_constraints: dict[str, int] | None = None,
        embargo_rows: int = 0,
        horizon_rows: int | None = None,
        window_rows: int | None = None,
        validation: ValidationStrategy = "auto",
        folds: int = 5,
        test_fraction: float = 0.2,
        seed: int = 42,
        search: tasks.Search | None = None,
        strategies: tasks.Strategies | None = None,
        models: tasks.Candidates | None = None,
        ensemble: tasks.Stacking | None = None,
        feature_graph: tasks.Derivation | None = None,
        selection: tasks.Selection | None = None,
        decision: tasks.Threshold | None = None,
        fairness: tasks.Parity | None = None,
        problem: tasks.Problem | None = None,
        estimand: parameter_estimation.Estimand | None = None,
        estimation_policy: parameter_estimation.Config | None = None,
        leakage_policy: LeakagePolicy = "warn",
        charter: str | pathlib.Path | None = None,
        leakage_attestations: str | pathlib.Path | None = None,
        baseline_column: str | None = None,
        baseline_subject: Literal["frozen_rule", "selected_model"] = (
            "frozen_rule"
        ),
        framing_draft: str | pathlib.Path | None = None,
        propensity_column: str | None = None,
        logged_action_column: str | None = None,
        ingest_plan: str | pathlib.Path | None = None,
        checkpoints: tuple[policies.Reviewer, ...] = (),
        output_dir: str | pathlib.Path = "runs",
    ) -> None:
        """Defines a project.

        An inconsistent configuration raises pydantic's `ValidationError` when
        the specification is built.

        Args:
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
            modality: Dataset family declaration. The supported beta path is
                `tabular`; other families fail closed until their complete
                lifecycle is registered.
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

        Raises:
            ConfigError: If only one of `estimand` and `estimation_policy` is
                given.
        """
        frame = (
            interchange.to_pandas(data) if interchange.tablelike(data) else None
        )
        self.spec = tasks.Config(
            data=tasks.Dataset(
                uri="<in-memory>" if frame is not None else str(data),
                target=target if isinstance(target, str) else target[0],
                modality=modality,
                targets=[] if isinstance(target, str) else list(target),
                id_columns=id_columns or [],
                group_column=group_column,
                time_column=time_column,
                time_as_feature=time_as_feature,
                excluded_columns=excluded_columns or [],
                unavailable_at_prediction=unavailable_at_prediction or [],
                feature_delays=feature_delays or {},
                sensitive_columns=sensitive_columns or [],
                weight_column=weight_column,
                event_column=event_column,
                query_column=query_column,
                memory_budget_gb=memory_budget_gb,
                training=training,
                treatment_column=treatment_column,
                horizon=horizon,
                seasonal_period=seasonal_period,
                forecast_lags=forecast_lags or [],
                forecast_windows=forecast_windows or [],
                forecast_calendar=forecast_calendar or [],
                text_encoding=text_encoding,
                embedding_model=embedding_model,
                ordinal_order=ordinal_order or [],
                unlabeled=unlabeled,
                reference_uri=reference,
                pseudo_label_confidence=pseudo_label_confidence,
                monotonic_constraints=monotonic_constraints or {},
                baseline_column=baseline_column,
                baseline_subject=baseline_subject,
                propensity_column=propensity_column,
                logged_action_column=logged_action_column,
                ingest_plan=None if ingest_plan is None else str(ingest_plan),
                framing_draft=(
                    None if framing_draft is None else str(framing_draft)
                ),
            ),
            task=task,
            metric=metric,
            validation=tasks.Validation(
                strategy=validation,
                embargo_rows=embargo_rows,
                horizon_rows=horizon_rows,
                window_rows=window_rows,
                folds=folds,
                test_fraction=test_fraction,
                seed=seed,
            ),
            resources=tasks.Resources(
                time_budget_seconds=time_budget,
                workers=workers,
                enforcement=enforcement,
                executor=executor,
                cluster_address=cluster_address,
                memory_limit_gb=memory_limit_gb,
            ),
            search=search or tasks.Search(),
            strategies=strategies or tasks.Strategies(),
            models=models or tasks.Candidates(),
            ensemble=ensemble or tasks.Stacking(),
            feature_graph=feature_graph or tasks.Derivation(),
            selection=selection or tasks.Selection(),
            decision=decision or tasks.Threshold(),
            fairness=fairness or tasks.Parity(),
            problem=problem or tasks.Problem(),
            leakage_policy=leakage_policy,
            charter_path=None if charter is None else str(charter),
            leakage_attestations=(
                None
                if leakage_attestations is None
                else str(leakage_attestations)
            ),
        )
        self.output_dir = pathlib.Path(output_dir)
        if (estimand is None) != (estimation_policy is None):
            raise errors.ConfigError(
                "parameter estimation requires both an explicit estimand "
                "and an estimation policy"
            )
        self.estimand = estimand
        self.estimation_policy = estimation_policy
        self.orchestrator = scmle.pipeline.orchestrator.Orchestrator(
            self.spec, frame, checkpoints=checkpoints
        )
        self.orchestrator.tracking_path = self.output_dir / "experiments.sqlite"
        self.profile: scmle.contracts.data.Portrait | None = None

    def plan(self) -> experiments.Plan:
        """Analyzes the data and proposes a workflow; trains nothing.

        Returns:
            The proposed experiment plan. The dataset profile is kept as
            `profile`.
        """
        context = self.orchestrator.prepare()
        try:
            self.profile = context.profile
            return scmle.pipeline.context.require(context.plan, "plan")
        finally:
            context.close()

    def recommendations(
        self,
    ) -> tuple[scmle.pipeline.recommendations.Recommendation, ...]:
        """Advice about the data and specification, from the profile.

        Returns:
            Recommendations for the user (never applied automatically).
        """
        context = self.orchestrator.prepare()
        try:
            return context.recommendations
        finally:
            context.close()

    def run(self, plan: experiments.Plan | None = None) -> results.Run:
        """Executes the workflow.

        If an estimand and policy were declared, the parameter estimation runs
        before the workflow and its evidence is bound to the result after it.

        Args:
            plan: An approved or modified plan restricting candidates.

        Returns:
            The completed run.
        """
        estimation = None
        if self.estimand is not None or self.estimation_policy is not None:
            estimation = self.estimate_parameters()
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
                ``parameter_estimation.json``.

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
        return estimate(
            self.orchestrator,
            self.estimand,
            self.estimation_policy,
            artifact_directory,
        )

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
        if not signing_key:
            raise errors.InputError("signing_key must not be empty")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise errors.KindError("seed must be an integer")
        if not isinstance(require_independent_ground_truth, bool):
            raise errors.KindError(
                "require_independent_ground_truth must be bool"
            )
        if not isinstance(resume, bool):
            raise errors.KindError("resume must be bool")
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

        A cancelled run raises `LifecycleCancelledError`; other lifecycle
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
