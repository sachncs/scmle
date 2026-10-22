"""Public API: `MLProject(...).plan()`, `.run()`, `result.predict()`."""

import dataclasses
import json
import pathlib
from typing import Any
from typing import Literal

import pandas as pd

from scml import durability
from scml.api import results
from scml.artifacts import reproducibility
from scml.contracts import data as data_contracts
from scml.contracts import errors
from scml.contracts import experiments
from scml.contracts import modalities
from scml.contracts import performance
from scml.contracts import tasks
from scml.core import context as context_module
from scml.core import orchestrator as orchestrator_module
from scml.core import policies as checkpoints_module
from scml.data.readers import interchange
from scml.data.readers import tabular
from scml.evaluation import parameter_estimation
from scml.governance import attestation
from scml.intelligence import recommendations as recommendations_module
from scml.models import modality_lifecycle

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


class MLProject:
    """A declarative ML project over a tabular dataset."""

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
        search: tasks.SearchSpec | None = None,
        strategies: tasks.StrategySpec | None = None,
        models: tasks.ModelSpec | None = None,
        ensemble: tasks.EnsembleSpec | None = None,
        feature_graph: tasks.FeatureGraphSpec | None = None,
        selection: tasks.SelectionSpec | None = None,
        decision: tasks.DecisionSpec | None = None,
        fairness: tasks.FairnessSpec | None = None,
        problem: tasks.ProblemSpec | None = None,
        estimand: parameter_estimation.EstimandSpec | None = None,
        estimation_policy: parameter_estimation.EstimationPolicy | None = None,
        leakage_policy: LeakagePolicy = "warn",
        checkpoints: tuple[checkpoints_module.PolicyCheckpoint, ...] = (),
        output_dir: str | pathlib.Path = "runs",
    ) -> None:
        """Defines a project.

        Args:
            data: Supported tabular path (CSV/TSV, JSON/JSONL, Excel,
                Parquet or Feather), or an in-memory table: pandas DataFrame,
                pyarrow Table, Polars frame, or anything implementing the
                Arrow stream protocol.
            target: Name of the target column, or for `multilabel` and
                `multioutput` tasks the list of all target columns.
            task: `auto`, `classification`, `regression`, or an explicit
                binary/multiclass task.
            modality: Dataset family declaration. The supported beta path is
                `tabular`; other families fail closed until their complete
                lifecycle is registered.
            metric: Primary metric; defaults per task.
            time_budget: Wall-clock budget in seconds.
            executor: `process` (default) or `ray`.
            cluster_address: Ray cluster to join (a local runtime if None).
            memory_limit_gb: Hard memory cap per trial.
            workers: Concurrent trial processes (1 = sequential).
            enforcement: `hard` supervises every trial in its own process and
                kills it past the time or memory limit; `cooperative` only
                stops new trials from starting (and allows in-process hooks).
            id_columns: Identifier columns excluded from features.
            group_column: Entity column that must not straddle splits.
            time_column: Column ordering rows in time.
            time_as_feature: Also let the model use the time column as a
                feature (off by default; models must extrapolate in time).
            excluded_columns: Other columns to ignore.
            unavailable_at_prediction: Columns known only after the outcome
                (excluded; the feature-availability contract).
            feature_delays: Columns whose value is known only that many rows
                after the row it describes; the model sees the value from
                that many rows earlier (needs `time_column`).
            sensitive_columns: Columns describing groups of people; the
                fairness report compares outcomes across them.
            event_column: For `task="survival"`, the 0/1 column saying
                whether the event was observed (1) or the row was censored
                (0); the target is the time to the event or to censoring.
            text_encoding: `tfidf` (default) or `embedding` for free-text
                columns (needs the `embeddings` extra; falls back to TF-IDF
                with a warning when unavailable).
            embedding_model: Sentence-embedding model for `embedding`.
            memory_budget_gb: Largest in-memory size of the dataset; a bigger
                Parquet / CSV file is streamed (see `DataSpec`).
            training: `sample` or `stream` for a streamed file.
            treatment_column: For `task="uplift"`, the 0/1 column saying
                whether a row was treated; the target is the outcome.
            horizon: For `task="forecast"`, how many steps ahead to predict
                (needs `time_column`; `target` is the series value).
            seasonal_period: Rows per season of a forecast series.
            forecast_lags: Lags of the value used as features (default
                1, 2, 3 and 7).
            forecast_windows: Rolling-window sizes (default 3 and 7).
            forecast_calendar: Date parts of the time column to add.
            query_column: For `task="ranking"`, the column identifying each
                row's query; the target is the relevance grade.
            ordinal_order: For `task="ordinal"`, the target's labels from
                lowest to highest.
            reference: Path of a recent sample of the population the model will
                serve; the development rows are compared with it.
            unlabeled: What to do with rows whose target is missing: `block`
                (default), `ignore`, or `pseudo_label` (classification; used
                only if it wins in cross-validation).
            pseudo_label_confidence: Smallest probability trusted as a label.
            weight_column: Numeric column of per-row training weights (never
                a feature; not needed at prediction time; test metrics
                are also reported weighted).
            monotonic_constraints: Numeric feature to +1 / -1: predictions
                must never fall / rise as the feature grows (binary
                classification and regression).
            embargo_rows: Rows dropped between training and validation
                windows in temporal validation.
            horizon_rows: Rolling validation: rows in each validation window.
            window_rows: Rolling validation: rows in each training window
                (all earlier rows when omitted).
            validation: Split strategy or `auto`.
            folds: Cross-validation folds.
            test_fraction: Share of rows locked away for the final test.
            seed: Random seed.
            search: Hyperparameter search settings; defaults to a small
                pruned random search. Use `SearchSpec(top_models=0)` to
                disable tuning.
            strategies: Which data strategies to evaluate; defaults to all.
                Use `StrategySpec(top_models=0)` to disable.
            models: Model names and families to exclude.
            ensemble: Stacking settings; `EnsembleSpec(size=0)` disables.
            feature_graph: Declarative derived features (row-wise, fitted and
                time-series operations), versioned with the configuration.
            selection: Serving constraints (max model size, p95 latency).
            problem: The decision, the population and the scores the model
                must reach on the locked rows (checked after the one test
                release).
            estimand: Explicit population quantity for a separate parameter
                estimation evidence track. It does not affect model
                selection.
            estimation_policy: Explicit likelihood, covariance, missingness,
                and uncertainty policy for parameter estimation.
            fairness: Optional group-threshold mitigation of outcome gaps
                across a sensitive column (binary tasks).
            decision: Threshold optimization for binary tasks (objective
                f1, accuracy, balanced_accuracy or an explicit cost).
            leakage_policy: `warn` or `block` on leakage hypotheses.
            checkpoints: Approval gates after planning and selection (the
                latter fires before the test set is opened).
            output_dir: Parent directory for run directories.

        Raises:
            pydantic.ValidationError: If the configuration is inconsistent.
        """
        frame = (
            interchange.to_pandas(data)
            if interchange.is_tabular(data)
            else None
        )
        self.spec = tasks.ProjectSpec(
            data=tasks.DataSpec(
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
            ),
            task=task,
            metric=metric,
            validation=tasks.ValidationSpec(
                strategy=validation,
                embargo_rows=embargo_rows,
                horizon_rows=horizon_rows,
                window_rows=window_rows,
                folds=folds,
                test_fraction=test_fraction,
                seed=seed,
            ),
            resources=tasks.ResourceSpec(
                time_budget_seconds=time_budget,
                workers=workers,
                enforcement=enforcement,
                executor=executor,
                cluster_address=cluster_address,
                memory_limit_gb=memory_limit_gb,
            ),
            search=search or tasks.SearchSpec(),
            strategies=strategies or tasks.StrategySpec(),
            models=models or tasks.ModelSpec(),
            ensemble=ensemble or tasks.EnsembleSpec(),
            feature_graph=feature_graph or tasks.FeatureGraphSpec(),
            selection=selection or tasks.SelectionSpec(),
            decision=decision or tasks.DecisionSpec(),
            fairness=fairness or tasks.FairnessSpec(),
            problem=problem or tasks.ProblemSpec(),
            leakage_policy=leakage_policy,
        )
        self.output_dir = pathlib.Path(output_dir)
        if (estimand is None) != (estimation_policy is None):
            raise errors.ConfigError(
                "parameter estimation requires both an explicit estimand "
                "and an estimation policy"
            )
        self.estimand = estimand
        self.estimation_policy = estimation_policy
        self.orchestrator = orchestrator_module.Orchestrator(
            self.spec, frame, checkpoints=checkpoints
        )
        self.orchestrator.tracking_path = self.output_dir / "experiments.sqlite"
        self.profile: data_contracts.DatasetProfile | None = None

    def plan(self) -> experiments.ExperimentPlan:
        """Analyzes the data and proposes a workflow; trains nothing."""
        context = self.orchestrator.prepare()
        try:
            self.profile = context.profile
            return context_module.require(context.plan, "plan")
        finally:
            context.close()

    def recommendations(
        self,
    ) -> tuple[recommendations_module.Recommendation, ...]:
        """Advice about the data and specification, from the profile.

        Returns:
            Recommendations for the user (never applied automatically).
        """
        context = self.orchestrator.prepare()
        try:
            return context.recommendations
        finally:
            context.close()

    def run(
        self, plan: experiments.ExperimentPlan | None = None
    ) -> results.RunResult:
        """Executes the workflow.

        Args:
            plan: An approved or modified plan restricting candidates.

        Returns:
            The completed run.
        """
        estimation = None
        if self.estimand is not None or self.estimation_policy is not None:
            estimation = self.estimate_parameters()
        result = self.orchestrator.run(self.output_dir, plan)
        if estimation is not None:
            artifact = result.run_dir / "parameter_estimation.json"
            estimation.write(artifact)
            # The result must expose the exact digest-verified artifact that
            # was persisted, rather than an unverified in-memory equivalent.
            result.estimation = parameter_estimation.EstimationResult.read(
                artifact
            )
        return result

    def estimate_parameters(
        self, artifact_directory: str | pathlib.Path | None = None
    ) -> parameter_estimation.EstimationResult:
        """Estimates declared parameters without changing model selection.

        The estimation track is intentionally separate from the model
        selection track. It only runs when both scientific declarations are
        present, and it fails before fitting when the dataset or declaration
        is incompatible.

        Args:
            artifact_directory: Optional directory in which to write
                ``parameter_estimation.json``.

        Returns:
            The declared parameter estimates and uncertainty evidence.

        Raises:
            ConfigError: If the estimand, policy, or modality is undeclared
                or incompatible with this tabular estimator.
            DataError: If the declared columns are absent.
        """
        if self.estimand is None or self.estimation_policy is None:
            raise errors.ConfigError(
                "parameter estimation is disabled: declare both estimand "
                "and estimation_policy explicitly"
            )
        if self.spec.data.modality != "tabular":
            raise errors.ConfigError(
                "parameter estimation is supported only for the declared "
                "tabular modality"
            )
        frame = self.orchestrator.frame
        if frame is None:
            frame = tabular.read_table(self.spec.data.uri)
        declared = [*self.estimand.feature_names, self.estimand.outcome]
        treatment_column = None
        if self.estimation_policy.estimator == "ipw_ate":
            treatment_column = self.spec.data.treatment_column
            if treatment_column is None:
                raise errors.ConfigError(
                    "ipw_ate parameter estimation requires treatment_column"
                )
            if treatment_column not in self.estimand.feature_names:
                raise errors.ConfigError(
                    "treatment_column must be included in estimand "
                    "feature_names"
                )
            treatment_position = self.estimand.feature_names.index(
                treatment_column
            )
            if self.estimation_policy.treatment_position != treatment_position:
                raise errors.ConfigError(
                    "treatment_position must match treatment_column in "
                    "estimand feature_names"
                )
            declared.append(treatment_column)
        if self.estimation_policy.cluster_name is not None:
            declared.append(self.estimation_policy.cluster_name)
        declared = list(dict.fromkeys(declared))
        missing = sorted(set(declared) - set(frame.columns))
        if missing:
            raise errors.DataError(
                f"parameter estimation columns are missing: {missing}"
            )
        missing_counts = frame.loc[:, declared].isna().sum()
        missing_values = {
            name: int(count)
            for name, count in missing_counts.items()
            if count
        }
        if missing_values and self.estimation_policy.missing == "reject":
            raise errors.DataError(
                "parameter estimation missing values require an explicit "
                f"complete_case policy: {missing_values}"
            )
        features = frame.loc[:, list(self.estimand.feature_names)]
        outcome = frame.loc[:, self.estimand.outcome]
        clusters = None
        if self.estimation_policy.cluster_name is not None:
            clusters = frame.loc[:, self.estimation_policy.cluster_name]
        cluster_values = None if clusters is None else clusters.tolist()
        estimation = parameter_estimation.estimate_parameters(
            features.to_numpy(),
            outcome.to_numpy(),
            self.estimand,
            self.estimation_policy,
            clusters=cluster_values,
        )
        estimation = dataclasses.replace(
            estimation,
            diagnostics={
                **estimation.diagnostics,
                "data_columns": {
                    "features": list(self.estimand.feature_names),
                    "outcome": self.estimand.outcome,
                    "treatment": treatment_column,
                    "cluster": self.estimation_policy.cluster_name,
                },
                "missingness": {
                    "policy": self.estimation_policy.missing,
                    "columns": declared,
                    "missing_counts": missing_values,
                },
            },
        )
        if artifact_directory is not None:
            artifact = (
                pathlib.Path(artifact_directory) / "parameter_estimation.json"
            )
            estimation.write(artifact)
            estimation = parameter_estimation.EstimationResult.read(artifact)
        return estimation

    @classmethod
    def resume(
        cls,
        run_directory: str | pathlib.Path,
        data: pd.DataFrame | None = None,
    ) -> results.RunResult:
        """Continues an interrupted run from its saved configuration.

        Completed trials are reused; the rest are computed.

        Args:
            run_directory: Directory of a cancelled, failed or crashed run.
            data: The original in-memory data, required only if the project
                was created from a DataFrame rather than a file.

        Returns:
            The completed run.
        """
        orchestrator = orchestrator_module.Orchestrator.for_run_directory(
            run_directory, frame=data
        )
        return orchestrator.resume(run_directory)

    @classmethod
    def verify_replay(
        cls,
        run_directory: str | pathlib.Path,
        data: pd.DataFrame | None = None,
        output_dir: str | pathlib.Path | None = None,
    ) -> reproducibility.ReplayVerdict:
        """Re-executes a finished run's configuration and compares the runs.

        Args:
            run_directory: A completed run directory.
            data: The original in-memory data, if the run used a DataFrame.
            output_dir: Where to write the rerun (default: next to the
                original).

        Returns:
            Whether the rerun is `exact`, `comparable` or `different`.
        """
        original = pathlib.Path(run_directory)
        orchestrator = orchestrator_module.Orchestrator.for_run_directory(
            original, frame=data
        )
        rerun = orchestrator.run(output_dir or original.parent)
        return reproducibility.compare_runs(
            reproducibility.load_run(original),
            reproducibility.load_run(rerun.run_dir),
        )

    def cancel(self) -> None:
        """Requests cancellation of a running workflow."""
        self.orchestrator.cancel()


class ModalityProject:
    """Runs a typed non-tabular dataset through the shared lifecycle.

    ``MLProject`` remains the supported tabular beta surface. This project
    surface accepts a concrete ``ModalityAdapter`` for experimental text,
    document, image, audio, video, graph, geospatial-vector, or raster data.
    It gives those datasets the same curation, explainability, packaging,
    inference, monitoring, documentation, and recovery evidence without
    pretending that a tabular estimator can decode arbitrary raw bytes.
    """

    def __init__(
        self,
        adapter: modality_lifecycle.ModalityAdapter,
        signing_key: str | bytes,
        *,
        output_dir: str | pathlib.Path = "runs",
        seed: int = 42,
        observed_adapter: modality_lifecycle.ModalityAdapter | None = None,
        performance_budget: performance.PerformanceBudget | None = None,
        performance_baseline: performance.PerformanceBaseline | None = None,
        manifest: data_contracts.ModalityManifest | None = None,
        require_independent_ground_truth: bool = False,
        cancellation: modality_lifecycle.LifecycleCancellation | None = None,
        checkpoint_path: str | pathlib.Path | None = None,
        resume: bool = False,
    ) -> None:
        """Defines an experimental lifecycle run.

        Args:
            adapter: Typed adapter that owns modality validation and features.
            signing_key: Non-empty key used to authenticate the package.
            output_dir: New directory for the package and lifecycle report.
            seed: Deterministic split and model seed.
            observed_adapter: Optional shifted dataset for monitoring.
            performance_budget: Optional declared performance budget.
            performance_baseline: Optional comparable-run baseline.
            manifest: Optional identity and governance contract.
            require_independent_ground_truth: Require a complete independent
                explanation reference suite before packaging.
            cancellation: Optional thread-safe signal for cooperative
                cancellation at lifecycle stage boundaries.
            checkpoint_path: Optional signed, atomic stage checkpoint path for
                durable modality recovery. Supplying this requires the
                lifecycle report path managed by this project.
            resume: Resume from ``checkpoint_path`` after verifying its
                signature, identity, and deterministic completed-stage replay.

        Raises:
            TypeError: If an object is not a modality adapter.
            ValueError: If the signing key is empty or the output is invalid.
        """
        if not isinstance(adapter, modality_lifecycle.ModalityAdapter):
            raise TypeError("adapter must be a ModalityAdapter")
        if not signing_key:
            raise ValueError("signing_key must not be empty")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if not isinstance(require_independent_ground_truth, bool):
            raise TypeError("require_independent_ground_truth must be bool")
        if not isinstance(resume, bool):
            raise TypeError("resume must be bool")
        if cancellation is not None and not isinstance(
            cancellation, modality_lifecycle.LifecycleCancellation
        ):
            raise TypeError("cancellation must be a LifecycleCancellation")
        if observed_adapter is not None and not isinstance(
            observed_adapter, modality_lifecycle.ModalityAdapter
        ):
            raise TypeError("observed_adapter must be a ModalityAdapter")
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
        self.cancellation = (
            cancellation or modality_lifecycle.LifecycleCancellation()
        )
        self.lifecycle = modality_lifecycle.ModalityLifecycle()
        self.report: modality_lifecycle.LifecycleReport | None = None

    def run(self) -> modality_lifecycle.LifecycleReport:
        """Run all fifteen stages and persist their evidence atomically."""
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise ValueError("output_dir must be a directory")
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
        """Request cooperative cancellation of the active lifecycle run."""
        self.cancellation.cancel()

    def declaration(self) -> modalities.ModalityDeclaration:
        """Return the experimental capability declaration after a run."""
        if self.report is None:
            raise RuntimeError("run must complete before declaration")
        return self.lifecycle.declaration(self.report)


class ModalityCurationProject:
    """Curate and document a modality without forcing a model fit.

    This is the data-understanding entry point for unlabeled, unsupervised,
    or domain-specific datasets. It persists signed curation and dataset-card
    evidence, but it never invents labels, features, metrics, or a model.
    """

    def __init__(
        self,
        adapter: modality_lifecycle.ModalityCurationAdapter,
        signing_key: str | bytes,
        *,
        output_dir: str | pathlib.Path = "runs",
        manifest: data_contracts.ModalityManifest | None = None,
    ) -> None:
        """Define a curation-only run over a typed modality adapter.

        Args:
            adapter: Adapter that owns source identity and curation policy.
            signing_key: Non-empty key used to authenticate the evidence.
            output_dir: Directory where curation evidence is written.
            manifest: Optional source, lineage, privacy, and task contract.

        Raises:
            TypeError: If the adapter or manifest has the wrong type.
            ValueError: If the signing key is empty or the output is invalid.
        """
        if not isinstance(adapter, modality_lifecycle.ModalityCurationAdapter):
            raise TypeError("adapter must be a ModalityCurationAdapter")
        if not signing_key:
            raise ValueError("signing_key must not be empty")
        if manifest is not None and not isinstance(
            manifest, data_contracts.ModalityManifest
        ):
            raise TypeError("manifest must be ModalityManifest")
        self.adapter = adapter
        self.signing_key = signing_key
        self.output_dir = pathlib.Path(output_dir)
        self.manifest = manifest

    def run(self) -> modality_lifecycle.ModalityCurationResult:
        """Persist signed curation evidence without entering modelling."""
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise ValueError("output_dir must be a directory")
        samples = self.adapter.samples
        modality = modalities.modality_value(self.adapter.modality)
        dataset_hash = self.adapter.dataset_hash
        sample_count = len(samples)
        if self.manifest is not None:
            if self.manifest.modality != modality:
                raise ValueError("manifest modality must match adapter")
            if self.manifest.content_hash != dataset_hash:
                raise ValueError("manifest content_hash must match adapter")
            if self.manifest.sample_count != sample_count:
                raise ValueError("manifest sample_count must match adapter")
        report = self.adapter.curate()
        if not isinstance(
            report,
            (
                modality_lifecycle.curation.ModalityCurationReport,
                modality_lifecycle.document.DocumentCurationReport,
            ),
        ):
            raise TypeError(
                "adapter curate() must return a modality curation report"
            )
        curation_payload = report.to_dict()
        reported_dataset_hash = curation_payload.get("dataset_hash")
        reported_sample_count = curation_payload.get("sample_count")
        if reported_dataset_hash != dataset_hash:
            raise ValueError("curation dataset_hash must match adapter")
        if reported_sample_count != sample_count:
            raise ValueError("curation sample_count must match adapter")
        card = modality_lifecycle.curation.ModalityDatasetCard.from_report(
            modality,
            curation_payload,
            self.adapter.modality_notes(),
            manifest=self.manifest,
            task="not_declared",
        )
        unsigned: dict[str, object] = {
            "schema_version": 1,
            "record_type": "modality_curation",
            "modality": modality,
            "dataset_hash": dataset_hash,
            "sample_count": sample_count,
            "curation": curation_payload,
            "dataset_card": card.to_dict(),
        }
        signature = attestation.sign_payload(unsigned, self.signing_key)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "modality_curation.json"
        if path.exists():
            raise FileExistsError(path)
        payload = {
            **unsigned,
            "path": str(path),
            "signature": {
                "algorithm": attestation.SIGNATURE_ALGORITHM,
                "value": signature,
            },
        }
        durability.write_atomically(
            path,
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
        return modality_lifecycle.ModalityCurationResult(
            modality=modality,
            dataset_hash=dataset_hash,
            sample_count=sample_count,
            curation=curation_payload,
            dataset_card=card.to_dict(),
            path=str(path),
            signature=signature,
        )
