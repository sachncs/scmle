"""Public API: `MLProject(...).plan()`, `.run()`, `result.predict()`."""

import contextlib
import dataclasses
import hashlib
import json
import os
import pathlib
import tempfile
from typing import Any
from typing import Literal

import pandas as pd

from scmle import durability
from scmle.api import results
from scmle.artifacts import bundle as bundle_module
from scmle.artifacts import reproducibility
from scmle.contracts import data as data_contracts
from scmle.contracts import errors
from scmle.contracts import experiments
from scmle.contracts import lineage
from scmle.contracts import modalities
from scmle.contracts import performance
from scmle.contracts import tasks
from scmle.core import context as context_module
from scmle.core import orchestrator as orchestrator_module
from scmle.core import policies as checkpoints_module
from scmle.data.readers import connectors
from scmle.data.readers import interchange
from scmle.data.readers import tabular
from scmle.evaluation import parameter_estimation
from scmle.governance import attestation
from scmle.inference import evidence as inference_evidence
from scmle.intelligence import recommendations as recommendations_module
from scmle.models import modality_lifecycle

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


def publish_curation_artifact(path: pathlib.Path, content: bytes) -> None:
    """Publish curation bytes atomically without replacing an artifact.

    A hard link from a fully flushed sibling file gives the destination a
    no-clobber, atomic first publication.  ``FileExistsError`` is deliberately
    preserved for the caller so a concurrent publisher can verify and reuse its
    result instead of replacing it.

    Args:
        path: Destination of the artifact; its directory is created.
        content: The bytes to publish.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, path)
        durability.sync_directory(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)


PARAMETER_ESTIMATION_DECLARATION_ARTIFACT = (
    "parameter_estimation_declaration.json"
)
PARAMETER_ESTIMATION_DECLARATION_LOCK = "parameter_estimation_declaration.lock"
PARAMETER_ESTIMATION_DECLARATION_SCHEMA = 1


def build_parameter_estimation_declaration(
    estimand: parameter_estimation.EstimandSpec | None,
    policy: parameter_estimation.EstimationPolicy | None,
) -> dict[str, Any]:
    """Builds the explicit scientific declaration stored with a run.

    The orchestrator adds the final project configuration hash and the content
    digest immediately after it allocates the run directory. Keeping this record
    separate from the model-selection specification lets resume reconstruct the
    scientific contract without changing the public project specification
    schema.

    Args:
        estimand: The declared population quantity, or None.
        policy: The declared estimation policy, or None.

    Returns:
        A record with the schema version, whether estimation is enabled, and the
        estimand and policy when it is.

    Raises:
        ConfigError: If only one of the estimand and the policy is given.
    """
    if (estimand is None) != (policy is None):
        raise errors.ConfigError(
            "parameter estimation requires both an explicit estimand "
            "and an estimation policy"
        )
    declaration: dict[str, Any] = {
        "schema_version": PARAMETER_ESTIMATION_DECLARATION_SCHEMA,
        "enabled": estimand is not None,
    }
    if estimand is not None and policy is not None:
        declaration["estimand"] = estimand.to_dict()
        declaration["policy"] = policy.to_dict()
    return declaration


def parameter_estimation_declaration_digest(
    declaration: dict[str, Any],
) -> str:
    """Computes the canonical digest for a declaration without its digest.

    Args:
        declaration: The declaration record, without a `digest` entry.

    Returns:
        The hex SHA-256 of its compact, key-sorted JSON.
    """
    return hashlib.sha256(
        json.dumps(declaration, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def restore_parameter_estimation_declaration(
    run_directory: str | pathlib.Path,
    expected_config_hash: str,
) -> tuple[
    parameter_estimation.EstimandSpec | None,
    parameter_estimation.EstimationPolicy | None,
]:
    """Validates and reconstructs the scientific declaration for resume.

    Args:
        run_directory: Directory of the run holding the declaration and its lock
            file.
        expected_config_hash: Configuration hash the declaration must have been
            written for.

    Returns:
        The estimand and the policy, or a pair of None if estimation was
        disabled for the run.

    Raises:
        ArtifactError: If the declaration or its lock is missing, unreadable,
            not an object, lacks a digest, fails the digest check, has an
            unsupported schema or another configuration hash, has fields that do
            not match its enabled state, or describes an invalid estimand or
            policy.
    """
    artifact_path = pathlib.Path(run_directory)
    report_path = artifact_path / PARAMETER_ESTIMATION_DECLARATION_ARTIFACT
    lock_path = artifact_path / PARAMETER_ESTIMATION_DECLARATION_LOCK
    try:
        envelope = json.loads(report_path.read_text(encoding="utf-8"))
        lock_digest = lock_path.read_text(encoding="utf-8").strip()
    except (OSError, TypeError, ValueError) as error:
        raise errors.ArtifactError(
            "parameter estimation declaration is missing or unreadable",
            {"run_dir": str(artifact_path)},
        ) from error
    if not isinstance(envelope, dict):
        raise errors.ArtifactError(
            "parameter estimation declaration must be a JSON object",
            {"run_dir": str(artifact_path)},
        )
    digest = envelope.pop("digest", None)
    if not isinstance(digest, str):
        raise errors.ArtifactError(
            "parameter estimation declaration digest is missing",
            {"run_dir": str(artifact_path)},
        )
    if (
        digest != parameter_estimation_declaration_digest(envelope)
        or lock_digest != digest
    ):
        raise errors.ArtifactError(
            "parameter estimation declaration digest mismatch",
            {"run_dir": str(artifact_path)},
        )
    if (
        envelope.get("schema_version")
        != PARAMETER_ESTIMATION_DECLARATION_SCHEMA
    ):
        raise errors.ArtifactError(
            "unsupported parameter estimation declaration schema",
            {"run_dir": str(artifact_path)},
        )
    if envelope.get("project_config_hash") != expected_config_hash:
        raise errors.ArtifactError(
            "parameter estimation declaration does not match the run "
            "configuration",
            {"run_dir": str(artifact_path)},
        )
    enabled = envelope.get("enabled")
    if enabled is False:
        if set(envelope) != {
            "schema_version",
            "enabled",
            "project_config_hash",
        }:
            raise errors.ArtifactError(
                "disabled parameter estimation declaration has extra fields",
                {"run_dir": str(artifact_path)},
            )
        return None, None
    if enabled is not True or set(envelope) != {
        "schema_version",
        "enabled",
        "project_config_hash",
        "estimand",
        "policy",
    }:
        raise errors.ArtifactError(
            "parameter estimation declaration is incomplete",
            {"run_dir": str(artifact_path)},
        )
    try:
        estimand_record = envelope["estimand"]
        estimand = parameter_estimation.EstimandSpec(
            name=estimand_record["name"],
            outcome=estimand_record["outcome"],
            population=estimand_record["population"],
            likelihood=estimand_record["likelihood"],
            feature_names=tuple(estimand_record["feature_names"]),
            assumptions=tuple(estimand_record["assumptions"]),
            description=estimand_record.get("description", ""),
        )
        policy_values = dict(envelope["policy"])
        if "prior" in policy_values:
            prior_values = dict(policy_values["prior"])
            policy_values["prior"] = parameter_estimation.BayesianPrior(
                mean=tuple(prior_values["mean"]),
                precision=tuple(
                    tuple(row) for row in prior_values["precision"]
                ),
                shape=prior_values.get("shape", 2.0),
                scale=prior_values.get("scale", 1.0),
            )
        for field in (
            "causal_assumptions",
            "continuous_treatment_assumptions",
            "hierarchical_assumptions",
        ):
            if field in policy_values:
                policy_values[field] = tuple(policy_values[field])
        if "reml_variance_bounds" in policy_values:
            policy_values["reml_variance_bounds"] = tuple(
                policy_values["reml_variance_bounds"]
            )
        policy = parameter_estimation.EstimationPolicy(**policy_values)
    except (KeyError, TypeError, ValueError) as error:
        raise errors.ArtifactError(
            "parameter estimation declaration is invalid",
            {"run_dir": str(artifact_path)},
        ) from error
    return estimand, policy


def bind_parameter_estimation_evidence(
    run_directory: pathlib.Path,
    artifact: pathlib.Path,
    estimation: parameter_estimation.EstimationResult,
) -> None:
    """Bind a verified parameter report into the run evidence graph.

    The graph is rebuilt from its validated content so the evaluation node's
    digest reflects the evaluation file after it gains the parameter-report
    reference. The parameter report remains a scientific sibling of model
    evaluation: the snapshot supplies its input, but model selection does not
    consume its estimates.

    Args:
        run_directory: Directory of the run.
        artifact: The persisted parameter-estimation report.
        estimation: The verified estimation result.

    Raises:
        ArtifactError: If the evidence graph lacks the evaluation or snapshot
            node.
    """
    graph_path = run_directory / "evidence_graph.json"
    graph = inference_evidence.InferenceEvidenceLinker.load(graph_path)
    evaluation_node = graph.nodes.get("evaluation")
    snapshot_node = graph.nodes.get("snapshot")
    if evaluation_node is None or snapshot_node is None:
        raise errors.ArtifactError(
            "parameter estimation cannot bind to an incomplete evidence graph",
            {"artifact": artifact.name, "graph": graph_path.name},
        )
    nodes = [
        node for node_id, node in graph.nodes.items() if node_id != "evaluation"
    ]
    updated = lineage.EvidenceGraph(nodes)
    updated.add_node(
        lineage.EvidenceNode(
            "evaluation",
            lineage.EvidenceNodeKind.EVALUATION,
            bundle_module.file_sha256(run_directory / "evaluation.json"),
            {"artifact": "evaluation.json"},
        )
    )
    updated.add_node(
        lineage.EvidenceNode(
            "parameter-estimation",
            lineage.EvidenceNodeKind.PARAMETER_ESTIMATION,
            estimation.digest(),
            {
                "artifact": artifact.name,
                "estimator": estimation.policy.estimator,
                "estimand": estimation.estimand.name,
            },
        )
    )
    for edge in graph.edges:
        updated.add_edge(edge)
    updated.add_edge(
        lineage.EvidenceEdge("snapshot", "parameter-estimation", "estimates")
    )
    updated.add_edge(
        lineage.EvidenceEdge("evaluation", "parameter-estimation", "records")
    )
    record = updated.to_dict()
    record["graph_digest"] = updated.digest()
    inference_evidence.InferenceEvidenceLinker.write_atomically(
        graph_path, record
    )


def persist_parameter_estimation_evidence(
    result: results.RunResult,
    estimation: parameter_estimation.EstimationResult,
) -> None:
    """Persists and binds a verified parameter report to a run result.

    The report is written, read back and compared with the digest of what was
    written. The run's evaluation record is then rewritten with a reference to
    the report, the evidence graph is updated, and the result exposes the
    verified report.

    Args:
        result: The completed run; its evaluation and estimation are updated.
        estimation: The estimation to persist.

    Raises:
        ArtifactError: If the written report cannot be read back or its digest
            differs from the one written.
    """
    artifact = result.run_dir / "parameter_estimation.json"
    expected_digest = estimation.write(artifact)
    try:
        verified = parameter_estimation.EstimationResult.read(artifact)
    except (OSError, TypeError, ValueError, KeyError) as error:
        raise errors.ArtifactError(
            "parameter estimation artifact cannot be read",
            {
                "run_dir": str(result.run_dir),
                "artifact": artifact.name,
            },
        ) from error
    if verified.digest() != expected_digest:
        raise errors.ArtifactError(
            "parameter estimation artifact does not match the persisted "
            "estimate",
            {
                "run_dir": str(result.run_dir),
                "artifact": artifact.name,
            },
        )
    parameter_record = {
        "artifact": artifact.name,
        "digest": verified.digest(),
    }
    evaluation = {
        **result.evaluation,
        "parameter_estimation": parameter_record,
    }
    durability.write_atomically(
        result.run_dir / "evaluation.json",
        (
            json.dumps(
                evaluation,
                default=str,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
    )
    bind_parameter_estimation_evidence(result.run_dir, artifact, verified)
    result.evaluation.clear()
    result.evaluation.update(evaluation)
    # The result must expose the exact digest-verified artifact that was
    # persisted, rather than an unverified in-memory equivalent.
    result.estimation = verified


def estimate_parameters_for_project(
    orchestrator: orchestrator_module.Orchestrator,
    estimand: parameter_estimation.EstimandSpec,
    policy: parameter_estimation.EstimationPolicy,
    artifact_directory: str | pathlib.Path | None = None,
) -> parameter_estimation.EstimationResult:
    """Estimates the declared parameters against an orchestrator dataset.

    Args:
        orchestrator: The orchestrator holding the project's data and spec.
        estimand: The declared population quantity.
        policy: The declared estimation policy.
        artifact_directory: If given, the report is written there as
            ``parameter_estimation.json`` and read back verified.

    Returns:
        The estimation result, with diagnostics naming the columns used, the
        missingness handling and the data identity.

    Raises:
        ConfigError: If the modality is not tabular, or a treatment-based
            estimator lacks a consistent `treatment_column`.
        DataError: If a declared column is absent, or has missing values under
            the `reject` policy.
    """
    if orchestrator.spec.data.modality != "tabular":
        raise errors.ConfigError(
            "parameter estimation is supported only for the declared "
            "tabular modality"
        )
    frame = orchestrator.frame
    if frame is None:
        frame = tabular.read_table(orchestrator.spec.data.uri)
    declared = [*estimand.feature_names, estimand.outcome]
    treatment_column = None
    if policy.estimator in {
        "bayesian_ate",
        "ipw_ate",
        "aipw_ate",
        "continuous_ipw_ate",
    }:
        treatment_column = orchestrator.spec.data.treatment_column
        if treatment_column is None:
            raise errors.ConfigError(
                f"{policy.estimator} parameter estimation requires "
                "treatment_column"
            )
        if treatment_column not in estimand.feature_names:
            raise errors.ConfigError(
                "treatment_column must be included in estimand feature_names"
            )
        treatment_position = estimand.feature_names.index(treatment_column)
        if policy.treatment_position != treatment_position:
            raise errors.ConfigError(
                "treatment_position must match treatment_column in "
                "estimand feature_names"
            )
        declared.append(treatment_column)
    if policy.cluster_name is not None:
        declared.append(policy.cluster_name)
    declared = list(dict.fromkeys(declared))
    missing = sorted(set(declared) - set(frame.columns))
    if missing:
        raise errors.DataError(
            f"parameter estimation columns are missing: {missing}"
        )
    missing_counts = frame.loc[:, declared].isna().sum()
    missing_values = {
        name: int(count) for name, count in missing_counts.items() if count
    }
    if missing_values and policy.missing == "reject":
        raise errors.DataError(
            "parameter estimation missing values require an explicit "
            f"complete_case policy: {missing_values}"
        )
    features = frame.loc[:, list(estimand.feature_names)]
    outcome = frame.loc[:, estimand.outcome]
    clusters = None
    if policy.cluster_name is not None:
        clusters = frame.loc[:, policy.cluster_name]
    cluster_values = None if clusters is None else clusters.tolist()
    snapshot_id = tabular.snapshot_identity(frame, orchestrator.spec.data.uri)
    source_identity = None
    if orchestrator.spec.data.uri not in {"<memory>", "<in-memory>"}:
        source_identity = connectors.source_identity(orchestrator.spec.data.uri)
    estimation = parameter_estimation.estimate_parameters(
        features.to_numpy(),
        outcome.to_numpy(),
        estimand,
        policy,
        clusters=cluster_values,
    )
    estimation = dataclasses.replace(
        estimation,
        diagnostics={
            **estimation.diagnostics,
            "data_columns": {
                "features": list(estimand.feature_names),
                "outcome": estimand.outcome,
                "treatment": treatment_column,
                "cluster": policy.cluster_name,
            },
            "missingness": {
                "policy": policy.missing,
                "columns": declared,
                "missing_counts": missing_values,
            },
            "data_identity": {
                "source_uri": orchestrator.spec.data.uri,
                "source_identity": source_identity,
                "snapshot_id": snapshot_id,
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


class MLProject:
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
        checkpoints: tuple[checkpoints_module.PolicyCheckpoint, ...] = (),
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
            task: `auto`, `classification`, `regression`, or an explicit
                binary/multiclass task.
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
                Parquet / CSV file is streamed (see `DataSpec`).
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
                random search. Use `SearchSpec(top_models=0)` to disable tuning.
            strategies: Which data strategies to evaluate; defaults to all. Use
                `StrategySpec(top_models=0)` to disable.
            models: Model names and families to exclude.
            ensemble: Stacking settings; `EnsembleSpec(size=0)` disables.
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
        self.orchestrator = orchestrator_module.Orchestrator(
            self.spec, frame, checkpoints=checkpoints
        )
        self.orchestrator.tracking_path = self.output_dir / "experiments.sqlite"
        self.profile: data_contracts.DatasetProfile | None = None

    def plan(self) -> experiments.ExperimentPlan:
        """Analyzes the data and proposes a workflow; trains nothing.

        Returns:
            The proposed experiment plan. The dataset profile is kept as
            `profile`.
        """
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

        If an estimand and policy were declared, the parameter estimation runs
        first and its evidence is bound to the result.

        Args:
            plan: An approved or modified plan restricting candidates.

        Returns:
            The completed run.
        """
        estimation = None
        if self.estimand is not None or self.estimation_policy is not None:
            estimation = self.estimate_parameters()
        declaration = build_parameter_estimation_declaration(
            self.estimand, self.estimation_policy
        )
        result = self.orchestrator.run(
            self.output_dir,
            plan,
            parameter_estimation_declaration=declaration,
        )
        if estimation is not None:
            persist_parameter_estimation_evidence(result, estimation)
        return result

    def estimate_parameters(
        self, artifact_directory: str | pathlib.Path | None = None
    ) -> parameter_estimation.EstimationResult:
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
            ConfigError: If the estimand, policy, or modality is undeclared or
                incompatible with this tabular estimator.
        """
        if self.estimand is None or self.estimation_policy is None:
            raise errors.ConfigError(
                "parameter estimation is disabled: declare both estimand "
                "and estimation_policy explicitly"
            )
        return estimate_parameters_for_project(
            self.orchestrator,
            self.estimand,
            self.estimation_policy,
            artifact_directory,
        )

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
            data: The original in-memory data, required only if the project was
                created from a DataFrame rather than a file.

        Returns:
            The completed run.
        """
        run_path = pathlib.Path(run_directory)
        orchestrator = orchestrator_module.Orchestrator.for_run_directory(
            run_path, frame=data
        )
        estimand, policy = restore_parameter_estimation_declaration(
            run_path,
            orchestrator.spec.config_hash(),
        )
        estimation = None
        if estimand is not None and policy is not None:
            estimation = estimate_parameters_for_project(
                orchestrator, estimand, policy
            )
        result = orchestrator.resume(run_path)
        if estimation is not None:
            persist_parameter_estimation_evidence(result, estimation)
        return result

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
        estimand, policy = restore_parameter_estimation_declaration(
            original,
            orchestrator.spec.config_hash(),
        )
        estimation = None
        declaration = build_parameter_estimation_declaration(estimand, policy)
        if estimand is not None and policy is not None:
            estimation = estimate_parameters_for_project(
                orchestrator, estimand, policy
            )
        rerun = orchestrator.run(
            output_dir or original.parent,
            parameter_estimation_declaration=declaration,
        )
        if estimation is not None:
            persist_parameter_estimation_evidence(rerun, estimation)
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
                cancellation at lifecycle checkpoints.
            checkpoint_path: Optional signed, atomic checkpoint path for durable
                modality recovery. Supplying this requires the lifecycle report
                path managed by this project.
            resume: Resume from ``checkpoint_path`` after verifying its
                signature, identity, and deterministic replay of the completed
                work.

        Raises:
            TypeError: If an object is not a modality adapter, a flag or the
                seed has the wrong type, or the cancellation is not a
                cancellation signal.
            ValueError: If the signing key is empty.
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
        """Runs the full lifecycle and persist its evidence atomically.

        Returns:
            The lifecycle report, also kept as `report`.

        Raises:
            ValueError: If `output_dir` exists and is not a directory.
        """
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
        """Returns the experimental capability declaration after a run.

        Returns:
            The declaration derived from the latest report.

        Raises:
            RuntimeError: If `run` has not completed.
        """
        if self.report is None:
            raise RuntimeError("run must complete before declaration")
        return self.lifecycle.declaration(self.report)


class ModalityCurationProject:
    """Curate and document a modality without forcing a model fit.

    This is the data-understanding entry point for unlabeled, unsupervised, or
    domain-specific datasets. It persists signed curation and dataset-card
    evidence, but it never invents labels, features, metrics, or a model.

    Attributes:
        adapter: Adapter that owns source identity and curation policy.
        signing_key: Key that authenticates the evidence.
        output_dir: Directory where curation evidence is written.
        manifest: Source, lineage, privacy and task contract, if any.
    """

    def __init__(
        self,
        adapter: modality_lifecycle.ModalityCurationAdapter,
        signing_key: str | bytes,
        *,
        output_dir: str | pathlib.Path = "runs",
        manifest: data_contracts.ModalityManifest | None = None,
    ) -> None:
        """Defines a curation-only run over a typed modality adapter.

        Args:
            adapter: Adapter that owns source identity and curation policy.
            signing_key: Non-empty key used to authenticate the evidence.
            output_dir: Directory where curation evidence is written.
            manifest: Optional source, lineage, privacy, and task contract.

        Raises:
            TypeError: If the adapter or manifest has the wrong type.
            ValueError: If the signing key is empty.
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
        """Persist signed curation evidence without entering modelling.

        If an artifact already exists it is verified and reused when it matches
        the current adapter, instead of being replaced.

        Returns:
            The signed curation result, either newly written or the matching
            existing one.

        Raises:
            ValueError: If `output_dir` exists and is not a directory, the
                manifest or the curation report disagrees with the adapter, or
                an existing artifact is unreadable, is not an object, or differs
                from the current adapter's modality, dataset hash, sample count
                or content.
            TypeError: If the adapter's `curate()` does not return a modality
                curation report.
        """
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
        candidate = modality_lifecycle.ModalityCurationResult(
            modality=modality,
            dataset_hash=dataset_hash,
            sample_count=sample_count,
            curation=curation_payload,
            dataset_card=card.to_dict(),
            path=str(path),
            signature=signature,
        )
        payload = candidate.to_dict()
        serialized_payload = (
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        try:
            publish_curation_artifact(path, serialized_payload)
        except FileExistsError:
            existing = modality_lifecycle.ModalityCurationResult.read_verified(
                path, self.signing_key
            )
            if existing.modality != candidate.modality:
                raise ValueError(
                    "existing curation modality does not match current adapter"
                ) from None
            if existing.dataset_hash != candidate.dataset_hash:
                raise ValueError(
                    "existing curation dataset_hash does not match current "
                    "adapter"
                ) from None
            if existing.sample_count != candidate.sample_count:
                raise ValueError(
                    "existing curation sample_count does not match current "
                    "adapter"
                ) from None
            try:
                existing_record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError) as error:
                raise ValueError(
                    "existing curation artifact is unreadable"
                ) from error
            if not isinstance(existing_record, dict):
                raise ValueError(
                    "existing curation artifact must be an object"
                ) from None
            if attestation.canonical_digest(existing_record) != (
                attestation.canonical_digest(candidate.to_dict())
            ):
                raise ValueError(
                    "existing curation canonical content does not match "
                    "current adapter"
                ) from None
            return existing
        return candidate
