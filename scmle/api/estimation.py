"""Parameter estimation for a project: declaration, estimate and binding.

The scientific declaration a run stores beside its configuration, the
estimate made against the project's dataset, and the binding of the verified
report into the run's evidence graph. Model selection never reads it.
"""

from collections.abc import Hashable
from collections.abc import Mapping
import dataclasses
import hashlib
import json
import pathlib
from typing import Any

import pandas

from scmle import durability
from scmle.artifacts import bundle
from scmle.artifacts import results
from scmle.contracts import errors
from scmle.contracts import lineage
from scmle.contracts import records
from scmle.data.readers import connectors
from scmle.data.readers import tabular
from scmle.evaluation import parameter_estimation
from scmle.evaluation.envelope import rename
from scmle.inference import evidence
import scmle.pipeline.orchestrator

PARAMETER_ESTIMATION_DECLARATION_ARTIFACT = (
    "parameter_estimation_declaration.json"
)
PARAMETER_ESTIMATION_DECLARATION_LOCK = "parameter_estimation_declaration.lock"
PARAMETER_ESTIMATION_DECLARATION_SCHEMA = 2
REPORT = "parameter_estimation.json"
REFUSAL = "parameter_estimation_refusal.json"
REFUSAL_SCHEMA = 1
REFUSALS = (errors.ConfigError, errors.DataError)
"""What `estimate` raises when the declaration cannot be estimated on the
data: a missing or inconsistent column, missing values under `reject`, a
likelihood the estimator does not fit, or an estimator's own refusal (too few
rows per arm, no overlap, a singular design)."""


def declare(
    estimand: parameter_estimation.Estimand | None,
    policy: parameter_estimation.Config | None,
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


def fingerprint(
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


def unwrap(
    artifact_path: pathlib.Path,
    expected_config_hash: str,
) -> dict[str, Any]:
    """Reads a run's declaration and checks its digest, schema and hash.

    Args:
        artifact_path: Directory of the run holding the declaration and its
            lock file.
        expected_config_hash: Configuration hash the declaration must have been
            written for.

    Returns:
        The declaration without its digest.

    Raises:
        ArtifactError: If the declaration or its lock is missing, unreadable,
            not an object, lacks a digest, fails the digest check, has an
            unsupported schema or has another configuration hash.
    """
    report_path = artifact_path / PARAMETER_ESTIMATION_DECLARATION_ARTIFACT
    lockfile = artifact_path / PARAMETER_ESTIMATION_DECLARATION_LOCK
    try:
        envelope = json.loads(report_path.read_text(encoding="utf-8"))
        lock_digest = lockfile.read_text(encoding="utf-8").strip()
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
    if digest != fingerprint(envelope) or lock_digest != digest:
        raise errors.ArtifactError(
            "parameter estimation declaration digest mismatch",
            {"run_dir": str(artifact_path)},
        )
    if envelope.get("schema_version") not in range(
        records.ORIGIN, PARAMETER_ESTIMATION_DECLARATION_SCHEMA + 1
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
    return envelope


def rebuild(
    envelope: Mapping[str, Any],
) -> tuple[parameter_estimation.Estimand, parameter_estimation.Config]:
    """Reconstructs the estimand and policy of an enabled declaration.

    The stored policy is plain JSON: the prior, the assumption lists and the
    variance bounds are restored to the types the configuration takes. A
    version-2 declaration holds the estimand and the policy in their
    single-word keys; both are moved back to version 1 through their
    migrations first, so declarations of either version rebuild.

    Args:
        envelope: The verified declaration, with `schema_version`,
            `estimand` and `policy`.

    Returns:
        The estimand and the policy.

    Raises:
        KeyError: If a required field is absent.
        TypeError: If a field has the wrong shape.
        ValueError: If the estimand or the policy is invalid.
    """
    estimand_record = dict(envelope["estimand"])
    policy_values = dict(envelope["policy"])
    if envelope.get("schema_version") != records.ORIGIN:
        estimand_record = records.downgrade(
            parameter_estimation.Estimand.migrations, estimand_record
        )
        policy_values = records.downgrade(
            parameter_estimation.Config.migrations, policy_values
        )
    estimand = parameter_estimation.Estimand(
        name=estimand_record["name"],
        outcome=estimand_record["outcome"],
        population=estimand_record["population"],
        likelihood=estimand_record["likelihood"],
        features=tuple(estimand_record["feature_names"]),
        assumptions=tuple(estimand_record["assumptions"]),
        description=estimand_record.get("description", ""),
    )
    if "prior" in policy_values:
        prior_values = dict(policy_values["prior"])
        policy_values["prior"] = parameter_estimation.Prior(
            mean=tuple(prior_values["mean"]),
            precision=tuple(tuple(row) for row in prior_values["precision"]),
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
    return estimand, parameter_estimation.Config(
        **rename(parameter_estimation.Config, policy_values)
    )


def restore(
    run_directory: str | pathlib.Path,
    expected_config_hash: str,
) -> tuple[
    parameter_estimation.Estimand | None,
    parameter_estimation.Config | None,
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
    envelope = unwrap(artifact_path, expected_config_hash)
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
    return decode(
        envelope, errors.ArtifactError, {"run_dir": str(artifact_path)}
    )


def decode(
    envelope: Mapping[str, Any],
    failure: type[errors.ScmleError],
    details: Mapping[str, str],
) -> tuple[parameter_estimation.Estimand, parameter_estimation.Config]:
    """Rebuilds a declaration, raising one typed error for any defect.

    Args:
        envelope: The declaration, with `schema_version`, `estimand` and
            `policy`.
        failure: The error raised when it is invalid.
        details: Where the declaration came from, for the error.

    Returns:
        The estimand and the policy.

    Raises:
        ScmleError: `failure`, when a field is absent, has the wrong shape
            or is invalid.
    """
    try:
        return rebuild(envelope)
    except (KeyError, TypeError, ValueError) as error:
        raise failure(
            "parameter estimation declaration is invalid", dict(details)
        ) from error


def load(
    path: str | pathlib.Path,
) -> tuple[parameter_estimation.Estimand, parameter_estimation.Config]:
    """Reads a parameter-estimation declaration file.

    The file is a JSON object with `estimand` and `policy` in the layout a
    run stores (single-word keys, `schema_version` 2, the default) or in the
    version-1 layout with `schema_version` 1. A run's own
    `parameter_estimation_declaration.json` is accepted as it is.

    Args:
        path: The declaration file.

    Returns:
        The estimand and the policy.

    Raises:
        ConfigError: If the file cannot be read, is not a JSON object with
            `estimand` and `policy`, has an unsupported `schema_version` or
            declares an invalid estimand or policy.
    """
    source = {"path": str(path)}
    try:
        envelope = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise errors.ConfigError(
            "parameter estimation declaration cannot be read", source
        ) from error
    if not isinstance(envelope, dict) or not {"estimand", "policy"} <= set(
        envelope
    ):
        raise errors.ConfigError(
            "parameter estimation declaration must be a JSON object with "
            "`estimand` and `policy`",
            source,
        )
    version = envelope.get(
        "schema_version", PARAMETER_ESTIMATION_DECLARATION_SCHEMA
    )
    if version not in range(
        records.ORIGIN, PARAMETER_ESTIMATION_DECLARATION_SCHEMA + 1
    ):
        raise errors.ConfigError(
            "unsupported parameter estimation declaration schema", source
        )
    return decode(
        {
            "schema_version": version,
            "estimand": envelope["estimand"],
            "policy": envelope["policy"],
        },
        errors.ConfigError,
        source,
    )


def refuse(
    directory: str | pathlib.Path,
    estimand: parameter_estimation.Estimand,
    policy: parameter_estimation.Config,
    error: errors.ScmleError,
) -> pathlib.Path:
    """Records why a declared estimate was refused.

    Writes `REFUSAL` atomically and removes a report an earlier estimate
    left in the same directory, so the directory never holds both.

    Args:
        directory: Where the record is written; created when missing.
        estimand: The declared population quantity.
        policy: The declared estimation policy.
        error: The refusal `estimate` raised.

    Returns:
        The path of the record, which holds `status` `refused`, the
        `reason`, its `category` and `details`, the estimand and the policy.
    """
    folder = pathlib.Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": REFUSAL_SCHEMA,
        "status": "refused",
        "reason": str(error),
        "category": error.category,
        "details": error.details,
        "estimand": estimand.to_dict(),
        "policy": policy.to_dict(),
    }
    path = folder / REFUSAL
    durability.write_atomically(
        path,
        (
            json.dumps(record, default=str, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    )
    (folder / REPORT).unlink(missing_ok=True)
    return path


def bind(
    run_directory: pathlib.Path,
    artifact: pathlib.Path,
    estimation: parameter_estimation.Estimation,
) -> None:
    """Binds a verified parameter report into the run evidence graph.

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
            node. A missing or corrupt graph file raises from the loader.
    """
    graph_path = run_directory / "evidence_graph.json"
    graph = evidence.Linker.load(graph_path)
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
    updated = lineage.Graph(nodes)
    updated.add_node(
        lineage.Node(
            "evaluation",
            lineage.Kind.EVALUATION,
            bundle.shasum(run_directory / "evaluation.json"),
            {"artifact": "evaluation.json"},
        )
    )
    updated.add_node(
        lineage.Node(
            "parameter-estimation",
            lineage.Kind.PARAMETER_ESTIMATION,
            estimation.digest(),
            {
                "artifact": artifact.name,
                "estimator": estimation.policy.estimator,
                "estimand": estimation.estimand.name,
            },
        )
    )
    for edge in graph.edges:
        updated.connect(edge)
    updated.connect(
        lineage.Edge("snapshot", "parameter-estimation", "estimates")
    )
    updated.connect(
        lineage.Edge("evaluation", "parameter-estimation", "records")
    )
    record = updated.to_dict()
    record["graph_digest"] = updated.digest()
    evidence.Linker.write(graph_path, record)


def persist(
    result: results.Run,
    estimation: parameter_estimation.Estimation,
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
        verified = parameter_estimation.Estimation.read(artifact)
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
    bind(result.run_dir, artifact, verified)
    result.evaluation.clear()
    result.evaluation.update(evaluation)
    # The result must expose the exact digest-verified artifact that was
    # persisted, rather than an unverified in-memory equivalent.
    result.estimation = verified


def locate(
    orchestrator: scmle.pipeline.orchestrator.Orchestrator,
    estimand: parameter_estimation.Estimand,
    policy: parameter_estimation.Config,
) -> str:
    """Returns the treatment column a treatment-based estimator needs.

    Args:
        orchestrator: The orchestrator holding the project's spec.
        estimand: The declared population quantity.
        policy: The declared estimation policy.

    Returns:
        The spec's treatment column.

    Raises:
        ConfigError: If the spec has no treatment column, the estimand does
            not list it as a feature, or the policy's treatment position does
            not match its place among the features.
    """
    treatment_column = orchestrator.spec.data.treatment_column
    if treatment_column is None:
        raise errors.ConfigError(
            f"{policy.estimator} parameter estimation requires treatment_column"
        )
    if treatment_column not in estimand.features:
        raise errors.ConfigError(
            "treatment_column must be included in estimand feature_names"
        )
    treatment_position = estimand.features.index(treatment_column)
    if policy.treatment != treatment_position:
        raise errors.ConfigError(
            "treatment_position must match treatment_column in "
            "estimand feature_names"
        )
    return treatment_column


def screen(
    orchestrator: scmle.pipeline.orchestrator.Orchestrator,
    estimand: parameter_estimation.Estimand,
    policy: parameter_estimation.Config,
    frame: pandas.DataFrame,
) -> tuple[list[str], str | None, dict[Hashable, int]]:
    """Checks the declared columns against the frame before fitting.

    Args:
        orchestrator: The orchestrator holding the project's spec.
        estimand: The declared population quantity.
        policy: The declared estimation policy.
        frame: The project's dataset.

    Returns:
        The declared columns in first-mention order, the treatment column (or
        None for an estimator without one), and the missing-value count of
        each declared column that has any.

    Raises:
        ConfigError: If a treatment-based estimator lacks a consistent
            `treatment_column`.
        DataError: If a declared column is absent, or has missing values under
            the `reject` policy.
    """
    declared = [*estimand.features, estimand.outcome]
    treatment_column = None
    if policy.estimator in {
        "bayesian_ate",
        "ipw_ate",
        "aipw_ate",
        "continuous_ipw_ate",
    }:
        treatment_column = locate(orchestrator, estimand, policy)
        declared.append(treatment_column)
    if policy.cluster is not None:
        declared.append(policy.cluster)
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
    return declared, treatment_column, missing_values


def trace(uri: str, frame: pandas.DataFrame) -> dict[str, str | None]:
    """Records which data an estimate was made from.

    Args:
        uri: The project's data URI, or an in-memory marker.
        frame: The project's dataset.

    Returns:
        The source URI, the source's identity (None for in-memory data) and
        the snapshot identifier of the frame.
    """
    snapshot_id = tabular.identify(frame, uri)
    source_identity: str | None = None
    if uri not in {"<memory>", "<in-memory>"}:
        source_identity = connectors.identify(uri)
    return {
        "source_uri": uri,
        "source_identity": source_identity,
        "snapshot_id": snapshot_id,
    }


def estimate(
    orchestrator: scmle.pipeline.orchestrator.Orchestrator,
    estimand: parameter_estimation.Estimand,
    policy: parameter_estimation.Config,
    artifact_directory: str | pathlib.Path | None = None,
) -> parameter_estimation.Estimation:
    """Estimates the declared parameters against an orchestrator dataset.

    Args:
        orchestrator: The orchestrator holding the project's spec and either
            its in-memory frame or the table at the spec's data URI.
        estimand: The declared population quantity.
        policy: The declared estimation policy.
        artifact_directory: If given, the report is written there as
            ``parameter_estimation.json`` and the returned result is the one
            read back through the digest-checking reader; a refusal record
            an earlier attempt left there is removed.

    Returns:
        The estimation result, with diagnostics naming the columns used, the
        missingness handling and the data identity.

    Raises:
        ConfigError: If the modality is not tabular, or a treatment-based
            estimator lacks a consistent `treatment_column`.
        DataError: If a declared column is absent, or has missing values under
            the `reject` policy.

    Invalid estimator inputs raise `ValueError` from
    `parameter_estimation.infer`.
    """
    if orchestrator.spec.data.modality != "tabular":
        raise errors.ConfigError(
            "parameter estimation is supported only for the declared "
            "tabular modality"
        )
    frame = orchestrator.frame
    if frame is None:
        frame = tabular.read(orchestrator.spec.data.uri)
    declared, treatment_column, missing_values = screen(
        orchestrator, estimand, policy, frame
    )
    features = frame.loc[:, list(estimand.features)]
    outcome = frame.loc[:, estimand.outcome]
    cluster_values = (
        None
        if policy.cluster is None
        else frame.loc[:, policy.cluster].tolist()
    )
    estimation = parameter_estimation.infer(
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
                "features": list(estimand.features),
                "outcome": estimand.outcome,
                "treatment": treatment_column,
                "cluster": policy.cluster,
            },
            "missingness": {
                "policy": policy.missing,
                "columns": declared,
                "missing_counts": missing_values,
            },
            "data_identity": trace(orchestrator.spec.data.uri, frame),
        },
    )
    if artifact_directory is not None:
        artifact = pathlib.Path(artifact_directory) / REPORT
        estimation.write(artifact)
        estimation = parameter_estimation.Estimation.read(artifact)
        (artifact.parent / REFUSAL).unlink(missing_ok=True)
    return estimation
