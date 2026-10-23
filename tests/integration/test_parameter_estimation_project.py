"""Project integration for the separate parameter-estimation evidence track."""

import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest

import scml
from scml.artifacts import bundle as bundle_module
from scml.contracts import errors
from scml.evaluation.parameter_estimation import EstimandSpec
from scml.evaluation.parameter_estimation import EstimationPolicy
from scml.evaluation.parameter_estimation import EstimationResult
from scml.inference.evidence import InferenceEvidenceLinker


def estimation_frame() -> pd.DataFrame:
    """Return a deterministic tabular dataset for project evidence tests."""
    rng = np.random.default_rng(21)
    signal = rng.normal(size=80)
    site = np.repeat(["a", "b", "c", "d"], 20)
    outcome = 1.25 + 2.0 * signal + rng.normal(scale=0.1, size=80)
    return pd.DataFrame({"signal": signal, "outcome": outcome, "site": site})


def declared_estimand() -> EstimandSpec:
    """Return the explicit scientific quantity used by the tests."""
    return EstimandSpec(
        name="signal_slope",
        outcome="outcome",
        population="the declared project dataset",
        likelihood="gaussian",
        feature_names=("signal",),
        assumptions=("the conditional mean is linear",),
    )


def project_with_estimation(output_dir) -> scml.MLProject:
    """Build the deterministic project used by artifact-binding tests."""
    return scml.MLProject(
        estimation_frame(),
        "outcome",
        task="regression",
        time_budget=30,
        estimand=declared_estimand(),
        estimation_policy=EstimationPolicy(estimator="linear"),
        output_dir=output_dir,
    )


def test_project_run_persists_separate_estimation_evidence(tmp_path):
    """Declared estimation is persisted without changing model selection."""
    project = scml.MLProject(
        estimation_frame(),
        "outcome",
        task="regression",
        time_budget=30,
        estimand=declared_estimand(),
        estimation_policy=EstimationPolicy(
            estimator="linear", covariance="clustered", cluster_name="site"
        ),
        output_dir=tmp_path,
    )

    result = project.run()

    artifact = result.run_dir / "parameter_estimation.json"
    assert artifact.exists()
    assert result.estimation is not None
    restored = EstimationResult.read(artifact)
    assert restored.digest() == result.estimation.digest()
    assert restored.policy.covariance == "clustered"
    assert result.evaluation["selected_model"] == result.selected_model
    assert result.evaluation["parameter_estimation"] == {
        "artifact": "parameter_estimation.json",
        "digest": restored.digest(),
    }
    persisted_evaluation = json.loads(
        (result.run_dir / "evaluation.json").read_text(encoding="utf-8")
    )
    assert (
        persisted_evaluation["parameter_estimation"]
        == result.evaluation["parameter_estimation"]
    )
    graph = InferenceEvidenceLinker.load(result.run_dir / "evidence_graph.json")
    parameter_node = graph.nodes["parameter-estimation"]
    assert parameter_node.digest == restored.digest()
    assert parameter_node.attributes["artifact"] == "parameter_estimation.json"
    assert graph.nodes["evaluation"].digest == bundle_module.file_sha256(
        result.run_dir / "evaluation.json"
    )
    assert graph.path_exists("snapshot", "parameter-estimation")
    assert graph.path_exists("evaluation", "parameter-estimation")
    identity = result.estimation.diagnostics["data_identity"]
    assert identity["source_identity"] is None
    assert identity["snapshot_id"].startswith("snap_")
    declaration = json.loads(
        (result.run_dir / "parameter_estimation_declaration.json").read_text(
            encoding="utf-8"
        )
    )
    assert declaration["enabled"] is True
    assert declaration["project_config_hash"] == project.spec.config_hash()
    assert declaration["digest"] == (
        (result.run_dir / "parameter_estimation_declaration.lock")
        .read_text(encoding="utf-8")
        .strip()
    )


def interrupted_estimation_run(tmp_path):
    """Create a completed run in the state expected by resume."""
    result = project_with_estimation(tmp_path).run()
    (result.run_dir / "evaluation.json").unlink()
    (result.run_dir / "parameter_estimation.json").unlink()
    shutil.rmtree(result.run_dir / "model")
    return result


def test_resume_reconstructs_estimation_and_rebinds_evidence(tmp_path):
    """Resume restores the declaration and follows the fresh-run path."""
    original = interrupted_estimation_run(tmp_path)

    resumed = scml.MLProject.resume(original.run_dir, data=estimation_frame())

    assert resumed.estimation is not None
    report = EstimationResult.read(
        original.run_dir / "parameter_estimation.json"
    )
    assert report.digest() == resumed.estimation.digest()
    assert report.estimand == declared_estimand()
    assert report.policy == EstimationPolicy(estimator="linear")
    assert resumed.evaluation["parameter_estimation"] == {
        "artifact": "parameter_estimation.json",
        "digest": report.digest(),
    }
    graph = InferenceEvidenceLinker.load(
        original.run_dir / "evidence_graph.json"
    )
    assert graph.nodes["parameter-estimation"].digest == report.digest()
    assert graph.path_exists("snapshot", "parameter-estimation")
    assert graph.path_exists("evaluation", "parameter-estimation")


def test_resume_fails_closed_when_estimation_declaration_is_missing(tmp_path):
    """Resume cannot infer a scientific policy after declaration loss."""
    original = interrupted_estimation_run(tmp_path)
    (original.run_dir / "parameter_estimation_declaration.json").unlink()

    with pytest.raises(errors.ArtifactError, match="declaration"):
        scml.MLProject.resume(original.run_dir, data=estimation_frame())


def test_resume_fails_closed_for_corrupt_estimation_declaration(tmp_path):
    """Malformed declaration JSON stops resume before execution."""
    original = interrupted_estimation_run(tmp_path)
    (original.run_dir / "parameter_estimation_declaration.json").write_text(
        "not json", encoding="utf-8"
    )

    with pytest.raises(errors.ArtifactError, match="declaration"):
        scml.MLProject.resume(original.run_dir, data=estimation_frame())


def test_resume_fails_closed_for_tampered_estimation_declaration(tmp_path):
    """A changed declaration fails digest verification before resume."""
    original = interrupted_estimation_run(tmp_path)
    path = original.run_dir / "parameter_estimation_declaration.json"
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["estimand"]["name"] = "tampered"
    path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(errors.ArtifactError, match="digest"):
        scml.MLProject.resume(original.run_dir, data=estimation_frame())


def test_project_run_rejects_missing_estimation_artifact(monkeypatch, tmp_path):
    """A missing persisted report cannot produce a completed result."""
    original_read = EstimationResult.read

    def remove_artifact(path):
        path.unlink()
        return original_read(path)

    monkeypatch.setattr(EstimationResult, "read", remove_artifact)

    with pytest.raises(errors.ArtifactError, match="cannot be read"):
        project_with_estimation(tmp_path).run()

    run_directory = next(tmp_path.glob("run_*"))
    evaluation = json.loads(
        (run_directory / "evaluation.json").read_text(encoding="utf-8")
    )
    assert "parameter_estimation" not in evaluation


def test_project_run_rejects_valid_digest_for_tampered_report(
    monkeypatch, tmp_path
):
    """A report changed and re-digested after writing is still rejected."""
    original_read = EstimationResult.read

    def tamper_artifact(path):
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelope["report"]["diagnostics"]["data_identity"]["source_uri"] = (
            "tampered"
        )
        envelope["digest"] = hashlib.sha256(
            json.dumps(
                envelope["report"], sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        path.write_text(json.dumps(envelope), encoding="utf-8")
        return original_read(path)

    monkeypatch.setattr(EstimationResult, "read", tamper_artifact)

    with pytest.raises(
        errors.ArtifactError, match="does not match the persisted estimate"
    ):
        project_with_estimation(tmp_path).run()

    run_directory = next(tmp_path.glob("run_*"))
    evaluation = json.loads(
        (run_directory / "evaluation.json").read_text(encoding="utf-8")
    )
    assert "parameter_estimation" not in evaluation


def test_estimation_fails_closed_without_both_declarations(tmp_path):
    """Undeclared estimation cannot silently infer a scientific policy."""
    project = scml.MLProject(estimation_frame(), "outcome", output_dir=tmp_path)

    with pytest.raises(errors.ConfigError, match="declare both"):
        project.estimate_parameters()


def test_partial_estimation_declaration_is_rejected(tmp_path):
    """An estimand without its policy is not a usable evidence contract."""
    with pytest.raises(errors.ConfigError, match="both"):
        scml.MLProject(
            estimation_frame(),
            "outcome",
            estimand=declared_estimand(),
            output_dir=tmp_path,
        )


def test_incompatible_likelihood_fails_before_model_selection(tmp_path):
    """Likelihood mismatch is rejected before the orchestrator is called."""
    estimand = EstimandSpec(
        name="wrong_likelihood",
        outcome="outcome",
        population="the declared project dataset",
        likelihood="bernoulli",
        feature_names=("signal",),
        assumptions=("the conditional mean is linear",),
    )
    project = scml.MLProject(
        estimation_frame(),
        "outcome",
        task="regression",
        time_budget=30,
        estimand=estimand,
        estimation_policy=EstimationPolicy(estimator="linear"),
        output_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="expected gaussian"):
        project.run()

    assert not list(tmp_path.glob("run_*/parameter_estimation.json"))


def test_ipw_declaration_binds_the_project_treatment_column(tmp_path):
    """IPW cannot infer or silently substitute a treatment column."""
    frame = estimation_frame().assign(treatment=(np.arange(80) % 2))
    estimand = EstimandSpec(
        name="treatment_effect",
        outcome="outcome",
        population="the declared project dataset",
        likelihood="gaussian",
        feature_names=("treatment", "signal"),
        assumptions=("the declared causal assumptions hold",),
    )
    policy = EstimationPolicy(
        estimator="ipw_ate",
        covariance="iid",
        treatment_position=0,
        causal_assumptions=(
            "consistency",
            "conditional_exchangeability",
            "positivity",
        ),
    )

    project = scml.MLProject(
        frame,
        "outcome",
        estimand=estimand,
        estimation_policy=policy,
        output_dir=tmp_path,
    )

    with pytest.raises(errors.ConfigError, match="treatment_column"):
        project.estimate_parameters()


def test_project_rejects_missing_cluster_and_missingness_columns(tmp_path):
    """Project declarations fail closed before estimation consumes rows."""
    project = scml.MLProject(
        estimation_frame(),
        "outcome",
        estimand=declared_estimand(),
        estimation_policy=EstimationPolicy(
            estimator="linear", covariance="clustered", cluster_name="missing"
        ),
        output_dir=tmp_path,
    )
    with pytest.raises(errors.DataError, match="columns are missing"):
        project.estimate_parameters()

    missing_frame = estimation_frame()
    missing_frame.loc[0, "signal"] = np.nan
    project = scml.MLProject(
        missing_frame,
        "outcome",
        estimand=declared_estimand(),
        estimation_policy=EstimationPolicy(estimator="linear"),
        output_dir=tmp_path,
    )
    with pytest.raises(errors.DataError, match="missing values"):
        project.estimate_parameters()


def test_complete_case_binds_missingness_and_run_artifact_to_result(tmp_path):
    """The result exposes the digest-verified, complete persisted report."""
    frame = estimation_frame()
    frame.loc[0, "signal"] = np.nan
    project = scml.MLProject(
        frame,
        "outcome",
        task="regression",
        time_budget=30,
        estimand=declared_estimand(),
        estimation_policy=EstimationPolicy(
            estimator="linear", missing="complete_case"
        ),
        output_dir=tmp_path,
    )

    result = project.run()

    artifact = result.run_dir / "parameter_estimation.json"
    restored = EstimationResult.read(artifact)
    assert result.estimation is not None
    assert result.estimation.digest() == restored.digest()
    assert result.estimation.rows_dropped == 1
    assert result.estimation.diagnostics["data_columns"]["outcome"] == (
        "outcome"
    )
    assert result.estimation.diagnostics["missingness"]["policy"] == (
        "complete_case"
    )


def test_project_estimation_records_file_source_identity(tmp_path):
    """File-backed estimation records source and snapshot identities."""
    source = tmp_path / "estimation.csv"
    frame = estimation_frame()
    frame.to_csv(source, index=False)
    project = scml.MLProject(
        source,
        "outcome",
        estimand=declared_estimand(),
        estimation_policy=EstimationPolicy(estimator="linear"),
        output_dir=tmp_path / "runs",
    )

    result = project.estimate_parameters()

    identity = result.diagnostics["data_identity"]
    assert identity["source_identity"].startswith("src_")
    assert identity["snapshot_id"].startswith("snap_")


def test_verify_replay_preserves_parameter_estimation_evidence(tmp_path):
    """Replay compares the scientific report as well as model evidence."""
    project = project_with_estimation(tmp_path)
    original = project.run()

    verdict = scml.MLProject.verify_replay(
        original.run_dir,
        data=estimation_frame(),
        output_dir=tmp_path / "replay",
    )

    assert verdict.level == "exact"
    assert (tmp_path / "replay").exists()
