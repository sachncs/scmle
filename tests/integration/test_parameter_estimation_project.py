"""Project integration for the separate parameter-estimation evidence track."""

import hashlib
import json

import numpy
import pandas
import pytest
import support

import scmle.api.project
from scmle.artifacts import bundle
from scmle.contracts import errors
from scmle.contracts import records
import scmle.evaluation.parameter_estimation
from scmle.evaluation.parameter_estimation import Config
from scmle.evaluation.parameter_estimation import Estimand
from scmle.evaluation.parameter_estimation import Estimation
from scmle.inference.evidence import Linker


def estimation_frame() -> pandas.DataFrame:
    """Return a deterministic tabular dataset for project evidence tests."""
    rng = numpy.random.default_rng(21)
    signal = rng.normal(size=80)
    site = numpy.repeat(["a", "b", "c", "d"], 20)
    outcome = 1.25 + 2.0 * signal + rng.normal(scale=0.1, size=80)
    return pandas.DataFrame(
        {"signal": signal, "outcome": outcome, "site": site}
    )


def declared_estimand() -> Estimand:
    """Return the explicit scientific quantity used by the tests."""
    return Estimand(
        name="signal_slope",
        outcome="outcome",
        population="the declared project dataset",
        likelihood="gaussian",
        features=("signal",),
        assumptions=("the conditional mean is linear",),
    )


def project_with_estimation(output_dir) -> scmle.api.project.Project:
    """Build the deterministic project used by artifact-binding tests."""
    return scmle.api.project.Project(
        estimation_frame(),
        "outcome",
        task="regression",
        budget=30,
        estimand=declared_estimand(),
        policy=Config(estimator="linear"),
        directory=output_dir,
    )


def test_project_run_persists_separate_estimation_evidence(tmp_path):
    """Declared estimation is persisted without changing model selection.

    Also enforces website/src/pages/docs/modeling.astro:15.
    Also enforces website/src/pages/docs/modeling.astro:231.
    """
    project = scmle.api.project.Project(
        estimation_frame(),
        "outcome",
        task="regression",
        budget=30,
        estimand=declared_estimand(),
        policy=Config(
            estimator="linear", covariance="clustered", cluster="site"
        ),
        directory=tmp_path,
    )

    result = project.run()

    artifact = result.directory / "parameter_estimation.json"
    assert artifact.exists()
    assert result.estimation is not None
    restored = Estimation.read(artifact)
    assert restored.digest() == result.estimation.digest()
    assert restored.policy.covariance == "clustered"
    assert result.evaluation["selected"]["model"] == result.model
    assert result.evaluation["parameter_estimation"] == {
        "artifact": "parameter_estimation.json",
        "digest": restored.digest(),
    }
    persisted_evaluation = json.loads(
        (result.directory / "evaluation.json").read_text(encoding="utf-8")
    )
    assert (
        persisted_evaluation["parameter_estimation"]
        == result.evaluation["parameter_estimation"]
    )
    graph = Linker.load(result.directory / "evidence_graph.json")
    parameter_node = graph.nodes["parameter-estimation"]
    assert parameter_node.digest == restored.digest()
    assert parameter_node.attributes["artifact"] == "parameter_estimation.json"
    assert graph.nodes["evaluation"].digest == bundle.shasum(
        result.directory / "evaluation.json"
    )
    assert graph.reachable("snapshot", "parameter-estimation")
    assert graph.reachable("evaluation", "parameter-estimation")
    identity = result.estimation.diagnostics["data_identity"]
    assert identity["source_identity"] is None
    assert identity["snapshot_id"].startswith("snap_")
    declaration = json.loads(
        (result.directory / "parameter_estimation_declaration.json").read_text(
            encoding="utf-8"
        )
    )
    assert declaration["enabled"] is True
    assert declaration["project_config_hash"] == project.spec.config_hash()
    assert declaration["digest"] == (
        (result.directory / "parameter_estimation_declaration.lock")
        .read_text(encoding="utf-8")
        .strip()
    )


def test_project_dispatches_and_serializes_bayesian_ate(tmp_path):
    """The generic project track binds a causal Bayesian declaration."""
    frame = estimation_frame().assign(
        treatment=numpy.arange(80, dtype=float) % 2.0,
    )
    estimand = Estimand(
        name="treatment_effect",
        outcome="outcome",
        population="the declared project dataset",
        likelihood="gaussian",
        features=("treatment", "signal"),
        assumptions=("the declared causal assumptions hold",),
    )
    prior = scmle.evaluation.parameter_estimation.Prior(
        mean=(0.0, 0.0, 0.0),
        precision=((0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.01)),
    )
    policy = Config(
        estimator="bayesian_ate",
        covariance="iid",
        prior=prior,
        treatment=0,
        causal_assumptions=(
            "linearity",
            "consistency",
            "conditional_exchangeability",
            "positivity",
        ),
    )
    project = scmle.api.project.Project(
        frame,
        "outcome",
        task="uplift",
        treatment="treatment",
        budget=30,
        estimand=estimand,
        policy=policy,
        directory=tmp_path,
    )

    result = project.run()

    assert result.estimation is not None
    assert result.estimation.policy == policy
    declaration = json.loads(
        (result.directory / "parameter_estimation_declaration.json").read_text(
            encoding="utf-8"
        )
    )
    assert declaration["policy"]["estimator"] == "bayesian_ate"
    assert declaration["policy"]["treatment"] == 0
    assert declaration["policy"]["prior"] == json.loads(
        json.dumps(prior.to_dict())
    )
    assert result.estimation.reference["passed"] is True


def interrupted_estimation_run(tmp_path):
    """A run that crashed before its test set opened, as resume expects."""
    return support.interrupt(project_with_estimation(tmp_path))


def test_resume_reconstructs_estimation_and_rebinds_evidence(tmp_path):
    """Resume restores the declaration and follows the fresh-run path."""
    original = interrupted_estimation_run(tmp_path)

    resumed = scmle.api.project.Project.resume(
        original, data=estimation_frame()
    )

    assert resumed.estimation is not None
    report = Estimation.read(original / "parameter_estimation.json")
    assert report.digest() == resumed.estimation.digest()
    assert report.estimand == declared_estimand()
    assert report.policy == Config(estimator="linear")
    assert resumed.evaluation["parameter_estimation"] == {
        "artifact": "parameter_estimation.json",
        "digest": report.digest(),
    }
    graph = Linker.load(original / "evidence_graph.json")
    assert graph.nodes["parameter-estimation"].digest == report.digest()
    assert graph.reachable("snapshot", "parameter-estimation")
    assert graph.reachable("evaluation", "parameter-estimation")


def test_resume_fails_closed_when_estimation_declaration_is_missing(tmp_path):
    """Resume cannot infer a scientific policy after declaration loss."""
    original = interrupted_estimation_run(tmp_path)
    (original / "parameter_estimation_declaration.json").unlink()

    with pytest.raises(errors.ArtifactError, match="declaration"):
        scmle.api.project.Project.resume(original, data=estimation_frame())


def test_resume_fails_closed_for_corrupt_estimation_declaration(tmp_path):
    """Malformed declaration JSON stops resume before execution.

    Also enforces website/src/pages/docs/modeling.astro:279.
    """
    original = interrupted_estimation_run(tmp_path)
    (original / "parameter_estimation_declaration.json").write_text(
        "not json", encoding="utf-8"
    )

    with pytest.raises(errors.ArtifactError, match="declaration"):
        scmle.api.project.Project.resume(original, data=estimation_frame())


def test_resume_fails_closed_for_tampered_estimation_declaration(tmp_path):
    """A changed declaration fails digest verification before resume."""
    original = interrupted_estimation_run(tmp_path)
    path = original / "parameter_estimation_declaration.json"
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["estimand"]["name"] = "tampered"
    path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(errors.ArtifactError, match="digest"):
        scmle.api.project.Project.resume(original, data=estimation_frame())


def test_project_run_rejects_missing_estimation_artifact(monkeypatch, tmp_path):
    """A missing persisted report cannot produce a completed result.

    Also enforces website/src/pages/docs/modeling.astro:279.
    """
    original_read = Estimation.read

    def remove_artifact(path):
        path.unlink()
        return original_read(path)

    monkeypatch.setattr(Estimation, "read", remove_artifact)

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
    """A report changed and re-digested after writing is still rejected.

    Also enforces website/src/pages/docs/modeling.astro:279.
    """
    original_read = Estimation.read

    def tamper_artifact(path):
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelope["report"]["diagnostics"]["data_identity"]["source_uri"] = (
            "tampered"
        )
        anchored = records.downgrade(Estimation.migrations, envelope["report"])
        envelope["digest"] = hashlib.sha256(
            json.dumps(anchored, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        path.write_text(json.dumps(envelope), encoding="utf-8")
        return original_read(path)

    monkeypatch.setattr(Estimation, "read", tamper_artifact)

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
    """Undeclared estimation cannot silently infer a scientific policy.

    Also enforces website/src/pages/docs/modeling.astro:167.
    """
    project = scmle.api.project.Project(
        estimation_frame(), "outcome", directory=tmp_path
    )

    with pytest.raises(errors.ConfigError, match="declare both"):
        project.estimate()


def test_partial_estimation_declaration_is_rejected(tmp_path):
    """An estimand without its policy is not a usable evidence contract.

    Also enforces website/src/pages/docs/modeling.astro:167.
    """
    with pytest.raises(errors.ConfigError, match="both"):
        scmle.api.project.Project(
            estimation_frame(),
            "outcome",
            estimand=declared_estimand(),
            directory=tmp_path,
        )


def test_incompatible_likelihood_fails_before_model_selection(tmp_path):
    """Likelihood mismatch is rejected before the orchestrator is called.

    Also enforces website/src/pages/docs/modeling.astro:231.
    """
    estimand = Estimand(
        name="wrong_likelihood",
        outcome="outcome",
        population="the declared project dataset",
        likelihood="bernoulli",
        features=("signal",),
        assumptions=("the conditional mean is linear",),
    )
    project = scmle.api.project.Project(
        estimation_frame(),
        "outcome",
        task="regression",
        budget=30,
        estimand=estimand,
        policy=Config(estimator="linear"),
        directory=tmp_path,
    )

    with pytest.raises(ValueError, match="expected gaussian"):
        project.run()

    assert not list(tmp_path.glob("run_*/parameter_estimation.json"))


def test_ipw_declaration_binds_the_project_treatment_column(tmp_path):
    """IPW cannot infer or silently substitute a treatment column."""
    frame = estimation_frame().assign(treatment=(numpy.arange(80) % 2))
    estimand = Estimand(
        name="treatment_effect",
        outcome="outcome",
        population="the declared project dataset",
        likelihood="gaussian",
        features=("treatment", "signal"),
        assumptions=("the declared causal assumptions hold",),
    )
    policy = Config(
        estimator="ipw_ate",
        covariance="iid",
        treatment=0,
        causal_assumptions=(
            "consistency",
            "conditional_exchangeability",
            "positivity",
        ),
    )

    project = scmle.api.project.Project(
        frame,
        "outcome",
        estimand=estimand,
        policy=policy,
        directory=tmp_path,
    )

    with pytest.raises(errors.ConfigError, match="treatment_column"):
        project.estimate()


def test_project_rejects_missing_cluster_and_missingness_columns(tmp_path):
    """Project declarations fail closed before estimation consumes rows."""
    project = scmle.api.project.Project(
        estimation_frame(),
        "outcome",
        estimand=declared_estimand(),
        policy=Config(
            estimator="linear", covariance="clustered", cluster="missing"
        ),
        directory=tmp_path,
    )
    with pytest.raises(errors.DataError, match="columns are missing"):
        project.estimate()

    missing_frame = estimation_frame()
    missing_frame.loc[0, "signal"] = numpy.nan
    project = scmle.api.project.Project(
        missing_frame,
        "outcome",
        estimand=declared_estimand(),
        policy=Config(estimator="linear"),
        directory=tmp_path,
    )
    with pytest.raises(errors.DataError, match="missing values"):
        project.estimate()


def test_complete_case_binds_missingness_and_run_artifact_to_result(tmp_path):
    """The result exposes the digest-verified, complete persisted report."""
    frame = estimation_frame()
    frame.loc[0, "signal"] = numpy.nan
    project = scmle.api.project.Project(
        frame,
        "outcome",
        task="regression",
        budget=30,
        estimand=declared_estimand(),
        policy=Config(estimator="linear", missing="complete_case"),
        directory=tmp_path,
    )

    result = project.run()

    artifact = result.directory / "parameter_estimation.json"
    restored = Estimation.read(artifact)
    assert result.estimation is not None
    assert result.estimation.digest() == restored.digest()
    assert result.estimation.rows.dropped == 1
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
    project = scmle.api.project.Project(
        source,
        "outcome",
        estimand=declared_estimand(),
        policy=Config(estimator="linear"),
        directory=tmp_path / "runs",
    )

    result = project.estimate()

    identity = result.diagnostics["data_identity"]
    assert identity["source_identity"].startswith("src_")
    assert identity["snapshot_id"].startswith("snap_")


def test_verify_replay_preserves_parameter_estimation_evidence(tmp_path):
    """Replay compares the scientific report as well as model evidence."""
    project = project_with_estimation(tmp_path)
    original = project.run()

    verdict = scmle.api.project.Project.verify(
        original.directory,
        data=estimation_frame(),
        output_dir=tmp_path / "replay",
    )

    assert verdict.level == "exact"
    assert (tmp_path / "replay").exists()
