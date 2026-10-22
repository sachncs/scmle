"""Project integration for the separate parameter-estimation evidence track."""

import numpy as np
import pandas as pd
import pytest

import scml
from scml.contracts import errors
from scml.evaluation.parameter_estimation import EstimandSpec
from scml.evaluation.parameter_estimation import EstimationPolicy
from scml.evaluation.parameter_estimation import EstimationResult


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
    assert "parameter_estimation" not in result.evaluation


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
