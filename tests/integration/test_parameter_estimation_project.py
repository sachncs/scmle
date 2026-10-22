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
