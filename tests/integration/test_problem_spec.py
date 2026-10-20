"""A run is judged against the success criteria its problem statement set."""

import numpy as np
import pandas as pd
import pytest
import support

from scml.contracts import errors
from scml.contracts import tasks


def frame(rows=700, seed=0):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {"a": rng.normal(size=rows), "b": rng.normal(size=rows)}
    )
    data["y"] = (data.a + rng.normal(0, 0.4, rows) > 0).astype(int)
    return data


def run(tmp_path, criteria):
    return support.make_project(
        frame(),
        "y",
        time_budget=300,
        output_dir=tmp_path,
        problem=tasks.ProblemSpec(
            decision="approve or decline",
            population="applicants",
            success_criteria=criteria,
        ),
    ).run()


def test_met_criteria_are_reported_and_the_summary_says_so(tmp_path):
    result = run(tmp_path, {"roc_auc": 0.8})

    assert result.meets_criteria
    assert result.evaluation["success_criteria"][0]["met"]
    assert result.evaluation["release_evaluation"]["passed"]
    assert result.evaluation["explanation_evidence"]
    assert result.evaluation["provenance"]["config_hash"]
    assert result.evaluation["provenance"]["dataset"]["snapshot_id"]
    assert result.evaluation["provenance"]["model"]["manifest_sha256"]
    assert (result.run_dir / "provenance.json").exists()
    assert result.evaluation["sbom"]["bomFormat"] == "CycloneDX"
    assert (result.run_dir / "sbom.json").exists()
    assert result.evaluation["reliability_plan"]["release_blocked"] is True
    assert (result.run_dir / "fault_injection_plan.json").exists()
    assert (
        result.run_dir / "explanations/permutation_importance.json"
    ).exists()
    assert (result.run_dir / "capability_registry.json").exists()
    assert (result.run_dir / "evaluation_policy.json").exists()
    assert (result.run_dir / "release_evidence.json").exists()
    assert "success criteria met" in result.summary()


def test_missed_criteria_are_named_with_the_gap(tmp_path):
    result = run(tmp_path, {"roc_auc": 0.9999, "log_loss": 0.0001})

    assert not result.meets_criteria
    assert not result.evaluation["release_evaluation"]["passed"]
    assert result.evaluation["release_evaluation"]["blocked"]
    text = result.summary()
    assert "NOT met" in text and "roc_auc" in text and "log_loss" in text


def test_no_criteria_means_nothing_to_miss(tmp_path):
    result = run(tmp_path, {})

    assert result.meets_criteria and result.evaluation["success_criteria"] == []
    assert result.evaluation["release_evaluation"]["status"] == (
        "policy_not_declared"
    )
    assert "criteria" not in result.summary()


def test_a_misspelled_metric_stops_the_run_before_the_test_is_opened(tmp_path):
    with pytest.raises(errors.ConfigError, match="unreported metrics"):
        run(tmp_path, {"rocauc": 0.8})
