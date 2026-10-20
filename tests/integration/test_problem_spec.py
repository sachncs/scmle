"""A run is judged against the success criteria its problem statement set."""

import json

import numpy as np
import pandas as pd
import pytest
import support

from scml.contracts import errors
from scml.contracts import evidence
from scml.contracts import tasks
from scml.governance import attestation


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
    assert not result.evaluation["release_evaluation"]["passed"]
    assert any(
        finding["kind"] == "reliability"
        for finding in result.evaluation["release_evaluation"]["findings"]
    )
    assert result.evaluation["explanation_evidence"]
    assert result.evaluation["provenance"]["config_hash"]
    assert result.evaluation["provenance"]["dataset"]["snapshot_id"]
    assert result.evaluation["provenance"]["model"]["manifest_sha256"]
    assert (result.run_dir / "provenance.json").exists()
    assert result.evaluation["sbom"]["bomFormat"] == "CycloneDX"
    assert (result.run_dir / "sbom.json").exists()
    assert result.evaluation["release_attestation"]["schema_version"] == 1
    assert (result.run_dir / "release_attestation.json").exists()
    assert result.evaluation["reliability_plan"]["release_blocked"] is True
    assert (result.run_dir / "fault_injection_plan.json").exists()
    source_registry = json.loads(
        (result.run_dir / "source_registry.json").read_text()
    )
    assert source_registry["schema_compatibility"]["status"] == "initial"
    model_card = json.loads((result.run_dir / "model_card.json").read_text())
    assert (
        model_card["snapshot_id"]
        == result.evaluation["provenance"]["dataset"]["snapshot_id"]
    )
    assert model_card["review_status"] == "machine_generated_unreviewed"
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


def test_packaged_release_attestation_is_signed_when_key_is_configured(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SCML_SIGNING_KEY", "release-key")

    result = run(tmp_path, {"roc_auc": 0.8})
    record = attestation.ReleaseAttestation.from_dict(
        result.evaluation["release_attestation"]
    )

    assert record.attestation_signature
    assert record.verify(
        result.evaluation["provenance"],
        result.evaluation["sbom"],
        signing_key="release-key",
    )
    evidence_index = evidence.EvidenceIndex.from_dict(
        result.evaluation["release_evidence"]
    )
    assert evidence_index.verify("release-key")
    assert (result.run_dir / "release_evidence.json").exists()
    assert not any(
        finding["kind"] == "security"
        for finding in result.evaluation["release_evaluation"]["findings"]
    )
