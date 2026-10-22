"""Tests for deterministic explanation quality measures."""

import pytest

from scml.explainability import quality


def report(*features: str) -> dict[str, object]:
    """Build a compact permutation-style explanation report."""
    return {
        "features": [
            {"feature": feature, "importance": 1.0 / (index + 1)}
            for index, feature in enumerate(features)
        ]
    }


def test_ranking_stability_measures_top_k_overlap():
    score = quality.ranking_stability(
        (report("signal", "age", "noise"), report("signal", "noise", "age")),
        top_k=2,
    )

    assert score == pytest.approx(1.0 / 3.0)


def test_linear_contribution_report_is_ranked_and_json_native():
    result = quality.linear_contribution_report(
        "sample-1",
        ("signal", "noise", "age"),
        (2.0, 10.0, -1.0),
        (0.5, 0.01, -2.0),
        top_k=2,
    )

    assert result == {
        "sample_id": "sample-1",
        "features": [
            {"feature": "age", "importance": 2.0},
            {"feature": "signal", "importance": 1.0},
        ],
        "terms": [["age", 2.0], ["signal", 1.0]],
    }


@pytest.mark.parametrize(
    "factory",
    (
        lambda: quality.linear_contribution_report(
            "", ("signal",), (1.0,), (1.0,)
        ),
        lambda: quality.linear_contribution_report(
            "sample", ("signal",), (1.0,), (1.0,), top_k=2
        ),
        lambda: quality.linear_contribution_report(
            "sample", ("signal",), (1.0,), (1.0, 2.0)
        ),
    ),
)
def test_linear_contribution_report_rejects_incomplete_contract(factory):
    with pytest.raises(ValueError):
        factory()


def test_fidelity_scores_reconstruction_against_mean_baseline():
    assert quality.fidelity_score((0.0, 1.0, 2.0), (0.0, 1.0, 2.0)) == 1.0
    assert quality.fidelity_score((0.0, 1.0, 2.0), (1.0, 1.0, 1.0)) == 0.0


def test_quality_evaluation_fails_closed_for_missing_required_measure():
    result = quality.evaluate(
        "permutation_importance",
        sample_count=2,
        thresholds=quality.QualityThresholds(stability_minimum=0.8),
    )

    assert result.passed is False
    assert result.findings == ("stability evidence is missing",)


def test_quality_evaluation_passes_declared_measures():
    result = quality.evaluate(
        "permutation_importance",
        sample_count=2,
        thresholds=quality.QualityThresholds(
            stability_minimum=0.8, fidelity_minimum=0.9
        ),
        stability=0.95,
        fidelity=0.92,
    )

    assert result.passed is True
    assert result.to_dict()["thresholds"] == {
        "stability_minimum": 0.8,
        "fidelity_minimum": 0.9,
    }


def test_independent_ground_truth_evaluation_is_explicit_and_deterministic():
    """Reference rankings are compared without making causal claims."""
    ground_truth = quality.ExplanationGroundTruth(
        method="annotated_feature_reference",
        reference="tests/fixtures/explanation-reference-v1.json",
        expected_features=("signal", "age"),
        minimum_overlap=1.0,
        limitations=("The reference covers this synthetic fixture only.",),
    )

    result = quality.evaluate_ground_truth(
        ground_truth,
        report("signal", "age", "noise"),
        "sample-1",
    )

    assert result.passed is True
    assert result.overlap_score == 1.0
    assert result.to_dict()["status"] == "available"
    assert result.to_dict()["reference"] == (
        "tests/fixtures/explanation-reference-v1.json"
    )


def test_independent_ground_truth_evaluation_fails_on_mismatched_ranking():
    """A supplied reference cannot be silently treated as satisfied."""
    ground_truth = quality.ExplanationGroundTruth(
        method="annotated_feature_reference",
        reference="fixture://reference",
        expected_features=("signal", "age"),
    )

    result = quality.evaluate_ground_truth(
        ground_truth,
        report("noise", "age", "signal"),
        "sample-1",
    )

    assert result.passed is False
    assert result.overlap_score == 0.5
    assert result.findings == (
        "independent explanation reference overlap is below threshold",
    )


def test_independent_ground_truth_suite_fails_closed_for_partial_references():
    """A suite cannot pass when only part of the held-out set is referenced."""
    ground_truth = quality.ExplanationGroundTruth(
        method="synthetic_oracle",
        reference="fixture://reference",
        expected_features=("signal",),
    )

    result = quality.evaluate_ground_truth_suite(
        sample_count=2,
        references=((ground_truth, report("signal", "noise"), "sample-1"),),
    )

    assert result.status == "partial"
    assert result.passed is False
    assert result.to_dict()["minimum_overlap_score"] == 1.0


@pytest.mark.parametrize(
    "factory",
    (
        lambda: quality.QualityThresholds(),
        lambda: quality.fidelity_score((1.0,), (1.0, 2.0)),
        lambda: quality.ranking_stability((report("signal"),)),
    ),
)
def test_quality_contract_rejects_incomplete_evidence(factory):
    with pytest.raises(ValueError):
        factory()
