"""Single-word keys on the write-only evaluation and model records (phase 5, F5).

Each record writes its new keys, and its `migrations` move the layout the
previous writer produced (spelled out here by hand) to the current one and
back. The three estimation envelopes are version 3; their digests stay on
the version-1 layout, so a report stored at any version still verifies. The
image model digest hashes its configuration at version 1, so a fitted model
keeps its digest.
"""

import dataclasses
import hashlib
import json

import numpy
import pytest
from sklearn.linear_model import LogisticRegression

from scmle.contracts import records
from scmle.evaluation import bias_variance
from scmle.evaluation import criteria
from scmle.evaluation import envelope
from scmle.evaluation import fit_health
from scmle.evaluation import learning_curve
from scmle.evaluation import longitudinal
from scmle.evaluation import mitigation
from scmle.evaluation import nested
from scmle.evaluation import parameter_estimation
from scmle.evaluation import release
from scmle.evaluation import resources
from scmle.evaluation import sequential
from scmle.evaluation import uplift
from scmle.evaluation.release import assessment
from scmle.explainability import quality
from scmle.models import documents
from scmle.models import images
from scmle.models.deeplearning import contracts
from tests.unit.test_evaluation_records import ESTIMATIONS
from tests.unit.test_evaluation_records import OLD
from tests.unit.test_evaluation_records import stored
from tests.unit.test_parameter_estimation import continuous_treatment_policy

DIGEST = "a" * 64


def moves(kind, old, new):
    """Asserts that `kind`'s single step maps `old` to `new` and back.

    Args:
        kind: A record class with a one-step `migrations`.
        old: The layout the previous writer produced.
        new: The layout the current writer produces.
    """
    assert len(kind.migrations) == 1
    assert records.upgrade(kind.migrations, dict(old), 1) == new
    assert records.downgrade(kind.migrations, dict(new)) == old


def writes(record, old, new):
    """Asserts that `record` writes `new` and its kind moves `old` to it.

    Args:
        record: The record.
        old: Its layout as the previous writer produced it.
        new: Its layout as the current writer produces it.
    """
    assert record.to_dict() == new
    moves(type(record), old, new)


def test_a_decomposition_writes_confounded_and_share():
    record = bias_variance.Decomposition(
        "squared_error", 10, 50, 0.3, 0.1, 0.2, 0.4, 0.7, 0.6, "balanced", "x"
    )
    common = {
        "kind": "squared_error",
        "replicates": 10,
        "rows": 50,
        "noise": 0.1,
        "bias": 0.2,
        "variance": 0.4,
        "total": 0.7,
        "verdict": "balanced",
        "advice": "x",
    }
    writes(
        record,
        {**common, "bias_plus_noise": 0.3, "variance_share": 0.6},
        {**common, "confounded": 0.3, "share": 0.6},
    )


@pytest.mark.parametrize(
    ("higher", "word"), [(True, "higher"), (False, "lower")]
)
def test_a_criterion_names_its_direction(higher, word):
    record = criteria.Criterion("roc_auc", 0.8, 0.9, higher, True)
    common = {"metric": "roc_auc", "required": 0.8, "achieved": 0.9}
    writes(
        record,
        {**common, "higher_is_better": higher, "met": True},
        {**common, "direction": word, "met": True},
    )


def test_a_health_writes_gains_and_noise():
    record = fit_health.Health("healthy", 0.3, 0.2, 0.66, 0.1, "fine")
    writes(
        record,
        {
            "verdict": "healthy",
            "train_gain": 0.3,
            "validation_gain": 0.2,
            "retained": 0.66,
            "noise_share": 0.1,
            "note": "fine",
        },
        {
            "verdict": "healthy",
            "retained": 0.66,
            "noise": 0.1,
            "note": "fine",
            "gains": {"training": 0.3, "validation": 0.2},
        },
    )


def test_a_curve_writes_its_gains():
    record = learning_curve.Curve((10, 20), (0.5, 0.6), "plateau", 0.1, 0.2)
    common = {"sizes": [10, 20], "scores": [0.5, 0.6], "verdict": "plateau"}
    writes(
        record,
        {**common, "last_gain": 0.1, "total_gain": 0.2},
        {**common, "gains": {"last": 0.1, "total": 0.2}},
    )


def test_a_mitigation_writes_gap_and_accuracy():
    record = mitigation.Mitigation(
        "group",
        "demographic_parity",
        {"a": 0.5},
        0.3,
        0.1,
        0.8,
        0.79,
        True,
        "r",
    )
    common = {
        "column": "group",
        "criterion": "demographic_parity",
        "thresholds": {"a": 0.5},
        "applied": True,
        "reason": "r",
    }
    writes(
        record,
        {
            **common,
            "gap_before": 0.3,
            "gap_after": 0.1,
            "accuracy_before": 0.8,
            "accuracy_after": 0.79,
        },
        {
            **common,
            "gap": {"before": 0.3, "after": 0.1},
            "accuracy": {"before": 0.8, "after": 0.79},
        },
    )


def test_nested_scores_write_outer_and_inner():
    record = nested.Scores((0.7, 0.8), (0.9, 0.85), 0.75, 0.125, 2, 4)
    common = {"mean": 0.75, "optimism": 0.125, "folds": 2, "configurations": 4}
    writes(
        record,
        {**common, "outer_scores": (0.7, 0.8), "inner_scores": (0.9, 0.85)},
        {**common, "outer": (0.7, 0.8), "inner": (0.9, 0.85)},
    )


def test_a_usage_writes_size_latency_and_timed():
    writes(
        resources.Usage(1.5, 2.5, 30, "m"),
        {
            "model_mb": 1.5,
            "p95_latency_ms": 2.5,
            "rows_timed": 30,
            "machine": "m",
        },
        {"size": 1.5, "latency": 2.5, "timed": 30, "machine": "m"},
    )


def test_a_propensity_writes_treated_and_balance():
    record = uplift.Propensity(0.5, 0.52, True, "n", 0.4)
    common = {"auc": 0.52, "randomized": True, "note": "n"}
    writes(
        record,
        {**common, "treated_share": 0.5, "balance_p_value": 0.4},
        {**common, "treated": 0.5, "balance": 0.4},
    )


def finding(kind, passed, blocking=True):
    """A release finding of `kind`."""
    return release.Finding(kind, f"{kind.value}.x", passed, blocking, "why")


def test_a_tally_writes_blockers():
    record = release.Tally(3, 1, 2, 2, 1)
    common = {"total": 3, "passed": 1, "failed": 2, "blocking": 2}
    writes(
        record,
        {**common, "blocking_failures": 1},
        {**common, "blockers": 1},
    )


def test_a_verdict_and_its_scorecard_write_single_words():
    findings = (
        finding(release.Kind.METRIC, False),
        finding(release.Kind.INPUT, False, blocking=False),
        finding(release.Kind.INPUT, True),
    )
    verdict = release.Verdict("binary", 3, False, findings, ("why",))
    layout = verdict.to_dict()
    assert layout["policy"] == 3
    assert layout["blockers"] == ["why"]
    readiness = layout["readiness"]
    assert readiness["findings"] == {
        "total": 3,
        "passed": 1,
        "failed": 2,
        "blocking": 2,
    }
    assert (readiness["blockers"], readiness["warnings"]) == (1, 1)
    assert readiness["kinds"]["input"]["blockers"] == 0
    old = records.downgrade(release.Verdict.migrations, layout)
    assert old["policy_schema_version"] == 3
    assert old["blocking_reasons"] == ["why"]
    scorecard = old["readiness"]
    assert scorecard["total_findings"] == 3
    assert scorecard["non_blocking_failures"] == 1
    assert scorecard["counts_by_kind"]["metric"]["blocking_failures"] == 1
    assert records.upgrade(release.Verdict.migrations, old, 1) == layout


def test_release_grades_both_quality_layouts():
    assert assessment.grade({"quality": {"stability": 0.9}}, "stability") == 0.9
    legacy = {"quality": {"fidelity_score": 0.8}}
    assert assessment.grade(legacy, "fidelity") == 0.8


def test_explanation_thresholds_and_evaluation_write_single_words():
    thresholds = quality.Thresholds(0.8, 0.9)
    moves(
        quality.Thresholds,
        {"stability_minimum": 0.8, "fidelity_minimum": 0.9},
        {"stability": 0.8, "fidelity": 0.9},
    )
    record = quality.Evaluation("m", 2, 0.95, 0.92, thresholds, True, ("ok",))
    common = {"method": "m", "passed": True, "findings": ["ok"]}
    writes(
        record,
        {
            **common,
            "sample_count": 2,
            "stability_score": 0.95,
            "fidelity_score": 0.92,
            "thresholds": {"stability_minimum": 0.8, "fidelity_minimum": 0.9},
        },
        {
            **common,
            "samples": 2,
            "stability": 0.95,
            "fidelity": 0.92,
            "thresholds": {"stability": 0.8, "fidelity": 0.9},
        },
    )


def test_a_truth_writes_expected_and_overlap():
    common = {"method": "m", "reference": "r", "limitations": []}
    writes(
        quality.Truth("m", "r", ("a", "b"), 0.5),
        {**common, "expected_features": ["a", "b"], "minimum_overlap": 0.5},
        {**common, "expected": ["a", "b"], "overlap": 0.5},
    )


def comparison():
    """A passing comparison of two features."""
    return quality.Comparison(
        "s1", "m", "r", ("a", "b"), ("a", "c"), 0.5, 0.5, True, ("ok",), ()
    )


def test_a_comparison_and_its_suite_write_single_words():
    common = {
        "status": "available",
        "method": "m",
        "reference": "r",
        "passed": True,
        "findings": ["ok"],
        "limitations": [],
    }
    old = {
        **common,
        "sample_id": "s1",
        "expected_features": ["a", "b"],
        "observed_features": ["a", "c"],
        "overlap_score": 0.5,
        "minimum_overlap": 0.5,
    }
    new = {
        **common,
        "sample": "s1",
        "expected": ["a", "b"],
        "observed": ["a", "c"],
        "agreement": 0.5,
        "overlap": 0.5,
    }
    writes(comparison(), old, new)
    suite = quality.Suite(2, 1, (comparison(),), False, ("partial",))
    layout = suite.to_dict()
    assert (layout["samples"], layout["references"]) == (2, 1)
    assert layout["evaluations"] == [new]
    before = records.downgrade(quality.Suite.migrations, layout)
    assert (before["sample_count"], before["reference_count"]) == (2, 1)
    assert before["evaluations"] == [old]
    assert records.upgrade(quality.Suite.migrations, before, 1) == layout


def test_a_deep_learning_boundary_writes_single_words():
    boundary = contracts.TABULAR_TRANSFORMER_CONTRACT
    layout = boundary.to_dict()
    assert layout["family"] == boundary.family
    assert layout["extras"] == list(boundary.extras)
    assert {"trusted", "atomic", "schema_version"} <= set(layout["checkpoint"])
    assert {"rank", "dtype"} <= set(layout["inference"])
    old = records.downgrade(contracts.Boundary.migrations, layout)
    assert old["model_family"] == boundary.family
    assert old["required_extras"] == list(boundary.extras)
    assert old["checkpoint"]["trusted_local_storage"] is True
    assert old["inference"]["input_rank"] == 2
    assert records.upgrade(contracts.Boundary.migrations, old, 1) == layout


def test_a_contribution_writes_features():
    record = contracts.Contribution(numpy.zeros((1, 2)), "m", "t", 2)
    common = {"method": "m", "target": "t", "row_count": 1, "values": [[0, 0]]}
    writes(record, {**common, "feature_count": 2}, {**common, "features": 2})


@pytest.mark.parametrize("module", [documents, images])
def test_model_evaluations_and_explanations_write_single_words(module):
    evaluation = module.Evaluation(4, 0.75, 0.7, 0.5, ("a", "b"))
    common = {
        "accuracy": 0.75,
        "macro_f1": 0.7,
        "log_loss": 0.5,
        "labels": ("a", "b"),
    }
    writes(evaluation, {**common, "sample_count": 4}, {**common, "samples": 4})
    explanation = module.Explanation(DIGEST, "b" * 64, "m", (("x", 1.0),), ())
    layout = explanation.to_dict()
    assert layout["digests"] == {"model": DIGEST, "payload": "b" * 64}
    old = records.downgrade(module.Explanation.migrations, layout)
    assert (old["model_digest"], old["input_hash"]) == (DIGEST, "b" * 64)
    assert records.upgrade(module.Explanation.migrations, old, 1) == layout


def test_a_document_drift_writes_digests_and_signals():
    record = documents.Drift(DIGEST, "b" * 64, 0.1, -0.2, 0.05, 0.25)
    old = {
        "reference_hash": DIGEST,
        "observed_hash": "b" * 64,
        "language_distance": 0.1,
        "mean_character_delta": -0.2,
        "duplicate_rate": 0.05,
        "threshold": 0.25,
        "shifted": False,
    }
    new = {
        "digests": {"reference": DIGEST, "observed": "b" * 64},
        "language": 0.1,
        "characters": -0.2,
        "duplicates": 0.05,
        "threshold": 0.25,
        "shifted": False,
    }
    writes(record, old, new)


def test_an_image_drift_writes_digests_and_signals():
    record = images.Drift(DIGEST, "b" * 64, 0.3, 0.05, 0.25)
    old = {
        "reference_hash": DIGEST,
        "observed_hash": "b" * 64,
        "mean_pixel_distance": 0.3,
        "duplicate_rate": 0.05,
        "threshold": 0.25,
        "shifted": True,
    }
    new = {
        "digests": {"reference": DIGEST, "observed": "b" * 64},
        "pixels": 0.3,
        "duplicates": 0.05,
        "threshold": 0.25,
        "shifted": True,
    }
    writes(record, old, new)


def fitted():
    """An image classifier fitted by hand on two flattened 2 by 2 pixels."""
    classifier = images.Classifier(images.Config(size=(2, 2)))
    features = numpy.array([[0.0] * 12, [1.0] * 12, [0.1] * 12, [0.9] * 12])
    classifier.estimator = LogisticRegression().fit(
        features, ["red", "blue", "red", "blue"]
    )
    classifier.names = tuple(f"p{index}" for index in range(12))
    return classifier


def test_an_image_config_writes_size_and_keeps_its_model_digest():
    config = images.Config(size=(2, 2), max_iter=200)
    assert config.to_dict()["size"] == [2, 2]
    old = records.downgrade(images.Config.migrations, config.to_dict())
    assert old["target_size"] == [2, 2] and "size" not in old
    classifier = fitted()
    estimator = classifier.estimator
    state = {
        "config": records.downgrade(
            images.Config.migrations, classifier.config.to_dict()
        ),
        "feature_names": classifier.names,
        "classes": estimator.classes_.tolist(),
        "coef": estimator.coef_.tolist(),
        "intercept": estimator.intercept_.tolist(),
    }
    assert "target_size" in state["config"]
    encoded = json.dumps(
        state, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert classifier.digest == hashlib.sha256(encoded).hexdigest()


def test_an_image_package_writes_version_two_and_reads_version_one(tmp_path):
    classifier = fitted()
    classifier.save(tmp_path / "new", signing_key="k")
    manifest = json.loads((tmp_path / "new" / "manifest.json").read_text())
    assert manifest["schema_version"] == images.archive.MANIFEST == 2
    assert manifest["config"]["size"] == [2, 2]
    loaded = images.Classifier.load(tmp_path / "new", signing_key="k")
    assert loaded.digest == classifier.digest
    older = tmp_path / "old"
    older.mkdir()
    (older / "model.joblib").write_bytes(
        (tmp_path / "new" / "model.joblib").read_bytes()
    )
    manifest = {
        key: value
        for key, value in manifest.items()
        if key in ("model_path", "artifact_digest", "model_digest")
    }
    manifest["schema_version"] = 1
    manifest["config"] = records.downgrade(
        images.Config.migrations, classifier.config.to_dict()
    )
    (older / "manifest.json").write_text(json.dumps(manifest))
    loaded = images.Classifier.load(older, trusted=True)
    assert loaded.digest == classifier.digest


LONGITUDINAL = {
    "consistency",
    "positivity",
    "conditional_exchangeability",
    "no_unmeasured_time_varying_confounding",
    "no_time_varying_covariates",
}


def test_longitudinal_declarations_write_single_words():
    estimand = longitudinal.Estimand(
        "e", "id", "t", "a", "y", ("x",), "p", tuple(sorted(LONGITUDINAL))
    )
    layout = estimand.to_dict()
    assert (layout["subject"], layout["time"], layout["treatment"]) == (
        "id",
        "t",
        "a",
    )
    old = records.downgrade(longitudinal.Estimand.migrations, layout)
    assert old["subject_identity"] == "id" and old["time_column"] == "t"
    assert (
        longitudinal.Estimand(**envelope.rename(longitudinal.Estimand, old))
        == estimand
    )
    policy = longitudinal.Config().to_dict()
    assert policy["minimums"] == {"period": 8, "arm": 2, "ess": 2.0}
    assert (policy["confidence"], policy["clip"]) == (0.95, 0.01)
    assert (policy["iterations"], policy["reference"]) == (100, 1e-6)
    old = records.downgrade(longitudinal.Config.migrations, policy)
    assert (
        longitudinal.Config(**envelope.rename(longitudinal.Config, old))
        == longitudinal.Config()
    )
    period = longitudinal.Period(
        "1", 0.5, 0.1, (0.3, 0.7), 10, 4, 6, 0.2, 0.8, 0.0, 3.5, 5.5, True, 4
    )
    layout = period.to_dict()
    assert (layout["se"], layout["treated"], layout["control"]) == (0.1, 4, 6)
    assert layout["propensity"] == {"low": 0.2, "high": 0.8}
    assert layout["ess"] == {"treated": 3.5, "control": 5.5}
    assert layout["clipped"] == 0.0
    old = records.downgrade(longitudinal.Period.migrations, layout)
    assert (
        longitudinal.Period(**envelope.rename(longitudinal.Period, old))
        == period
    )


SEQUENTIAL = {
    "consistency",
    "positivity",
    "sequential_exchangeability",
    "no_unmeasured_time_varying_confounding",
    "time_varying_covariates_are_pre_treatment",
    "complete_follow_up",
}


def test_sequential_declarations_write_single_words():
    estimand = sequential.Estimand(
        "e",
        "id",
        "t",
        "a",
        "y",
        ("x",),
        ("z",),
        "p",
        tuple(sorted(SEQUENTIAL)),
    )
    layout = estimand.to_dict()
    assert layout["covariates"] == {"baseline": ("x",), "varying": ("z",)}
    assert layout["regimes"] == {"target": 1, "comparator": 0}
    old = records.downgrade(sequential.Estimand.migrations, layout)
    assert (
        sequential.Estimand(**envelope.rename(sequential.Estimand, old))
        == estimand
    )
    policy = sequential.Config().to_dict()
    assert policy["minimums"] == {"subjects": 30, "arm": 2, "ess": 5.0}
    assert policy["weight"] == 20.0
    old = records.downgrade(sequential.Config.migrations, policy)
    assert (
        sequential.Config(**envelope.rename(sequential.Config, old))
        == sequential.Config()
    )
    period = sequential.Diagnostics(
        "1", 10, 4, 6, 0.2, 0.8, 0.3, 0.7, 0.1, 25.0, 20.0, 7.5
    )
    layout = period.to_dict()
    assert layout["denominator"] == {"low": 0.2, "high": 0.8}
    assert layout["numerator"] == {"low": 0.3, "high": 0.7}
    assert (layout["truncated"], layout["untruncated"]) == (0.1, 25.0)
    assert (layout["weight"], layout["ess"]) == (20.0, 7.5)
    old = records.downgrade(sequential.Diagnostics.migrations, layout)
    assert (
        sequential.Diagnostics(**envelope.rename(sequential.Diagnostics, old))
        == period
    )
    regime = sequential.Estimate(
        "always_treated", 1, 0.5, 0.1, (0.3, 0.7), 5, 6.0, 4.5
    )
    layout = regime.to_dict()
    assert (layout["se"], layout["weights"], layout["ess"]) == (0.1, 6.0, 4.5)
    old = records.downgrade(sequential.Estimate.migrations, layout)
    assert (
        sequential.Estimate(**envelope.rename(sequential.Estimate, old))
        == regime
    )


def test_parameter_declarations_write_single_words():
    estimand = parameter_estimation.Estimand(
        "e", "y", "p", "gaussian", ("x", "z"), ("a",)
    )
    layout = estimand.to_dict()
    assert layout["features"] == ("x", "z") and "feature_names" not in layout
    estimate = parameter_estimation.Estimate("x", 0.5, 0.1, (0.3, 0.7), 0.01)
    layout = estimate.to_dict()
    assert (layout["se"], layout["p_value"]) == (0.1, 0.01)
    policy = continuous_treatment_policy("clustered")
    layout = policy.to_dict()
    assert layout["cluster"] == "site_id"
    assert (layout["treatment"], layout["cap"]) == (0, 8.0)
    assert layout["assumptions"] == {
        "continuous": policy.continuous_treatment_assumptions
    }
    assert layout["confidence"] == 0.95
    old = records.downgrade(parameter_estimation.Config.migrations, layout)
    assert old["continuous_weight_cap"] == 8.0 and "assumptions" not in old
    assert (
        parameter_estimation.Config(
            **envelope.rename(parameter_estimation.Config, old)
        )
        == policy
    )
    plain = parameter_estimation.Config(estimator="linear").to_dict()
    assert "assumptions" not in plain


@pytest.mark.parametrize("name", sorted(ESTIMATIONS))
def test_an_estimation_of_each_version_verifies_its_anchored_digest(
    name, tmp_path
):
    kind = ESTIMATIONS[name]
    old = stored(name)
    result = kind.read(OLD / f"{name}.json")
    current = tmp_path / "current.json"
    assert result.write(current) == old["digest"]
    written = json.loads(current.read_text("utf-8"))
    assert written[records.KEY] == envelope.VERSION == 3
    assert "policy_schema_version" not in json.dumps(written)
    middle = {
        records.KEY: 2,
        "report": kind.migrations[1].downgrade(written["report"]),
        "digest": old["digest"],
    }
    path = tmp_path / "middle.json"
    path.write_text(json.dumps(middle), encoding="utf-8")
    assert kind.read(path) == result
    assert kind.read(current) == result


@pytest.mark.parametrize("name", sorted(ESTIMATIONS))
def test_an_estimation_newer_than_the_build_is_refused(name, tmp_path):
    envelope_record = {**stored(name), records.KEY: 4}
    path = tmp_path / "newer.json"
    path.write_text(json.dumps(envelope_record), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        ESTIMATIONS[name].read(path)


def test_estimation_parts_are_written_in_single_words():
    result = sequential.Estimation.read(OLD / "sequential.json")
    layout = result.to_dict()
    assert "subject" in layout["estimand"]
    assert "minimums" in layout["policy"]
    assert all("ess" in period for period in layout["periods"])
    assert all("weights" in regime for regime in layout["regimes"])
    replayed = dataclasses.replace(result)
    assert sequential.Estimation.from_dict(layout) == replayed
