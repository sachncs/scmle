"""`evaluation.json` and six write-only kinds write single-word keys (phase 5).

`evaluation.json` moves to version 2 (`schema_version` 2); its reader,
`results.evaluation`, takes a version-1 file (no version key) or a version-2
one to the version-2 layout, and the step downgrades back. The six F6 kinds
(the non-tabular `Scope`, the bundle `Manifest`, the claims-audit
`Verification`, the inference `Link`, the modality `Package` and the retry
policy `Retries`) route `to_dict` through their `KEYS` migration; the ones
that are read back accept either layout, and the hashed or signed ones
(`Link`, `Package` inside a lifecycle report) keep their version-1 anchor.
"""

import json

import pytest

from scmle.api import non_tabular
from scmle.artifacts import reproducibility
from scmle.artifacts import results
from scmle.artifacts.bundle import provenance
from scmle.artifacts.claims_audit import verdict
from scmle.contracts import records
from scmle.inference import evidence
from scmle.pipeline.modality_lifecycle import containers
from scmle.pipeline.modality_lifecycle import reporting
from scmle.runtime import retry
from scmle.trail import reports

NOW = "2026-01-01T00:00:00+00:00"
DIGEST = "a" * 64
LINK_DIGEST = "7ce5adbe5f79c2e07aff5c1a7cae04c461a734612d856fd38032530e7e147302"


def original() -> dict:
    """An `evaluation.json` as version 1 wrote it (no version key)."""
    return {
        "run_id": "run-1",
        "task": "binary",
        "metric": "roc_auc",
        "metric_greater_is_better": True,
        "selection_policy": "best mean",
        "selected_model": "logistic",
        "selected_trial_id": "trial_0001",
        "selected_params": {"C": 1.0},
        "selected_strategy": "default",
        "selected_cv_score": 0.8,
        "selected_cv_std": 0.01,
        "fit_health": {"verdict": "healthy", "note": "fine"},
        "cv_confidence": {"interval": [0.7, 0.9]},
        "baseline_cv_score": 0.5,
        "test_metrics": {"roc_auc": 0.81},
        "weighted_test_metrics": None,
        "semi_supervised": None,
        "fairness_mitigation": None,
        "execution_backend": "local",
        "pareto_front": [],
        "evaluation_prevalence": {"natural": True},
        "success_criteria": [],
        "streamed_locked": None,
        "uplift": None,
        "n_dev": 80,
        "n_test": 20,
        "split_fingerprint": "sha256:split",
        "split_strategy": "stratified",
        "decision": None,
        "net_benefit": None,
        "resources": {},
        "resource_limits": {},
        "skipped_for_time": [],
        "test_set_opened": 1,
        "diagnostics": [],
        "elapsed_seconds": 3.5,
        "reproducibility": "same seed",
        "learning_curve": None,
        "nested_cv": None,
        "bias_variance": None,
        "sensitivity": None,
        "forecast_backtest": None,
        "quantile_coverage": None,
        "model_card": {"schema_version": 2},
        "reliability_plan": {},
        "sbom": {},
        "provenance": {},
        "release_attestation": {"signed": False},
        "release_policy": None,
        "release_evaluation": {"status": "policy_not_declared"},
        "release_evidence": {"records": [], "ready": False},
        "explanation_evidence": {},
        "artifact_verification": {"passed": True},
    }


def current() -> dict:
    """The same record at version 2."""
    return {
        "schema_version": 2,
        "run": "run-1",
        "task": "binary",
        "metric": "roc_auc",
        "direction": "higher",
        "policy": "best mean",
        "selected": {
            "model": "logistic",
            "trial": "trial_0001",
            "params": {"C": 1.0},
            "strategy": "default",
            "score": 0.8,
            "deviation": 0.01,
        },
        "health": {"verdict": "healthy", "note": "fine"},
        "confidence": {"interval": [0.7, 0.9]},
        "baseline": 0.5,
        "metrics": {"roc_auc": 0.81},
        "weighted": None,
        "unlabeled": None,
        "mitigation": None,
        "backend": "local",
        "pareto": [],
        "prevalence": {"natural": True},
        "criteria": [],
        "streamed": None,
        "uplift": None,
        "rows": {"development": 80, "test": 20},
        "digests": {"split": "sha256:split"},
        "validation": "stratified",
        "decision": None,
        "benefit": None,
        "resources": {},
        "limits": {},
        "skipped": [],
        "opened": 1,
        "diagnostics": [],
        "elapsed": 3.5,
        "reproducibility": "same seed",
        "learning": None,
        "nested": None,
        "decomposition": None,
        "sensitivity": None,
        "backtest": None,
        "quantiles": None,
        "card": {"schema_version": 2},
        "reliability": {},
        "sbom": {},
        "provenance": {},
        "release": {
            "attestation": {"signed": False},
            "policy": None,
            "evaluation": {"status": "policy_not_declared"},
            "evidence": {"records": [], "ready": False},
        },
        "explanations": {},
        "verification": {"passed": True},
    }


def test_evaluation_reads_version_one_as_version_two():
    assert results.evaluation(original()) == current()


def test_evaluation_reads_version_two_unchanged():
    assert results.evaluation(current()) == current()


def test_evaluation_direction_is_lower_for_a_loss():
    record = original() | {"metric_greater_is_better": False}
    assert results.evaluation(record)["direction"] == "lower"


def test_evaluation_downgrades_to_version_one():
    layout = current()
    del layout[records.KEY]
    assert records.downgrade(results.MIGRATIONS, layout) == original()


def test_evaluation_refuses_a_newer_version():
    with pytest.raises(ValueError, match="newer"):
        results.evaluation(current() | {"schema_version": 3})


def test_evaluation_keys_are_single_words():
    stored = results.evaluation(original())
    groups = ("selected", "rows", "digests", "release")
    keys = [*stored, *(key for name in groups for key in stored[name])]
    assert [k for k in keys if "_" in k] == [records.KEY]


@pytest.mark.parametrize("layout", [original, current])
def test_reread_reads_either_version(tmp_path, layout):
    (tmp_path / "trials").mkdir()
    for name, body in {
        "config.json": {},
        "dataset_manifest.json": {},
        "evaluation.json": layout(),
    }.items():
        (tmp_path / name).write_text(json.dumps(body), encoding="utf-8")
    read = reproducibility.reread(tmp_path)
    assert read["split"] == "sha256:split"
    assert read["selected"] == ("logistic", "default", '{"C": 1.0}')


@pytest.mark.parametrize("layout", [original, current])
def test_experiment_report_reads_either_version(tmp_path, layout):
    base = tmp_path / "trail" / "retained" / "run"
    (base / "trials").mkdir(parents=True)
    (base / "evaluation.json").write_text(
        json.dumps(layout()), encoding="utf-8"
    )
    body = reports.experiments(tmp_path / "trail")
    assert body["selected_trial"] == "trial_0001"
    assert body["selected_model"] == "logistic"
    assert body["selection_policy"] == "best mean"
    assert body["cv_interval"] == [0.7, 0.9]
    assert body["test_metrics"] == {"roc_auc": 0.81}


def test_scope_writes_single_word_keys():
    scope = non_tabular.Scope(
        modality="image",
        representation="pixels",
        decoder_policy="caller decodes",
        limitations=("small",),
    )
    written = scope.to_dict()
    assert written["decoder"] == "caller decodes"
    assert written["family"] == scope.model_family
    assert "decoder_policy" not in written
    assert "model_family" not in written
    assert non_tabular.Scope.KEYS.downgrade(written) == {
        "modality": "image",
        "representation": "pixels",
        "decoder_policy": "caller decodes",
        "model_family": scope.model_family,
        "status": "experimental",
        "limitations": ["small"],
    }


def manifest() -> provenance.Manifest:
    return provenance.Manifest(
        "logistic",
        "v1",
        "roc_auc",
        model_params={"C": 1.0},
        decision_threshold=0.4,
    )


def test_manifest_writes_single_word_keys():
    written = manifest().to_dict()
    assert written["model"] == "logistic"
    assert written["version"] == "v1"
    assert written["params"] == {"C": 1.0}
    assert written["threshold"] == 0.4
    assert "scmle" in written
    for old in (
        "model_name",
        "model_version",
        "scmle_version",
        "model_params",
        "decision_threshold",
    ):
        assert old not in written


def test_manifest_reads_either_layout():
    info = manifest()
    written = info.to_dict()
    assert provenance.Manifest.read(written) == info
    assert (
        provenance.Manifest.read(provenance.Manifest.KEYS.downgrade(written))
        == info
    )


def test_verification_writes_directory():
    written = verdict.Verification("runs/r1", ()).to_dict()
    assert written["directory"] == "runs/r1"
    assert "run_dir" not in written


def link() -> evidence.Link:
    return evidence.Link(
        "e1",
        "sha256:" + "a" * 64,
        "sha256:" + "b" * 64,
        "sha256:" + "c" * 64,
        NOW,
        "sha256:" + "d" * 64,
    )


def test_link_writes_digests():
    assert link().to_dict() == {
        "event": "e1",
        "digests": {
            "payload": "sha256:" + "a" * 64,
            "prediction": "sha256:" + "b" * 64,
            "bundle": "sha256:" + "c" * 64,
            "graph": "sha256:" + "d" * 64,
        },
        "timestamp": NOW,
    }


def test_link_node_digest_is_anchored_at_version_one():
    node = link().node()
    assert node.digest == LINK_DIGEST
    assert set(node.attributes) == {
        "event_id",
        "input_fingerprint",
        "output_fingerprint",
        "bundle_digest",
        "timestamp",
    }


def test_package_writes_model_digest():
    written = containers.Package("pkg", DIGEST, "sig").to_dict()
    assert written == {
        "path": "pkg",
        "digests": {"model": DIGEST},
        "signature": "sig",
        "schema_version": 1,
    }


@pytest.mark.parametrize(
    "layout",
    [
        {
            "path": "pkg",
            "digests": {"model": DIGEST},
            "signature": "sig",
            "schema_version": 1,
        },
        {
            "path": "pkg",
            "model_sha256": DIGEST,
            "signature": "sig",
            "schema_version": 1,
        },
    ],
)
def test_package_unpacks_either_layout(layout):
    assert reporting.unpack(layout) == containers.Package("pkg", DIGEST, "sig")


def test_retries_writes_single_word_keys():
    written = retry.Retries(4, 0.25, 3.0, 9.0).to_dict()
    assert written == {
        "attempts": 4,
        "delays": {"base": 0.25, "cap": 9.0},
        "factor": 3.0,
    }


def test_retries_reads_either_layout():
    policy = retry.Retries(4, 0.25, 3.0, 9.0)
    assert retry.Retries.read(policy.to_dict()) == policy
    assert (
        retry.Retries.read(
            {
                "max_attempts": 4,
                "base_delay_seconds": 0.25,
                "factor": 3.0,
                "max_delay_seconds": 9.0,
            }
        )
        == policy
    )
