"""The coverage claim is a hashed document fixed before the rows are read."""

import json

import numpy as np
import pytest

from scmle.artifacts import claims_audit
from scmle.data.custody import sealed
from scmle.data.ledger import ledger as ledger_module
from scmle.evaluation import coverage_run


def make_claim(**overrides):
    values = {
        "task": "regression",
        "levels": (0.5, 0.9),
        "tolerance": 0.02,
        "split_fingerprint": "abc",
        "charter_hash": None,
    }
    values.update(overrides)
    return coverage_run.claim_document(**values)


def test_the_claim_names_levels_methods_and_the_rule_and_hashes_stably():
    claim = make_claim()

    assert claim["levels"] == [0.5, 0.9]
    assert claim["methods"] == ["residual", "cv_plus", "quantile"]
    assert claim["tolerance"] == 0.02
    assert claim["rows_read"] == "once, by the final evaluation"
    assert coverage_run.claim_hash(claim) == coverage_run.claim_hash(
        make_claim()
    )
    assert coverage_run.claim_hash(claim) == ledger_module.digest(claim)
    assert coverage_run.claim_hash(claim) != coverage_run.claim_hash(
        make_claim(levels=(0.5, 0.95))
    )


def test_classification_claims_name_set_modes():
    claim = make_claim(task="binary_classification")
    assert claim["methods"] == ["pooled", "class_conditional"]


def test_a_task_without_intervals_has_no_claim():
    assert make_claim(task="ranking") is None


class FakeBundle:
    conformal = None
    quantile = object()

    def predict_interval(self, features, coverage, method):
        from scmle.contracts import errors

        if method != "quantile":
            raise errors.ArtifactError("none")
        import pandas as pd

        return pd.DataFrame(
            {"lower": np.full(len(features), -1e9), "upper": 1e9}
        )


def test_the_report_carries_the_hash_of_the_claim():
    import pandas as pd

    features = pd.DataFrame({"a": range(40)})
    report = coverage_run.run_report(
        FakeBundle(), features, np.zeros(40), (0.5,), claim_sha256="f" * 64
    )
    assert report["claim_sha256"] == "f" * 64


KEY = "k"


def run_folder(tmp_path, order=("coverage_claim_registered",)):
    claim = make_claim()
    (tmp_path / "coverage_claim.json").write_text(json.dumps(claim))
    report = {"claim_sha256": coverage_run.claim_hash(claim), "methods": {}}
    (tmp_path / "coverage_report.json").write_text(json.dumps(report))
    log = sealed.OpenLog(tmp_path / claims_audit.CUSTODY_FILE, KEY)
    for event in order:
        log.append(
            "final_test",
            event,
            "scmle.coverage",
            {"sha256": coverage_run.claim_hash(claim)},
        )
    return claim


STEP = "coverage claim: registered before the rows were read"


def test_a_consistent_claim_verifies(tmp_path):
    run_folder(tmp_path)
    found = claims_audit.check_coverage_claim(tmp_path, KEY)
    assert found.status == "pass"


def test_an_edited_claim_fails(tmp_path):
    run_folder(tmp_path)
    claim = make_claim(levels=(0.5, 0.6))
    (tmp_path / "coverage_claim.json").write_text(json.dumps(claim))
    found = claims_audit.check_coverage_claim(tmp_path, KEY)
    assert found.status == "fail"
    assert (
        "coverage_claim.json: differs from the registered hash" in found.detail
    )


def test_a_report_for_another_claim_fails(tmp_path):
    run_folder(tmp_path)
    (tmp_path / "coverage_report.json").write_text(
        json.dumps({"claim_sha256": "0" * 64})
    )
    found = claims_audit.check_coverage_claim(tmp_path, KEY)
    assert found.status == "fail"
    assert "coverage_report.json: names another claim" in found.detail


def test_a_report_without_a_registration_fails(tmp_path):
    run_folder(tmp_path, order=("other",))
    found = claims_audit.check_coverage_claim(tmp_path, KEY)
    assert found.status == "fail"
    assert "no registration" in found.detail


def test_no_coverage_files_means_no_step(tmp_path):
    assert claims_audit.check_coverage_claim(tmp_path, KEY) is None


@pytest.mark.parametrize("extra", [1])
def test_two_registrations_fail(tmp_path, extra):
    run_folder(tmp_path, order=("coverage_claim_registered",) * 2)
    assert claims_audit.check_coverage_claim(tmp_path, KEY).status == "fail"
