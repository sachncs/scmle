"""Frozen reference outputs, their integrity and what they check."""

import json
import math

import pytest
from scipy import stats

from benchmarks.qualification import make_oracles
from benchmarks.qualification import oracles
from scmle.contracts import qualification
from scmle.data.detectors import certificates
from scmle.data.ledger import budget

REL = 1e-8


def test_every_declared_oracle_has_a_verified_file():
    assert set(oracles.NAMES) == set(make_oracles.FAMILIES)
    for name in oracles.NAMES:
        document = oracles.load_oracle(name)
        assert document.name == name
        assert document.cases
        assert document.tool["scipy"]
        assert document.generator_sha256 == oracles.generator_sha256()


def test_a_tampered_fixture_is_rejected(tmp_path):
    source = oracles.ORACLE_DIR / "wilson_interval.json"
    record = json.loads(source.read_text(encoding="utf-8"))
    record["cases"][0]["expected"]["low"] += 1e-6
    (tmp_path / "wilson_interval.json").write_text(json.dumps(record))
    with pytest.raises(oracles.OracleError, match="hash"):
        oracles.load_oracle("wilson_interval", tmp_path)


def test_a_fixture_without_a_hash_or_with_unknown_fields_is_rejected(tmp_path):
    source = oracles.ORACLE_DIR / "wilson_interval.json"
    record = json.loads(source.read_text(encoding="utf-8"))
    without = {k: v for k, v in record.items() if k != "content_sha256"}
    (tmp_path / "wilson_interval.json").write_text(json.dumps(without))
    with pytest.raises(oracles.OracleError, match="content_sha256"):
        oracles.load_oracle("wilson_interval", tmp_path)
    (tmp_path / "wilson_interval.json").write_text(
        json.dumps(record | {"extra": 1})
    )
    with pytest.raises(oracles.OracleError, match="unexpected"):
        oracles.load_oracle("wilson_interval", tmp_path)
    with pytest.raises(oracles.OracleError, match="no such oracle file"):
        oracles.load_oracle("kruskal", tmp_path)
    with pytest.raises(oracles.OracleError, match="not a known"):
        oracles.load_oracle("../etc", tmp_path)


def test_the_frozen_outputs_match_a_recomputation_from_the_stored_inputs():
    """Guards against drift from the installed tool or an edited generator.

    Only the expected values are recomputed. The inputs stay as stored, so a
    change in a random-number stream cannot move a reference.
    """
    for name in oracles.NAMES:
        for case in oracles.load_oracle(name).cases:
            fresh = make_oracles.FAMILIES[name].compute(case["inputs"])
            oracles.assert_agrees(
                case["expected"],
                fresh,
                label=f"{name}/{case['id']}",
                rel_tol=REL,
                abs_tol=1e-15,
            )


def test_agreement_helper_accepts_close_values_and_names_the_worst_offender():
    oracles.assert_agrees(1.0 + 1e-10, 1.0, rel_tol=REL)
    oracles.assert_agrees({"a": [1.0, 2.0]}, {"a": [1.0, 2.0 + 1e-12]})
    oracles.assert_agrees(float("nan"), float("nan"))
    oracles.assert_agrees(float("inf"), float("inf"))
    oracles.assert_agrees(0.0, 0.0)
    oracles.assert_agrees(1e-12, 0.0, abs_tol=1e-9)
    with pytest.raises(AssertionError, match=r"\[1\]"):
        oracles.assert_agrees([1.0, 2.0, 3.0], [1.0, 2.1, 3.0], label="x")
    with pytest.raises(AssertionError, match="shape"):
        oracles.assert_agrees([1.0], [1.0, 2.0])
    with pytest.raises(AssertionError, match="keys"):
        oracles.assert_agrees({"a": 1.0}, {"b": 1.0})
    with pytest.raises(AssertionError):
        oracles.assert_agrees(1e-12, 0.0)
    with pytest.raises(AssertionError):
        oracles.assert_agrees(float("nan"), 1.0)
    with pytest.raises(AssertionError):
        oracles.assert_agrees(float("inf"), float("-inf"))


def test_the_wilson_interval_agrees_with_scipy_on_every_frozen_case():
    for case in oracles.load_oracle("wilson_interval").cases:
        given = case["inputs"]
        low, high = certificates.wilson_interval(
            given["successes"], given["n"], given["confidence"]
        )
        oracles.assert_agrees(
            {"low": low, "high": high},
            case["expected"],
            label=case["id"],
            rel_tol=REL,
            abs_tol=1e-15,
        )
        record = qualification.Certificate(
            detector_id="d",
            defect_type="t",
            planted=given["n"],
            detected=given["successes"],
            clean_rows=1,
            false_alarms=0,
            seed_path=(0, "x", 0),
        )
        oracles.assert_agrees(
            record.recall_lower_bound(given["confidence"]),
            case["expected"]["low"],
            label=case["id"],
            rel_tol=REL,
            abs_tol=1e-15,
        )


def test_the_harmonic_number_agrees_with_the_digamma_identity():
    for case in oracles.load_oracle("harmonic_number").cases:
        n = case["inputs"]["n"]
        oracles.assert_agrees(
            budget.harmonic(n),
            case["expected"]["value"],
            label=case["id"],
            rel_tol=REL,
        )


@pytest.mark.parametrize("n", [1, 2, 5, 20, 100, 1000])
def test_the_rank_one_thresholds_reproduce_alpha_under_scipys_adjustment(n):
    plan = budget.ErrorBudget.create(0.05, 0.1)
    for rule, method in (
        (budget.Correction.BY, "by"),
        (budget.Correction.BH, "bh"),
    ):
        level = budget.confirmation_alpha(plan, n, rule)
        others = [1.0] * (n - 1)
        adjusted = stats.false_discovery_control(
            [level, *others], method=method
        )
        oracles.assert_agrees(
            float(adjusted[0]), 0.05, label=f"{method} n={n}", rel_tol=1e-9
        )


def test_the_frozen_p_value_oracles_cover_the_planned_tests():
    expected = {
        "chi2_contingency",
        "fisher_exact",
        "ks_2samp",
        "kruskal",
        "mannwhitneyu",
        "pearsonr",
        "spearmanr",
        "kendalltau",
        "fdr_control",
        "clopper_pearson",
    }
    assert expected <= set(oracles.NAMES)
    for name in expected:
        for case in oracles.load_oracle(name).cases:
            for value in case["expected"].values():
                flat = value if isinstance(value, list) else [value]
                assert all(math.isfinite(v) for v in flat)


def test_the_unavailable_r_oracles_are_declared_not_faked():
    assert set(oracles.PENDING_R) == {"onlineFDR", "knockoff"}
    for name in oracles.PENDING_R:
        assert name not in oracles.NAMES
        assert oracles.PENDING_R[name]
