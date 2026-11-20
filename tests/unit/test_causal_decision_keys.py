"""The causal and decision records write single-word keys (stage B, phase 5).

Each write-only record routes `to_dict` through its `KEYS` migration: the
new keys are written, and `KEYS.downgrade` gives back the version-1 layout,
so a version-1 sample still reads. None of these records hashes itself; the
rollout journal hashes each row as stored, so a version-1 row still
verifies.
"""

import dataclasses

import pytest

from scmle.causal.identification import aipw
from scmle.causal.identification import attest
from scmle.causal.identification import battery
from scmle.causal.identification import certified
from scmle.causal.identification import did
from scmle.causal.identification import iv
from scmle.causal.identification import rd
from scmle.causal.identification import select
from scmle.causal.identification import sensitivity
from scmle.causal.identification import synth
from scmle.decision import netbenefit
from scmle.decision import ope
from scmle.decision import rollout
from scmle.decision import threshold
from scmle.decision.constraints import satisfaction


def filled(kind, **given):
    """An instance whose every field holds its own name, unless given."""
    values = {f.name: f.name for f in dataclasses.fields(kind)}
    values.update(given)
    return kind(**values)


# record, the written keys it gains, the version-1 keys it loses
CASES = {
    "aipw": (
        filled(aipw.Estimate),
        {"received", "digests"},
        {"n_input", "nuisance_digest"},
    ),
    "replication": (
        filled(battery.Replication, strata={"a": [1.0, 0.1]}),
        {"sign", "agreeing", "heterogeneity"},
        {"pooled_sign", "same_sign", "p_heterogeneity"},
    ),
    "gate": (
        filled(certified.Gate),
        {"rows"},
        {"min_rows"},
    ),
    "did": (
        filled(did.Estimate),
        {"untreated", "pretrend", "drift", "horizon"},
        {"n_never", "pretrend_p", "max_pre_change", "mean_horizon"},
    ),
    "iv": (
        filled(iv.Estimate),
        {"wald", "stage", "fstat", "ar"},
        {
            "wald_low",
            "wald_high",
            "first_stage",
            "f_stat",
            "ar_kind",
            "ar_low",
            "ar_high",
        },
    ),
    "rd": (
        filled(rd.Estimate),
        {"corrected", "counts", "density"},
        {"bias_corrected", "n_left", "n_right", "density_p"},
    ),
    "sensitivity": (
        filled(sensitivity.Sensitivity),
        {"rvalpha"},
        {"rv_alpha"},
    ),
    "synth": (
        filled(synth.Estimate),
        {"rmspe", "floor", "placebos"},
        {"pre_rmspe", "post_rmspe", "min_p", "placebo_ratios"},
    ),
    "satisfaction": (
        filled(satisfaction.Satisfaction),
        {"gap", "estimate", "upper", "small"},
        {"max_gap", "point_gap", "upper_bound", "small_groups"},
    ),
    "benefit": (
        filled(netbenefit.Benefit),
        {"everyone", "useful"},
        {"treat_all", "useful_range"},
    ),
    "ope": (
        filled(ope.Estimate),
        {"ess", "truncation"},
        {"ess_ratio", "clipped_share"},
    ),
    "rollout": (
        filled(rollout.Decision),
        {"evalues", "lower", "challenger"},
        {"log_e", "lower_bound", "challenger_units"},
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_record_writes_its_new_keys_and_downgrades_to_version_one(name):
    record, gained, lost = CASES[name]
    written = record.to_dict()
    assert gained <= set(written)
    assert not lost & set(written)
    older = record.KEYS.downgrade(written)
    assert lost <= set(older)
    assert record.KEYS.upgrade(older) == written


def test_nested_keys_hold_the_old_values():
    assert CASES["aipw"][0].to_dict()["digests"] == {
        "nuisance": "nuisance_digest"
    }
    found = CASES["iv"][0].to_dict()
    assert found["wald"] == {"low": "wald_low", "high": "wald_high"}
    assert found["ar"] == {
        "kind": "ar_kind",
        "low": "ar_low",
        "high": "ar_high",
    }
    assert CASES["rd"][0].to_dict()["counts"] == {
        "left": "n_left",
        "right": "n_right",
    }
    assert CASES["synth"][0].to_dict()["rmspe"] == {
        "pre": "pre_rmspe",
        "post": "post_rmspe",
    }


def test_iv_writes_infinite_ends_as_text_inside_the_nested_objects():
    record = filled(
        iv.Estimate,
        wald_low=float("-inf"),
        wald_high=1.0,
        ar_low=float("-inf"),
        ar_high=float("inf"),
    )
    found = record.to_dict()
    assert found["wald"] == {"low": "-inf", "high": 1.0}
    assert found["ar"]["low"] == "-inf"
    assert found["ar"]["high"] == "inf"


def test_the_applied_attestations_write_the_evaluation_date_as_evaluated():
    applied = attest.Applied({}, (), {"a": "none"}, "2026-01-02")
    written = applied.to_dict()
    assert written["evaluated"] == "2026-01-02"
    assert "evaluated_on" not in written
    old = applied.KEYS.downgrade(written)
    assert old == {
        "evaluated_on": "2026-01-02",
        "accepted": [],
        "rejected": {"a": "none"},
    }
    assert applied.KEYS.upgrade(old) == written


def test_a_verdict_nests_what_is_missing_and_reads_version_one():
    verdict = select.Verdict("iv", False, ("a",), ("b",), checked=True)
    written = verdict.to_dict()
    assert written["missing"] == {"assumptions": ["a"], "design": ["b"]}
    old = select.Verdict.KEYS.downgrade(written)
    assert old["missing_assumptions"] == ["a"]
    assert old["missing_design"] == ["b"]
    assert "missing" not in old


def test_a_plan_writes_chartered_and_reads_version_one():
    plan = select.Plan("identified", "rd", (), (), ("rd",), True)
    written = plan.to_dict()
    assert written["chartered"] is True
    assert "order_chartered" not in written
    old = select.Plan.KEYS.downgrade(written)
    assert old["order_chartered"] is True


def test_the_thresholding_record_writes_cost_interval_and_stable():
    check = filled(threshold.Calibration)
    record = threshold.Thresholding(
        threshold.Threshold(0.3, "bayes"),
        "attested",
        check,
        None,
        1.5,
        (1.0, 2.0),
        (0.2, 0.4),
    )
    written = record.to_dict()
    assert written["cost"] == 1.5
    assert written["interval"] == [1.0, 2.0]
    assert written["stable"] == [0.2, 0.4]
    old = record.KEYS.downgrade(written)
    assert old["expected_cost"] == 1.5
    assert old["cost_interval"] == [1.0, 2.0]
    assert old["stable_range"] == [0.2, 0.4]


def test_a_journal_holding_a_version_one_decision_row_still_verifies():
    decision = filled(rollout.Decision)
    log = rollout.Journal()
    log.append("decision", decision.KEYS.downgrade(decision.to_dict()))
    log.append("decision", decision.to_dict())
    assert log.verify()
    assert "lower_bound" in log.rows[0]["body"]
    assert "lower" in log.rows[1]["body"]
