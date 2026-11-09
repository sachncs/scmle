"""Wiring of the robustness checks onto the exploration set, and their reports."""

import copy
import json

import numpy
import pandas
import pytest

from scmle.data.charter import contracts
from scmle.data.detectors import certificates
from scmle.data.robustness import qualify
from scmle.data.robustness import report
from scmle.data.robustness import stability

RICH = (
    contracts.Choice("imputation", ("median", "complete_case"), "median"),
    contracts.Choice("outlier_rule", ("none", "winsorize", "iqr_drop"), "none"),
    contracts.Choice("inclusion", ("all", "first_half", "second_half"), "all"),
)


def world(rows=300, seed=0):
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame(
        {
            "a": rng.normal(size=rows),
            "b": rng.normal(size=rows),
            "g": rng.choice(["u", "v"], rows),
        }
    )
    codes = pandas.Series(frame["a"] * 0.8 + rng.normal(size=rows), name="y")
    return frame, codes


def run(frame, codes, tmp_path, **options):
    base = {
        "survivors": ["a"],
        "multiclass": False,
        "choices": RICH,
        "seed": 1,
        "permutations": 20,
        "candidates": ["a", "b", "g"],
        "directory": tmp_path,
    }
    base.update(options)
    return qualify.run(frame, codes, **base)


def test_a_missing_store_is_descriptive_and_the_survivor_is_assessed(tmp_path):
    frame, codes = world()
    found = run(frame, codes, tmp_path)

    assert found.store_note and "descriptive" in found.store_note
    assert found.stability_gate is certificates.GateStatus.DESCRIPTIVE
    assert found.c2st_gate is certificates.GateStatus.DESCRIPTIVE
    (finding,) = found.findings
    assert finding.name == "a" and finding.status == stability.STABLE
    assert len(finding.specs) == 12 and found.grid.total == 18
    assert found.fragile == ()


def test_the_shipped_certificates_are_found_from_a_checkout():
    frame, codes = world()
    found = qualify.run(
        frame,
        codes,
        survivors=[],
        multiclass=False,
        choices=RICH,
        seed=1,
        permutations=20,
    )
    assert found.store_note == ""
    assert found.c2st_gate is certificates.GateStatus.GATING
    assert found.stability_gate is certificates.GateStatus.GATING


def test_nothing_to_vary_or_nothing_found_gives_a_reason_not_a_verdict(
    tmp_path,
):
    frame, codes = world()
    default = (contracts.Choice("imputation", ("median", "mice"), "median"),)
    few = run(frame, codes, tmp_path, choices=default)
    assert few.findings == () and "specification" in few.stability_note
    assert any("mice" in note for note in few.notes)
    empty = run(frame, codes, tmp_path, survivors=[])
    assert empty.findings == () and "survived" in empty.stability_note
    many = run(frame, codes, tmp_path, multiclass=True)
    assert many.findings == () and "classes" in many.stability_note


def test_only_the_first_findings_are_assessed_and_the_cap_is_said(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(qualify, "MAX_FINDINGS", 3)
    frame, codes = world(rows=120)
    names = [f"n{k}" for k in range(qualify.MAX_FINDINGS + 2)]
    rng = numpy.random.default_rng(3)
    wide = pandas.concat(
        [
            pandas.DataFrame(rng.normal(size=(120, len(names))), columns=names),
            frame,
        ],
        axis=1,
    )
    found = run(wide, codes, tmp_path, survivors=names, candidates=["a"])
    assert len(found.findings) == qualify.MAX_FINDINGS
    assert any("first 3 of 5" in note for note in found.notes)


def test_no_reference_is_unverifiable_and_a_reference_is_tested(tmp_path):
    frame, codes = world()
    none = run(frame, codes, tmp_path)
    assert none.shift.status.value == "unverifiable"
    reference = frame.copy()
    reference["a"] += 2.0
    shifted = run(frame, codes, tmp_path, reference=reference)
    assert shifted.shift.status.value == "fail"
    same = run(frame, codes, tmp_path, reference=world(seed=5)[0])
    assert same.shift.status.value == "pass"


def test_subgroup_coverage_uses_the_reference_shares_when_there_is_one(
    tmp_path,
):
    frame, codes = world()
    without = run(frame, codes, tmp_path, sensitive=["g", "missing"])
    assert {r.status.value for r in without.coverage} == {"unverifiable"}
    reference = frame.copy()
    reference["g"] = "u"
    withref = run(frame, codes, tmp_path, sensitive=["g"], reference=reference)
    assert {r.level: r.status.value for r in withref.coverage}["v"] == "fail"


def test_the_forward_check_is_not_run_in_a_run_and_says_why(tmp_path):
    frame, codes = world()
    plain = run(frame, codes, tmp_path)
    assert (
        plain.forward.status == "skipped" and "temporal" in plain.forward.reason
    )
    timed = run(frame, codes, tmp_path, temporal=True)
    assert timed.forward.status == "skipped"
    assert "final test set" in timed.forward.reason and not timed.forward.opened


def test_the_reports_validate_and_carry_the_curve_and_attribution(tmp_path):
    frame, codes = world()
    found = run(frame, codes, tmp_path)
    stability_document = json.loads(
        json.dumps(report.stability_report(found), allow_nan=False)
    )
    shift_document = json.loads(
        json.dumps(report.shift_report(found), allow_nan=False)
    )
    report.validate_stability(stability_document)
    report.validate_shift(shift_document)
    assert stability_document["status"] == "evaluated"
    assert stability_document["counts"]["stable"] == 1
    finding = stability_document["findings"][0]
    assert finding["specs"] and finding["attribution"] and finding["curve"]
    assert shift_document["status"] == "unverifiable"
    assert shift_document["periods"]["status"] == "not_run"


def test_validators_reject_missing_keys_wrong_versions_and_unearned_passes(
    tmp_path,
):
    frame, codes = world()
    found = run(frame, codes, tmp_path)
    good = report.stability_report(found)
    for key in ("findings", "rule", "guarantee"):
        broken = {k: v for k, v in good.items() if k != key}
        with pytest.raises(ValueError, match="missing"):
            report.validate_stability(broken)
    with pytest.raises(ValueError, match="schema_version"):
        report.validate_stability({**good, "schema_version": 9})
    with pytest.raises(ValueError, match="status"):
        report.validate_stability({**good, "status": "great"})
    odd = copy.deepcopy(good)
    odd["findings"][0]["status"] = "robust"
    with pytest.raises(ValueError, match="finding"):
        report.validate_stability(odd)
    shifted = report.shift_report(found)
    with pytest.raises(ValueError, match="pass needs"):
        report.validate_shift({**shifted, "status": "pass"})
    with pytest.raises(ValueError, match="status"):
        report.validate_shift({**shifted, "status": "ok"})
    with pytest.raises(ValueError, match="missing"):
        report.validate_shift({"schema_version": 1})


def test_a_stability_report_without_findings_says_why(tmp_path):
    frame, codes = world()
    found = run(frame, codes, tmp_path, survivors=[])
    document = report.stability_report(found)
    assert document["status"] == "not_evaluated"
    assert document["reason"] == found.stability_note
    report.validate_stability(document)
