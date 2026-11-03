"""Subgroup coverage and the forward-in-time check."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from scmle.contracts import errors
from scmle.data.custody import sealed
from scmle.data.robustness import coverage
from scmle.data.robustness import forward
from scmle.data.robustness import shift

PASS = shift.Status.PASS
FAIL = shift.Status.FAIL


def test_without_a_reference_every_level_is_unverifiable_never_pass():
    sample = pd.Series(["a"] * 30 + ["b"] * 70, name="g")
    rows = coverage.subgroup_coverage(sample, None)
    assert {r.status for r in rows} == {shift.Status.UNVERIFIABLE}
    assert [r.level for r in rows] == ["a", "b"]


def test_the_interval_is_the_exact_one_and_decides_pass_or_fail():
    sample = pd.Series(["a"] * 30 + ["b"] * 70, name="g")
    rows = {
        r.level: r
        for r in coverage.subgroup_coverage(sample, {"a": 0.3, "b": 0.7})
    }
    exact = stats.binomtest(30, 100).proportion_ci(0.95, method="exact")
    assert rows["a"].interval == pytest.approx((exact.low, exact.high))
    assert rows["a"].status is PASS and rows["a"].share == 0.3
    gap = {
        r.level: r
        for r in coverage.subgroup_coverage(sample, {"a": 0.6, "b": 0.4})
    }
    assert gap["a"].status is FAIL and gap["b"].status is FAIL


def test_a_level_missing_from_the_sample_or_the_deployment_is_listed():
    sample = pd.Series(["a"] * 50, name="g")
    rows = {
        r.level: r
        for r in coverage.subgroup_coverage(sample, {"a": 0.5, "z": 0.5})
    }
    assert rows["z"].count == 0 and rows["z"].status is FAIL
    only_sample = {
        r.level: r for r in coverage.subgroup_coverage(sample, {"a": 1.0})
    }
    assert only_sample["a"].status is PASS


def test_missing_values_are_their_own_level_and_input_is_validated():
    sample = pd.Series(["a", None, "a", None], name="g")
    assert {r.level for r in coverage.subgroup_coverage(sample, None)} == {
        "a",
        coverage.MISSING_LEVEL,
    }
    assert coverage.shares(sample)["a"] == 0.5
    with pytest.raises(ValueError, match="empty"):
        coverage.subgroup_coverage(pd.Series([], dtype=object), None)
    with pytest.raises(ValueError, match="alpha"):
        coverage.subgroup_coverage(sample, None, alpha=1.0)
    record = coverage.subgroup_coverage(sample, {"a": 0.5})[0].to_dict()
    assert record["status"] in {"pass", "fail"}


def forward_set(rows=300, beta=0.8, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=rows)
    frame = pd.DataFrame({"f": x, "g": rng.normal(size=rows)})
    target = pd.Series(beta * x + rng.normal(size=rows), name="y")
    vault = sealed.CustodyVault(sealed.OpenLog(), process=False)
    held = vault.seal("final_test", frame, target, tuple(range(rows)))
    return vault, held


def encode(series):
    return series.to_numpy(dtype=float)


def opens(vault):
    return [e for e in vault.log.entries if e["event"] == "open"]


def test_f_is_opened_once_and_logged_under_the_forward_actor():
    vault, held = forward_set()
    result = forward.forward_check(
        held, {"f": 0.5, "g": 0.0}, encode, temporal=True, seed=0, bootstrap=100
    )
    assert result.status == "checked" and result.opened
    assert len(opens(vault)) == 1
    assert opens(vault)[0]["actor"] == forward.ACTOR
    by_name = {f.name: f for f in result.findings}
    assert by_name["f"].label == forward.HOLDS
    assert by_name["g"].label == forward.INCONCLUSIVE
    check = by_name["f"].check
    assert check.set_id.value == "F" and check.tier.value == "unverified"
    with pytest.raises(errors.LeakageError):
        held.open(held.issue_token())
    vault.log.verify()


def test_a_reversed_effect_is_labelled_unstable_over_time():
    held = forward_set(beta=-0.8)[1]
    result = forward.forward_check(
        held, {"f": 0.5}, encode, temporal=True, seed=0, bootstrap=100
    )
    (finding,) = result.findings
    assert finding.label == forward.UNSTABLE
    assert finding.check.decision.value == "warn"


def test_a_negative_effect_on_x_that_holds_forward_holds():
    held = forward_set(beta=-0.8)[1]
    (finding,) = forward.forward_check(
        held, {"f": -0.5}, encode, temporal=True, seed=0, bootstrap=100
    ).findings
    assert finding.label == forward.HOLDS


def test_non_temporal_data_a_missing_set_and_no_findings_skip_without_opening():
    vault, held = forward_set()
    for kwargs, text in (
        ({"temporal": False}, "not temporal"),
        ({"temporal": True, "source": None}, "no forward set"),
    ):
        source = kwargs.pop("source", held)
        result = forward.forward_check(
            source, {"f": 0.5}, encode, seed=0, **kwargs
        )
        assert result.status == "skipped" and not result.opened
        assert text in result.reason
    empty = forward.forward_check(held, {}, encode, temporal=True, seed=0)
    assert empty.status == "skipped" and "not opened" in empty.reason
    assert opens(vault) == []


def test_a_finding_whose_column_is_absent_is_inconclusive():
    held = forward_set()[1]
    result = forward.forward_check(
        held, {"gone": 0.5}, encode, temporal=True, seed=0, bootstrap=20
    )
    assert result.findings[0].label == forward.INCONCLUSIVE
    assert result.findings[0].check is None
    assert result.to_dict()["findings"][0]["effect_f"] is None


def test_labels_follow_the_interval_and_the_sign_on_x():
    assert forward.label_for(0.3, 0.1, 0.4) == forward.HOLDS
    assert forward.label_for(0.3, -0.4, -0.1) == forward.UNSTABLE
    assert forward.label_for(-0.3, -0.4, -0.1) == forward.HOLDS
    assert forward.label_for(-0.3, 0.1, 0.4) == forward.UNSTABLE
    assert forward.label_for(0.3, -0.1, 0.4) == forward.INCONCLUSIVE
