"""Replication across strata and the Holm family of subgroups."""

import pandas
import pytest

from scmle.assurance.charter import causal
from scmle.causal.identification import battery
from scmle.causal.identification import certified
from scmle.causal.identification.run import estimator
from scmle.causal.identification.run import strata


def block(**changes):
    values = {
        "treatment": "t",
        "outcome": "y",
        "estimand": "ATE",
        "assumptions": ("unconfoundedness_given:a", "positivity"),
        "strategies": ("backdoor_aipw",),
        "strata": ("region",),
    }
    values.update(changes)
    return causal.Question(**values)


class Fixed(estimator.Estimator):
    """Estimates read from the rows: column `value` and `se`."""

    name = "fixed"

    def __init__(self, *args):
        super().__init__(*args)
        self.calls = []

    def estimate(self, c, level):
        self.calls.append((len(c), level))
        self.design["touched"] = True
        self.weak = True
        if "boom" in c.columns and c["boom"].iloc[0]:
            raise ValueError("cannot here")
        value = float(c["value"].iloc[0]) if "value" in c.columns else 1.0
        se = float(c["se"].iloc[0]) if "se" in c.columns else 0.1
        return estimator.Est(value, se, value - 2 * se, value + 2 * se, len(c))


class Student(estimator.Estimator):
    """An estimate whose interval uses a t reference on 6 degrees of freedom."""

    name = "student"

    def estimate(self, c, level):
        return estimator.Est(1.0, 0.4, 0.0, 2.0, len(c), df=6.0)


def adapter(spec=None):
    return Fixed(spec or block(), pandas.DataFrame(), 0, "linear", 0.05)


def frame(**sizes):
    parts = [
        pandas.DataFrame({"region": name}, index=range(n))
        for name, n in sizes.items()
    ]
    return pandas.concat(parts, ignore_index=True)


def test_the_minimum_stratum_size_is_fifty():
    assert strata.MIN_STRATUM_ROWS == 50


def test_a_replication_tests_every_usable_stratum():
    found = adapter()
    out = strata.replication(found, frame(a=50, b=70), 0.9, 1.0, 0.05)
    expected = battery.replicate(
        {"region=a": (1.0, 0.1), "region=b": (1.0, 0.1)}, 1.0, 0.05
    ).to_dict()
    assert out == {"by_column": {"region": expected}, "passed": True}
    assert found.calls == [(50, 0.9), (70, 0.9)]


def test_a_replication_restores_the_design_and_weak_flag():
    found = adapter()
    found.design = {"keep": 1}
    strata.replication(found, frame(a=50, b=70), 0.9, 1.0, 0.05)
    assert found.design == {"keep": 1}
    assert found.weak is False


def test_a_replication_skips_thin_absent_and_failing_strata():
    spec = block(strata=("gone", "region"))
    found = adapter(spec)
    rows = frame(a=60, b=49, c=60)
    rows["boom"] = rows["region"] == "c"
    out = strata.replication(found, rows, 0.9, 1.0, 0.05)
    assert out == {
        "by_column": {
            "gone": {"skipped": "column not in C"},
            "region": {"skipped": "fewer than two usable strata"},
        },
        "passed": None,
    }


def test_a_replication_drops_strata_without_a_positive_error():
    rows = frame(a=60, b=60)
    for se in (0.0, float("nan"), float("inf")):
        out = strata.replication(adapter(), rows.assign(se=se), 0.9, 1.0, 0.05)
        assert out["passed"] is None


def test_a_replication_fails_when_a_stratum_disagrees_in_sign():
    rows = frame(a=60, b=60)
    rows["value"] = [1.0] * 60 + [-1.0] * 60
    out = strata.replication(adapter(), rows, 0.9, 1.0, 0.05)
    assert out["passed"] is False
    both = strata.replication(
        adapter(block(strata=("region", "other"))),
        rows.assign(other=["p", "q"] * 60),
        0.9,
        1.0,
        0.05,
    )
    assert both["passed"] is False


def test_an_examination_of_an_absent_column_is_skipped_with_p_one():
    row = strata.examine(adapter(), frame(a=60), 0.9, "gone", "x")
    assert row == {
        "column": "gone",
        "value": "x",
        "skipped": "column not in C",
        "p_value": 1.0,
    }


def test_an_examination_of_a_thin_subgroup_is_skipped():
    row = strata.examine(adapter(), frame(a=49), 0.9, "region", "a")
    assert row["skipped"] == "fewer than 50 rows"
    assert row["p_value"] == 1.0
    assert "estimate" not in row


def test_an_examination_records_the_wald_test_of_the_subgroup():
    found = adapter()
    row = strata.examine(found, frame(a=50, b=60), 0.9, "region", "a")
    test = battery.wald("region=a", "subgroup", 1.0, 0.1)
    assert row == {
        "column": "region",
        "value": "a",
        "skipped": None,
        "estimate": 1.0,
        "se": 0.1,
        "low": pytest.approx(0.8),
        "high": pytest.approx(1.2),
        "n": 50,
        "p_value": test.p_value,
    }
    assert found.calls == [(50, 0.9)]


def test_an_examination_refers_the_wald_test_to_the_estimates_t():
    found = Student(block(), pandas.DataFrame(), 0, "linear", 0.05)
    row = strata.examine(found, frame(a=50, b=60), 0.9, "region", "a")
    assert (
        row["p_value"] == battery.wald("x", "subgroup", 1.0, 0.4, 6.0).p_value
    )
    assert row["p_value"] > battery.wald("x", "subgroup", 1.0, 0.4).p_value


def test_an_examination_matches_values_as_text():
    rows = pandas.DataFrame({"code": [1] * 60 + [2] * 60})
    row = strata.examine(adapter(), rows, 0.9, "code", "2")
    assert row["n"] == 60


def test_an_examination_records_an_estimation_error_or_unusable_error():
    rows = frame(a=60).assign(boom=True)
    row = strata.examine(adapter(), rows, 0.9, "region", "a")
    assert row["skipped"] == "cannot here"
    assert row["p_value"] == 1.0
    flat = strata.examine(
        adapter(), frame(a=60).assign(se=0.0), 0.9, "region", "a"
    )
    assert flat["skipped"] == "no usable standard error"
    assert flat["p_value"] == 1.0


def test_a_stratification_adjusts_the_family_by_holm():
    rows = frame(a=60, b=60)
    rows["value"] = [1.0] * 60 + [0.01] * 60
    found = adapter()
    found.design = {"keep": 2}
    groups = (causal.Subgroup("region", ("a", "b", "c")),)
    out = strata.stratify(found, rows, 0.95, groups)
    assert out["rule"] == "holm"
    assert out["alpha"] == pytest.approx(0.05)
    assert out["family_size"] == 3
    assert out["statement"] == "FWER <= alpha under arbitrary dependence"
    assert out["rejected"] == ["region=a"]
    names = [(h["value"], h["rejected"]) for h in out["hypotheses"]]
    assert names == [("a", True), ("b", False), ("c", False)]
    p_a = out["hypotheses"][0]["p_value"]
    assert out["hypotheses"][0]["adjusted_p"] == pytest.approx(
        min(1.0, 3 * p_a)
    )
    assert out["hypotheses"][2]["adjusted_p"] == 1.0
    assert found.design == {"keep": 2}


def stratified():
    rows = frame(a=60, b=500)
    groups = (causal.Subgroup("region", ("a", "b")),)
    return strata.stratify(adapter(), rows, 0.95, groups)


def test_a_certification_by_name_marks_the_family_gating():
    found = stratified()
    strata.certify(found, ("subgroup_holm",))
    rows = certified.gate(strata.stratify).rows
    assert found["gate"] == {
        "status": "gating",
        "reason": "named among the certified gates",
        "rows": rows,
    }
    assert [h["certified_size"] for h in found["hypotheses"]] == [
        rows is None or size >= rows for size in (60, 500)
    ]
    assert found["statement_gate"].startswith("gating: the family-wise error")


def test_a_certification_without_the_name_is_descriptive():
    found = stratified()
    strata.certify(found, ())
    assert found["gate"]["status"] == "descriptive"
    assert found["gate"]["reason"] == (
        "the caller named the certified gates and this is not one"
    )
    assert [h["certified_size"] for h in found["hypotheses"]] == [False, False]
    assert found["statement_gate"].endswith(
        "read a rejected level as exploratory; the caller named the "
        "certified gates and this is not one"
    )


def test_a_certification_from_the_shipped_store_follows_that_store():
    found = stratified()
    strata.certify(found, None)
    gate = certified.gate(strata.stratify)
    assert found["gate"] == gate.to_dict()
    gating = gate.status == "gating"
    assert found["statement_gate"].startswith(
        "gating:" if gating else "descriptive:"
    )
    assert found["hypotheses"][0]["certified_size"] is (
        gating and (gate.rows is None or gate.rows <= 60)
    )


def test_subgroup_results_are_none_without_subgroups_or_with_one_unit():
    found = adapter()
    assert strata.estimate(found, block(), frame(a=60), 0.95, ()) is None
    spec = block(subgroups=(causal.Subgroup("region", ("a",)),))
    single = adapter(spec)
    single.unit = True
    assert strata.estimate(single, spec, frame(a=60), 0.95, ()) is None


def test_subgroup_results_restore_weakness_and_are_labelled():
    spec = block(subgroups=(causal.Subgroup("region", ("a",)),))
    found = adapter(spec)
    out = strata.estimate(found, spec, frame(a=60), 0.95, ())
    assert found.weak is False
    assert out["gate"]["status"] == "descriptive"
    assert out["family_size"] == 1
