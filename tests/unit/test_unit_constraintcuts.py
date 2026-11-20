"""Group thresholds that meet a fairness constraint on development data."""

import numpy
import pytest

from scmle.assurance.charter import decision
from scmle.assurance.framing import costs
from scmle.contracts import errors
from scmle.decision.constraints import cuts
from scmle.decision.constraints import rates


def constraint(criterion="demographic_parity", gap=0.1):
    return decision.Constraint(
        "g", criterion, gap, "group_thresholds", "counsel"
    )


def shifted(n=2000, seed=0, shift=0.3):
    rng = numpy.random.default_rng(seed)
    groups = numpy.repeat(["a", "b"], n // 2)
    y = (rng.random(n) < 0.4).astype(float)
    score = numpy.clip(
        0.3 + 0.3 * y + rng.normal(0, 0.15, n) - shift * (groups == "b"), 0, 1
    )
    return score, y, groups


def test_the_grid_has_ninety_nine_points():
    assert cuts.GRID == 99


def test_thresholds_act_on_each_groups_cut():
    found = cuts.Thresholds({"a": 0.5, "1": 0.2}, None, 0.0, 0.0, False)
    acts = found.act([0.5, 0.49, 0.2, 0.19], ["a", "a", 1, 1])
    assert list(acts) == [1.0, 0.0, 1.0, 0.0]
    assert acts.dtype == float


def test_a_development_gap_is_the_largest_rate_difference():
    y = numpy.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=float)
    a = numpy.array([1, 1, 1, 0, 1, 0, 0, 0], dtype=float)
    g = numpy.array(list("aaaabbbb"))
    assert cuts.contrast(y, a, g, "demographic_parity") == 0.5
    assert cuts.contrast(y, a, g, "equal_opportunity") == 0.5
    assert cuts.contrast(y, a, g, "equalized_odds") == 0.5


def test_an_empty_denominator_counts_as_rate_zero():
    y = numpy.array([1, 1, 0, 0], dtype=float)
    a = numpy.array([1, 1, 0, 0], dtype=float)
    g = numpy.array(["a", "a", "b", "b"])
    assert cuts.contrast(y, a, g, "equal_opportunity") == 1.0


def test_an_alignment_cuts_each_group_at_its_quantile():
    s = numpy.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    y = numpy.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    g = numpy.array(list("aaaabbbb"))
    found = cuts.align(s, y, g, ["a", "b"], 0.25, positive=False)
    assert found == {
        "a": pytest.approx(numpy.quantile([0.1, 0.2, 0.3, 0.4], 0.75)),
        "b": pytest.approx(numpy.quantile([0.5, 0.6, 0.7, 0.8], 0.75)),
    }
    positives = cuts.align(s, y, g, ["a", "b"], 0.5, positive=True)
    assert positives == {"a": pytest.approx(0.3), "b": pytest.approx(0.7)}


def test_an_alignment_with_an_empty_group_is_none():
    s = numpy.array([0.1, 0.2])
    y = numpy.array([0.0, 0.0])
    g = numpy.array(["a", "b"])
    assert cuts.align(s, y, g, ["a", "b"], 0.5, positive=True) is None
    assert cuts.align(s, y, g, ["a", "c"], 0.5, positive=False) is None


def test_an_equalization_keeps_the_common_cut_when_it_already_meets_the_bound():
    s, y, g = shifted(shift=0.0)
    found = cuts.equalize(s, y, g, constraint(gap=0.2), 0.5)
    common = (s >= 0.5).astype(float)
    assert found.cuts == {"a": 0.5, "b": 0.5}
    assert found.target is None
    assert found.constrained is False
    assert found.benefit == costs.net_benefit(y, common, 0.5)
    assert found.gap == cuts.contrast(y, common, g, "demographic_parity")


def test_an_equalization_sweeps_when_the_common_cut_is_unfair():
    s, y, g = shifted()
    found = cuts.equalize(s, y, g, constraint(gap=0.05), 0.4)
    assert found.constrained is True
    assert found.gap <= 0.025
    assert found.cuts["b"] < found.cuts["a"]
    acts = found.act(s, g)
    assert found.benefit == pytest.approx(costs.net_benefit(y, acts, 0.4))
    assert 0.01 <= found.target <= 0.99
    best = cuts.sweep((s, y, g), ["a", "b"], "demographic_parity", 0.4, 0.025)
    assert best == found


def test_the_margin_scales_the_bound():
    s, y, g = shifted()
    loose = cuts.equalize(s, y, g, constraint(gap=0.05), 0.4, margin=1.0)
    assert loose.gap <= 0.05
    assert (
        loose.benefit
        >= cuts.equalize(s, y, g, constraint(gap=0.05), 0.4).benefit
    )


def test_an_equalization_refuses_predictive_parity():
    s, y, g = shifted()
    with pytest.raises(
        errors.InputError, match="cannot target predictive parity"
    ):
        cuts.equalize(s, y, g, constraint("predictive_parity"), 0.4)


def test_an_unreachable_bound_is_infeasible():
    s = numpy.array([0.1, 0.9, 0.2, 0.8])
    y = numpy.array([0.0, 1.0, 0.0, 1.0])
    g = numpy.array(["a", "a", "b", "c"])
    with pytest.raises(rates.InfeasibleError) as caught:
        cuts.equalize(s, y, g, constraint("equal_opportunity", 0.01), 0.5)
    assert str(caught.value) == (
        "no group thresholds meet the development bound 0.0050 for "
        "equal_opportunity"
    )


def test_a_frontier_reports_each_bound_in_order():
    s, y, g = shifted()
    rows = cuts.frontier(s, y, g, constraint(), 0.4, [0.5, 0.05])
    assert [r["max_gap"] for r in rows] == [0.5, 0.05]
    assert all(r["feasible"] for r in rows)
    fitted = cuts.equalize(s, y, g, constraint(gap=0.05), 0.4)
    assert rows[1] == {
        "max_gap": 0.05,
        "feasible": True,
        "net_benefit": fitted.benefit,
        "development_gap": fitted.gap,
    }


def test_a_frontier_records_infeasible_bounds():
    s = numpy.array([0.1, 0.9, 0.2, 0.8])
    y = numpy.array([0.0, 1.0, 0.0, 1.0])
    g = numpy.array(["a", "a", "b", "c"])
    rows = cuts.frontier(s, y, g, constraint("equal_opportunity"), 0.5, [0.01])
    assert rows == [{"max_gap": 0.01, "feasible": False}]


def test_a_toll_prices_the_constraint_in_net_benefit():
    y = numpy.array([1, 0, 1, 0, 1, 0] * 20, dtype=float)
    free = numpy.array([1, 0, 1, 1, 1, 0] * 20, dtype=float)
    bound = numpy.array([1, 0, 0, 0, 1, 0] * 20, dtype=float)
    out = cuts.toll(y, free, bound, 0.5, replicates=200, seed=1)
    expected = costs.net_benefit(y, free, 0.5) - costs.net_benefit(
        y, bound, 0.5
    )
    assert out["price"] == pytest.approx(expected)
    assert out["lower"] <= out["price"] <= out["upper"]
    assert set(out) == {"price", "lower", "upper"}
    again = cuts.toll(y, free, bound, 0.5, replicates=200, seed=1)
    assert again == out


def test_a_toll_of_identical_rules_is_zero():
    y = numpy.array([1, 0] * 30, dtype=float)
    acts = numpy.array([1, 1, 0] * 20, dtype=float)
    out = cuts.toll(
        y, acts, acts, 0.3, clusters=numpy.arange(60) // 3, replicates=50
    )
    assert out == {"price": 0.0, "lower": 0.0, "upper": 0.0}
