"""Thresholding with a calibration gate, and net benefit with a band."""

import numpy as np
import pytest

from scmle.data.framing import costs
from scmle.decision import netbenefit
from scmle.decision import threshold


def model(fp=5.0, fn=100.0, lo=0.02, hi=0.15):
    return costs.DecisionCostModel(
        0.0,
        fn,
        fp,
        0.0,
        lo,
        hi,
        (costs.CostSource.HUMAN,) * 4,
        costs.CostSource.HUMAN,
    )


def world(n, seed, skew=1.0):
    rng = np.random.default_rng(seed)
    p = rng.beta(1, 6, n)
    y = (rng.random(n) < p).astype(float)
    return np.clip(p**skew, 0, 1), y


def test_the_bayes_threshold_is_the_closed_form():
    assert threshold.bayes_threshold(5, 100) == pytest.approx(5 / 105)
    with pytest.raises(ValueError):
        threshold.bayes_threshold(0, 1)


def test_the_net_benefit_identity_holds_on_random_tables():
    rng = np.random.default_rng(0)
    for ignored in range(20):
        y = (rng.random(300) < rng.uniform(0.1, 0.6)).astype(float)
        a = (rng.random(300) < 0.4).astype(float)
        fp, fn = rng.uniform(1, 9), rng.uniform(1, 90)
        m = costs.DecisionCostModel(
            0,
            fn,
            fp,
            0,
            0.001,
            0.999,
            (costs.CostSource.HUMAN,) * 4,
            costs.CostSource.HUMAN,
        )
        t = fp / (fp + fn)
        none = fn * y.mean()
        lhs = costs.net_benefit(y, a, t)
        rhs = (none - costs.expected_cost(y, a, m)) / fn
        assert lhs == pytest.approx(rhs, abs=1e-12)


def test_a_calibrated_score_passes_and_takes_the_bayes_path():
    p, y = world(6000, 1)
    decision = threshold.choose_threshold(p, y, model(), seed=1)
    assert decision.rule.path == "bayes"
    assert decision.rule.threshold == pytest.approx(5 / 105)
    assert decision.tier == "attested"
    low, high = decision.cost_interval
    assert low <= decision.expected_cost <= high
    assert decision.to_dict()["rule_hash"] == decision.rule.digest


def test_a_miscalibrated_score_is_recalibrated_or_falls_back():
    p, y = world(6000, 2, skew=0.5)
    decision = threshold.choose_threshold(p, y, model(), seed=2)
    assert decision.check.calibrated is False
    assert decision.rule.path in ("recalibrated", "grid")
    if decision.rule.path == "recalibrated":
        assert decision.recheck is not None and decision.recheck.calibrated
        assert decision.rule.recalibrator is not None
    else:
        assert decision.tier == "unverified"


def test_a_cost_that_is_not_human_lowers_the_tier():
    p, y = world(4000, 3)
    weak = costs.DecisionCostModel(
        0,
        100.0,
        5.0,
        0,
        0.02,
        0.15,
        (costs.CostSource.TEMPLATE,) * 4,
        costs.CostSource.HUMAN,
    )
    assert threshold.choose_threshold(p, y, weak).tier == "unverified"


def test_platt_scaling_recovers_a_planted_map():
    rng = np.random.default_rng(4)
    raw = rng.uniform(0.02, 0.98, 20000)
    truth = 1 / (1 + np.exp(-(0.4 + 1.5 * np.log(raw / (1 - raw)))))
    y = (rng.random(raw.size) < truth).astype(float)
    fitted = threshold.fit_recalibrator(raw, y)
    assert fitted.intercept == pytest.approx(0.4, abs=0.08)
    assert fitted.slope == pytest.approx(1.5, abs=0.08)


def test_the_calibration_check_refuses_small_or_bad_inputs():
    with pytest.raises(ValueError):
        threshold.calibration_check([0.5] * 10, [1.0, 0.0] * 5)
    with pytest.raises(ValueError):
        threshold.calibration_check([2.0] * 100, [1.0, 0.0] * 50)
    with pytest.raises(ValueError):
        threshold.calibration_check([0.5] * 100, [1.0, 0.0] * 50, level=1.0)


def test_cluster_intervals_are_wider_than_independent_ones():
    rng = np.random.default_rng(5)
    clusters = np.repeat(np.arange(100), 20)
    effect = np.repeat(rng.normal(0, 0.15, 100), 20)
    p = np.clip(0.2 + effect, 0.01, 0.99)
    y = (rng.random(2000) < p).astype(float)
    acts = (p > 0.2).astype(float)
    m = model()
    flat = threshold.cost_interval(y, acts, m, None, 400, 0)
    grouped = threshold.cost_interval(y, acts, m, clusters, 400, 0)
    assert grouped[1] - grouped[0] > flat[1] - flat[0]


def test_a_useful_rule_is_called_useful_and_a_random_one_is_not():
    rng = np.random.default_rng(6)
    n = 6000
    y = (rng.random(n) < 0.2).astype(float)
    good = np.where(y == 1, rng.random(n) < 0.9, rng.random(n) < 0.1).astype(
        float
    )
    rule = netbenefit.net_benefit_band(y, good, (0.1, 0.4), seed=1)
    assert rule.claim == "useful" and rule.useful_range == (0.1, 0.4)
    assert rule.incumbent is None and rule.baselines == (
        "treat_none",
        "treat_all",
    )
    noise = (rng.random(n) < 0.5).astype(float)
    assert netbenefit.net_benefit_band(y, noise, (0.1, 0.4), seed=1).claim == (
        "not_useful_in_range"
    )


def test_the_band_matches_point_net_benefit_and_includes_the_incumbent():
    rng = np.random.default_rng(7)
    y = (rng.random(2000) < 0.3).astype(float)
    a = (rng.random(2000) < 0.4).astype(float)
    b = (rng.random(2000) < 0.5).astype(float)
    report = netbenefit.net_benefit_band(y, a, (0.1, 0.3), incumbent=b, seed=2)
    assert report.baselines[-1] == "incumbent"
    t = report.thresholds[3]
    assert report.rule[3] == pytest.approx(costs.net_benefit(y, a, t))
    assert report.treat_all[3] == pytest.approx(costs.net_benefit_all(y, t))
    assert report.difference[2][3] == pytest.approx(
        costs.net_benefit(y, a, t) - costs.net_benefit(y, b, t)
    )
    assert all(
        lo <= d
        for lo, d in zip(report.lower[0], report.difference[0], strict=True)
    )
    assert report.critical >= 0


def test_clusters_widen_the_band():
    rng = np.random.default_rng(8)
    clusters = np.repeat(np.arange(100), 20)
    base = np.repeat(rng.uniform(0.05, 0.6, 100), 20)
    y = (rng.random(2000) < base).astype(float)
    a = (rng.random(2000) < 0.3 + 0.4 * base).astype(float)
    flat = netbenefit.net_benefit_band(y, a, (0.1, 0.3), seed=3)
    grouped = netbenefit.net_benefit_band(
        y, a, (0.1, 0.3), clusters=clusters, seed=3
    )
    assert grouped.critical > 0 and flat.clusters == 2000
    assert grouped.clusters == 100
    assert (
        grouped.difference[0][0] - grouped.lower[0][0]
        > flat.difference[0][0] - flat.lower[0][0]
    )


def test_the_final_table_is_evaluated_once_for_the_registered_rule():
    y = np.array([1.0, 0.0] * 100)
    a = y.copy()
    guard = netbenefit.FinalNetBenefit("abc")
    with pytest.raises(netbenefit.SecondEvaluationError, match="differs"):
        guard.evaluate("other", y, a, (0.1, 0.3))
    first = guard.evaluate("abc", y, a, (0.1, 0.3), replicates=50)
    assert first.claim == "useful"
    with pytest.raises(netbenefit.SecondEvaluationError, match="already"):
        guard.evaluate("abc", y, a, (0.1, 0.3))
    with pytest.raises(ValueError):
        netbenefit.FinalNetBenefit(" ")


@pytest.mark.parametrize(
    "args",
    [
        {"t_range": (0.0, 0.3)},
        {"t_range": (0.4, 0.3)},
        {"t_range": (0.1, 0.3), "alpha": 1.0},
        {"t_range": (0.1, 0.3), "replicates": 5},
    ],
)
def test_bad_band_inputs_are_refused(args):
    y = np.array([1.0, 0.0] * 20)
    with pytest.raises(ValueError):
        netbenefit.net_benefit_band(y, y, **args)
