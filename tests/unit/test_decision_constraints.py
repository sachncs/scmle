"""Fairness constraints: satisfaction, group thresholds, feasibility."""

import numpy as np
import pytest

from scmle.data.charter import decision as charter
from scmle.decision import constraints as c


def constraint(criterion="equal_opportunity", gap=0.05, **kwargs):
    return charter.FairnessConstraint(
        "sex", criterion, gap, "group_thresholds", "counsel", **kwargs
    )


def binormal(n, seed, shift=0.6, rates=(0.3, 0.3)):
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, 2, n)
    prevalence = np.where(groups == 0, rates[0], rates[1])
    y = (rng.random(n) < prevalence).astype(float)
    centre = y * 1.5 - groups * shift * y
    score = 1 / (1 + np.exp(-(rng.normal(centre - 0.75, 1.0))))
    return score, y, groups


def test_newcombe_matches_a_hand_computed_interval():
    lo, hi = c.newcombe(40, 100, 20, 100, 0.95)
    # Wilson limits (40/100: .3094,.4973; 20/100: .1334,.2889), then
    # d -/+ sqrt of the summed squared distances (Newcombe 1998, method 10)
    assert lo == pytest.approx(0.07312, abs=1e-4)
    assert hi == pytest.approx(0.31851, abs=1e-4)


def test_equal_rates_are_satisfied_and_unequal_rates_are_not():
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 2, 20000)
    y = (rng.random(20000) < 0.4).astype(float)
    fair = (rng.random(20000) < 0.2 + 0.5 * y).astype(float)
    ok = c.satisfaction_test(y, fair, groups, constraint(gap=0.08))
    assert ok.status == c.SATISFIED and ok.upper_bound <= 0.08
    unfair = (rng.random(20000) < 0.2 + 0.5 * y - 0.15 * groups * y).astype(
        float
    )
    bad = c.satisfaction_test(y, unfair, groups, constraint(gap=0.08))
    assert bad.status == c.NOT_SATISFIED
    assert bad.point_gap == pytest.approx(0.15 / 0.7 * 0.7, abs=0.05)


def test_a_point_estimate_below_the_bound_is_not_satisfaction():
    rng = np.random.default_rng(1)
    groups = rng.integers(0, 2, 400)
    y = (rng.random(400) < 0.5).astype(float)
    acts = (rng.random(400) < 0.3 + 0.4 * y).astype(float)
    result = c.satisfaction_test(y, acts, groups, constraint(gap=0.1))
    assert result.point_gap < 0.1 < result.upper_bound
    assert result.status == c.NOT_SATISFIED


def test_small_groups_are_not_assessable_never_passing():
    groups = np.array([0] * 300 + [1] * 10)
    y = np.tile([1.0, 0.0], 155)
    acts = y.copy()
    result = c.satisfaction_test(y, acts, groups, constraint(min_group=30))
    assert result.status == c.NOT_ASSESSABLE
    assert result.small_groups == ("1",)
    assert result.point_gap is None


def test_each_criterion_reads_its_own_rate():
    y = np.array([1.0, 1.0, 0.0, 0.0] * 50)
    groups = np.array([0, 1] * 100)
    acts = np.array([1.0, 0.0, 1.0, 0.0] * 50)
    assert c.development_gap(y, acts, groups, "demographic_parity") == 1.0
    assert c.development_gap(y, acts, groups, "equal_opportunity") == 1.0
    assert c.development_gap(y, acts, groups, "equalized_odds") == 1.0
    # group 1 selects nobody, so its precision is read as zero; group 0 has 0.5
    assert c.development_gap(y, acts, groups, "predictive_parity") == 0.5


def test_group_thresholds_close_a_gap_and_the_price_is_reported():
    score, y, groups = binormal(30000, 2)
    t = 0.3
    free = (score >= t).astype(float)
    assert c.development_gap(y, free, groups, "equal_opportunity") > 0.1
    fitted = c.fit_group_thresholds(score, y, groups, constraint(gap=0.06), t)
    assert fitted.constrained and fitted.development_gap <= 0.03
    acts = fitted.act(score, groups)
    result = c.satisfaction_test(y, acts, groups, constraint(gap=0.06))
    assert result.status == c.SATISFIED
    price = c.price_of_constraint(y, free, acts, t, replicates=200)
    assert price["lower"] <= price["price"] <= price["upper"]
    assert np.isfinite(price["price"])


def test_an_already_fair_rule_keeps_the_common_threshold():
    score, y, groups = binormal(20000, 3, shift=0.0)
    fitted = c.fit_group_thresholds(score, y, groups, constraint(gap=0.2), 0.3)
    assert not fitted.constrained and fitted.cuts == {"0": 0.3, "1": 0.3}


def test_frontier_marks_unreachable_points():
    score, y, groups = binormal(10000, 4)
    rows = c.frontier(score, y, groups, constraint(), 0.3, [0.001, 0.05, 0.3])
    assert [r["max_gap"] for r in rows] == [0.001, 0.05, 0.3]
    assert rows[-1]["feasible"]
    assert all("net_benefit" in r for r in rows if r["feasible"])


def test_predictive_parity_cannot_be_reached_by_group_thresholds():
    score, y, groups = binormal(5000, 5)
    with pytest.raises(ValueError, match="predictive parity"):
        c.fit_group_thresholds(
            score, y, groups, constraint("predictive_parity"), 0.3
        )


def test_incompatible_criteria_are_refused_only_when_base_rates_differ():
    pair = (
        constraint("demographic_parity"),
        constraint("equalized_odds"),
    )
    ignored, y, groups = binormal(20000, 6, rates=(0.2, 0.5))
    with pytest.raises(c.InfeasibleConstraintsError, match="cannot both hold"):
        c.check_feasibility(pair, y, groups)
    ignored, y_equal, groups_equal = binormal(20000, 7, rates=(0.3, 0.3))
    report = c.check_feasibility(pair, y_equal, groups_equal)
    assert report["differ"] is False and report["conflicts"]
    single = c.check_feasibility(pair[:1], y, groups)
    assert single["conflicts"] == []


def test_inputs_are_validated():
    with pytest.raises(ValueError):
        c.satisfaction_test([1.0, 0.0], [1.0, 0.0], [0], constraint())
    with pytest.raises(ValueError):
        c.satisfaction_test(
            [1.0, 0.0], [1.0, 0.0], [0, 1], constraint(), alpha=1.0
        )


def clustered_world(seed, gap=0.06, clusters=30, size=100, spread=0.2):
    """Selection varies by cluster; each cluster belongs to one group."""
    rng = np.random.default_rng(seed)
    group_of = np.repeat([0, 1], clusters)
    mean = np.where(group_of == 0, 0.30, 0.30 + gap)
    prob = np.clip(mean + spread * rng.standard_normal(2 * clusters), 0.0, 1.0)
    cluster = np.repeat(np.arange(2 * clusters), size)
    groups = np.repeat(group_of, size)
    acts = (rng.random(len(cluster)) < prob[cluster]).astype(float)
    outcomes = (rng.random(len(cluster)) < 0.4).astype(float)
    return outcomes, acts, groups, cluster


def test_clustered_rows_are_not_declared_satisfied_more_often_than_alpha():
    # Pre-registered before the first run: the true gap is 0.06 against a
    # bound of 0.05, so every "satisfied" is a false claim; the cluster-aware
    # test may make it in at most 7 percent of 400 worlds, and the row-wise
    # test, which treats the rows as independent, makes it at least 3 points
    # more often.
    rule = constraint("demographic_parity", gap=0.05, min_group=30)
    worlds = 400
    rows = cluster_aware = 0
    for seed in range(worlds):
        y, acts, groups, cluster = clustered_world(seed)
        rows += c.satisfaction_test(y, acts, groups, rule).status == (
            c.SATISFIED
        )
        cluster_aware += c.cluster_satisfaction_test(
            y, acts, groups, cluster, rule, draws=400, seed=seed
        ).status == (c.SATISFIED)
    assert cluster_aware / worlds <= 0.07
    assert rows / worlds >= cluster_aware / worlds + 0.03


def test_a_clearly_fair_clustered_rule_is_satisfied():
    y, acts, groups, cluster = clustered_world(1, gap=0.0, spread=0.02)
    result = c.cluster_satisfaction_test(
        y, acts, groups, cluster, constraint("demographic_parity", gap=0.2)
    )
    assert result.status == c.SATISFIED
    assert result.point_gap is not None and result.upper_bound is not None
    assert result.upper_bound >= result.point_gap


def test_cluster_aware_small_groups_and_bad_inputs():
    y, acts, groups, cluster = clustered_world(2)
    small = c.cluster_satisfaction_test(
        y,
        acts,
        groups,
        cluster,
        constraint("demographic_parity", min_group=5000),
    )
    assert small.status == c.NOT_ASSESSABLE
    with pytest.raises(ValueError):
        c.cluster_satisfaction_test(
            y, acts, groups, cluster[:-1], constraint("demographic_parity")
        )
    with pytest.raises(ValueError):
        c.cluster_satisfaction_test(
            y, acts, groups, cluster, constraint("demographic_parity"), alpha=2
        )
    with pytest.raises(ValueError):
        c.cluster_satisfaction_test(
            y, acts, groups, cluster, constraint("demographic_parity"), draws=0
        )


def test_the_cluster_test_is_reproducible_and_reads_every_criterion():
    y, acts, groups, cluster = clustered_world(3)
    for criterion in (
        "demographic_parity",
        "equal_opportunity",
        "equalized_odds",
        "predictive_parity",
    ):
        rule = constraint(criterion, gap=0.5, min_group=30)
        first = c.cluster_satisfaction_test(
            y, acts, groups, cluster, rule, draws=200, seed=4
        )
        again = c.cluster_satisfaction_test(
            y, acts, groups, cluster, rule, draws=200, seed=4
        )
        assert first == again
        assert first.criterion == criterion
