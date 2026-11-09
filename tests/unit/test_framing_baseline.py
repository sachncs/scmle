import numpy
import pytest
from scipy import stats

from benchmarks.qualification import framing_certify
from scmle.contracts import qualification
from scmle.data.framing import baseline
from scmle.data.framing import costs

HUMAN = (costs.CostSource.HUMAN,) * 4
MODEL = costs.DecisionCostModel(
    0.0, 5.0, 1.0, 0.0, 0.1, 0.4, HUMAN, costs.CostSource.HUMAN
)
RULE = {"column": "score", "threshold": 0.4}


def reference_cluster_se(d, clusters):
    """Independent loop sandwich estimator."""
    ids = sorted(set(clusters))
    n, g = len(d), len(ids)
    mean = sum(d) / n
    total = 0.0
    for c in ids:
        s = sum(x - mean for x, k in zip(d, clusters, strict=True) if k == c)
        total += s * s
    return (g / (g - 1) * total) ** 0.5 / n


def test_cluster_test_matches_reference_loop():
    rng = numpy.random.default_rng(3)
    d = rng.normal(0.1, 1.0, 120)
    clusters = rng.integers(0, 12, 120)
    got = baseline.cluster_test(d, clusters)
    assert got.se == pytest.approx(
        reference_cluster_se(list(d), list(clusters)), rel=1e-12
    )
    assert got.p_value == pytest.approx(
        stats.t.sf(got.mean / got.se, got.df), rel=1e-12
    )
    assert got.df == len(set(clusters)) - 1


def test_cluster_test_needs_two_clusters_and_aligned_inputs():
    with pytest.raises(ValueError):
        baseline.cluster_test([1.0, 2.0], [0, 0])
    with pytest.raises(ValueError):
        baseline.cluster_test([1.0, 2.0], [0, 1, 2])


def test_delta_is_linear_in_the_weight():
    rng = numpy.random.default_rng(4)
    y = (rng.random(300) < 0.3).astype(float)
    a = (rng.random(300) < 0.5).astype(float)
    b = (rng.random(300) < 0.4).astype(float)

    def delta(t):
        return baseline.paired_difference(y, a, b, t).mean()

    lo, mid, hi = 0.1, 0.25, 0.4
    w = [costs.odds(t) for t in (lo, mid, hi)]
    interpolated = delta(lo) + (delta(hi) - delta(lo)) * (w[1] - w[0]) / (
        w[2] - w[0]
    )
    assert delta(mid) == pytest.approx(interpolated, abs=1e-12)
    expected = costs.net_benefit(y, a, mid) - costs.net_benefit(y, b, mid)
    assert delta(mid) == pytest.approx(expected, abs=1e-12)


def test_intersection_union_size_in_dev_worlds():
    for kind in ("null_hi", "null_lo", "null_both"):
        se_a, fp_a = framing_certify.null_parameters(kind)
        runs = 400
        rejected = sum(
            framing_certify.claim_p(
                framing_certify.DEV_SEED_BASE + s,
                baseline.MIN_CLUSTERS,
                se_a,
                fp_a,
            )
            <= 0.05
            for s in range(runs)
        )
        assert rejected / runs <= framing_certify.size_bound(runs)


def test_power_where_the_closed_form_allows_it():
    row = framing_certify.power_rows(framing_certify.DEV_SEED_BASE, 300)
    assert row["attainable"]
    assert row["empirical_power"] >= row["floor"]


def test_selective_labels_flag_and_clean_worlds():
    flagged = sum(
        baseline.selective_labels(
            *framing_certify.label_world(True, 10_000 + s)
        ).flagged
        for s in range(60)
    )
    clean = sum(
        baseline.selective_labels(
            *framing_certify.label_world(False, 10_000 + s)
        ).flagged
        for s in range(60)
    )
    assert flagged == 60
    assert clean == 0


def test_selective_labels_small_arm_is_not_flagged():
    got = baseline.selective_labels([1, 0, 1, 1], [1, 0, 1, 1])
    assert not got.flagged and got.p_value == 1.0


def rows(seed=0, g=60, se_a=0.85, fp_a=0.1):
    return framing_certify.world(seed, g, se_a, fp_a)


def guard(alpha=0.05):
    return baseline.Evaluator(baseline.register(MODEL, RULE, alpha))


def test_claim_is_issued_with_a_real_gain_and_only_once():
    y, a, b, group = rows()
    g = guard()
    report = g.evaluate(y, a, b, group, rule=RULE, model=MODEL)
    assert report.issued and report.status == "issued"
    assert report.claim.tier is qualification.Tier.CONFIRMED
    assert {c.set_id for c in report.checks} == {
        qualification.SetId.CONFIRMATION
    }
    assert report.to_dict()["registration_hash"] == (
        g.registration.registration_hash
    )
    with pytest.raises(baseline.BaselineError, match="already evaluated"):
        g.evaluate(y, a, b, group, rule=RULE, model=MODEL)


def test_changed_rule_or_costs_after_registration_raise():
    y, a, b, group = rows()
    with pytest.raises(baseline.BaselineError, match="differs"):
        guard().evaluate(y, a, b, group, rule={"column": "other"}, model=MODEL)
    other = costs.DecisionCostModel(
        0.0, 4.0, 1.0, 0.0, 0.1, 0.4, HUMAN, costs.CostSource.HUMAN
    )
    with pytest.raises(baseline.BaselineError, match="differs"):
        guard().evaluate(y, a, b, group, rule=RULE, model=other)


def test_no_gain_is_not_shown_and_stays_unverified():
    y, a, b, group = rows(se_a=0.6, fp_a=0.2)
    report = guard().evaluate(y, a, b, group, rule=RULE, model=MODEL)
    assert not report.issued and report.status == "not_shown"
    assert report.claim.tier is qualification.Tier.UNVERIFIED
    assert report.claim.adjusted_p_value is None


def test_selective_labels_block_the_claim():
    y, a, b, group = rows()
    y = y.copy()
    y[b == 0] = numpy.nan
    report = guard().evaluate(y, a, b, group, rule=RULE, model=MODEL)
    assert report.status == "not_issued" and not report.endpoints
    assert "selective labels" in report.reasons[0]
    assert report.claim.tier is qualification.Tier.UNVERIFIED


def test_too_few_clusters_do_not_issue_a_claim():
    y, a, b, group = rows(g=baseline.MIN_CLUSTERS - 1)
    report = guard().evaluate(y, a, b, group, rule=RULE, model=MODEL)
    assert report.status == "not_issued"
    assert "clusters" in report.reasons[0]


def test_input_tier_caps_the_claim():
    y, a, b, group = rows()
    report = guard().evaluate(
        y,
        a,
        b,
        group,
        rule=RULE,
        model=MODEL,
        input_tier=qualification.Tier.UNVERIFIED,
    )
    assert report.issued
    assert report.claim.tier is qualification.Tier.UNVERIFIED


def test_profile_on_x_withholds_a_biased_baseline():
    y, b = rows()[0], rows()[2]
    clean = baseline.profile_on_exploration(y, b, MODEL)
    assert len(clean["net_benefit_baseline"]) == 5
    y2 = y.copy()
    y2[b == 0] = numpy.nan
    biased = baseline.profile_on_exploration(y2, b, MODEL)
    assert biased["net_benefit_baseline"] == baseline.NOT_COMPUTED
    assert biased["selective_labels"]["flagged"]
