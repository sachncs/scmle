"""Pins the baseline claim: tests, finding, registration, evaluator and report."""

import hashlib
import math

import numpy
import pytest
from scipy import stats

from scmle.contracts import qualification
from scmle.contracts import signing
from scmle.data.framing import baseline
from scmle.data.framing import costs

HUMAN = (costs.CostSource.HUMAN,) * 4
MODEL = costs.DecisionCostModel(
    0.0, 5.0, 1.0, 0.0, 0.1, 0.4, HUMAN, costs.CostSource.HUMAN
)
RULE = {"column": "score", "threshold": 0.4}


def test_the_constants_are_the_documented_ones():
    assert baseline.DETECTOR_ID == "framing-selective-labels"
    assert baseline.DEFECT_TYPE == "selective_labels"
    assert baseline.LABEL_TOLERANCE == 0.2
    assert baseline.LABEL_ALPHA == 0.01
    assert baseline.MIN_ARM_ROWS == 5
    assert baseline.MIN_CLUSTERS == 20
    assert baseline.CHECK_PREFIX == "baseline"
    assert baseline.CLAIM_ID == "baseline_vs_current_process"
    assert baseline.NOT_COMPUTED == "not_computed"
    assert baseline.MIN_VARIANCE_CLUSTERS == 2


def test_vector_converts_and_names_the_bad_argument():
    result = baseline.vector([1, 2, 3], "d")
    assert result.dtype == float and result.tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match=r"^d must be a non-empty 1-d array$"):
        baseline.vector([], "d")
    with pytest.raises(ValueError, match=r"^zz must be a non-empty 1-d array$"):
        baseline.vector([[1.0, 2.0]], "zz")
    assert baseline.vector([5.0], "d").tolist() == [5.0]


def test_the_paired_difference_is_the_documented_gain_per_row():
    y = [1, 1, 0, 0, 1]
    a = [1, 0, 1, 0, 1]
    b = [0, 1, 0, 1, 1]
    got = baseline.paired_difference(y, a, b, 0.25)
    w = 0.25 / 0.75
    assert got.tolist() == pytest.approx([1.0, -1.0, -w, w, 0.0])


def test_the_paired_difference_checks_lengths_and_names_arguments():
    with pytest.raises(ValueError, match=r"^y, a and b must have the same"):
        baseline.paired_difference([1, 0], [1, 0, 1], [1, 0], 0.2)
    with pytest.raises(ValueError, match=r"^y must"):
        baseline.paired_difference([2, 0], [1, 0], [1, 0], 0.2)
    with pytest.raises(ValueError, match=r"^a must"):
        baseline.paired_difference([1, 0], [2, 0], [1, 0], 0.2)
    with pytest.raises(ValueError, match=r"^b must"):
        baseline.paired_difference([1, 0], [1, 0], [2, 0], 0.2)
    with pytest.raises(ValueError, match="strictly between"):
        baseline.paired_difference([1, 0], [1, 0], [1, 0], 1.0)


def test_cluster_test_returns_every_field_for_a_hand_worked_case():
    got = baseline.cluster_test([1.0, 2.0, 3.0, 4.0], ["a", "a", "b", "b"])
    assert got.mean == 2.5
    assert got.se == pytest.approx(1.0)
    assert got.df == 1
    assert got.statistic == pytest.approx(2.5)
    assert got.p_value == pytest.approx(0.5 - math.atan(2.5) / math.pi)
    assert got.n == 4
    assert got.clusters == 2


def test_cluster_test_three_clusters_uses_the_small_sample_factor():
    d = [1.0, 0.0, 2.0, 3.0, 1.0, 5.0]
    got = baseline.cluster_test(d, [0, 0, 1, 1, 2, 2])
    mean = sum(d) / 6
    sums = [
        (1 - mean) + (0 - mean),
        (2 - mean) + (3 - mean),
        (1 - mean) + (5 - mean),
    ]
    variance = 3 / 2 * sum(s * s for s in sums) / 36
    assert got.se == pytest.approx(math.sqrt(variance))
    assert got.df == 2 and got.clusters == 3 and got.n == 6
    assert got.p_value == pytest.approx(stats.t.sf(mean / got.se, 2))


def test_cluster_test_with_no_spread_decides_by_the_sign_of_the_mean():
    positive = baseline.cluster_test([1.0, 1.0, 1.0, 1.0], [0, 0, 1, 1])
    assert positive.se == 0.0
    assert positive.statistic == math.inf and positive.p_value == 0.0
    negative = baseline.cluster_test([-1.0, -1.0, -1.0, -1.0], [0, 0, 1, 1])
    assert negative.statistic == -math.inf and negative.p_value == 1.0
    zero = baseline.cluster_test([0.0, 0.0, 0.0, 0.0], [0, 0, 1, 1])
    assert zero.statistic == 0.0 and zero.p_value == 0.5
    assert zero.mean == 0.0


def test_cluster_test_errors_say_what_is_wrong():
    with pytest.raises(ValueError, match=r"^at least two clusters are needed$"):
        baseline.cluster_test([1.0, 2.0], [0, 0])
    with pytest.raises(
        ValueError, match=r"^d and clusters must have the same length$"
    ):
        baseline.cluster_test([1.0, 2.0], [0, 1, 2])
    with pytest.raises(ValueError, match=r"^d must be"):
        baseline.cluster_test([], [])


def test_factorize_codes_in_sorted_order_and_counts_clusters():
    codes, count = baseline.factorize(["b", "a", "b", "c"])
    assert codes.tolist() == [1, 0, 1, 2]
    assert codes.dtype == numpy.intp
    assert count == 3
    codes, count = baseline.factorize([7])
    assert codes.tolist() == [0] and count == 1
    codes, count = baseline.factorize([])
    assert codes.tolist() == [] and count == 0
    with pytest.raises(ValueError, match=r"^clusters must be 1-d$"):
        baseline.factorize([[1, 2], [3, 4]])


def test_selective_labels_reports_every_field_in_the_flagged_direction():
    observed = [True] * 10 + [False] * 10 + [True] * 18 + [False] * 2
    base = [0] * 20 + [1] * 20
    got = baseline.selective_labels(observed, base)
    assert got.missing_rate_b0 == 0.5 and got.missing_rate_b1 == 0.1
    assert got.difference == pytest.approx(0.4)
    expected = stats.fisher_exact([[10, 10], [2, 18]], alternative="greater")[1]
    assert got.p_value == pytest.approx(expected)
    assert got.flagged is True
    assert got.tolerance == 0.2 and got.alpha == 0.01
    assert got.to_dict() == {
        "flagged": True,
        "missing_rate_b0": 0.5,
        "missing_rate_b1": 0.1,
        "difference": got.difference,
        "p_value": got.p_value,
        "tolerance": 0.2,
        "alpha": 0.01,
    }


def test_selective_labels_flags_the_other_direction_too():
    observed = [True] * 18 + [False] * 2 + [True] * 10 + [False] * 10
    base = [0] * 20 + [1] * 20
    got = baseline.selective_labels(observed, base)
    assert got.difference == pytest.approx(-0.4)
    expected = stats.fisher_exact([[2, 18], [10, 10]], alternative="less")[1]
    assert got.p_value == pytest.approx(expected)
    assert got.flagged is True


def test_selective_labels_needs_both_the_size_and_the_significance():
    wide = [True] * 3 + [False] * 2 + [True] * 5 + [False] * 0
    small = baseline.selective_labels(wide, [0] * 5 + [1] * 5)
    assert small.difference == pytest.approx(0.4)
    assert small.p_value > baseline.LABEL_ALPHA and small.flagged is False
    big = 1000
    observed = (
        [True] * (big - 150)
        + [False] * 150
        + [True] * (big - 100)
        + [False] * 100
    )
    mild = baseline.selective_labels(observed, [0] * big + [1] * big)
    assert mild.difference == pytest.approx(0.05)
    assert mild.p_value < baseline.LABEL_ALPHA and mild.flagged is False


def test_selective_labels_flags_exactly_at_the_tolerance():
    observed = [True] * 600 + [False] * 400 + [True] * 800 + [False] * 200
    got = baseline.selective_labels(observed, [0] * 1000 + [1] * 1000)
    assert got.difference == pytest.approx(0.2)
    assert got.p_value < baseline.LABEL_ALPHA
    assert got.flagged is True


def test_selective_labels_flags_when_p_equals_the_level(monkeypatch):
    monkeypatch.setattr(
        stats, "fisher_exact", lambda table, alternative: (1.0, 0.01)
    )
    observed = [True] * 10 + [False] * 10 + [True] * 18 + [False] * 2
    got = baseline.selective_labels(observed, [0] * 20 + [1] * 20)
    assert got.p_value == 0.01 and got.flagged is True


def test_selective_labels_with_a_small_arm_says_nothing():
    got = baseline.selective_labels([1, 0, 1, 1], [1, 0, 1, 1])
    assert got.flagged is False and got.p_value == 1.0
    assert got.difference == 0.0
    assert math.isnan(got.missing_rate_b0) and math.isnan(got.missing_rate_b1)
    assert got.tolerance == 0.2 and got.alpha == 0.01


def test_an_arm_of_exactly_the_minimum_size_is_assessed():
    observed = [False] * 5 + [True] * 5 + [True] * 5
    got = baseline.selective_labels(observed, [0] * 5 + [1] * 5 + [1] * 5)
    assert got.missing_rate_b0 == 1.0 and got.missing_rate_b1 == 0.0
    assert got.p_value < 1.0
    other = baseline.selective_labels(
        [False] * 5 + [True] * 5 + [True] * 5, [1] * 5 + [0] * 5 + [0] * 5
    )
    assert other.missing_rate_b1 == 1.0 and other.missing_rate_b0 == 0.0
    assert other.p_value < 1.0
    four = baseline.selective_labels([True] * 9, [0] * 4 + [1] * 5)
    assert math.isnan(four.missing_rate_b0)
    four = baseline.selective_labels([True] * 9, [0] * 5 + [1] * 4)
    assert math.isnan(four.missing_rate_b0)


def test_selective_labels_errors():
    with pytest.raises(
        ValueError, match=r"^observed and b must have the same length$"
    ):
        baseline.selective_labels([True, False], [1, 0, 1])
    with pytest.raises(ValueError, match=r"^b must hold only 0 and 1$"):
        baseline.selective_labels([True, False], [2, 0])


def profile_world():
    outcomes = numpy.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0] * 4, dtype=float)
    decisions = numpy.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0] * 4, dtype=float)
    return outcomes, decisions


def test_the_exploration_profile_lists_thresholds_and_default_net_benefits():
    outcomes, decisions = profile_world()
    got = baseline.profile_on_exploration(outcomes, decisions, MODEL)
    grid = numpy.linspace(0.1, 0.4, 5)
    assert got["thresholds"] == pytest.approx(grid.tolist())
    assert got["net_benefit_none"] == [0.0] * 5
    assert got["net_benefit_all"] == pytest.approx(
        [0.5 - 0.5 * t / (1 - t) for t in grid]
    )
    assert got["net_benefit_baseline"] == pytest.approx(
        [costs.net_benefit(outcomes, decisions, t) for t in grid]
    )
    assert "note" not in got
    assert got["selective_labels"]["flagged"] is False
    assert set(got) == {
        "selective_labels",
        "thresholds",
        "net_benefit_none",
        "net_benefit_all",
        "net_benefit_baseline",
    }


def test_the_exploration_profile_uses_only_the_observed_rows():
    outcomes, decisions = profile_world()
    hidden = outcomes.copy()
    hidden[[0, 7, 11, 20]] = numpy.nan
    got = baseline.profile_on_exploration(hidden, decisions, MODEL)
    seen = ~numpy.isnan(hidden)
    assert got["net_benefit_baseline"] == pytest.approx(
        [
            costs.net_benefit(hidden[seen], decisions[seen], t)
            for t in got["thresholds"]
        ]
    )
    assert got["net_benefit_all"] == pytest.approx(
        [costs.net_benefit_all(hidden[seen], t) for t in got["thresholds"]]
    )


def test_an_explicit_observation_mask_overrides_the_missing_values():
    outcomes, decisions = profile_world()
    mask = numpy.arange(40) % 3 != 0
    got = baseline.profile_on_exploration(
        outcomes, decisions, MODEL, observed=mask
    )
    assert got["net_benefit_baseline"] == pytest.approx(
        [
            costs.net_benefit(outcomes[mask], decisions[mask], t)
            for t in got["thresholds"]
        ]
    )
    assert got["net_benefit_all"] == pytest.approx(
        [costs.net_benefit_all(outcomes[mask], t) for t in got["thresholds"]]
    )
    ints = baseline.profile_on_exploration(
        outcomes, decisions, MODEL, observed=mask.astype(int)
    )
    assert ints["net_benefit_baseline"] == got["net_benefit_baseline"]


def test_selective_labels_withhold_the_baseline_with_a_reason():
    outcomes, decisions = profile_world()
    biased = outcomes.copy()
    biased[decisions == 0] = numpy.nan
    got = baseline.profile_on_exploration(biased, decisions, MODEL)
    assert got["net_benefit_baseline"] == "not_computed"
    assert got["note"] == (
        "outcome observation depends on the baseline decision: its net "
        "benefit is biased and not reported"
    )
    assert got["selective_labels"]["flagged"] is True
    assert "net_benefit_all" in got


def test_no_observed_outcome_withholds_the_baseline_and_the_treat_all_value():
    decisions = numpy.array([1.0, 0.0] * 20)
    got = baseline.profile_on_exploration(
        numpy.full(40, numpy.nan), decisions, MODEL
    )
    assert got["net_benefit_baseline"] == "not_computed"
    assert got["note"] == "no outcome is observed"
    assert "net_benefit_all" not in got
    assert got["net_benefit_none"] == [0.0] * 5


def test_registration_hash_covers_every_field_and_the_rule_digest_is_sha256():
    registration = baseline.register(MODEL, RULE, 0.05)
    assert registration.alpha == 0.05
    assert (registration.t_lo, registration.t_hi) == (0.1, 0.4)
    assert registration.cost_model_hash == MODEL.model_hash
    expected_rule = hashlib.sha256(signing.canonical_payload(RULE)).hexdigest()
    assert registration.rule_hash == expected_rule == baseline.rule_digest(RULE)
    payload = {
        "cost_model_hash": MODEL.model_hash,
        "rule_hash": expected_rule,
        "alpha": 0.05,
        "t_lo": 0.1,
        "t_hi": 0.4,
    }
    assert (
        registration.registration_hash
        == hashlib.sha256(signing.canonical_payload(payload)).hexdigest()
    )
    assert baseline.register(MODEL, RULE, 0.06).registration_hash != (
        registration.registration_hash
    )


def test_the_level_must_be_strictly_inside_the_unit_interval():
    for bad in (0.0, 1.0, -0.1, 1.5, math.nan):
        with pytest.raises(ValueError):
            baseline.register(MODEL, RULE, bad)
    with pytest.raises(ValueError, match=r"^alpha must be strictly between"):
        baseline.register(MODEL, RULE, 0.0)
    with pytest.raises(ValueError, match=r"^alpha must be strictly between"):
        baseline.register(MODEL, RULE, 1.0)
    assert baseline.register(MODEL, RULE, 0.5).alpha == 0.5


def world(groups, per_group=4, gain=True, seed=0):
    """Clusters with a model that clearly beats the baseline (or not)."""
    rng = numpy.random.default_rng(seed)
    size = groups * per_group
    y = (rng.random(size) < 0.5).astype(float)
    b = numpy.zeros(size)
    a = y.copy() if gain else b.copy()
    cluster = numpy.repeat(numpy.arange(groups), per_group)
    return y, a, b, cluster


def run(g, y, a, b, cluster, **options):
    return g.evaluate(y, a, b, cluster, rule=RULE, model=MODEL, **options)


def guard(alpha=0.05):
    return baseline.Evaluator(baseline.register(MODEL, RULE, alpha))


def test_the_evaluator_starts_unused_and_keeps_its_registration():
    g = guard()
    assert g.used is False
    assert g.registration.alpha == 0.05
    y, a, b, cluster = world(30)
    run(g, y, a, b, cluster)
    assert g.used is True


def test_a_second_evaluation_and_a_changed_input_are_refused_with_reasons():
    y, a, b, cluster = world(30)
    g = guard()
    run(g, y, a, b, cluster)
    with pytest.raises(baseline.BaselineError) as second:
        run(g, y, a, b, cluster)
    assert str(second.value) == (
        "the baseline claim was already evaluated on C; a second look at the "
        "confirmation set is not allowed"
    )
    fresh = guard()
    with pytest.raises(baseline.BaselineError) as changed:
        fresh.evaluate(y, a, b, cluster, rule={"column": "x"}, model=MODEL)
    assert str(changed.value) == (
        "the rule or the cost model differs from what was registered before "
        "C was opened"
    )
    assert fresh.used is False


def test_an_issued_claim_carries_every_check_and_the_claim_fields():
    y, a, b, cluster = world(30)
    g = guard()
    report = run(g, y, a, b, cluster, seed=7)
    assert report.issued is True and report.status == "issued"
    assert report.reasons == ()
    assert len(report.endpoints) == 2
    lo = baseline.cluster_test(
        baseline.paired_difference(y, a, b, 0.1), cluster
    )
    hi = baseline.cluster_test(
        baseline.paired_difference(y, a, b, 0.4), cluster
    )
    assert report.endpoints == (lo, hi)
    assert report.p_value == max(lo.p_value, hi.p_value)
    first, second = report.checks
    assert first.check_id == "baseline:delta@t=0.1000"
    assert second.check_id == "baseline:delta@t=0.4000"
    assert first.null_hypothesis == "Delta(t) <= 0 at t = 0.1000"
    assert second.null_hypothesis == "Delta(t) <= 0 at t = 0.4000"
    assert first.seed_path == (7, "framing.baseline", 0)
    assert second.seed_path == (7, "framing.baseline", 1)
    assert first.p_value == lo.p_value and first.effect == lo.mean
    assert first.n == lo.n == 120
    assert first.interval is None
    assert first.set_id is qualification.SetId.CONFIRMATION
    assert first.tier is qualification.Tier.CONFIRMED
    assert first.decision is qualification.Decision.WARN
    assert first.statistic == lo.statistic
    claim = report.claim
    assert claim.claim_id == "baseline_vs_current_process"
    assert claim.statement == (
        "net benefit of the model exceeds net benefit of the current process "
        "over thresholds [0.1000, 0.4000]"
    )
    assert claim.check_ids == (first.check_id, second.check_id)
    assert claim.adjusted_p_value == report.p_value
    assert claim.effect == min(lo.mean, hi.mean)
    assert claim.tier is qualification.Tier.CONFIRMED
    assert claim.assumptions == (
        "outcomes observed independently of the baseline decision",
        "the action rule was frozen on X before C was opened",
        "cluster-robust standard error, asymptotic in the clusters",
        f"cost model {MODEL.model_hash[:12]} fixed before C",
    )


def test_the_report_document_states_the_guarantee():
    y, a, b, cluster = world(30)
    g = guard()
    report = run(g, y, a, b, cluster)
    document = report.to_dict()
    assert document["schema_version"] == 1
    assert document["status"] == "issued" and document["issued"] is True
    assert document["reasons"] == []
    assert document["p_value_intersection_union"] == report.p_value
    assert [e["clusters"] for e in document["endpoints"]] == [30, 30]
    assert document["selective_labels"] == report.selectivity.to_dict()
    assert document["registration_hash"] == g.registration.registration_hash
    assert [c["check_id"] for c in document["checks"]] == [
        c.check_id for c in report.checks
    ]
    assert document["claim"]["claim_id"] == baseline.CLAIM_ID
    assert document["guarantee"] == (
        "Pr(claim | no true gain at some t in the range) <= alpha under "
        "independent C, read once, a rule frozen on X and outcomes observed "
        "independently of the baseline; the cluster-robust error is "
        "asymptotic in the clusters"
    )


def test_no_gain_is_not_shown_and_the_checks_pass():
    y, a, b, cluster = world(30, gain=False)
    report = run(guard(), y, a, b, cluster)
    assert report.status == "not_shown" and report.issued is False
    wording = (
        "the gain is not shown to be positive at both ends of the cost range "
        "at level 0.05"
    )
    assert report.reasons == (wording,)
    assert all(c.decision is qualification.Decision.PASS for c in report.checks)
    assert all(c.tier is qualification.Tier.UNVERIFIED for c in report.checks)
    assert report.claim.adjusted_p_value is None
    assert report.p_value == 0.5
    assert report.claim.effect == 0.0
    assert report.claim.tier is qualification.Tier.UNVERIFIED
    assert report.claim.check_ids == (
        "baseline:delta@t=0.1000",
        "baseline:delta@t=0.4000",
    )
    assert report.checks[0].statistic == 0.0


def test_the_p_value_equal_to_the_level_still_issues(monkeypatch):
    y, a, b, cluster = world(30)
    exact = baseline.ClusterTest(0.1, 0.05, 29, 2.0, 0.05, 120, 30)
    monkeypatch.setattr(baseline, "cluster_test", lambda d, c: exact)
    report = run(guard(0.05), y, a, b, cluster)
    assert report.issued is True and report.p_value == 0.05
    assert all(c.decision is qualification.Decision.WARN for c in report.checks)


def test_the_larger_endpoint_p_value_decides():
    y, a, b, cluster = world(30)
    # beats the baseline for t=0.1 only: extra false positives hurt at t=0.4
    noisy = a.copy()
    noisy[(y == 0) & (numpy.arange(y.size) % 3 == 0)] = 1.0
    report = run(guard(), y, noisy, b, cluster)
    assert report.p_value == max(e.p_value for e in report.endpoints)
    assert report.endpoints[0].p_value != report.endpoints[1].p_value
    assert report.claim.effect == min(e.mean for e in report.endpoints)


def test_an_infinite_statistic_is_recorded_as_zero_in_the_check():
    y = numpy.tile([1.0, 0.0], 30)
    a = y.copy()
    b = numpy.zeros(60)
    cluster = numpy.repeat(numpy.arange(30), 2)
    report = run(guard(), y, a, b, cluster)
    assert numpy.isinf(report.endpoints[0].statistic)
    assert report.checks[0].statistic == 0.0
    assert report.issued is True


def test_too_few_clusters_are_reported_with_the_count_and_the_minimum():
    y, a, b, cluster = world(19)
    report = run(guard(), y, a, b, cluster)
    assert report.status == "not_issued" and report.issued is False
    assert report.reasons == (
        "19 clusters is below the measured minimum of 20 for a calibrated test",
    )
    assert report.endpoints == () and report.checks == ()
    assert report.p_value is None
    assert report.claim.check_ids == ("baseline:not_computed",)
    assert report.claim.adjusted_p_value is None
    assert report.claim.effect == 0.0
    assert report.claim.tier is qualification.Tier.UNVERIFIED
    exact = world(20)
    assert run(guard(), *exact).status == "issued"


def test_clusters_are_counted_among_the_observed_rows_only():
    y, a, b, cluster = world(30)
    y = y.copy()
    y[cluster >= 15] = numpy.nan
    report = run(guard(), y, a, b, cluster)
    assert report.status == "not_issued"
    assert report.reasons[0].startswith("15 clusters is below")
    none = run(guard(), numpy.full(y.size, numpy.nan), a, b, cluster)
    assert any(r.startswith("0 clusters is below") for r in none.reasons)


def test_selective_labels_add_a_reason_and_run_no_test():
    y, a, b, cluster = world(30)
    b = (numpy.arange(y.size) % 2).astype(float)
    y = y.copy()
    y[b == 0] = numpy.where(
        numpy.arange((b == 0).sum()) % 2 == 0, numpy.nan, 1.0
    )
    report = run(guard(), y, a, b, cluster)
    assert report.status == "not_issued"
    assert report.reasons[0] == (
        "outcomes are observed depending on the baseline decision (selective "
        "labels); the naive comparison is biased"
    )
    assert report.selectivity.flagged is True
    assert report.endpoints == ()


def test_an_explicit_mask_selects_the_rows_that_are_tested():
    y, a, b, cluster = world(30)
    mask = numpy.arange(y.size) % 4 != 0
    other = y.copy()
    other[~mask] = 5.0  # would be rejected as non-binary if it were read
    report = run(guard(), other, a, b, cluster, observed=mask)
    assert report.status == "issued"
    assert report.endpoints[0].n == int(mask.sum())
    wanted = baseline.cluster_test(
        baseline.paired_difference(y[mask], a[mask], b[mask], 0.1),
        cluster[mask],
    )
    assert report.endpoints[0] == wanted
    ints = run(guard(), other, a, b, cluster, observed=mask.astype(int))
    assert ints.endpoints == report.endpoints


def test_the_input_tier_is_kept_only_when_the_claim_is_issued():
    y, a, b, cluster = world(30)
    report = run(
        guard(), y, a, b, cluster, input_tier=qualification.Tier.ATTESTED
    )
    assert report.claim.tier is qualification.Tier.ATTESTED
    assert {c.tier for c in report.checks} == {qualification.Tier.ATTESTED}
    y2, a2, b2, cluster2 = world(30, gain=False)
    lost = run(
        guard(),
        y2,
        a2,
        b2,
        cluster2,
        input_tier=qualification.Tier.ATTESTED,
    )
    assert lost.claim.tier is qualification.Tier.UNVERIFIED


def test_the_default_seed_is_zero():
    y, a, b, cluster = world(30)
    report = run(guard(), y, a, b, cluster)
    assert report.checks[0].seed_path == (0, "framing.baseline", 0)


def test_the_default_input_tier_is_confirmed():
    y, a, b, cluster = world(30)
    assert run(guard(), y, a, b, cluster).claim.tier is (
        qualification.Tier.CONFIRMED
    )


def test_object_outcomes_with_missing_entries_are_read_as_floats():
    outcomes, decisions = profile_world()
    as_objects = [None if i % 5 == 0 else v for i, v in enumerate(outcomes)]
    got = baseline.profile_on_exploration(as_objects, decisions, MODEL)
    seen = numpy.array([i % 5 != 0 for i in range(40)])
    assert got["net_benefit_all"] == pytest.approx(
        [costs.net_benefit_all(outcomes[seen], t) for t in got["thresholds"]]
    )


def test_equal_missing_rates_use_the_greater_tail_of_the_exact_test():
    observed = [True] * 8 + [False] * 2 + [True] * 16 + [False] * 4
    got = baseline.selective_labels(observed, [0] * 10 + [1] * 20)
    assert got.difference == 0.0
    greater = stats.fisher_exact([[2, 8], [4, 16]], alternative="greater")[1]
    less = stats.fisher_exact([[2, 8], [4, 16]], alternative="less")[1]
    assert greater != less
    assert got.p_value == pytest.approx(greater)


def test_a_bad_level_is_reported_under_the_name_alpha():
    for bad in (math.nan, 2.0, -1.0):
        with pytest.raises(ValueError, match=r"^alpha must"):
            baseline.register(MODEL, RULE, bad)


def test_object_outcomes_with_missing_entries_are_evaluated():
    y, a, b, cluster = world(30)
    as_objects = numpy.array(
        [None if i % 9 == 0 else v for i, v in enumerate(y)], dtype=object
    )
    report = run(guard(), as_objects, a, b, cluster)
    keep = numpy.array([i % 9 != 0 for i in range(y.size)])
    assert report.endpoints[0].n == int(keep.sum())
    assert report.endpoints[0] == baseline.cluster_test(
        baseline.paired_difference(y[keep], a[keep], b[keep], 0.1),
        cluster[keep],
    )


def test_a_finite_statistic_is_recorded_as_it_is():
    y, a, b, cluster = world(30)
    report = run(guard(), y, a, b, cluster)
    for check, endpoint in zip(report.checks, report.endpoints, strict=True):
        assert math.isfinite(endpoint.statistic) and endpoint.statistic != 0.0
        assert check.statistic == endpoint.statistic
