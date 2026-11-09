"""The survival rule, the transformations and the curve test."""

import math

import numpy
import pytest
from scipy import stats

from scmle.contracts import qualification
from scmle.data.charter import contracts
from scmle.data.detectors import certificates
from scmle.data.ledger import bivariate
from scmle.data.robustness import multiverse
from scmle.data.robustness import stability

CHOICES = (
    contracts.Choice("imputation", ("median", "complete_case"), "median"),
    contracts.Choice("outlier_rule", ("none", "winsorize", "iqr_drop"), "none"),
    contracts.Choice("inclusion", ("all", "first_half", "second_half"), "all"),
)


def grid(seed=0, size=18):
    return multiverse.enumerate_grid(CHOICES, seed, size)


def data(rows=300, beta=0.6, seed=0, missing=0.1):
    rng = numpy.random.default_rng(seed)
    x = rng.normal(size=rows)
    y = beta * x + rng.normal(size=rows)
    x[rng.random(rows) < missing] = numpy.nan
    return x, y


def spec_of(**values):
    return multiverse.Specification(tuple(sorted(values.items())))


def test_rank_effect_matches_scipy_spearman_and_the_ledger_statistic():
    x, y = data(seed=1, missing=0.0)
    assert stability.rank_effect(x, y) == pytest.approx(
        stats.spearmanr(x, y).statistic
    )
    assert stability.rank_effect(x, y) == pytest.approx(
        bivariate.spearman(x, y).effect
    )


def test_rank_effect_is_nan_when_undefined():
    assert math.isnan(
        stability.rank_effect(numpy.arange(5.0), numpy.arange(5.0))
    )
    assert math.isnan(stability.rank_effect(numpy.ones(20), numpy.arange(20.0)))


def test_each_specification_does_what_it_says():
    x = numpy.array([1.0, 2.0, 3.0, 4.0, 100.0, numpy.nan, numpy.nan, 5.0])
    y = numpy.arange(8.0)
    position = numpy.arange(8)

    def apply(**values):
        return stability.prepare(x, y, position, 8, spec_of(**values))

    median_x = apply(imputation="median")[0]
    assert median_x[5] == numpy.median([1, 2, 3, 4, 100, 5])
    mean_x = apply(imputation="mean")[0]
    assert mean_x[5] == pytest.approx(numpy.mean([1, 2, 3, 4, 100, 5]))
    zero_x = apply(imputation="zero")[0]
    assert zero_x[5] == 0.0
    complete_x, complete_y = apply(imputation="complete_case")
    assert len(complete_x) == 6 and 5 not in complete_y
    clipped = apply(outlier_rule="winsorize", imputation="complete_case")[0]
    assert clipped.max() < 100.0
    dropped = apply(outlier_rule="iqr_drop", imputation="complete_case")[0]
    assert 100.0 not in dropped and len(dropped) == 5
    first_y = apply(inclusion="first_half")[1]
    assert set(first_y) == {0.0, 1.0, 2.0, 3.0}
    second_y = apply(inclusion="second_half")[1]
    assert set(second_y) == {4.0, 5.0, 6.0, 7.0}


def test_inclusion_follows_the_original_positions_after_resampling():
    x = numpy.arange(10.0)
    y = numpy.arange(10.0)
    pick = numpy.array([9, 9, 1, 8, 0])
    kept = stability.prepare(
        x[pick], y[pick], pick, 10, spec_of(inclusion="first_half")
    )[0]
    assert sorted(kept) == [0.0, 1.0]


def test_a_missing_target_drops_the_row():
    x = numpy.arange(6.0)
    y = numpy.array([0.0, 1.0, numpy.nan, 3.0, 4.0, 5.0])
    kept = stability.prepare(x, y, numpy.arange(6), 6, spec_of())[1]
    assert len(kept) == 5


def test_only_implemented_choices_and_options_are_run_and_the_rest_noted():
    declared = (
        contracts.Choice("imputation", ("median", "mice"), "median"),
        contracts.Choice("encoding", ("one_hot", "target"), "one_hot"),
        contracts.Choice("outlier_rule", ("none", "robust_z"), "robust_z"),
        contracts.Choice("mnar_delta", ("0", "1"), "0"),
    )
    kept, notes = stability.applicable_choices(declared)
    assert [(k.name, k.options) for k in kept] == [("imputation", ("median",))]
    text = " ".join(notes)
    assert "encoding" in text and "mice" in text and "robust_z" in text
    assert "mnar_delta" in text


def test_a_strong_and_robust_effect_is_stable():
    x, y = data(rows=400, beta=1.0)
    result = stability.assess_finding("x", x, y, grid(), 0, bootstrap=60)
    assert result.status == stability.STABLE, result.reason
    assert result.check.decision is qualification.Decision.PASS
    assert result.check.set_id is qualification.SetId.EXPLORATION
    assert result.check.interval[0] > 0.0
    assert result.headline == pytest.approx(
        next(
            s.effect
            for s in result.specs
            if s.choices == grid().default.to_dict()
        )
    )


def test_a_sign_that_flips_with_the_declared_choice_is_fragile():
    rng = numpy.random.default_rng(3)
    rows = 400
    x = rng.normal(size=rows)
    half = numpy.arange(rows) < rows / 2
    y = numpy.where(half, 1.0, -1.0) * x + 0.5 * rng.normal(size=rows)
    result = stability.assess_finding("x", x, y, grid(), 0, bootstrap=40)
    assert result.status == stability.FRAGILE
    assert "sign" in result.reason or "interval" in result.reason
    assert result.numbers["sign_share"] < 0.8
    assert result.attribution["inclusion"]["eta_squared"] > 0.5
    assert result.check.decision is qualification.Decision.WARN


def test_a_fragile_finding_quarantines_only_when_the_detector_is_gating():
    rng = numpy.random.default_rng(4)
    x = rng.normal(size=300)
    y = rng.normal(size=300)
    gate = certificates.GateStatus.GATING
    result = stability.assess_finding(
        "x", x, y, grid(), 0, bootstrap=30, gate=gate
    )
    assert result.status == stability.FRAGILE
    assert result.check.decision is qualification.Decision.QUARANTINE


def test_a_flipped_sign_in_more_than_the_allowed_share_is_fragile_by_rule():
    def row(effect, low, high):
        return stability.SpecEffect("h", {}, effect, low, high)

    rule = stability.Rule(s_min=0.8, band=0.5, q_min=0.8)
    good = [row(0.5, 0.3, 0.7)] * 10
    verdict = stability.verdict(good, 0.5, rule)
    assert verdict[0] == stability.STABLE
    assert verdict[2]["sign_share"] == 1.0
    flipped = [row(0.5, 0.3, 0.7)] * 7 + [row(-0.4, -0.6, -0.2)] * 3
    status, reason, numbers = stability.verdict(flipped, 0.5, rule)
    assert status == stability.FRAGILE and "sign" in reason
    assert numbers["sign_share"] == pytest.approx(0.7)
    edge = [row(0.5, 0.3, 0.7)] * 8 + [row(-0.4, -0.6, -0.2)] * 2
    assert stability.verdict(edge, 0.5, rule)[0] == stability.STABLE


def test_size_band_and_interval_share_each_fail_on_their_own():
    def row(effect, low, high):
        return stability.SpecEffect("h", {}, effect, low, high)

    rule = stability.Rule()
    drifted = [row(0.2, 0.1, 0.3)] * 10
    status, reason = stability.verdict(drifted, 0.5, rule)[:2]
    assert status == stability.FRAGILE and "headline" in reason
    wide = [row(0.5, -0.1, 0.9)] * 10
    status, reason = stability.verdict(wide, 0.5, rule)[:2]
    assert status == stability.FRAGILE and "excludes zero" in reason


def test_too_few_specifications_cannot_be_evaluated():
    row = stability.SpecEffect("h", {}, 0.5, 0.3, 0.7)
    assert stability.verdict([row] * 3, 0.5, stability.Rule())[0] == (
        stability.NOT_EVALUABLE
    )
    assert stability.verdict([row] * 9, None, stability.Rule())[0] == (
        stability.NOT_EVALUABLE
    )
    single = multiverse.enumerate_grid((), 0)
    x, y = data()
    result = stability.assess_finding("x", x, y, single, 0, bootstrap=10)
    assert result.status == stability.NOT_EVALUABLE and result.check is None


def test_the_rule_validates_its_thresholds():
    with pytest.raises(ValueError, match="s_min"):
        stability.Rule(s_min=0.0)
    with pytest.raises(ValueError, match="band"):
        stability.Rule(band=0.0)


def test_the_same_seed_gives_the_same_verdict_and_a_cache_resumes_it():
    x, y = data(seed=5)
    cache = multiverse.ResultCache()
    first = stability.assess_finding(
        "x", x, y, grid(), 7, bootstrap=20, cache=cache
    )
    again = stability.assess_finding(
        "x", x, y, grid(), 7, bootstrap=20, cache=cache
    )
    plain = stability.assess_finding("x", x, y, grid(), 7, bootstrap=20)
    assert first.to_dict() == again.to_dict() == plain.to_dict()
    assert len(cache.items) == 18


def test_the_time_budget_records_the_specifications_it_skipped():
    x, y = data(seed=6)
    ticks = iter(range(1000))
    result = stability.assess_finding(
        "x",
        x,
        y,
        grid(),
        0,
        bootstrap=10,
        budget_seconds=5.5,
        clock=lambda: float(next(ticks)),
    )
    assert result.skipped and len(result.specs) + len(result.skipped) == 18


def test_cluster_bootstrap_draws_whole_groups():
    labels = numpy.repeat(numpy.arange(5), 4)
    rng = numpy.random.default_rng(0)
    pick = stability.resample(20, labels, rng)
    counts = numpy.bincount(labels[pick], minlength=5)
    assert set(counts) <= {0, 4, 8, 12, 16, 20}
    assert len(stability.resample(7, None, rng)) == 7


def test_the_curve_test_separates_a_real_effect_from_noise():
    x, y = data(rows=300, beta=0.8, seed=7)
    real = stability.curve_test(x, y, grid(), 0, "x", permutations=49)
    assert real.p_median <= 0.05 and real.p_share <= 0.05
    rng = numpy.random.default_rng(8)
    noise = stability.curve_test(
        rng.normal(size=300),
        rng.normal(size=300),
        grid(),
        0,
        "n",
        permutations=49,
    )
    assert noise.p_median > 0.05


def test_a_finding_with_a_curve_test_reports_its_p_value_on_the_check():
    x, y = data(rows=300, beta=0.8, seed=9)
    result = stability.assess_finding(
        "x", x, y, grid(), 0, bootstrap=20, curve_permutations=19
    )
    assert result.curve is not None
    assert result.check.p_value == result.curve.p_median


def test_stable_findings_under_a_permuted_target_stay_under_the_floor():
    stable = 0
    for seed in range(20):
        x, y = data(rows=250, beta=0.0, seed=seed, missing=0.1)
        result = stability.assess_finding(
            "x", x, y, grid(seed), seed, bootstrap=30
        )
        stable += result.status == stability.STABLE
    assert stable <= 2


def test_attribution_names_the_choice_that_moves_the_estimate():
    effects = [
        stability.SpecEffect(f"s{k}", {"a": a, "b": b}, v, None, None)
        for k, (a, b, v) in enumerate(
            [
                ("p", "u", 0.1),
                ("p", "v", 0.1),
                ("q", "u", 0.5),
                ("q", "v", 0.5),
            ]
        )
    ]
    result = stability.attribution(effects)
    assert result["a"]["eta_squared"] == pytest.approx(1.0)
    assert result["b"]["eta_squared"] == pytest.approx(0.0)
    assert result["a"]["option_means"] == {"p": 0.1, "q": 0.5}
    assert stability.attribution([]) == {}


def test_gate_status_needs_every_defect_type_current_and_passing():
    assert stability.gate_status(None) is certificates.GateStatus.DESCRIPTIVE
    assert (
        stability.gate_status(certificates.CertificateStore())
        is certificates.GateStatus.DESCRIPTIVE
    )
