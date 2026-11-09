"""Two-sample shift tests: differential checks against scipy and sklearn."""

import numpy
import pandas
import pytest
from scipy import spatial
from scipy import stats
from sklearn import metrics

from scmle.contracts import qualification
from scmle.data.detectors import certificates
from scmle.data.robustness import shift

GATING = certificates.GateStatus.GATING


def frames(rows=120, shift_by=0.0, seed=0, columns=4):
    rng = numpy.random.default_rng(seed)
    names = [f"f{k}" for k in range(columns)]
    left = pandas.DataFrame(rng.normal(size=(rows, columns)), columns=names)
    right = pandas.DataFrame(rng.normal(size=(rows, columns)), columns=names)
    right["f0"] += shift_by
    return left, right


def brute_force_auc(scores, side):
    positive = scores[side == 1]
    negative = scores[side == 0]
    wins = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return (wins + 0.5 * ties) / (len(positive) * len(negative))


def brute_force_delong_variance(scores, side):
    positive = scores[side == 1]
    negative = scores[side == 0]
    psi = (positive[:, None] > negative[None, :]) + 0.5 * (
        positive[:, None] == negative[None, :]
    )
    v10 = psi.mean(axis=1)
    v01 = psi.mean(axis=0)
    return v10.var(ddof=1) / len(positive) + v01.var(ddof=1) / len(negative)


def test_the_auc_matches_sklearn_and_the_pairwise_definition_with_ties():
    rng = numpy.random.default_rng(1)
    scores = numpy.round(rng.normal(size=80), 1)
    side = rng.integers(0, 2, 80)
    auc, interval = shift.auc_delong(scores, side)
    assert auc == pytest.approx(metrics.roc_auc_score(side, scores))
    assert auc == pytest.approx(brute_force_auc(scores, side))
    assert interval[0] <= auc <= interval[1]


def test_the_delong_variance_matches_the_pairwise_definition():
    rng = numpy.random.default_rng(2)
    scores = numpy.round(rng.normal(size=90), 1) + numpy.arange(90) % 2 * 0.4
    side = numpy.arange(90) % 2
    low, high = shift.auc_delong(scores, side, confidence=0.9)[1]
    variance = brute_force_delong_variance(scores, side)
    z = stats.norm.ppf(0.95)
    assert high - low == pytest.approx(2 * z * numpy.sqrt(variance))


def test_delong_rejects_a_class_with_one_row_and_a_bad_level():
    with pytest.raises(ValueError, match="two rows"):
        shift.auc_delong(numpy.array([0.1, 0.2, 0.3]), numpy.array([1, 0, 0]))
    with pytest.raises(ValueError, match="confidence"):
        shift.auc_delong(numpy.arange(4.0), numpy.array([0, 1, 0, 1]), 1.0)


def test_a_perfect_separation_gives_a_degenerate_interval():
    auc, interval = shift.auc_delong(numpy.arange(10.0), numpy.arange(10) // 5)
    assert auc == 1.0 and interval == (1.0, 1.0)


def test_the_permutation_p_value_is_the_add_one_formula():
    null = numpy.array([0.1, 0.2, 0.3, 0.4])
    assert shift.permutation_p_value(0.3, null) == pytest.approx(3 / 5)
    assert shift.permutation_p_value(9.0, null) == pytest.approx(1 / 5)


def test_the_mmd_statistic_matches_the_definition_from_pairwise_distances():
    left, right = frames(rows=40, shift_by=0.5, columns=3)
    both = pandas.concat([left, right], ignore_index=True)
    design = shift.design_matrix(both, len(both))
    bandwidth = shift.median_bandwidth(design)
    kernel = shift.gaussian_kernel(design, bandwidth)
    expected = numpy.exp(
        -spatial.distance.cdist(design, design, "sqeuclidean")
        / (2 * bandwidth**2)
    )
    assert numpy.allclose(kernel, expected)
    indicator = numpy.r_[numpy.ones(40), numpy.zeros(40)]
    got = shift.mmd_statistics(kernel, indicator[:, None])[0]
    xx = expected[:40, :40]
    yy = expected[40:, 40:]
    xy = expected[:40, 40:]
    direct = (
        (xx.sum() - numpy.trace(xx)) / (40 * 39)
        + (yy.sum() - numpy.trace(yy)) / (40 * 39)
        - 2 * xy.mean()
    )
    assert got == pytest.approx(direct)


def test_the_median_heuristic_is_the_median_pairwise_distance():
    rng = numpy.random.default_rng(3)
    design = rng.normal(size=(30, 2))
    assert shift.median_bandwidth(design) == pytest.approx(
        numpy.median(spatial.distance.pdist(design))
    )
    assert shift.median_bandwidth(numpy.zeros((5, 2))) == 1.0


def test_the_mmd_permutation_p_value_agrees_with_scipy_permutation_test():
    left, right = frames(rows=60, shift_by=0.9, columns=2, seed=5)
    mine = shift.mmd_test(left, right, 0, permutations=499)
    both = pandas.concat([left, right], ignore_index=True)
    design = shift.design_matrix(both, len(both))
    kernel = shift.gaussian_kernel(design, shift.median_bandwidth(design))
    index = numpy.arange(len(both), dtype=float)

    def statistic(first, second):
        labels = numpy.isin(index, first).astype(float)
        return shift.mmd_statistics(kernel, labels[:, None])[0]

    reference = stats.permutation_test(
        (index[:60], index[60:]),
        statistic,
        permutation_type="independent",
        n_resamples=999,
        alternative="greater",
        rng=numpy.random.default_rng(0),
    )
    assert mine.statistic == pytest.approx(reference.statistic)
    assert abs(mine.p_value - reference.pvalue) < 0.04


def test_a_clear_shift_is_found_by_both_tests_and_no_shift_is_not():
    left, right = frames(rows=300, shift_by=1.0, seed=7)
    for test in (shift.c2st, shift.mmd_test):
        found = test(left, right, 0, permutations=99)
        assert found.p_value <= 0.05
        assert found.decision is qualification.Decision.WARN
    same_left, same_right = frames(rows=300, seed=8)
    for test in (shift.c2st, shift.mmd_test):
        found = test(same_left, same_right, 0, permutations=99)
        assert found.p_value > 0.05
        assert found.decision is qualification.Decision.PASS


def test_the_checks_are_exploration_checks_with_the_documented_fields():
    left, right = frames(rows=100, shift_by=1.5)
    check = shift.c2st(left, right, 3, permutations=49, gate=GATING)
    assert check.set_id is qualification.SetId.EXPLORATION
    assert check.tier is qualification.Tier.UNVERIFIED
    assert check.check_id == shift.C2ST_ID
    assert check.effect == pytest.approx(check.statistic - 0.5)
    assert check.interval[0] <= check.statistic <= check.interval[1]
    assert check.seed_path[0] == 3
    again = shift.c2st(left, right, 3, permutations=49, gate=GATING)
    assert again == check


def test_a_descriptive_test_never_blocks_and_a_gating_one_does_what_asked():
    left, right = frames(rows=200, shift_by=1.5)
    block = qualification.Decision.BLOCK
    quiet = shift.c2st(left, right, 0, permutations=49, on_shift=block)
    loud = shift.c2st(
        left, right, 0, permutations=49, on_shift=block, gate=GATING
    )
    assert quiet.decision is qualification.Decision.WARN
    assert loud.decision is block


def test_delta_requires_the_auc_lower_bound_to_clear_the_margin():
    left, right = frames(rows=200, shift_by=0.8)
    plain = shift.c2st(left, right, 0, permutations=49)
    strict = shift.c2st(left, right, 0, permutations=49, delta=0.45)
    assert plain.decision is qualification.Decision.WARN
    assert strict.decision is qualification.Decision.PASS
    assert strict.p_value == plain.p_value


def test_strata_permute_labels_within_blocks_only():
    left, right = frames(rows=100, seed=2)
    strata = numpy.r_[numpy.arange(100) % 10, numpy.arange(100) % 10]
    check = shift.c2st(left, right, 0, permutations=49, strata=strata)
    other = shift.mmd_test(left, right, 0, permutations=49, strata=strata)
    assert 0 < check.p_value <= 1 and 0 < other.p_value <= 1


def test_mismatched_columns_and_small_samples_are_refused():
    left, right = frames()
    with pytest.raises(ValueError, match="same columns"):
        shift.c2st(left, right.drop(columns="f0"), 0)
    with pytest.raises(ValueError, match="at least"):
        shift.mmd_test(left.head(10), right, 0)


def test_categorical_and_missing_values_are_handled():
    rng = numpy.random.default_rng(4)
    rows = 150

    def table(weights):
        return pandas.DataFrame(
            {
                "n": numpy.where(
                    rng.random(rows) < 0.1, numpy.nan, rng.normal(size=rows)
                ),
                "c": rng.choice(["a", "b", "c"], rows, p=weights),
            }
        )

    left, right = table([0.8, 0.1, 0.1]), table([0.1, 0.1, 0.8])
    assert shift.c2st(left, right, 0, permutations=49).p_value <= 0.05
    assert shift.mmd_test(left, right, 0, permutations=49).p_value <= 0.05


def test_feature_tests_match_scipy_and_adjust_with_bh():
    left, right = frames(rows=200, shift_by=0.7, seed=9)
    left["g"] = numpy.where(numpy.arange(200) % 2 == 0, "u", "v")
    right["g"] = numpy.where(numpy.arange(200) % 3 == 0, "u", "v")
    rows = {r.column: r for r in shift.feature_tests(left, right)}
    ks = stats.ks_2samp(left["f0"], right["f0"])
    assert rows["f0"].statistic == pytest.approx(ks.statistic)
    assert rows["f0"].p_value == pytest.approx(ks.pvalue)
    table = pandas.crosstab(
        pandas.concat([left["g"], right["g"]]),
        numpy.r_[numpy.zeros(200), numpy.ones(200)],
    ).to_numpy()
    chi = stats.chi2_contingency(table, correction=False)
    assert rows["g"].statistic == pytest.approx(chi.statistic)
    assert rows["g"].p_value == pytest.approx(chi.pvalue)
    assert rows["g"].effect == pytest.approx(numpy.sqrt(chi.statistic / 400))
    adjusted = stats.false_discovery_control(
        [r.p_value for r in rows.values()], method="bh"
    )
    assert [r.adjusted_p_value for r in rows.values()] == pytest.approx(
        list(adjusted)
    )
    assert rows["f0"].rejected and not rows["f3"].rejected


def test_a_column_with_one_level_and_an_empty_column_have_p_one():
    left = pandas.DataFrame({"c": ["a"] * 60, "n": [numpy.nan] * 60})
    right = pandas.DataFrame({"c": ["a"] * 60, "n": [numpy.nan] * 60})
    rows = {r.column: r for r in shift.feature_tests(left, right)}
    assert rows["c"].p_value == 1.0 and rows["n"].p_value == 1.0


def test_no_reference_is_unverifiable_never_pass():
    left = frames()[0]
    result = shift.assess(left, None, 0)
    assert result.status is shift.Status.UNVERIFIABLE
    assert result.c2st is None and result.agreement is None
    assert "no deployment sample" in result.reason


def test_a_reference_without_shared_columns_or_rows_is_unverifiable():
    left, right = frames()
    other = right.rename(columns=lambda name: "z" + name)
    assert shift.assess(left, other, 0).status is shift.Status.UNVERIFIABLE
    small = shift.assess(left, right.head(5), 0)
    assert small.status is shift.Status.UNVERIFIABLE
    assert "at least" in small.reason


def test_assess_reports_fail_pass_and_whether_the_tests_agree():
    left, right = frames(rows=300, shift_by=1.0, seed=1)
    failed = shift.assess(left, right, 0, permutations=99)
    assert failed.status is shift.Status.FAIL and failed.agreement is True
    assert failed.features[0].column == "f0" and failed.features[0].rejected
    same_left, same_right = frames(rows=300, seed=2)
    passed = shift.assess(same_left, same_right, 0, permutations=99)
    assert passed.status is shift.Status.PASS
    record = passed.to_dict()
    assert record["status"] == "pass" and record["c2st"]["check_id"]


def test_the_digest_is_a_sha256_of_the_module():
    assert len(shift.code_digest()) == 64
