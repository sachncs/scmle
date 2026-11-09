"""Size of the hypothesis tests the package implements itself.

Each test has two parts. Identical inputs go to the repository function and
to `scipy.stats` and the p-values must agree to the stated tolerance. Then
data are simulated under the null and the rejection rate at alpha = 0.05
must not exceed alpha by more than 3.5 binomial standard errors. Discrete
tests are conservative, so only the upper side is checked.
"""

import math

import numpy
import pandas
from scipy import stats
import support_statistics

from scmle.data.admissibility import leakage
from scmle.data.framing import baseline
from scmle.data.ledger import bivariate
from scmle.data.robustness import shift
from scmle.decision import design
from scmle.evaluation import balance
from scmle.evaluation import coverage
from scmle.evaluation import coverage_confirm
from scmle.evaluation import exchangeability
from scmle.governance import escalation

ALPHA = 0.05
AGREEMENT = 1e-9
RUNS = 4000


def test_spearman_p_values_match_scipy_in_every_direction():
    rng = numpy.random.default_rng(0)
    for ignored in range(200):
        x = rng.normal(size=int(rng.integers(5, 60)))
        y = 0.3 * x + rng.normal(size=len(x))
        for direction in bivariate.DIRECTIONS:
            found = bivariate.spearman(x, y, direction).p
            expected = stats.spearmanr(x, y, alternative=direction).pvalue
            assert abs(found - float(expected)) <= AGREEMENT, direction


def test_spearman_has_the_nominal_size_with_continuous_and_tied_data():
    rng = numpy.random.default_rng(1)
    for name, draw in {
        "continuous": lambda: (rng.normal(size=30), rng.normal(size=30)),
        "ties": lambda: (
            rng.integers(0, 3, size=40),
            rng.integers(0, 2, size=40),
        ),
    }.items():
        for direction in bivariate.DIRECTIONS:
            rejected = 0
            for ignored in range(RUNS):
                x, y = draw()
                rejected += bivariate.spearman(x, y, direction).p <= ALPHA
            support_statistics.assert_rate_at_most(
                rejected / RUNS, ALPHA, RUNS, f"spearman {name} {direction}"
            )


def test_kruskal_matches_scipy_and_has_the_nominal_size():
    rng = numpy.random.default_rng(2)
    rejected = 0
    for ignored in range(RUNS):
        x = rng.normal(size=60)
        labels = rng.integers(0, 3, size=60)
        found = bivariate.kruskal(x, labels).p
        expected = stats.kruskal(*[x[labels == k] for k in range(3)]).pvalue
        assert abs(found - float(expected)) <= AGREEMENT
        rejected += found <= ALPHA
    support_statistics.assert_rate_at_most(
        rejected / RUNS, ALPHA, RUNS, "kruskal"
    )


def test_t_tail_probabilities_match_scipy():
    for statistic in (-3.1, -0.4, 0.0, 1.7, 4.2):
        for freedom in (3, 12, 80):
            two = bivariate.tail_probability(statistic, freedom, "two-sided")
            up = bivariate.tail_probability(statistic, freedom, "greater")
            down = bivariate.tail_probability(statistic, freedom, "less")
            assert abs(two - 2 * stats.t.sf(abs(statistic), freedom)) <= 1e-12
            assert abs(up - stats.t.sf(statistic, freedom)) <= 1e-12
            assert abs(down - stats.t.cdf(statistic, freedom)) <= 1e-12


def test_binomial_p_values_match_scipy_exact_tests():
    for covered, rows, rate in [(8, 10, 0.9), (88, 100, 0.9), (430, 500, 0.9)]:
        less = stats.binomtest(covered, rows, rate, alternative="less").pvalue
        greater = stats.binomtest(
            covered, rows, rate, alternative="greater"
        ).pvalue
        two = stats.binomtest(covered, rows, rate).pvalue
        found = {
            "escalation": escalation.binomial_p_value(covered, rows, rate),
            "undercoverage": coverage.undercoverage_p_value(
                covered, rows, rate
            ),
        }
        for name, value in found.items():
            assert abs(value - float(less)) <= AGREEMENT, name
        floor = coverage_confirm.floor_p_value(covered, rows, rate)
        assert abs(floor - float(greater)) <= AGREEMENT
        assert abs(design.srm_p_value(covered, rows, rate) - two) <= AGREEMENT


def test_exact_binomial_tests_never_exceed_their_level_at_the_boundary():
    rng = numpy.random.default_rng(3)
    rows, rate = 400, 0.88
    counts = rng.binomial(rows, rate, RUNS)
    tests = {
        "undercoverage": lambda c: coverage.undercoverage_p_value(
            c, rows, rate
        ),
        "escalation": lambda c: escalation.binomial_p_value(c, rows, rate),
        "floor": lambda c: coverage_confirm.floor_p_value(c, rows, rate),
        "srm": lambda c: design.srm_p_value(c, rows, rate),
    }
    for name, test in tests.items():
        rejected = sum(test(int(c)) <= ALPHA for c in counts)
        support_statistics.assert_rate_at_most(
            rejected / RUNS, ALPHA, RUNS, f"exact binomial {name}"
        )


def test_the_coverage_claim_is_confirmed_at_most_alpha_of_the_time_at_the_floor():
    rng = numpy.random.default_rng(4)
    choice = {"level": 0.9, "tolerance": 0.02}
    rows = 500
    counts = rng.binomial(rows, 0.88, RUNS)
    confirmed = sum(
        coverage_confirm.decide(int(c), rows, choice)["status"]
        == coverage_confirm.CONFIRMED
        for c in counts
    )
    support_statistics.assert_rate_at_most(
        confirmed / RUNS, ALPHA, RUNS, "coverage claim confirmed at the floor"
    )


def test_permutation_p_values_match_scipy_on_the_same_null_draws():
    rng = numpy.random.default_rng(5)
    for ignored in range(20):
        first = rng.normal(size=15)
        second = rng.normal(size=15) + 0.5
        result = stats.permutation_test(
            (first, second),
            lambda a, b, axis: (
                numpy.mean(a, axis=axis) - numpy.mean(b, axis=axis)
            ),
            n_resamples=199,
            alternative="less",
            random_state=int(rng.integers(1 << 30)),
        )
        # `less`: the observed statistic against the null, signs flipped
        observed = -float(result.statistic)
        null = -numpy.asarray(result.null_distribution)
        expected = float(result.pvalue)
        assert abs(leakage.permutation_p(observed, null) - expected) <= 1e-12
        found = shift.permutation_p_value(observed, null)
        assert abs(found - expected) <= 1e-12


def test_permutation_p_values_are_valid_under_exchangeable_nulls():
    rng = numpy.random.default_rng(6)
    permutations = 99
    for name, function in {
        "leakage": leakage.permutation_p,
        "shift": shift.permutation_p_value,
    }.items():
        rejected = 0
        for ignored in range(RUNS):
            draws = rng.normal(size=permutations + 1)
            rejected += function(draws[0], draws[1:]) <= ALPHA
        support_statistics.assert_rate_at_most(
            rejected / RUNS, ALPHA, RUNS, f"permutation p {name}"
        )


def test_the_exchangeability_flag_is_raised_at_most_alpha_of_the_time():
    rng = numpy.random.default_rng(7)
    runs = 2000
    flagged = 0
    for ignored in range(runs):
        first = rng.normal(size=300)
        second = rng.normal(size=300)
        report = exchangeability.compare(first, second)
        flagged += report["flag"]
        assert report["ks_p_value"] == stats.ks_2samp(first, second).pvalue
        assert (
            report["mann_whitney_p_value"]
            == stats.mannwhitneyu(first, second).pvalue
        )
    support_statistics.assert_rate_at_most(
        flagged / runs, ALPHA, runs, "exchangeability flag"
    )


def test_the_balance_test_has_the_nominal_size_under_random_assignment():
    rng = numpy.random.default_rng(8)
    runs = 1000
    rejected = 0
    for run in range(runs):
        features = pandas.DataFrame(
            rng.normal(size=(100, 3)), columns=["a", "b", "c"]
        )
        arm = rng.permutation(numpy.r_[numpy.ones(50), numpy.zeros(50)])
        outcome = balance.balance_test(features, arm.astype(int), seed=run)
        rejected += outcome["p_value"] <= ALPHA
    support_statistics.assert_rate_at_most(
        rejected / runs, ALPHA, runs, "balance test"
    )


def test_the_sample_ratio_check_is_valid_however_often_it_is_read():
    rng = numpy.random.default_rng(9)
    runs, steps, probability = 4000, 3000, 0.5
    counts = numpy.cumsum(rng.random((runs, steps)) < probability, axis=1)
    totals = numpy.arange(1, steps + 1)[None, :]
    evidence = design.srm_log_bayes_factor(counts, totals, probability)
    crossed = (evidence.max(axis=1) >= -math.log(ALPHA)).mean()
    support_statistics.assert_rate_at_most(
        float(crossed), ALPHA, runs, "anytime sample-ratio check"
    )


def test_the_cluster_robust_t_test_is_a_t_test_on_cluster_means_when_equal():
    rng = numpy.random.default_rng(10)
    clusters, size = 25, 6
    labels = numpy.repeat(numpy.arange(clusters), size)
    for shift_size in (0.0, 0.1, 0.4):
        values = rng.normal(size=clusters * size) + shift_size
        found = baseline.cluster_test(values, labels)
        means = values.reshape(clusters, size).mean(axis=1)
        expected = stats.ttest_1samp(means, 0.0, alternative="greater")
        assert abs(found.p_value - float(expected.pvalue)) <= AGREEMENT
        assert abs(found.statistic - float(expected.statistic)) <= 1e-9


def test_the_cluster_robust_test_has_the_nominal_size_with_equal_clusters():
    rng = numpy.random.default_rng(11)
    runs = 20000
    for clusters in (20, 50):
        labels = numpy.repeat(numpy.arange(clusters), 5)
        rejected = 0
        for ignored in range(runs):
            effect = numpy.repeat(rng.normal(size=clusters), 5)
            values = effect + rng.normal(size=clusters * 5)
            rejected += baseline.cluster_test(values, labels).p_value <= ALPHA
        support_statistics.assert_rate_at_most(
            rejected / runs, ALPHA, runs, f"cluster test, G={clusters}"
        )
