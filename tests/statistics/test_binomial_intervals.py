"""Wilson and Clopper-Pearson intervals: endpoints and coverage.

Every implementation in the package is compared with
`scipy.stats.binomtest(...).proportion_ci` on identical inputs, to 1e-9, and
its coverage is measured by simulation from a known true proportion, with
the observed rate within 3.5 binomial standard errors of the target.
"""

import numpy
import pytest
from scipy import stats
import support_statistics

from scmle.data.charter import feasibility
from scmle.data.detectors import certificates
from scmle.data.detectors import duplicates
from scmle.data.framing import refusal
from scmle.decision import constraints
from scmle.evaluation import coverage

TOLERANCE = 1e-9
LEVEL = 0.95
SIMULATIONS = 2000
CASES = [(1, 10), (3, 10), (10, 10), (0, 10), (17, 60), (120, 400), (399, 400)]


def reference(successes, trials, method):
    result = stats.binomtest(successes, trials).proportion_ci(
        LEVEL, method=method
    )
    return float(result.low), float(result.high)


@pytest.mark.parametrize(("successes", "trials"), CASES)
def test_wilson_endpoints_match_scipy(successes, trials):
    expected = reference(successes, trials, "wilson")
    z = float(stats.norm.ppf(0.975))
    found = {
        "certificates": certificates.wilson_interval(successes, trials),
        "constraints": constraints.wilson(successes, trials, z),
    }
    for name, (low, high) in found.items():
        assert low == pytest.approx(expected[0], abs=TOLERANCE), name
        assert high == pytest.approx(expected[1], abs=TOLERANCE), name


@pytest.mark.parametrize(("successes", "trials"), CASES)
def test_wilson_half_width_matches_scipy(successes, trials):
    low, high = reference(successes, trials, "wilson")
    half = feasibility.wilson_half_width(successes / trials, trials)
    assert half == pytest.approx((high - low) / 2.0, abs=TOLERANCE)


@pytest.mark.parametrize(("successes", "trials"), CASES)
def test_clopper_pearson_endpoints_match_scipy(successes, trials):
    expected = reference(successes, trials, "exact")
    found = {
        "coverage": coverage.clopper_pearson(successes, trials, LEVEL),
        "duplicates": duplicates.clopper_pearson(successes, trials, LEVEL),
    }
    for name, (low, high) in found.items():
        assert low == pytest.approx(expected[0], abs=TOLERANCE), name
        assert high == pytest.approx(expected[1], abs=TOLERANCE), name


@pytest.mark.parametrize(("errors", "trials"), [(0, 25), (2, 25), (30, 200)])
def test_one_sided_upper_bound_is_the_two_sided_bound_at_double_tail(
    errors, trials
):
    # An upper bound at level 0.95 is the upper end of the 0.90 two-sided one.
    expected = stats.binomtest(errors, trials).proportion_ci(0.90).high
    found = refusal.clopper_pearson_upper(errors, trials, 0.95)
    assert found == pytest.approx(float(expected), abs=TOLERANCE)


def exact_coverage(interval, trials, proportion):
    """Probability that the interval from a Binomial draw holds the truth."""
    counts = numpy.arange(trials + 1)
    mass = stats.binom.pmf(counts, trials, proportion)
    holds = numpy.array(
        [
            low <= proportion <= high
            for low, high in (interval(int(k), trials) for k in counts)
        ]
    )
    return float(mass[holds].sum())


@pytest.mark.parametrize("trials", [5, 20, 100])
def test_clopper_pearson_never_covers_below_nominal(trials):
    # Clopper-Pearson is exact: coverage is at least 0.95 at every p.
    def interval(count, size):
        return coverage.clopper_pearson(count, size, LEVEL)

    for proportion in numpy.linspace(0.02, 0.98, 49):
        found = exact_coverage(interval, trials, float(proportion))
        assert found >= LEVEL - 1e-9, (trials, proportion, found)


@pytest.mark.parametrize("trials", [100, 400])
def test_wilson_exact_coverage_stays_near_nominal(trials):
    # Wilson is approximate; Brown, Cai and DasGupta (2001) show its coverage
    # oscillates around nominal. The stated tolerance is 0.03 for n >= 100
    # and p in [0.1, 0.9].
    def interval(count, size):
        return certificates.wilson_interval(count, size)

    for proportion in numpy.linspace(0.1, 0.9, 33):
        found = exact_coverage(interval, trials, float(proportion))
        assert abs(found - LEVEL) <= 0.03, (trials, proportion, found)


def simulated_coverage(interval, trials, proportion, seed):
    rng = numpy.random.default_rng(seed)
    draws = rng.binomial(trials, proportion, SIMULATIONS)
    cache = {}
    hits = 0
    for count in draws.tolist():
        if count not in cache:
            low, high = interval(count, trials)
            cache[count] = low <= proportion <= high
        hits += cache[count]
    return hits / SIMULATIONS


@pytest.mark.parametrize(("trials", "proportion"), [(200, 0.5), (1000, 0.3)])
def test_wilson_coverage_is_near_nominal_in_simulation(trials, proportion):
    for name, interval in {
        "certificates": certificates.wilson_interval,
        "constraints": lambda k, n: constraints.wilson(
            k, n, float(stats.norm.ppf(0.975))
        ),
    }.items():
        observed = simulated_coverage(interval, trials, proportion, seed=11)
        support_statistics.assert_rate_near(
            observed, LEVEL, SIMULATIONS, f"wilson {name} n={trials}"
        )


@pytest.mark.parametrize(
    ("trials", "proportion"), [(30, 0.3), (200, 0.5), (500, 0.02)]
)
def test_clopper_pearson_coverage_is_at_least_nominal_in_simulation(
    trials, proportion
):
    for name, interval in {
        "coverage": lambda k, n: coverage.clopper_pearson(k, n, LEVEL),
        "duplicates": lambda k, n: duplicates.clopper_pearson(k, n, LEVEL),
    }.items():
        observed = simulated_coverage(interval, trials, proportion, seed=12)
        support_statistics.assert_rate_at_least(
            observed, LEVEL, SIMULATIONS, f"clopper-pearson {name}"
        )


@pytest.mark.parametrize(("trials", "proportion"), [(25, 0.1), (200, 0.15)])
def test_one_sided_upper_bound_covers_at_least_nominal(trials, proportion):
    def interval(count, size):
        return 0.0, refusal.clopper_pearson_upper(count, size, LEVEL)

    observed = simulated_coverage(interval, trials, proportion, seed=13)
    support_statistics.assert_rate_at_least(
        observed, LEVEL, SIMULATIONS, "one-sided upper bound"
    )
