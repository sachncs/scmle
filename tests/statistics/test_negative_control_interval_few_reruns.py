"""Coverage of the negative control's 95th-percentile interval.

`controls.percentile_interval` builds a distribution-free interval for the
0.95 quantile with nominal coverage 0.90. For a continuous distribution the
largest of `B` draws lies above the 0.95 quantile with probability only
`1 - 0.95 ** B`, which is 0.64 for `B = 20` and 0.78 for `B = 30`, so an
interval capped at the largest draw cannot reach 0.90 with few re-runs. The
control therefore reports the coverage it really has, `interval_coverage(b)`.
These tests simulate the interval on normal draws, whose 0.95 quantile is
known, and check that the reported coverage is the measured one, with the same
3.5 standard errors as every other interval, and that enough re-runs do reach
the nominal level.
"""

import numpy
from scipy import stats
import support_statistics

from scmle.data.ledger import controls

RUNS = 4000


def coverage(reruns, seed):
    rng = numpy.random.default_rng(seed)
    truth = float(stats.norm.ppf(controls.QUANTILE))
    hits = 0
    for ignored in range(RUNS):
        counts = tuple(rng.normal(size=reruns).tolist())
        low, high = controls.percentile_interval(counts)
        hits += low <= truth <= high
    return hits / RUNS


def test_the_reported_coverage_with_twenty_re_runs_is_the_measured_one():
    support_statistics.assert_rate_near(
        coverage(20, seed=2),
        controls.interval_coverage(20),
        RUNS,
        "p95 interval, normal draws, B=20",
    )


def test_the_reported_coverage_with_thirty_re_runs_is_the_measured_one():
    support_statistics.assert_rate_near(
        coverage(30, seed=3),
        controls.interval_coverage(30),
        RUNS,
        "p95 interval, normal draws, B=30",
    )


def test_a_hundred_re_runs_cover_what_the_interval_claims():
    support_statistics.assert_rate_at_least(
        coverage(100, seed=4),
        controls.INTERVAL_CONFIDENCE,
        RUNS,
        "p95 interval, normal draws, B=100",
    )
    assert controls.interval_coverage(100) >= controls.INTERVAL_CONFIDENCE


def test_few_re_runs_cannot_reach_the_nominal_confidence():
    assert controls.interval_coverage(20) < controls.INTERVAL_CONFIDENCE - 0.2
    assert controls.interval_coverage(30) < controls.INTERVAL_CONFIDENCE - 0.1


def test_the_coverage_of_a_non_positive_number_of_re_runs_is_refused():
    try:
        controls.interval_coverage(0)
    except ValueError:
        return
    raise AssertionError("interval_coverage(0) did not raise")
