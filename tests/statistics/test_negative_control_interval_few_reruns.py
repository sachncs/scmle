"""Coverage of the negative control's 95th-percentile interval at few re-runs.

`controls.percentile_interval` documents itself as an approximate
distribution-free interval for the 0.95 quantile with nominal coverage 0.90
(`negative_control.json` reports `p95_interval_confidence: 0.9`). The
smallest allowed control has 20 re-runs, and the docstring says that with few
repeats the interval "is wide and reaches the largest count; that is the
honest statement of the error". For a continuous distribution the largest of
`B` draws lies above the 0.95 quantile with probability only `1 - 0.95 ** B`,
which is 0.64 for `B = 20` and 0.78 for `B = 30`, so no interval capped at the
largest draw can reach 0.90 there. This simulates the interval on normal
draws, whose 0.95 quantile is known, and checks the coverage against the
stated 0.90 with the same 3.5 standard errors as every other interval.
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


def test_the_interval_with_the_minimum_twenty_re_runs_covers_what_it_claims():
    support_statistics.assert_rate_at_least(
        coverage(20, seed=2),
        controls.INTERVAL_CONFIDENCE,
        RUNS,
        "p95 interval, normal draws, B=20",
    )


def test_the_interval_with_thirty_re_runs_covers_what_it_claims():
    support_statistics.assert_rate_at_least(
        coverage(30, seed=3),
        controls.INTERVAL_CONFIDENCE,
        RUNS,
        "p95 interval, normal draws, B=30",
    )
