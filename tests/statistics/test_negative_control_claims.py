"""The negative-control gate and its Monte Carlo interval.

`scmle.data.ledger.controls` promises three things about re-run counts
(`B` permuted-target discovery counts) and an observed count:

* if the observed count is exchangeable with the re-run counts, the gate
  `observed > 95th percentile` fires with probability at most 1/21 for
  `B = 20` and at most 0.05 in general;
* the exceedance p-value `(1 + #{permuted >= observed}) / (B + 1)` is valid;
* the 95th percentile has an approximate distribution-free interval whose
  nominal coverage, reported in `negative_control.json`, is 0.90.

All three are simulated with exchangeable draws from a known distribution.
Counts are integers (Poisson) and continuous values (normal); the interval
claim is distribution-free, so it must hold for both.
"""

import numpy
from scipy import stats
import support_statistics

from scmle.data.ledger import controls

RUNS = 4000


def draws(rng, kind, size):
    if kind == "poisson":
        return rng.poisson(8.0, size)
    return rng.normal(size=size)


def test_the_gate_fires_at_most_as_often_as_promised_when_exchangeable():
    rng = numpy.random.default_rng(0)
    for kind in ("poisson", "normal"):
        for reruns, bound in [(20, 1 / 21), (99, 0.05)]:
            fired = pvalue_rejections = 0
            for ignored in range(RUNS):
                sample = draws(rng, kind, reruns + 1)
                observed, counts = sample[0], tuple(sample[1:].tolist())
                fired += observed > controls.percentile(counts)
                exceed = sum(1 for c in counts if c >= observed)
                pvalue_rejections += (1 + exceed) / (reruns + 1) <= 0.05
            support_statistics.assert_rate_at_most(
                fired / RUNS, bound, RUNS, f"gate {kind} B={reruns}"
            )
            support_statistics.assert_rate_at_most(
                pvalue_rejections / RUNS,
                0.05,
                RUNS,
                f"exceedance p {kind} B={reruns}",
            )


def interval_coverage(rng, kind, reruns, runs):
    truth = (
        float(stats.poisson.ppf(controls.QUANTILE, 8.0))
        if kind == "poisson"
        else float(stats.norm.ppf(controls.QUANTILE))
    )
    hits = 0
    for ignored in range(runs):
        counts = tuple(draws(rng, kind, reruns).tolist())
        low, high = controls.percentile_interval(counts)
        hits += low <= truth <= high
    return hits / runs


def test_the_interval_reaches_its_nominal_coverage_with_enough_re_runs():
    rng = numpy.random.default_rng(1)
    for kind in ("poisson", "normal"):
        for reruns in (100, 200):
            observed = interval_coverage(rng, kind, reruns, RUNS)
            support_statistics.assert_rate_at_least(
                observed,
                controls.INTERVAL_CONFIDENCE,
                RUNS,
                f"p95 interval {kind} B={reruns}",
            )
