"""The closed-form power and sample-size formulas against simulated power.

`scmle.data.charter.feasibility` states how many confirmation rows promise a
given power at the Bonferroni level `alpha / K`. Each formula is applied at
its own required size, data are simulated with exactly that effect, and the
share of rejections at the Bonferroni level must be within 3.5 binomial
standard errors of the promised 0.80.
"""

import math

import numpy
from scipy import stats
import support_statistics

from scmle.data.charter import feasibility

ALPHA = 0.05
POWER = 0.8


def critical_value(claims):
    return float(stats.norm.ppf(1.0 - ALPHA / (2 * claims)))


def test_mean_effect_size_gives_the_promised_power():
    rng = numpy.random.default_rng(0)
    runs = 4000
    for delta, claims in [(0.2, 1), (0.2, 5), (0.3, 10)]:
        rows = feasibility.n_required_mean(
            delta, alpha=ALPHA, k_max=claims, power=POWER
        )
        z = rng.normal(delta, 1.0, (runs, rows)).mean(axis=1) * math.sqrt(rows)
        observed = float((numpy.abs(z) > critical_value(claims)).mean())
        support_statistics.assert_rate_near(
            observed, POWER, runs, f"mean effect {delta}, K={claims}"
        )
        closed = feasibility.power_mean(delta, rows, alpha=ALPHA, k_max=claims)
        support_statistics.assert_rate_near(
            observed, closed, runs, f"power_mean {delta}, K={claims}"
        )


def test_correlation_size_gives_the_promised_power():
    rng = numpy.random.default_rng(1)
    runs = 3000
    for correlation, claims in [(0.2, 1), (0.3, 5)]:
        rows = feasibility.n_required_correlation(
            correlation, alpha=ALPHA, k_max=claims, power=POWER
        )
        rejected = 0
        for ignored in range(runs):
            x = rng.normal(size=rows)
            noise = rng.normal(size=rows)
            y = correlation * x + math.sqrt(1.0 - correlation**2) * noise
            z = math.atanh(numpy.corrcoef(x, y)[0, 1]) * math.sqrt(rows - 3)
            rejected += abs(z) > critical_value(claims)
        support_statistics.assert_rate_near(
            rejected / runs,
            POWER,
            runs,
            f"correlation {correlation}, K={claims}",
        )


def test_two_proportion_size_gives_the_promised_power():
    rng = numpy.random.default_rng(2)
    runs = 3000
    for first, second, claims in [(0.3, 0.4, 1), (0.1, 0.15, 3)]:
        rows = feasibility.n_required_two_proportions(
            first, second, alpha=ALPHA, k_max=claims, power=POWER
        )
        hits_first = rng.binomial(rows, first, runs)
        hits_second = rng.binomial(rows, second, runs)
        pooled = (hits_first + hits_second) / (2.0 * rows)
        z = (
            (hits_second - hits_first)
            / rows
            / numpy.sqrt(2.0 * pooled * (1.0 - pooled) / rows)
        )
        observed = float((numpy.abs(z) > critical_value(claims)).mean())
        support_statistics.assert_rate_near(
            observed, POWER, runs, f"proportions {first}/{second}, K={claims}"
        )
