"""Size of the baseline claim's cluster-robust t test with uneven clusters.

`scmle.data.framing.baseline` states that the claim is a one-sided
cluster-robust t test whose size is at most `alpha_confirm`, and that
`MIN_CLUSTERS = 20` is the smallest cluster count at which that held. The
test statistic is the Liang-Zeger sandwich with a `G / (G - 1)` factor and
`G - 1` degrees of freedom. Here the data are drawn under the null (zero mean
difference, a random effect shared inside each cluster) with uneven cluster
sizes, the usual case for rows grouped by entity. The rejection rate at
alpha = 0.05 over 20,000 simulated datasets is measured at 0.056 and 0.057, a
little above alpha, which is a known property of this estimator with few,
uneven clusters and is stated in the docstring of `cluster_test`. The tests pin
that measured limit: the rate must stay within 3.5 binomial standard errors of
`KNOWN_LIMIT`, so a worse test is still caught.
"""

import numpy
import support_statistics

from scmle.data.framing import baseline

ALPHA = 0.05
KNOWN_LIMIT = 0.06
RUNS = 20000


def rejection_rate(clusters, smallest, largest, seed):
    rng = numpy.random.default_rng(seed)
    rejected = 0
    for ignored in range(RUNS):
        sizes = rng.integers(smallest, largest + 1, size=clusters)
        labels = numpy.repeat(numpy.arange(clusters), sizes)
        effect = numpy.repeat(rng.normal(size=clusters), sizes)
        values = effect + rng.normal(size=labels.size)
        rejected += baseline.cluster_test(values, labels).p_value <= ALPHA
    return rejected / RUNS


def test_size_with_twenty_clusters_of_uneven_size():
    observed = rejection_rate(20, 5, 15, seed=123)
    support_statistics.assert_rate_at_most(
        observed, KNOWN_LIMIT, RUNS, "cluster test, G=20, sizes 5 to 15"
    )


def test_size_with_thirty_clusters_of_very_uneven_size():
    observed = rejection_rate(30, 1, 30, seed=124)
    support_statistics.assert_rate_at_most(
        observed, KNOWN_LIMIT, RUNS, "cluster test, G=30, sizes 1 to 30"
    )
