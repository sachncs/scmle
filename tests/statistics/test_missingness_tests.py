"""Size of Little's MCAR test under data that are missing completely at random.

`little_mcar_test` is the package's own implementation (EM estimate, pattern
chi-square). With multivariate normal data and values deleted at random it
must reject at level alpha no more often than alpha plus 3.5 binomial
standard errors, for independent and for correlated columns, at the module's
own level and at 0.05.
"""

import numpy
import pandas
import support_statistics

from scmle.data.detectors import missingness

RUNS = 1000


def mcar_p_value(rng, rho):
    covariance = numpy.full((3, 3), rho) + (1.0 - rho) * numpy.eye(3)
    values = rng.multivariate_normal(numpy.zeros(3), covariance, 200)
    gaps = rng.random(values.shape) < 0.2
    gaps[:, 0] = False
    values[gaps] = numpy.nan
    result = missingness.little_mcar_test(
        pandas.DataFrame(values, columns=["a", "b", "c"])
    )
    return None if result is None else result.p_value


def test_little_mcar_test_has_the_nominal_size_when_data_are_mcar():
    rng = numpy.random.default_rng(0)
    for rho in (0.0, 0.5):
        p_values = [mcar_p_value(rng, rho) for ignored in range(RUNS)]
        tested = [p for p in p_values if p is not None]
        assert len(tested) > 0.95 * RUNS
        for level in (missingness.ALPHA, 0.05):
            rejected = sum(p <= level for p in tested)
            support_statistics.assert_rate_at_most(
                rejected / len(tested),
                level,
                len(tested),
                f"little mcar rho={rho} level={level}",
            )
