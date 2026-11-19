"""The duplicate-row test keeps its size on sparse tails."""

import math

import numpy
from scipy import special

from scmle.assurance.admissibility.screen import records


def boundary(singles: float, triples: float, rows: int) -> float:
    """The pairs' mean on the Schwarz boundary.

    Args:
        singles: Mean of the singles.
        triples: Mean of the triples.
        rows: Rows counted.

    Returns:
        The largest mean of the pairs a sampled population allows.
    """
    c = records.SCHWARZ * (rows - 1) / (rows - 2)
    return math.sqrt(c * singles * triples)


def test_the_exact_p_value_is_the_conditional_tail_on_the_boundary():
    singles, pairs, triples, rows = 322, 118, 14, 777
    c = records.SCHWARZ * (rows - 1) / (rows - 2)
    upper, lower = 2 * singles + pairs, 2 * triples + pairs
    j = numpy.arange(pairs % 2, min(upper, lower) + 1, 2)
    weights = (
        0.5 * j * math.log(c)
        - special.gammaln((upper - j) / 2 + 1)
        - special.gammaln(j + 1)
        - special.gammaln((lower - j) / 2 + 1)
    )
    tail = special.logsumexp(weights[j >= pairs])
    expected = math.exp(tail - special.logsumexp(weights))
    assert records.exact(singles, pairs, triples, c) == expected
    assert records.excess(singles, pairs, triples, rows)[0] == expected


def test_the_test_keeps_its_size_when_triples_are_rare():
    # Counts on the boundary of the null with a sparse tail (a mean of 4
    # triples) is where the half chi-square rejected 1.9 percent at 0.01.
    rng = numpy.random.default_rng(0)
    rows = 10**6
    means = (330.0, boundary(330.0, 4.0, rows), 4.0)
    draws = rng.poisson(means, size=(2000, 3))
    chances = numpy.array([records.excess(*d, rows)[0] for d in draws])
    for level in (0.05, 0.01):
        # Three binomial standard errors above the level.
        slack = 3 * math.sqrt(level * (1 - level) / len(draws))
        assert (chances < level).mean() <= level + slack


def test_copies_of_unique_rows_are_still_found():
    assert records.excess(2800, 1200, 0, 5200)[0] < 1e-12
    assert records.excess(420, 180, 0, 780)[0] < 1e-12
