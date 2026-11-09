"""Permutation p-values are uniform on same-distribution splits.

The acceptance rule is a Kolmogorov-Smirnov test at level 0.01 on null
replicates, including grouped and temporal data. Here each replicate is
one random split of a single sample into two sides, with labels permuted
within the blocks the split respects.
"""

import numpy
import pandas
from scipy import stats

from scmle.data.robustness import shift

LEVEL = 0.01
RUNS = 200


def iid_split(seed, rows=60):
    rng = numpy.random.default_rng(seed)
    both = pandas.DataFrame(rng.normal(size=(2 * rows, 2)), columns=["a", "b"])
    return both.iloc[:rows], both.iloc[rows:].reset_index(drop=True), None


def grouped_split(seed, groups=8, per=10):
    """Groups share a random effect; every group is split across sides."""
    rng = numpy.random.default_rng(seed)
    effect = numpy.repeat(rng.normal(scale=2.0, size=groups), 2 * per)
    data = effect[:, None] + rng.normal(size=(2 * per * groups, 2))
    both = pandas.DataFrame(data, columns=["a", "b"])
    label = numpy.repeat(numpy.arange(groups), 2 * per)
    side = numpy.zeros(len(both), dtype=bool)
    for g in range(groups):
        rows = numpy.flatnonzero(label == g)
        side[rng.permutation(rows)[:per]] = True
    left, right = both[side], both[~side]
    strata = numpy.r_[label[side], label[~side]]
    return left.reset_index(drop=True), right.reset_index(drop=True), strata


def temporal_split(seed, blocks=8, per=10):
    """An autoregressive series; sides are interleaved inside time blocks."""
    rng = numpy.random.default_rng(seed)
    total = 2 * per * blocks
    noise = rng.normal(size=(total, 2))
    series = numpy.zeros((total, 2))
    for t in range(1, total):
        series[t] = 0.9 * series[t - 1] + noise[t]
    both = pandas.DataFrame(series, columns=["a", "b"])
    label = numpy.repeat(numpy.arange(blocks), 2 * per)
    side = numpy.zeros(total, dtype=bool)
    for b in range(blocks):
        rows = numpy.flatnonzero(label == b)
        side[rng.permutation(rows)[:per]] = True
    strata = numpy.r_[label[side], label[~side]]
    return (
        both[side].reset_index(drop=True),
        both[~side].reset_index(drop=True),
        strata,
    )


def uniform_p(builder, test, runs=RUNS, **options):
    pvalues = []
    for seed in range(runs):
        left, right, strata = builder(seed)
        check = test(
            left,
            right,
            seed,
            permutations=99,
            strata=strata,
            **options,
        )
        pvalues.append(check.p_value)
    return pvalues


def ks_uniform_p(pvalues):
    """KS against the discrete grid (k / 100): use the continuous version
    on jittered ranks so the discreteness of 99 permutations is not read as
    non-uniformity."""
    rng = numpy.random.default_rng(0)
    jittered = (
        numpy.asarray(pvalues) * 100 - rng.uniform(size=len(pvalues))
    ) / 100
    return stats.kstest(jittered, "uniform").pvalue


def test_mmd_p_values_are_uniform_on_exchangeable_splits():
    pvalues = uniform_p(iid_split, shift.mmd_test)
    assert ks_uniform_p(pvalues) > LEVEL


def test_mmd_p_values_are_uniform_on_grouped_data_with_group_strata():
    pvalues = uniform_p(grouped_split, shift.mmd_test)
    assert ks_uniform_p(pvalues) > LEVEL


def test_mmd_p_values_are_uniform_on_temporal_data_with_block_strata():
    pvalues = uniform_p(temporal_split, shift.mmd_test)
    assert ks_uniform_p(pvalues) > LEVEL


def test_ignoring_the_blocks_makes_grouped_p_values_wrong():
    pvalues = []
    for seed in range(RUNS):
        left, right = grouped_split(seed)[:2]
        # Sides are balanced within groups, so a global permutation breaks
        # that balance and the test is too conservative.
        pvalues.append(
            shift.mmd_test(left, right, seed, permutations=99).p_value
        )
    assert ks_uniform_p(pvalues) < LEVEL


def test_c2st_p_values_are_uniform_on_exchangeable_splits():
    pvalues = []
    for seed in range(30):
        left, right = iid_split(seed)[:2]
        pvalues.append(shift.c2st(left, right, seed, permutations=99).p_value)
    assert ks_uniform_p(pvalues) > LEVEL
