"""Bivariate statistics against the frozen scipy oracles and by hand."""

import math

import numpy
import pytest
from scipy import stats

from benchmarks.qualification import oracles
from scmle.data.ledger import bivariate

REL = 1e-8


def test_spearman_matches_the_frozen_scipy_oracle():
    for case in oracles.load_oracle("spearmanr").cases:
        x = numpy.array(case["inputs"]["x"])
        y = numpy.array(case["inputs"]["y"])
        out = bivariate.spearman(x, y)
        oracles.assert_agrees(
            {"pvalue": out.p, "statistic": out.statistic},
            case["expected"],
            case["id"],
            rel_tol=REL,
        )


def test_kruskal_matches_the_frozen_scipy_oracle():
    for case in oracles.load_oracle("kruskal").cases:
        groups = case["inputs"]["groups"]
        x = numpy.concatenate([numpy.array(g) for g in groups])
        labels = numpy.concatenate(
            [numpy.full(len(g), i) for i, g in enumerate(groups)]
        )
        out = bivariate.kruskal(x, labels)
        oracles.assert_agrees(
            {"pvalue": out.p, "statistic": out.statistic},
            case["expected"],
            case["id"],
            rel_tol=REL,
        )
        assert out.effect == pytest.approx(out.statistic / (len(x) - 1))


def test_spearman_one_sided_p_values_split_the_two_sided_one():
    rng = numpy.random.default_rng(3)
    x = rng.normal(size=60)
    y = x + rng.normal(size=60)
    both = bivariate.spearman(x, y).p
    up = bivariate.spearman(x, y, bivariate.GREATER).p
    down = bivariate.spearman(x, y, bivariate.LESS).p
    assert up == pytest.approx(both / 2)
    assert down == pytest.approx(1.0 - both / 2)
    reference = stats.spearmanr(x, y, alternative="greater")
    assert up == pytest.approx(reference.pvalue, rel=REL)


def test_perfect_monotone_relation_has_a_tiny_p_in_the_right_tail():
    x = numpy.arange(10.0)
    assert bivariate.spearman(x, x).p < 1e-12
    assert bivariate.spearman(x, x, bivariate.GREATER).p < 1e-12
    assert bivariate.spearman(x, x, bivariate.LESS).p > 1.0 - 1e-12
    assert bivariate.spearman(x, -x, bivariate.LESS).p < 1e-12
    assert bivariate.spearman(x, x).effect == pytest.approx(1.0)


def test_a_rho_of_exactly_one_gives_the_limit_p_values():
    x = numpy.array([1.0, 2.0, 3.0])
    up = bivariate.spearman(x, x)
    assert up.statistic == 1.0  # three points rank exactly
    assert up.p == 0.0
    assert bivariate.spearman(x, x, bivariate.GREATER).p == 0.0
    assert bivariate.spearman(x, x, bivariate.LESS).p == 1.0
    down = bivariate.spearman(x, -x)
    assert down.statistic == -1.0 and down.p == 0.0
    assert bivariate.spearman(x, -x, bivariate.LESS).p == 0.0
    assert bivariate.spearman(x, -x, bivariate.GREATER).p == 1.0


def test_missing_values_drop_pairs_and_untestable_pairs_get_p_one():
    x = numpy.array([1.0, 2.0, numpy.nan, 4.0, 5.0, 6.0])
    y = numpy.array([2.0, 1.0, 3.0, numpy.nan, 5.0, 7.0])
    assert bivariate.spearman(x, y).n == 4
    few = bivariate.spearman(numpy.array([1.0, 2.0]), numpy.array([1.0, 3.0]))
    assert few.p == 1.0 and few.note
    constant = bivariate.spearman(numpy.ones(10), numpy.arange(10.0))
    assert constant.p == 1.0 and "constant" in constant.note
    one_class = bivariate.kruskal(numpy.arange(5.0), numpy.zeros(5))
    assert one_class.p == 1.0


def test_bad_arguments_are_refused():
    with pytest.raises(ValueError, match="direction"):
        bivariate.spearman(numpy.arange(5.0), numpy.arange(5.0), "sideways")
    with pytest.raises(ValueError, match="two-sided"):
        bivariate.kruskal(
            numpy.arange(5.0), numpy.arange(5.0), bivariate.GREATER
        )
    with pytest.raises(ValueError, match="equally long"):
        bivariate.spearman(numpy.arange(5.0), numpy.arange(4.0))
    with pytest.raises(ValueError, match="statistic"):
        bivariate.run_statistic("t-test", numpy.arange(5.0), numpy.arange(5.0))


def test_code_hash_is_a_stable_sha256_and_differs_by_statistic():
    first = bivariate.code_hash("spearman")
    assert len(first) == 64 and first == bivariate.code_hash("spearman")
    assert first != bivariate.code_hash("kruskal")
    with pytest.raises(ValueError):
        bivariate.code_hash("nope")


def test_null_p_values_are_uniform_by_a_ks_test():
    rng = numpy.random.default_rng(11)
    draws = [
        bivariate.spearman(rng.normal(size=40), rng.normal(size=40)).p
        for draw in range(3000)
    ]
    assert stats.kstest(draws, "uniform").pvalue > 1e-3
    assert math.isclose(
        float(numpy.mean(numpy.array(draws) <= 0.05)), 0.05, abs_tol=0.02
    )
