"""Exact statistics of the record-level leakage hypotheses."""

import datetime
import math

import numpy
import pandas
import pytest
from scipy import stats

from scmle.assurance.admissibility import leakage
from scmle.assurance.admissibility.screen import records


def config():
    return leakage.Config(skill_ceiling=0.9, association_ceiling=0.5)


def test_keys_sort_numbers_by_value_not_by_text():
    frame = pandas.DataFrame({"x": [9, 10, 2]})

    found = records.keys(frame, ["x"])

    assert found[0].tolist() == [1, 2, 0]
    assert found[1].tolist() == [0, 1, 2]


def test_keys_sort_mixed_types_by_their_text():
    frame = pandas.DataFrame(
        {"x": pandas.Series([datetime.date(2020, 1, 1), 1, "a"], dtype=object)}
    )

    found = records.keys(frame, ["x"])

    assert found[0].tolist() == [1, 0, 2]


def test_precision_uses_the_satterthwaite_freedom():
    labels = numpy.array([0, 0, 0, 1, 1])
    outcome = numpy.array([0.0, 1.0, 1.0, 0.0, 1.0])

    ratio, groups, freedom = records.precision(
        labels, outcome, numpy.full(5, 0.5)
    )

    assert ratio == pytest.approx(0.1)
    assert groups == 2
    assert freedom == pytest.approx(25 / 11)


def test_judge_gives_an_infinite_bound_to_an_exact_column():
    labels = numpy.array([0, 0, 1, 1])
    outcome = numpy.array([0.0, 1.0, 0.0, 1.0])

    (pending,) = records.judge(
        "x", numpy.full(4, 0.5), [labels], [outcome], config()
    )

    assert pending.observed == math.inf
    assert pending.bound == math.inf


def test_excess_puts_the_constrained_means_on_the_schwarz_boundary():
    singles, pairs, triples, rows = 1000, 200, 5, 1400
    c = records.SCHWARZ * (rows - 1) / (rows - 2)

    ratio = records.excess(singles, pairs, triples, rows)[1]

    middle = pairs / ratio
    shift = (pairs - middle) / 2
    assert middle**2 == pytest.approx(c * (singles + shift) * (triples + shift))


def test_duplication_counts_singles_pairs_and_triples():
    values = list(range(50)) + [100 + i for i in range(30)] * 2
    values += [200, 200, 200, 201, 201, 201]
    frame = pandas.DataFrame({"x": values, "y": [0] * len(values)})
    rows = len(values)
    cfg = config()

    found = records.duplication(frame, ["x"], "y", cfg)

    p, ratio = records.excess(50, 30, 2, rows)
    c = records.SCHWARZ * (rows - 1) / (rows - 2)
    least = stats.chi2.ppf(1.0 - cfg.confidence, 60) / 2
    most = [stats.chi2.ppf(cfg.confidence, 2 * f + 2) / 2 for f in (50, 2)]
    assert found.p_value == p
    assert found.observed == ratio
    assert found.bound == pytest.approx(
        least / math.sqrt(c * most[0] * most[1])
    )
