"""Iterative stratification keeps rare labels in every subset."""

import numpy as np

from scmle.data.splitting import iterative


def labels(rows=1000, seed=0):
    rng = np.random.default_rng(seed)
    rates = np.array([0.5, 0.3, 0.1, 0.05, 0.02])
    return (rng.random((rows, len(rates))) < rates).astype(int)


def test_every_subset_gets_its_share_of_every_label():
    matrix = labels()

    assigned = iterative.iterative_assignment(matrix, [0.2] * 5, seed=0)

    for subset in range(5):
        rows = assigned == subset
        assert abs(rows.sum() - 200) <= 2
        expected = matrix.sum(axis=0) / 5
        assert np.all(np.abs(matrix[rows].sum(axis=0) - expected) <= 2)


def test_every_row_is_assigned_exactly_once():
    assigned = iterative.iterative_assignment(labels(), [0.7, 0.3], seed=1)

    assert set(assigned) == {0, 1} and len(assigned) == 1000


def test_the_result_depends_only_on_the_seed():
    matrix = labels()

    first = iterative.iterative_assignment(matrix, [0.5, 0.5], seed=3)
    second = iterative.iterative_assignment(matrix, [0.5, 0.5], seed=3)
    other = iterative.iterative_assignment(matrix, [0.5, 0.5], seed=4)

    assert (first == second).all() and (first != other).any()


def test_rows_with_no_label_are_still_placed():
    matrix = np.zeros((50, 3), dtype=int)

    assigned = iterative.iterative_assignment(matrix, [0.5, 0.5], seed=0)

    assert abs((assigned == 0).sum() - 25) <= 1
