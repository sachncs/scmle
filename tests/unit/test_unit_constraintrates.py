"""Group rates, their intervals and the feasibility of criteria."""

import math

import numpy
import pytest

from scmle.assurance.charter import decision
from scmle.contracts import errors
from scmle.decision.constraints import rates


def constraint(criterion):
    return decision.Constraint(
        "g", criterion, 0.1, "group_thresholds", "counsel"
    )


Y = numpy.array([1, 1, 0, 0, 1, 0, 0, 0], dtype=float)
A = numpy.array([1, 0, 1, 0, 1, 1, 0, 0], dtype=float)
G = numpy.array(["a", "a", "a", "a", "b", "b", "b", "b"])


def test_the_incompatible_pairs_are_fixed():
    assert (
        frozenset({"demographic_parity", "predictive_parity"}),
        frozenset({"demographic_parity", "equalized_odds"}),
        frozenset({"equalized_odds", "predictive_parity"}),
    ) == rates.INCOMPATIBLE
    assert issubclass(rates.InfeasibleError, ValueError)


def test_a_preparation_returns_float_arrays_and_labels():
    y, a, g = rates.prepared([1, 0], [0, 1], ["x", "y"])
    assert y.dtype == float and a.dtype == float
    assert list(y) == [1.0, 0.0] and list(a) == [0.0, 1.0]
    assert list(g) == ["x", "y"]


def test_a_preparation_rejects_misaligned_groups():
    with pytest.raises(
        errors.InputError, match="groups must have the length of y"
    ):
        rates.prepared([1, 0], [0, 1], ["x"])


def test_a_wilson_interval_matches_the_formula():
    low, high = rates.wilson(40, 100, 1.96)
    p, n, z = 0.4, 100, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = (
        z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    )
    assert low == pytest.approx(centre - half)
    assert high == pytest.approx(centre + half)
    assert rates.wilson(0, 10, 1.96)[0] == pytest.approx(0.0, abs=1e-12)
    assert rates.wilson(10, 10, 1.96)[1] == pytest.approx(1.0)


def test_a_newcombe_interval_matches_the_hand_computation():
    low, high = rates.newcombe(40, 100, 20, 100, 0.95)
    assert low == pytest.approx(0.07312, abs=1e-4)
    assert high == pytest.approx(0.31851, abs=1e-4)
    flipped = rates.newcombe(20, 100, 40, 100, 0.95)
    assert flipped[0] == pytest.approx(-high)
    assert flipped[1] == pytest.approx(-low)


def test_a_wider_confidence_widens_the_newcombe_interval():
    narrow = rates.newcombe(40, 100, 20, 100, 0.8)
    wide = rates.newcombe(40, 100, 20, 100, 0.99)
    assert wide[0] < narrow[0] and wide[1] > narrow[1]


def test_a_tabulation_counts_each_rate_per_group():
    assert rates.tabulate(Y, A, G, "demographic_parity") == {
        "selection": {"a": (2.0, 4.0), "b": (2.0, 4.0)}
    }
    assert rates.tabulate(Y, A, G, "equal_opportunity") == {
        "tpr": {"a": (1.0, 2.0), "b": (1.0, 1.0)}
    }
    assert rates.tabulate(Y, A, G, "equalized_odds") == {
        "tpr": {"a": (1.0, 2.0), "b": (1.0, 1.0)},
        "fpr": {"a": (1.0, 2.0), "b": (1.0, 3.0)},
    }
    assert rates.tabulate(Y, A, G, "predictive_parity") == {
        "ppv": {"a": (1.0, 2.0), "b": (1.0, 2.0)}
    }


def test_a_tabulation_of_an_unknown_criterion_raises_a_key_error():
    with pytest.raises(KeyError):
        rates.tabulate(Y, A, G, "calibration")


def test_the_rate_parts_give_numerators_and_denominators():
    y = numpy.array([1.0, 0.0, 1.0])
    a = numpy.array([1.0, 1.0, 0.0])
    for name, (num, den) in {
        "selection": ([1, 1, 0], [1, 1, 1]),
        "tpr": ([1, 0, 0], [1, 0, 1]),
        "fpr": ([0, 1, 0], [0, 1, 0]),
        "ppv": ([1, 0, 0], [1, 1, 0]),
    }.items():
        got = rates.RATE_PARTS[name](y, a)
        assert list(got[0]) == num and list(got[1]) == den
    assert rates.CRITERION_RATES == {
        "demographic_parity": ("selection",),
        "equal_opportunity": ("tpr",),
        "equalized_odds": ("tpr", "fpr"),
        "predictive_parity": ("ppv",),
    }


def test_base_rates_count_positives_and_rows_by_text_label():
    assert rates.count([1, 0, 1, 1], [1, 1, 2, 2]) == {
        "1": (1.0, 2.0),
        "2": (2.0, 2.0),
    }
    with pytest.raises(errors.InputError, match="same length"):
        rates.count([1, 0], [1])


def big(prevalence_a, prevalence_b, n=2000):
    y = numpy.r_[
        numpy.ones(int(n * prevalence_a)),
        numpy.zeros(n - int(n * prevalence_a)),
        numpy.ones(int(n * prevalence_b)),
        numpy.zeros(n - int(n * prevalence_b)),
    ]
    groups = numpy.r_[numpy.zeros(n), numpy.ones(n)].astype(int)
    return y, groups


def test_compatible_criteria_are_reported_feasible():
    y, groups = big(0.5, 0.2)
    out = rates.check([constraint("equal_opportunity")], y, groups)
    assert out == {
        "base_rates": {"0": 0.5, "1": 0.2},
        "differ": True,
        "conflicts": [],
        "refused": False,
    }


def test_incompatible_criteria_with_equal_base_rates_are_feasible():
    y, groups = big(0.3, 0.3)
    out = rates.check(
        [constraint("demographic_parity"), constraint("predictive_parity")],
        y,
        groups,
    )
    assert out["differ"] is False
    assert out["conflicts"] == [["demographic_parity", "predictive_parity"]]
    assert out["refused"] is False


def test_incompatible_criteria_with_differing_base_rates_are_refused():
    y, groups = big(0.5, 0.2)
    with pytest.raises(rates.InfeasibleError) as caught:
        rates.check(
            [constraint("predictive_parity"), constraint("equalized_odds")],
            y,
            groups,
        )
    assert str(caught.value) == (
        "equalized_odds and predictive_parity cannot both hold: the base "
        "rates differ ({'0': 0.5, '1': 0.2})"
    )


def test_a_small_base_rate_difference_does_not_differ():
    y, groups = big(0.31, 0.3, n=200)
    out = rates.check([constraint("demographic_parity")], y, groups)
    assert out["differ"] is False


def test_the_level_is_split_over_group_pairs():
    # with alpha split three ways, a borderline difference stops differing
    rng = numpy.random.default_rng(0)
    y = numpy.r_[
        numpy.ones(56), numpy.zeros(44), numpy.ones(42), numpy.zeros(58)
    ]
    groups = numpy.r_[numpy.zeros(100), numpy.ones(100)]
    two = rates.check([], y, groups)
    assert two["differ"] is True
    third = numpy.r_[groups, numpy.full(100, 2)]
    y3 = numpy.r_[y, (rng.random(100) < 0.49).astype(float)]
    three = rates.check([], y3, third)
    assert three["differ"] is False
