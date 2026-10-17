"""Intervals and verdicts over repeated benchmark seeds."""

import pytest

from benchmarks import intervals


def test_the_interval_matches_a_hand_computed_student_t():
    mean, low, high = intervals.mean_interval([1.0, 2.0, 3.0])

    assert mean == 2.0
    # t(0.975, 2) = 4.3027; s = 1; margin = 4.3027 / sqrt(3)
    assert high - mean == pytest.approx(2.4843, abs=1e-3)
    assert mean - low == pytest.approx(2.4843, abs=1e-3)


def test_one_seed_has_no_interval():
    assert intervals.mean_interval([0.7]) == (0.7, 0.7, 0.7)


def test_advantage_follows_the_metric_direction():
    assert intervals.advantage("roc_auc", 0.9, 0.8) == pytest.approx(0.1)
    assert intervals.advantage("rmse", 8.0, 10.0) == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [(0.01, 0.05, "ahead"), (-0.05, -0.01, "behind"), (-0.02, 0.03, "tie")],
)
def test_verdicts_come_from_the_interval(low, high, expected):
    assert intervals.verdict(low, high, 5) == expected


def test_a_single_seed_is_labelled_as_such():
    assert intervals.verdict(0.02, 0.02, 1) == "ahead (1 seed)"
