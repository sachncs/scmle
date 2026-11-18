"""The adversary's deadline arithmetic, its odds verdict and its sampling."""

import dataclasses

import numpy
import pandas
import pytest

from scmle.evaluation import diagnostics
from scmle.evaluation import shift
from scmle.evaluation import shift_weights
from tests.unit.test_diagnostics import Clock
from tests.unit.test_shift import shifted_inputs


def timed(monkeypatch, durations: list[float]) -> Clock:
    """Replaces the clock and makes fold `i` take `durations[i]` seconds."""
    clock = Clock()
    monkeypatch.setattr(diagnostics, "clock", clock)
    original = shift.Adversary.fold
    spent = iter(durations)

    def fold(self, *args):
        clock.now += next(spent, 0.0)
        return original(self, *args)

    monkeypatch.setattr(shift.Adversary, "fold", fold)
    return clock


def test_the_first_fold_needs_a_fifth_of_the_estimate(monkeypatch) -> None:
    clock = timed(monkeypatch, [])
    data = shifted_inputs(1.0)
    first = shift.Adversary().estimate(data) / shift.FOLDS
    report = shift.Adversary().compute(
        dataclasses.replace(data, deadline=clock.now + 2 * first)
    )
    assert report["available"] is True


def test_no_time_for_the_first_fold_is_said_word_for_word(monkeypatch) -> None:
    clock = timed(monkeypatch, [])
    data = dataclasses.replace(shifted_inputs(0.0), deadline=clock.now - 1.0)
    assert shift.Adversary().compute(data) == {
        "available": False,
        "skipped": True,
        "reason": "the deadline leaves no time for the first fold",
    }


def test_the_next_fold_is_predicted_by_the_longest_so_far(monkeypatch) -> None:
    clock = timed(monkeypatch, [10.0, 1.0])
    # After the second fold the clock reads +11 s; the longest fold (10 s)
    # with the margin no longer fits, while the last one (1 s) would.
    deadline = clock.now + 10.0 + diagnostics.MARGIN * 10.0 + 0.5
    data = dataclasses.replace(shifted_inputs(1.0), deadline=deadline)
    report = shift.Adversary().compute(data)
    assert report["reason"] == (
        f"2 of {shift.FOLDS} folds ran; the next takes up to 10.00 s and "
        "the deadline came first"
    )


def test_a_small_reference_effective_size_refuses_weighting() -> None:
    rows = int(shift_weights.MIN_EFFECTIVE_SIZE) - 1
    scores = numpy.full(2 * rows, 0.5)
    found = shift.Adversary.odds(scores, rows)
    assert found["reference"]["effective_size"] == pytest.approx(rows)
    assert found["weighted_conformal"] == "refused"


def test_a_small_frame_keeps_its_columns_under_a_fresh_index() -> None:
    frame = pandas.DataFrame({"x": [1.0, 2.0]}, index=[7, 9])
    found = shift.Adversary.sample(frame, numpy.random.default_rng(0))
    assert list(found.columns) == ["x"]
    assert list(found.index) == [0, 1]
