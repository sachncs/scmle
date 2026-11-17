"""Policy contrasts on a four-row log whose weights are known exactly."""

import numpy
import pytest

from scmle.decision import ope

ROWS = 4


def log() -> ope.Bandits:
    behaviour = numpy.array([[0.5, 0.5]] * 2 + [[0.95, 0.05]] * 2)
    target = numpy.array([[1.0, 0.0]] * ROWS)
    return ope.Bandits(
        numpy.zeros(ROWS, dtype=int),
        numpy.array([1.0, 0.0, 1.0, 1.0]),
        behaviour,
        target,
    )


def test_an_ipw_contrast_uses_ipw_terms() -> None:
    half = numpy.full((ROWS, 2), 0.5)
    found = ope.contrast(log(), half, "ipw")
    weights = numpy.array([2.0, 2.0, 1 / 0.95, 1 / 0.95])
    rewards = numpy.array([1.0, 0.0, 1.0, 1.0])
    expected = float(numpy.mean(weights * rewards - 0.5 * weights * rewards))
    assert found.status == ope.ISSUED
    assert found.value == pytest.approx(expected)


def test_a_refused_baseline_keeps_the_declared_clip() -> None:
    risky = numpy.array([[0.0, 1.0]] * ROWS)
    found = ope.contrast(log(), risky, "ipw", clip=5.0, min_propensity=0.1)
    assert found.status == ope.REFUSED
    assert found.clipped is True
