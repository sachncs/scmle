"""Synthetic control: weights against a convex reference, p-values, refusal."""

import numpy
import pandas
import pytest
from scipy import optimize

from scmle.data.identification import synth


def panel(
    seed: int, donors: int = 24, tau: float = 3.0, pre: int = 12, post: int = 4
):
    rng = numpy.random.default_rng(seed)
    factors = rng.standard_normal((pre + post, 2)).cumsum(axis=0) * 0.5
    loadings = rng.uniform(0, 1, (donors + 1, 2))
    rows = []
    for u in range(donors + 1):
        for t in range(pre + post):
            y = factors[t] @ loadings[u] + rng.normal(scale=0.1)
            if u == 0 and t >= pre:
                y += tau
            rows.append((u, t, y))
    return pandas.DataFrame(rows, columns=["u", "t", "y"]), pre


def test_weights_match_a_convex_program_reference():
    rng = numpy.random.default_rng(0)
    donors = rng.standard_normal((15, 6))
    true = numpy.array([0.5, 0.3, 0.2, 0, 0, 0])
    target = donors @ true + rng.normal(scale=0.01, size=15)
    w = synth.fit_weights(target, donors)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()
    reference = optimize.minimize(
        lambda v: numpy.sum((target - donors @ v) ** 2),
        numpy.full(6, 1 / 6),
        method="trust-constr",
        bounds=[(0, 1)] * 6,
        constraints=[optimize.LinearConstraint(numpy.ones((1, 6)), 1, 1)],
        options={"gtol": 1e-12, "xtol": 1e-12, "maxiter": 5000},
    ).x
    assert numpy.allclose(w, reference, atol=1e-4)
    assert numpy.allclose(w, true, atol=0.05)


def test_a_clear_effect_gets_the_smallest_attainable_p_value():
    frame, pre = panel(1)
    out = synth.estimate(frame, "u", "t", "y", 0, pre)
    assert out.effect == pytest.approx(3.0, abs=0.4)
    assert out.p_value == pytest.approx(out.min_p)
    assert out.min_p == pytest.approx(1 / 25)
    assert out.pre_rmspe < 0.3


def test_no_effect_rejects_at_about_the_nominal_rate():
    rejected = 0
    runs = 60
    for seed in range(runs):
        frame, pre = panel(100 + seed, tau=0.0)
        rejected += synth.estimate(frame, "u", "t", "y", 0, pre).p_value <= 0.05
    assert rejected / runs <= 0.15


def test_too_few_donors_refuse():
    frame, pre = panel(2, donors=18)
    with pytest.raises(ValueError, match="at least 19"):
        synth.estimate(frame, "u", "t", "y", 0, pre)
    ok, pre = panel(2, donors=19)
    assert synth.estimate(ok, "u", "t", "y", 0, pre).donors == 19
    assert synth.min_donors(0.05) == 19 and synth.min_donors(0.1) == 9


def test_bad_panels_are_refused():
    frame, pre = panel(3)
    with pytest.raises(ValueError, match="balanced"):
        synth.estimate(frame.iloc[1:], "u", "t", "y", 0, pre)
    with pytest.raises(ValueError, match="not in the panel"):
        synth.estimate(frame, "u", "t", "y", 99, pre)
    with pytest.raises(ValueError, match="pre-periods"):
        synth.estimate(frame, "u", "t", "y", 0, 1)
