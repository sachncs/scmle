"""IV: closed-form 2x2 cases, Anderson-Rubin inversion, weak instruments."""

import math

import numpy
import pandas
import pytest

from scmle.data.identification import iv


def world(
    n: int, seed: int, strength: float, beta: float = 2.0, direct: float = 0.0
):
    rng = numpy.random.default_rng(seed)
    z = rng.standard_normal(n)
    u = rng.standard_normal(n)
    d = strength * z + u + rng.standard_normal(n)
    y = beta * d + direct * z + 1.5 * u + rng.standard_normal(n)
    return pandas.DataFrame({"z": z, "d": d, "y": y})


def test_the_wald_ratio_equals_the_closed_form_two_by_two_case():
    frame = pandas.DataFrame(
        {
            "z": [0, 0, 0, 0, 1, 1, 1, 1],
            "d": [0, 0, 1, 0, 1, 1, 0, 1],
            "y": [1.0, 2.0, 4.0, 1.0, 6.0, 7.0, 2.0, 5.0],
        }
    )
    out = iv.estimate(frame, "d", "y", "z")
    reduced = (
        frame.loc[frame.z == 1, "y"].mean()
        - frame.loc[frame.z == 0, "y"].mean()
    )
    first = (
        frame.loc[frame.z == 1, "d"].mean()
        - frame.loc[frame.z == 0, "d"].mean()
    )
    assert out.estimate == pytest.approx(reduced / first, abs=1e-12)
    assert out.first_stage == pytest.approx(first, abs=1e-12)


def test_controls_are_partialled_out_with_an_intercept():
    rng = numpy.random.default_rng(0)
    n = 4000
    w = rng.standard_normal(n)
    z = 0.8 * w + rng.standard_normal(n)
    u = rng.standard_normal(n)
    d = z + 0.5 * w + u + rng.standard_normal(n)
    y = 2.0 * d + w + 1.5 * u + rng.standard_normal(n)
    frame = pandas.DataFrame({"z": z, "d": d, "y": y, "w": w})
    out = iv.estimate(frame, "d", "y", "z", ("w",))
    assert out.wald_low < 2.0 < out.wald_high
    assert (
        not iv.estimate(frame, "d", "y", "z").wald_low
        < 2.0
        < iv.estimate(frame, "d", "y", "z").wald_high
    )


def test_a_strong_instrument_covers_and_uses_the_wald_interval():
    hits = 0
    for seed in range(100):
        out = iv.estimate(world(500, seed, 1.0), "d", "y", "z")
        assert out.headline == "wald" and not out.weak
        hits += out.low <= 2.0 <= out.high
    assert hits >= 90


def test_anderson_rubin_covers_when_the_instrument_is_weak():
    hits = weak = 0
    for seed in range(300):
        out = iv.estimate(world(300, seed, 0.08), "d", "y", "z")
        weak += out.weak
        hits += (
            not math.isnan(out.ar_low)
            and (out.ar_kind != "bounded" or out.ar_low <= 2.0 <= out.ar_high)
            and (
                out.ar_kind != "complement"
                or not out.ar_low < 2.0 < out.ar_high
            )
        )
    assert weak > 250
    assert hits >= 0.92 * 300


def test_the_ar_set_is_the_inverted_test():
    frame = world(400, 3, 0.15)
    out = iv.estimate(frame, "d", "y", "z")
    grid = numpy.linspace(-6, 10, 801)
    z, d, y = frame.z.to_numpy(), frame.d.to_numpy(), frame.y.to_numpy()
    z, d, y = iv.residualise([z, d, y], numpy.empty((400, 0)))
    szz = z @ z
    inside = []
    for b in grid:
        e = y - b * d
        pi = (z @ e) / szz
        r = e - pi * z
        stat = pi**2 / (numpy.sum(z**2 * r**2) / szz**2)
        inside.append(stat <= 3.841458820694124)
    inside = numpy.array(inside)
    if out.ar_kind == "bounded":
        mask = (grid >= out.ar_low) & (grid <= out.ar_high)
    elif out.ar_kind == "complement":
        mask = ~((grid > out.ar_low) & (grid < out.ar_high))
    else:
        mask = numpy.ones_like(inside)
    assert (mask == inside).mean() > 0.99


def test_an_invalid_instrument_biases_the_estimate_by_gamma_over_pi():
    out = iv.estimate(world(200000, 4, 1.0, direct=0.3), "d", "y", "z")
    assert out.estimate - 2.0 == pytest.approx(
        iv.exclusion_bound(out, 0.3), abs=0.03
    )


def test_degenerate_inputs_are_refused():
    frame = world(100, 5, 1.0)
    with pytest.raises(ValueError, match="first-stage"):
        iv.estimate(frame.assign(z=1.0), "d", "y", "z")
    with pytest.raises(ValueError, match="missing"):
        iv.estimate(frame.assign(y=numpy.nan), "d", "y", "z")


def test_excludes_follows_the_headline_set():
    out = iv.estimate(world(500, 6, 1.0), "d", "y", "z")
    assert out.excludes(0.0) and not out.excludes(2.0)
    weak = iv.estimate(world(300, 7, 0.0), "d", "y", "z")
    assert weak.headline == "anderson_rubin" and not weak.excludes(2.0)
    assert set(weak.to_dict()) >= {"low", "high"}
