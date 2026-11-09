"""Overidentification tests for several instruments (Sargan and Hansen J)."""

import numpy
import pytest

from scmle.data.identification import overid


def world(seed, n=500, k=3, direct=0.0, controls=False, heavy=False):
    rng = numpy.random.default_rng(seed)
    z = rng.standard_normal((n, k))
    u = rng.standard_normal(n)
    d = z @ numpy.full(k, 0.5) + u + rng.standard_normal(n)
    noise = rng.standard_normal(n)
    if heavy:
        noise = noise * (0.5 + numpy.abs(z[:, 0]))
    y = 1.0 * d + 0.8 * u + direct * z[:, k - 1] + noise
    w = rng.standard_normal((n, 2)) if controls else None
    return y, d, z, w


def test_valid_instruments_are_rejected_at_most_at_the_level():
    # Pre-registered before the first run: over 1000 worlds with three valid
    # instruments the false-alarm rate of each test stays within three
    # standard errors of 0.05.
    worlds = 1000
    bound = 0.05 + 3 * numpy.sqrt(0.05 * 0.95 / worlds)
    sargan = hansen = 0
    for seed in range(worlds):
        y, d, z, ignored = world(seed)
        sargan += overid.sargan(y, d, z)["p_value"] <= 0.05
        hansen += overid.hansen_j(y, d, z)["p_value"] <= 0.05
    assert sargan / worlds <= bound
    assert hansen / worlds <= bound


def test_an_instrument_with_a_direct_effect_is_detected():
    # Pre-registered: at least 80 percent of 300 worlds in which one of three
    # instruments moves the outcome directly (0.5 per unit).
    found_s = found_h = 0
    for seed in range(300):
        y, d, z, ignored = world(seed, direct=0.5)
        found_s += overid.sargan(y, d, z)["p_value"] <= 0.05
        found_h += overid.hansen_j(y, d, z)["p_value"] <= 0.05
    assert found_s / 300 >= 0.80
    assert found_h / 300 >= 0.80


def test_the_robust_test_keeps_its_size_under_heteroskedasticity():
    worlds = 600
    hits = sum(
        overid.hansen_j(*world(seed, heavy=True)[:3])["p_value"] <= 0.05
        for seed in range(worlds)
    )
    assert hits / worlds <= 0.05 + 3 * numpy.sqrt(0.05 * 0.95 / worlds)


def test_controls_are_partialled_out():
    y, d, z, w = world(3, controls=True)
    with_controls = overid.sargan(y, d, z, w)
    assert with_controls["degrees_of_freedom"] == 2
    assert 0.0 <= with_controls["p_value"] <= 1.0


def test_one_instrument_cannot_be_tested():
    y, d, z, ignored = world(1, k=1)
    for test in (overid.sargan, overid.hansen_j):
        found = test(y, d, z)
        assert found["p_value"] is None and found["status"] == "not_testable"


def test_the_result_says_what_a_pass_does_not_show():
    y, d, z, ignored = world(2)
    found = overid.sargan(y, d, z)
    assert found["status"] == "described"
    assert "does not show" in found["note"]
    assert found["degrees_of_freedom"] == 2 and found["statistic"] >= 0.0


def test_inputs_are_checked():
    y, d, z, ignored = world(4)
    with pytest.raises(ValueError):
        overid.sargan(y[:-1], d, z)
    with pytest.raises(ValueError):
        overid.hansen_j(y, d, z[:-1])
