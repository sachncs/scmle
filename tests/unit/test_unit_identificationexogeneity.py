"""Instrument and discontinuity adapters match their engines."""

import numpy
import pandas
import pytest

from scmle.assurance.charter import causal
from scmle.causal.identification import iv
from scmle.causal.identification import rd
from scmle.causal.identification.run import exogeneity
from scmle.contracts import errors


def iv_world(n, seed, strength=1.0):
    rng = numpy.random.default_rng(seed)
    z = rng.standard_normal(n)
    u = rng.standard_normal(n)
    w = rng.standard_normal(n)
    d = strength * z + u + 0.3 * w + rng.standard_normal(n)
    y = 2 * d + 1.5 * u + w + rng.standard_normal(n)
    nco = u + rng.standard_normal(n)
    return pandas.DataFrame({"z": z, "d": d, "y": y, "w": w, "nco": nco})


def iv_block(**changes):
    values = {
        "treatment": "d",
        "outcome": "y",
        "estimand": "LATE",
        "instrument": "z",
        "assumptions": (
            "instrument_relevance",
            "exclusion",
            "independence",
            "monotonicity",
            "unconfoundedness_given:w",
        ),
        "strategies": ("iv",),
        "controls": (
            causal.Control("outcome", "nco"),
            causal.Control("exposure", "w"),
        ),
    }
    values.update(changes)
    return causal.Question(**values)


def test_an_instrument_reads_its_controls_from_the_assumptions():
    adapter = exogeneity.Instrument(
        iv_block(), iv_world(50, 0), 0, "linear", 0.01
    )
    assert exogeneity.Instrument.name == "iv"
    assert adapter.controls_columns() == ("w",)
    bare = exogeneity.Instrument(
        iv_block(assumptions=("exclusion",)), iv_world(50, 0), 0, "linear", 0.01
    )
    assert bare.controls_columns() == ()


def test_an_instrument_without_a_declared_instrument_raises():
    spec = iv_block(instrument=None)
    adapter = exogeneity.Instrument(spec, iv_world(50, 0), 0, "linear", 0.01)
    with pytest.raises(errors.InputError, match="iv needs an instrument"):
        adapter.prepare()


def test_an_instrument_prepare_records_the_first_stage_on_x():
    x = iv_world(2000, 1)
    adapter = exogeneity.Instrument(iv_block(), x, 0, "linear", 0.01)
    adapter.prepare()
    fit = iv.estimate(x, "d", "y", "z", ("w",), 0.95)
    assert adapter.design == {
        "first_stage_F_on_x": fit.f_stat,
        "threshold": iv.F_THRESHOLD,
        "weak_on_x": fit.weak,
        "overidentification": "not applicable: one instrument, just identified",
    }
    assert adapter.weak is False


def test_a_weak_instrument_marks_the_adapter_weak():
    x = iv_world(200, 2, strength=0.0)
    adapter = exogeneity.Instrument(iv_block(), x, 0, "linear", 0.01)
    adapter.prepare()
    assert adapter.weak is True
    assert adapter.design["weak_on_x"] is True


def test_an_instrument_estimate_equals_the_engine_and_keeps_weakness():
    x, c = iv_world(2000, 3), iv_world(2000, 4)
    adapter = exogeneity.Instrument(iv_block(), x, 0, "linear", 0.01)
    adapter.prepare()
    est = adapter.estimate(c, 0.9)
    fit = iv.estimate(c, "d", "y", "z", ("w",), 0.9)
    assert (est.estimate, est.se, est.low, est.high, est.n) == (
        fit.estimate,
        fit.se,
        fit.low,
        fit.high,
        fit.n,
    )
    assert est.detail == {
        "headline": fit.headline,
        "ar_kind": fit.ar_kind,
        "estimand": "LATE",
    }
    assert adapter.design["first_stage_F_on_c"] == fit.f_stat
    assert adapter.weak is False
    adapter.weak = True
    adapter.estimate(c, 0.9)
    assert adapter.weak is True


def test_an_instrument_reckoning_uses_the_two_stage_interval():
    c = iv_world(1500, 5)
    adapter = exogeneity.Instrument(iv_block(), c, 0, "linear", 0.01)
    est = adapter.reckon(c, 0.9, "nco")
    fit = iv.estimate(c, "d", "nco", "z", ("w",), 0.9)
    assert (est.estimate, est.se, est.low, est.high, est.n) == (
        fit.estimate,
        fit.se,
        fit.wald_low,
        fit.wald_high,
        fit.n,
    )
    assert adapter.reckon(c, 0.9).estimate == adapter.estimate(c, 0.9).estimate


def test_instrument_controls_test_only_negative_outcomes():
    c = iv_world(1500, 6)
    adapter = exogeneity.Instrument(iv_block(), c, 0, "linear", 0.01)
    tests = adapter.controls(c, 0.9)
    assert [(t.name, t.kind) for t in tests] == [
        ("negative_outcome:nco", "negative_outcome")
    ]
    assert tests[0].estimate == adapter.reckon(c, 0.9, "nco").estimate


def rd_world(n, seed, jump=1.0):
    rng = numpy.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    y = 2 * x + jump * (x >= 0) + rng.normal(scale=0.5, size=n)
    pre = x + rng.normal(scale=0.5, size=n)
    return pandas.DataFrame({"x": x, "y": y, "pre": pre})


def rd_block(**changes):
    values = {
        "treatment": "d",
        "outcome": "y",
        "estimand": "LATE_RD",
        "running_variable": "x",
        "cutoff": 0.0,
        "assumptions": ("continuity_at_cutoff", "no_manipulation"),
        "strategies": ("rd",),
        "controls": (
            causal.Control("outcome", "pre"),
            causal.Control("exposure", "x"),
        ),
    }
    values.update(changes)
    return causal.Question(**values)


def test_a_discontinuity_starts_with_no_bandwidth_and_lists_pretreatment():
    adapter = exogeneity.RdEstimator(
        rd_block(), rd_world(50, 0), 0, "linear", 0.01
    )
    assert exogeneity.RdEstimator.name == "rd"
    assert adapter.bandwidth == 0.0
    assert adapter.pretreatment() == ("pre",)


def test_a_discontinuity_without_running_variable_or_cutoff_raises():
    for spec in (rd_block(running_variable=None), rd_block(cutoff=None)):
        adapter = exogeneity.RdEstimator(
            spec, rd_world(50, 0), 0, "linear", 0.01
        )
        with pytest.raises(errors.InputError, match="rd needs a running"):
            adapter.prepare()
        with pytest.raises(errors.InputError, match="rd needs a running"):
            adapter.fit(rd_world(50, 0), 0.9)


def test_a_discontinuity_prepare_freezes_the_bandwidth_from_x():
    x = rd_world(3000, 1)
    adapter = exogeneity.RdEstimator(rd_block(), x, 0, "linear", 0.01)
    adapter.prepare()
    fit = rd.estimate(x, "x", "y", 0.0, covariates=("pre",))
    assert adapter.bandwidth == fit.bandwidth
    assert adapter.design == {
        "bandwidth": fit.bandwidth,
        "density_p_on_x": fit.density_p,
        "alpha_density": 0.01,
        "n_left_in_bandwidth": fit.n_left,
        "n_right_in_bandwidth": fit.n_right,
    }
    assert adapter.refusal is None


def test_a_manipulated_running_variable_is_refused():
    x = rd_world(3000, 2)
    x.loc[x.x.between(-0.2, 0) & (numpy.arange(3000) % 2 == 0), "x"] *= -1
    adapter = exogeneity.RdEstimator(rd_block(), x, 0, "linear", 0.01)
    adapter.prepare()
    assert adapter.design["density_p_on_x"] < 0.01
    assert adapter.refusal == (
        "the density check rejects at the cutoff on X: possible "
        "manipulation of the running variable"
    )


def prepared_rd():
    x, c = rd_world(3000, 3), rd_world(3000, 4)
    adapter = exogeneity.RdEstimator(rd_block(), x, 0, "linear", 0.01)
    adapter.prepare()
    return adapter, c


def test_a_discontinuity_estimate_is_bias_corrected_at_the_frozen_bandwidth():
    adapter, c = prepared_rd()
    est = adapter.estimate(c, 0.9)
    fit = rd.estimate(c, "x", "y", 0.0, 0.9, adapter.bandwidth)
    assert (est.estimate, est.se, est.low, est.high, est.n) == (
        fit.bias_corrected,
        fit.se,
        fit.low,
        fit.high,
        fit.n_left + fit.n_right,
    )
    assert est.detail == {
        "conventional": fit.estimate,
        "bandwidth": adapter.bandwidth,
    }
    other = adapter.reckon(c, 0.9, "pre")
    pre = rd.estimate(c, "x", "pre", 0.0, 0.9, adapter.bandwidth)
    assert (other.estimate, other.se, other.n) == (
        pre.bias_corrected,
        pre.se,
        pre.n_left + pre.n_right,
    )
    assert other.detail == {}


def test_discontinuity_controls_are_placebos_then_balance_tests():
    adapter, c = prepared_rd()
    tests = adapter.controls(c, 0.9)
    fit = rd.estimate(c, "x", "y", 0.0, 0.9, adapter.bandwidth, ("pre",))
    expected = [
        (f"placebo_cutoff:{name}", "placebo", float(row["p"]))
        for name, row in fit.placebos.items()
    ] + [
        (f"covariate_balance:{name}", "negative_outcome", float(row["p"]))
        for name, row in fit.balance.items()
    ]
    assert [(t.name, t.kind, t.p_value) for t in tests] == expected
    assert tests[-1].name == "covariate_balance:pre"


def test_discontinuity_sensitivity_reports_bandwidth_stability():
    adapter, c = prepared_rd()
    out = adapter.sensitivity(c, 0.9)
    fit = rd.estimate(c, "x", "y", 0.0, 0.9, adapter.bandwidth)
    stable = all(
        (r["low"] <= 0 <= r["high"]) == (fit.low <= 0 <= fit.high)
        for r in fit.sensitivity.values()
    )
    assert out == {
        "holds": True,
        "applicable": False,
        "note": "RD has no unmeasured-confounding bound; the bandwidth "
        "sensitivity is descriptive",
        "bandwidth_sensitivity": fit.sensitivity,
        "same_conclusion_at_all_multiples": stable,
    }
    assert out["same_conclusion_at_all_multiples"] is True
