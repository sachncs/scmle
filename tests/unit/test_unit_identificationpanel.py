"""Difference-in-differences and synthetic-control adapters."""

import math

import numpy
import pandas
import pytest

from scmle.assurance.charter import causal
from scmle.causal.identification import did
from scmle.causal.identification import synth
from scmle.causal.identification.run import panel
from scmle.contracts import errors
from tests.unit.test_identification_run import did_frame


def did_block(**changes):
    values = {
        "treatment": "g",
        "outcome": "y",
        "estimand": "ATT_DID",
        "unit": "id",
        "time": "t",
        "adoption": "g",
        "assumptions": (
            "parallel_trends",
            "no_anticipation",
            "no_interference",
        ),
        "strategies": ("did",),
        "controls": (
            causal.Control("outcome", "nco"),
            causal.Control("exposure", "t"),
        ),
        "sensitivity": causal.Confounding(multiple=1.0),
    }
    values.update(changes)
    return causal.Question(**values)


@pytest.mark.parametrize("missing", ["unit", "time", "adoption"])
def test_a_did_without_panel_columns_raises(missing):
    adapter = panel.Differences(
        did_block(**{missing: None}), did_frame(1, 20), 0, "linear", 0.01
    )
    with pytest.raises(
        errors.InputError, match="did needs unit, time and adoption"
    ):
        adapter.columns()


def test_a_did_reads_its_columns():
    adapter = panel.Differences(
        did_block(), did_frame(1, 20), 0, "linear", 0.01
    )
    assert panel.Differences.name == "did"
    assert adapter.columns() == ("id", "t", "g")


def test_a_did_prepare_records_the_panel_from_x():
    x = did_frame(1, 200)
    adapter = panel.Differences(did_block(), x, 0, "linear", 0.01)
    adapter.prepare()
    fit = did.estimate(x, "id", "t", "g", "y")
    assert adapter.design == {
        "groups": fit.groups,
        "n_never_treated": fit.untreated,
        "events": list(fit.events),
        "pretrend_p_on_x": fit.pretrend,
        "pre_periods_support_sensitivity": True,
    }


def test_a_did_estimate_equals_the_engine():
    x, c = did_frame(1, 200), did_frame(2, 200)
    adapter = panel.Differences(did_block(), x, 0, "linear", 0.01)
    est = adapter.estimate(c, 0.9)
    fit = did.estimate(c, "id", "t", "g", "y", 0.9)
    assert (est.estimate, est.se, est.low, est.high, est.n) == (
        fit.estimate,
        fit.se,
        fit.low,
        fit.high,
        len(c),
    )
    assert est.detail == {
        "pretrend_p": fit.pretrend,
        "event_study": dict(zip(fit.events, fit.coefficients, strict=True)),
    }
    other = adapter.reckon(c, 0.9, "nco")
    nco = did.estimate(c, "id", "t", "g", "nco", 0.9)
    assert (other.estimate, other.se, other.n) == (nco.estimate, nco.se, len(c))
    assert other.detail == {}
    assert adapter.reckon(c, 0.9).estimate == fit.estimate


def test_did_controls_are_the_pretrend_then_negative_outcomes():
    c = did_frame(3, 200)
    adapter = panel.Differences(did_block(), c, 0, "linear", 0.01)
    tests = adapter.controls(c, 0.9)
    fit = did.estimate(c, "id", "t", "g", "y", 0.9)
    assert [(t.name, t.kind) for t in tests] == [
        ("pretrend_wald", "placebo"),
        ("negative_outcome:nco", "negative_outcome"),
    ]
    assert tests[0].p_value == fit.pretrend


def test_did_sensitivity_widens_by_the_relative_magnitude():
    c = did_frame(4, 300)
    adapter = panel.Differences(did_block(), c, 0, "linear", 0.01)
    out = adapter.sensitivity(c, 0.9)
    fit = did.estimate(c, "id", "t", "g", "y", 0.9)
    low, high = fit.sensitivity(1.0)
    assert out == {
        "holds": bool(low > 0 or high < 0),
        "m_max": 1.0,
        "low": low,
        "high": high,
        "breakdown_m": fit.breakdown(),
        "max_pre_change": fit.drift,
        "note": "simplified relative-magnitudes bound, not HonestDiD",
    }


def test_did_sensitivity_notes_a_missing_block():
    c = did_frame(4, 50)
    adapter = panel.Differences(
        did_block(sensitivity=None), c, 0, "linear", 0.01
    )
    assert adapter.sensitivity(c, 0.9) == {
        "holds": None,
        "note": "no sensitivity block in the charter",
    }


def test_did_sensitivity_needs_two_pre_periods():
    c = did_frame(5, 100)
    c["g"] = c["g"].where(c["g"].isna(), 1.0)
    adapter = panel.Differences(did_block(), c, 0, "linear", 0.01)
    assert adapter.sensitivity(c, 0.9) == {
        "holds": None,
        "note": "fewer than two pre-periods",
    }


def synth_panel(units=25, treated=(0,)):
    rng = numpy.random.default_rng(3)
    factors = rng.standard_normal((16, 2)).cumsum(axis=0) * 0.5
    loadings = rng.uniform(0, 1, (units, 2))
    rows = []
    for u in range(units):
        for t in range(16):
            y = factors[t] @ loadings[u] + rng.normal(scale=0.1)
            flag = float(u in treated and t >= 12)
            rows.append((u, t, y + 3.0 * flag, flag))
    return pandas.DataFrame(rows, columns=["u", "t", "y", "d"])


def synth_block(**changes):
    values = {
        "treatment": "d",
        "outcome": "y",
        "estimand": "ATT_SC",
        "unit": "u",
        "time": "t",
        "assumptions": ("synthetic_hull", "no_interference"),
        "strategies": ("synth",),
    }
    values.update(changes)
    return causal.Question(**values)


def test_a_synthetic_control_is_a_single_unit_design_on_x():
    x = synth_panel()
    adapter = panel.Synth(synth_block(), x, 0, "linear", 0.01)
    assert panel.Synth.name == "synth"
    assert adapter.unit is True
    est = adapter.estimate(x.iloc[:0], 0.9)
    result = synth.estimate(x, "u", "t", "y", 0, 12, 1 - 0.9)
    assert est.estimate == result.effect
    assert math.isnan(est.se) and math.isnan(est.low) and math.isnan(est.high)
    assert est.n == len(x)
    assert est.detail == {"p_value": result.p_value, "ratio": result.ratio}
    assert adapter.design == {
        "donors": result.donors,
        "min_attainable_p": result.floor,
        "pre_rmspe": result.pre_rmspe,
        "weights": result.weights,
    }
    assert est.estimate == pytest.approx(3.0, abs=0.5)


def test_a_synthetic_control_needs_unit_and_time():
    for spec in (synth_block(unit=None), synth_block(time=None)):
        adapter = panel.Synth(spec, synth_panel(), 0, "linear", 0.01)
        with pytest.raises(
            errors.InputError, match="synth needs unit and time"
        ):
            adapter.estimate(None, 0.9)


def test_a_synthetic_control_needs_exactly_one_treated_unit():
    for treated in ((), (0, 1)):
        adapter = panel.Synth(
            synth_block(), synth_panel(treated=treated), 0, "linear", 0.01
        )
        with pytest.raises(errors.InputError, match="exactly one treated unit"):
            adapter.estimate(None, 0.9)


def test_a_synthetic_control_has_no_other_outcome_estimate():
    adapter = panel.Synth(synth_block(), synth_panel(), 0, "linear", 0.01)
    with pytest.raises(NotImplementedError):
        adapter.reckon(None, 0.9, "y")
