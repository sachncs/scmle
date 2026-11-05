"""Snapshots of the F10 estimators on fixed inputs (mutation pins).

See `tests/unit/golden.py` and `tests/unit/MUTANTS_identification.md`.
Regenerate with SCMLE_WRITE_GOLDEN=1 only after a reviewed change.
"""

from pathlib import Path

import numpy as np
import pytest

from scmle.contracts.qualification import Tier
from scmle.data.identification import aipw
from scmle.data.identification import battery
from scmle.data.identification import did
from scmle.data.identification import iv
from scmle.data.identification import rd
from scmle.data.identification import sensitivity as s
from scmle.data.identification import synth
from scmle.data.identification import tier
from tests.unit import golden
from tests.unit import test_identification_aipw as aipw_tests
from tests.unit import test_identification_did as did_tests
from tests.unit import test_identification_iv as iv_tests
from tests.unit import test_identification_rd as rd_tests
from tests.unit import test_identification_sensitivity as sens_tests
from tests.unit import test_identification_synth as synth_tests
from tests.unit import test_identification_tier as tier_tests

FIXTURE = Path(__file__).parents[1] / "fixtures" / "identification_golden.json"
COLUMNS = ("a", "b", "c")


# Boosted trees differ between platforms by more than any useful tolerance,
# so only the linear learner is pinned; the boosted one is checked for sanity.
PINNED_LEARNERS = ("linear",)


def case_aipw():
    out = {}
    for learner in PINNED_LEARNERS:
        nuis = aipw.fit_nuisances(
            aipw_tests.planted(600, 1), COLUMNS, "t", "y", 0, learner
        )
        for estimand in ("ATE", "ATT"):
            out[f"{learner}_{estimand}"] = aipw.estimate(
                nuis, aipw_tests.planted(800, 2), "t", "y", estimand
            )
    out["naive"] = aipw.difference_in_means(
        aipw_tests.planted(800, 3, confound=1.5), "t", "y"
    )
    t = np.array([1.0, 0.0, 1.0, 0.0])
    y = np.array([3.0, 1.0, 4.0, 0.5])
    e = np.array([0.5, 0.25, 0.5, 0.25])
    out["scores"] = {
        name: aipw.scores(name, t, y, e, np.ones(4), np.zeros(4))
        for name in ("ATE", "ATT")
    }
    out["ess"] = aipw.effective_sample_size(np.array([1.0, 2.0, 3.0]))
    out["smd"] = aipw.weighted_smd(
        np.array([[1.0, 5.0], [2.0, 4.0], [3.0, 7.0], [4.0, 1.0]]),
        t,
        np.array([0.5, 0.25, 0.75, 0.4]),
    )
    return out


def case_did():
    return {
        "plain": did.estimate(did_tests.panel(1), "id", "t", "g", "y"),
        "second": did.estimate(
            did_tests.panel(2, second=True, drift=0.1), "id", "t", "g", "y"
        ),
        "wide": did.wide_panel(
            did_tests.panel(3, 4, 4, 4, 2), "id", "t", "y"
        ).iloc[:5],
    }


def case_iv():
    return {
        "strong": iv.estimate(iv_tests.world(600, 1, 1.0), "d", "y", "z"),
        "weak": iv.estimate(iv_tests.world(300, 2, 0.08), "d", "y", "z"),
        "direct": iv.estimate(
            iv_tests.world(600, 3, 1.0, direct=0.3), "d", "y", "z"
        ),
        "bound": iv.exclusion_bound(
            iv.estimate(iv_tests.world(600, 1, 1.0), "d", "y", "z"), 0.2
        ),
    }


def case_rd():
    return {
        "plain": rd.estimate(rd_tests.world(1200, 1), "x", "y", 0.0),
        "sorted": rd.estimate(
            rd_tests.world(1200, 2, sort=0.15), "x", "y", 0.0
        ),
        "band": rd.estimate(
            rd_tests.world(1200, 3), "x", "y", 0.0, 0.9, 0.5, ("z",)
        ),
    }


def case_synth():
    frame, pre = synth_tests.panel(1)
    rng = np.random.default_rng(0)
    donors = rng.standard_normal((15, 6))
    target = donors @ np.array([0.5, 0.3, 0.2, 0, 0, 0])
    return {
        "estimate": synth.estimate(frame, "u", "t", "y", 0, pre),
        "weights": synth.fit_weights(target, donors),
        "donors": [synth.min_donors(a) for a in (0.05, 0.1, 0.2)],
    }


def case_sensitivity():
    frame = sens_tests.synthetic(1200, 1, 0.5, 0.5).drop(columns="u")
    base = ("x1", "x2", "x3")
    fit = s.ols_fit(frame, "d", "y", base)
    return {
        "evalue": [s.evalue(v) for v in (0.5, 0.8, 1.0, 1.25, 3.9)],
        "interval": [
            s.evalue_interval(1.8, 1.2, 2.7),
            s.evalue_interval(0.5, 0.3, 0.8),
            s.evalue_interval(1.3, 0.9, 1.9),
        ],
        "smd": [s.smd_to_rr(d) for d in (0.2, -0.5)],
        "partial": s.partial_r2(4.184, 783),
        "rv": [
            s.robustness_value(4.184, 783, 1.0),
            s.robustness_value(4.184, 783, 0.5, 0.05),
            s.robustness_value(4.184, 783, 1.0, 0.05),
            s.robustness_value(0.5, 100, 1.0, 0.05),
        ],
        "fit": fit,
        "bias": s.bias_bound(fit, 0.1, 0.2),
        "interval_adj": s.adjusted_interval(fit, 0.1, 0.2, 0.9),
        "bench": s.benchmark_strength(0.1, 0.2, 1.0, 1.5),
        "assess": s.assess(frame, "d", "y", base, strength=1.0),
        "assess_fixed": s.assess(frame, "d", "y", base, "fixed", 0.01),
    }


def case_tier():
    out = {}
    changes = (
        {},
        {"identified": False},
        {"estimated_on_c": False},
        {"design_passed": False},
        {"battery_ran": False},
        {"battery_passed": False},
        {"negative_control_declared": False},
        {"replication_passed": False},
        {"sensitivity_holds": False},
        {"single_unit": True},
        {"weak_instrument": True},
        {"certified": frozenset()},
        {"attestors": {}},
    )
    for index, change in enumerate(changes):
        out[str(index)] = tier.decide(tier_tests.facts(**change))
    out["tiers"] = [str(Tier.CONFIRMED)]
    return out


def case_battery():
    tests = [
        battery.wald_test("a", "negative_outcome", 0.1, 1.0),
        battery.wald_test("b", "negative_exposure", 5.0, 1.0),
        battery.wald_test("c", "placebo", -1.5, 1.0),
    ]
    return {
        "report": battery.run_battery(tests, 0.05),
        "clean": battery.run_battery(tests[:1], 0.05),
        "empty": battery.run_battery([], 0.05),
        "replicate": battery.replicate(
            {"x": (1.0, 0.2), "y": (1.2, 0.3), "z": (0.8, 0.25)}, 1.0
        ),
        "split": battery.replicate(
            {"x": (1.0, 0.2), "y": (-1.2, 0.3)}, 1.0, 0.1
        ),
    }


CASES = {
    "aipw": case_aipw,
    "did": case_did,
    "iv": case_iv,
    "rd": case_rd,
    "synth": case_synth,
    "sensitivity": case_sensitivity,
    "tier": case_tier,
    "battery": case_battery,
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_estimators_reproduce_their_recorded_outputs(name):
    expected, actual = golden.check(FIXTURE, name, CASES[name])
    assert golden.same(expected, actual)


def test_the_boosted_learner_gives_a_finite_estimate():
    nuis = aipw.fit_nuisances(
        aipw_tests.planted(600, 1), COLUMNS, "t", "y", 0, "boosted"
    )
    result = aipw.estimate(
        nuis, aipw_tests.planted(800, 2), "t", "y", "ATE"
    )
    assert np.isfinite(golden.snapshot(result)["estimate"])
