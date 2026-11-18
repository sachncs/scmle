"""Back-door AIPW and randomized difference adapters match their engines."""

import numpy
import pandas
import pytest

from scmle.assurance.charter import causal
from scmle.causal.identification import aipw
from scmle.causal.identification import sensitivity
from scmle.causal.identification.run import adjustment
from scmle.contracts import errors
from tests.unit.test_identification_run import observational

COVARIATES = "unconfoundedness_given:a,b"


def block(**changes):
    values = {
        "treatment": "t",
        "outcome": "y",
        "estimand": "ATE",
        "assumptions": (COVARIATES, "positivity", "no_interference"),
        "strategies": ("backdoor_aipw",),
        "controls": (
            causal.Control("outcome", "nco"),
            causal.Control("exposure", "e"),
        ),
        "sensitivity": causal.Confounding("strongest_observed", 1.0),
    }
    values.update(changes)
    return causal.Question(**values)


def worlds():
    x = observational(1500, 1)
    c = observational(1500, 2)
    rng = numpy.random.default_rng(5)
    x["e"] = (rng.uniform(size=len(x)) < 0.5).astype(float)
    c["e"] = (rng.uniform(size=len(c)) < 0.5).astype(float)
    return x, c


def test_a_backdoor_adapter_reads_its_covariates_and_starts_unfitted():
    x, ignored = worlds()
    adapter = adjustment.Backdoor(block(), x, 0, "linear", 0.01)
    assert adjustment.Backdoor.name == "backdoor_aipw"
    assert adapter.covariates == ("a", "b")
    assert adapter.nuisances is None
    none = adjustment.Backdoor(
        block(assumptions=("positivity",)), x, 0, "linear", 0.01
    )
    assert none.covariates == ()


def test_a_backdoor_estimate_before_prepare_raises():
    x, c = worlds()
    adapter = adjustment.Backdoor(block(), x, 0, "linear", 0.01)
    with pytest.raises(errors.InputError, match="prepare\\(\\) must run first"):
        adapter.estimate(c, 0.95)


def test_a_backdoor_sensitivity_before_prepare_raises():
    x, c = worlds()
    adapter = adjustment.Backdoor(block(), x, 0, "linear", 0.01)
    with pytest.raises(errors.InputError, match="prepare\\(\\) must run first"):
        adapter.sensitivity(c, 0.95)


def test_a_backdoor_prepare_records_overlap_and_the_nuisance_digest():
    x, ignored = worlds()
    adapter = adjustment.Backdoor(block(), x, 4, "linear", 0.01)
    adapter.prepare()
    assert set(adapter.design) == {
        "overlap_on_x",
        "nuisance_digest",
        "learner",
    }
    assert adapter.design["learner"] == "linear"
    assert adapter.design["nuisance_digest"] == adapter.nuisances.digest
    assert adapter.design["overlap_on_x"]["passed"] is True
    assert adapter.refusal is None


def test_a_backdoor_estimate_equals_the_aipw_engine_on_the_same_rows():
    x, c = worlds()
    spec = block()
    adapter = adjustment.Backdoor(spec, x, 0, "linear", 0.01)
    adapter.prepare()
    est = adapter.estimate(c, 0.9)
    nuisances = aipw.prefit(x, ("a", "b"), "t", "y", 0, "linear")
    direct = aipw.estimate(
        nuisances, c, "t", "y", "ATE", 0.9, spec.trim, spec.retained
    )
    assert (est.estimate, est.se, est.low, est.high, est.n) == (
        direct.estimate,
        direct.se,
        direct.low,
        direct.high,
        direct.n,
    )
    assert est.detail == {"population": direct.population}
    assert adapter.design["overlap_on_c"] == direct.overlap
    assert adapter.design["population"] == direct.population
    assert 1.5 < est.estimate < 2.5


def test_a_backdoor_reckoning_refits_for_another_outcome():
    x, c = worlds()
    spec = block()
    adapter = adjustment.Backdoor(spec, x, 0, "linear", 0.01)
    est = adapter.reckon(c, 0.9, "nco")
    nuisances = aipw.prefit(x, ("a", "b"), "t", "nco", 0, "linear")
    direct = aipw.estimate(
        nuisances, c, "t", "nco", "ATE", 0.9, spec.trim, spec.retained
    )
    assert (est.estimate, est.se, est.low, est.high, est.n) == (
        direct.estimate,
        direct.se,
        direct.low,
        direct.high,
        direct.n,
    )
    assert est.detail == {}
    default = adapter.reckon(c, 0.9)
    assert default.estimate == pytest.approx(
        adapter.reckon(c, 0.9, "y").estimate
    )


def test_backdoor_controls_test_each_outcome_and_exposure():
    x, c = worlds()
    adapter = adjustment.Backdoor(block(), x, 0, "linear", 0.01)
    tests = adapter.controls(c, 0.9)
    assert [(t.name, t.kind) for t in tests] == [
        ("negative_outcome:nco", "negative_outcome"),
        ("negative_exposure:e", "negative_exposure"),
    ]
    nco = adapter.reckon(c, 0.9, "nco")
    assert tests[0].estimate == nco.estimate
    assert tests[0].se == nco.se


def test_backdoor_sensitivity_notes_a_missing_block_or_covariate():
    x, c = worlds()
    plain = adjustment.Backdoor(block(sensitivity=None), x, 0, "linear", 0.01)
    assert plain.sensitivity(c, 0.9) == {
        "holds": None,
        "note": "no sensitivity block in the charter",
    }
    bare = adjustment.Backdoor(
        block(assumptions=("unconfoundedness_given:none", "positivity")),
        x,
        0,
        "linear",
        0.01,
    )
    assert bare.sensitivity(c, 0.9) == {
        "holds": None,
        "note": "benchmarking needs a covariate",
    }


def test_backdoor_sensitivity_assesses_the_linear_summary_on_c():
    x, c = worlds()
    adapter = adjustment.Backdoor(block(), x, 0, "linear", 0.01)
    adapter.prepare()
    out = adapter.sensitivity(c, 0.9)
    matrix = adapter.nuisances.design.matrix(c)
    names = tuple(f"x{j}" for j in range(matrix.shape[1]))

    frame = pandas.DataFrame(matrix, columns=list(names))
    frame["t"] = c["t"].to_numpy(dtype=float)
    frame["y"] = c["y"].to_numpy(dtype=float)
    expected = sensitivity.assess(
        frame, "t", "y", names, "strongest_observed", 1.0, 0.9
    ).to_dict()
    scale = out.pop("scale")
    assert out == expected
    assert scale.startswith("linear-model coefficient of the treatment")


def randomized():
    spec = block(
        strategies=("randomized_diff",),
        assumptions=("randomized_assignment", "no_interference"),
    )
    c = observational(800, 3)
    c["t"] = (numpy.arange(len(c)) % 2).astype(float)
    return adjustment.RandomizedEstimator(spec, c, 0, "linear", 0.01), c


def test_a_randomized_estimate_is_the_difference_in_means():
    adapter, c = randomized()
    assert adjustment.RandomizedEstimator.name == "randomized_diff"
    est = adapter.estimate(c, 0.9)
    direct = aipw.difference_in_means(c, "t", "y", 0.9)
    assert (est.estimate, est.se, est.low, est.high, est.n) == (
        direct.estimate,
        direct.se,
        direct.low,
        direct.high,
        direct.n,
    )
    other = adapter.reckon(c, 0.9, "nco")
    assert (
        other.estimate == aipw.difference_in_means(c, "t", "nco", 0.9).estimate
    )
    assert adapter.reckon(c, 0.9).estimate == est.estimate


def test_randomized_controls_skip_exposures():
    adapter, c = randomized()
    tests = adapter.controls(c, 0.9)
    assert [t.name for t in tests] == ["negative_outcome:nco"]
    assert tests[0].kind == "negative_outcome"


def test_randomized_sensitivity_is_not_applicable():
    adapter, c = randomized()
    assert adapter.sensitivity(c, 0.9) == {
        "holds": True,
        "note": "randomized assignment: no unmeasured confounding to bound",
        "applicable": False,
    }
