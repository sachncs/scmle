"""Unit tests of the stability assessment, imported by its full name."""

import math

import numpy
import pytest

from scmle.assurance.charter import contracts
from scmle.assurance.detectors import certificates
from scmle.assurance.robustness import multiverse
from scmle.assurance.robustness.stability import assessment
from scmle.assurance.robustness.stability import estimation
from scmle.contracts import errors
from scmle.contracts import qualification

CHOICES = (
    contracts.Choice("imputation", ("median", "complete_case"), "median"),
    contracts.Choice("outlier_rule", ("none", "winsorize", "iqr_drop"), "none"),
    contracts.Choice("inclusion", ("all", "first_half", "second_half"), "all"),
)


def grid():
    return multiverse.universe(CHOICES, 0, 18)


def data(rows=300, beta=0.6, seed=0):
    rng = numpy.random.default_rng(seed)
    x = rng.normal(size=rows)
    y = beta * x + rng.normal(size=rows)
    x[rng.random(rows) < 0.1] = numpy.nan
    return x, y


def variant(effect, low, high, sha="s"):
    return estimation.Variant(sha, {}, effect, low, high)


def test_assess_needs_matching_lengths():
    with pytest.raises(errors.InputError, match="one entry per row"):
        assessment.assess(
            "f", numpy.zeros(3), numpy.zeros(4), grid(), 0, bootstrap=2
        )


def test_a_strong_finding_is_stable_and_passes():
    x, y = data()
    out = assessment.assess("f", x, y, grid(), 0, bootstrap=20)
    assert out.status == estimation.STABLE
    assert out.name == "f"
    assert out.headline is not None
    assert out.headline > 0.4
    assert len(out.specs) == 18
    assert out.curve is None
    check = out.check
    assert check.identifier == "f6-stability:f"
    assert check.decision is qualification.Decision.PASS
    assert check.n == 300
    assert check.statistic == out.numbers["sign_share"]
    assert check.effect == out.numbers["median"]
    assert check.interval == (
        min(s.low for s in out.specs),
        max(s.high for s in out.specs),
    )
    assert check.seeds == (0, "f6.stability.f", 0)
    assert check.subset is qualification.Subset.EXPLORATION
    assert check.tier is qualification.Tier.UNVERIFIED


def test_a_curve_test_runs_only_when_asked():
    x, y = data()
    out = assessment.assess(
        "f", x, y, grid(), 0, bootstrap=5, curve_permutations=20
    )
    assert out.curve is not None
    assert out.check.p_value == out.curve.p_median


def test_too_few_rows_are_not_evaluable_without_a_check():
    x, y = data(rows=8)
    out = assessment.assess(
        "f", x, y, grid(), 0, bootstrap=5, curve_permutations=10
    )
    assert out.status == estimation.NOT_EVALUABLE
    assert out.check is None
    assert out.curve is None


def test_certify_is_none_for_a_finding_that_is_not_evaluable():
    out = assessment.certify(
        "f",
        (),
        {},
        estimation.NOT_EVALUABLE,
        None,
        numpy.zeros(3),
        0,
        certificates.Authority.GATING,
    )
    assert out is None


@pytest.mark.parametrize(
    ("gate", "decision"),
    [
        (certificates.Authority.GATING, qualification.Decision.QUARANTINE),
        (certificates.Authority.DESCRIPTIVE, qualification.Decision.WARN),
    ],
)
def test_a_fragile_finding_is_quarantined_only_when_gated(gate, decision):
    check = assessment.certify(
        "f",
        (variant(0.1, -0.2, 0.3), variant(-0.1, -0.4, 0.05)),
        {"sign_share": 0.5, "median": 0.0},
        estimation.FRAGILE,
        None,
        numpy.zeros(10),
        1,
        gate,
    )
    assert check.decision is decision
    assert check.interval == (-0.4, 0.3)
    assert check.statistic == 0.5


def test_certify_falls_back_to_the_median_as_interval():
    check = assessment.certify(
        "f",
        (variant(0.2, None, None),),
        {"sign_share": None, "median": 0.2},
        estimation.STABLE,
        None,
        numpy.zeros(5),
        0,
        certificates.Authority.DESCRIPTIVE,
    )
    assert check.interval == (0.2, 0.2)
    assert check.statistic == 0.0
    assert check.p_value is None
    unmeasured = assessment.certify(
        "f",
        (variant(0.2, None, None),),
        {"sign_share": 1.0, "median": None},
        estimation.STABLE,
        None,
        numpy.zeros(5),
        0,
        certificates.Authority.DESCRIPTIVE,
    )
    assert unmeasured.interval == (0.0, 0.0)
    assert unmeasured.effect == 0.0


def test_to_dict_writes_non_finite_numbers_as_none():
    verdict = assessment.Stability(
        "f",
        "fragile",
        "why",
        0.1,
        {"median": math.nan, "sign_share": 0.5, "ratio": None},
        (variant(0.1, 0.0, 0.2),),
        (("h", "budget"),),
        {},
        None,
        None,
    )
    out = verdict.to_dict()
    assert out["numbers"] == {"median": None, "sign_share": 0.5, "ratio": None}
    assert out["skipped"] == [["h", "budget"]]
    assert out["specs"] == [variant(0.1, 0.0, 0.2).to_dict()]
    assert out["curve"] is None
    assert out["check"] is None
    assert (out["name"], out["status"], out["reason"]) == (
        "f",
        "fragile",
        "why",
    )


def test_code_digest_is_a_stable_sha256():
    digest = assessment.fingerprint()
    assert len(digest) == 64
    assert digest == assessment.fingerprint()


class Fake:
    """A store that answers one status for every defect type."""

    def __init__(self, answers):
        self.answers = answers
        self.asked = []

    def status(self, detector, digest, defect):
        self.asked.append((detector, defect))
        return self.answers.get(defect, certificates.Authority.DESCRIPTIVE)


def test_adjudication_gates_only_when_every_defect_is_certified():
    assert assessment.adjudication(None) is certificates.Authority.DESCRIPTIVE
    assert (
        assessment.adjudication(certificates.Store())
        is certificates.Authority.DESCRIPTIVE
    )
    full = Fake(
        dict.fromkeys(estimation.DEFECT_TYPES, certificates.Authority.GATING)
    )
    assert assessment.adjudication(full) is certificates.Authority.GATING
    assert [pair[1] for pair in full.asked] == list(estimation.DEFECT_TYPES)
    partial = Fake({estimation.DEFECT_TYPES[0]: certificates.Authority.GATING})
    assert (
        assessment.adjudication(partial) is certificates.Authority.DESCRIPTIVE
    )
