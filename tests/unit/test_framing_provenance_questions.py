import numpy

from scmle.assurance.framing import provenance
from scmle.assurance.framing import questions
from scmle.contracts import qualification


def test_unsigned_llm_field_caps_the_claim_and_a_signature_lifts_it():
    # Enforces docs/ARTIFACTS.md:82.
    sources = [provenance.Filling.TEMPLATE, provenance.Filling.LLM]
    assert (
        provenance.grade(qualification.Tier.CONFIRMED, sources, signed=False)
        is qualification.Tier.UNVERIFIED
    )
    assert (
        provenance.grade(qualification.Tier.CONFIRMED, sources, signed=True)
        is qualification.Tier.CONFIRMED
    )
    human = [provenance.Filling.HUMAN]
    assert (
        provenance.grade(qualification.Tier.CONFIRMED, human, signed=False)
        is qualification.Tier.CONFIRMED
    )
    assert (
        provenance.grade(
            qualification.Tier.CONFIRMED,
            human,
            signed=True,
            cost_from_data=True,
        )
        is qualification.Tier.UNVERIFIED
    )


def flip_at(point):
    return lambda v: "a" if v < point else "b"


def test_tipping_point_is_found_and_robust_unknowns_are_not_asked():
    flips = questions.Unknown("t", 0.05, 0.5, "a threshold", flip_at(0.3))
    steady = questions.Unknown("u", 0.0, 1.0, "a number", lambda v: "same")
    asked, robust = questions.ask([flips, steady])
    assert [q.slot for q in asked] == ["t"]
    assert abs(asked[0].tipping_value - 0.3) < 1e-9
    assert asked[0].verdicts == ("a", "b")
    assert robust == [{"slot": "u", "low": 0.0, "high": 1.0, "verdict": "same"}]


def test_derivable_unknowns_are_not_asked():
    flips = questions.Unknown("t", 0.05, 0.5, "a threshold", flip_at(0.3))
    assert questions.ask([flips], derivable={"t"}) == ([], [])


def test_gain_sign_verdict_flips_with_the_threshold():
    rng = numpy.random.default_rng(5)
    y = (rng.random(2000) < 0.3).astype(float)
    a = (rng.random(2000) < numpy.where(y == 1, 0.8, 0.45)).astype(float)
    b = (rng.random(2000) < numpy.where(y == 1, 0.55, 0.15)).astype(float)
    found = questions.tipover(
        questions.Unknown("t", 0.05, 0.5, "x", questions.judge(y, a, b))
    )
    assert found is not None and 0.05 < found.tipping_value < 0.5
