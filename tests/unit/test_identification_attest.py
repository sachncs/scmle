"""Signed attestations of the identification assumptions (F10)."""

import json

import pytest

from scmle.data.charter import causal
from scmle.data.identification import attest
from scmle.data.identification import run
from tests.unit.test_identification_run import ASSUMPTIONS
from tests.unit.test_identification_run import block
from tests.unit.test_identification_run import BUDGET
from tests.unit.test_identification_run import GATES
from tests.unit.test_identification_run import observational

KEY = "attestor-key"
SCOPE = "ab" * 32
TODAY = "2026-11-06"
NEEDED = ASSUMPTIONS
ATTESTORS = (
    causal.Attestor("Dr. A. Rivera", "study statistician", NEEDED[:2]),
    causal.Attestor("J. Okafor", "domain lead", NEEDED[2:]),
)


def record(assumption, attestor="Dr. A. Rivera", **changes):
    values = {
        "assumption": assumption,
        "attestor": attestor,
        "statement": "I judge this assumption to hold for this population",
        "evidence": "design memo 2026-10",
        "date": "2026-11-01",
        "expires": "2027-11-01",
        "scope": SCOPE,
    }
    values.update(changes)
    return attest.Attestation(**values).signed(KEY)


def everything():
    return (
        record(NEEDED[0]),
        record(NEEDED[1]),
        record(NEEDED[2], "J. Okafor"),
    )


def apply(records, key=KEY, today=TODAY, attestors=ATTESTORS):
    return attest.apply(attestors, records, NEEDED, SCOPE, today, key)


def test_valid_signed_attestations_name_every_assumption():
    found = apply(everything())
    assert found.rejected == {}
    assert found.accepted == {
        NEEDED[0]: "Dr. A. Rivera",
        NEEDED[1]: "Dr. A. Rivera",
        NEEDED[2]: "J. Okafor",
    }


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (
            lambda r: attest.Attestation(**{**r.to_dict(), "statement": "x"}),
            "signature",
        ),
        (lambda r: record(r.assumption, scope="cd" * 32), "scope"),
        (lambda r: record(r.assumption, expires="2026-11-02"), "valid"),
        (lambda r: record(r.assumption, date="2026-12-01"), "valid"),
        (lambda r: record(r.assumption, "J. Okafor"), "not permitted"),
        (lambda r: record(r.assumption, "Someone Else"), "not named"),
    ],
)
def test_an_attestation_that_does_not_count_is_rejected_with_a_reason(
    mutate, reason
):
    first = everything()[0]
    found = apply((mutate(first), *everything()[1:]))
    assert NEEDED[0] not in found.accepted
    assert reason in found.rejected[NEEDED[0]]
    assert len(found.accepted) == 2


def test_a_wrong_or_missing_key_accepts_nothing():
    assert apply(everything(), key="other").accepted == {}
    missing = apply(everything(), key=None)
    assert missing.accepted == {} and "key" in missing.rejected[NEEDED[0]]


def test_no_attestation_for_an_assumption_is_reported():
    found = apply(everything()[:2])
    assert "no signed attestation" in found.rejected[NEEDED[2]]


def test_the_file_round_trips_and_refuses_duplicates(tmp_path):
    path = tmp_path / "attestations.json"
    items = everything()
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "attestations": [i.to_dict() for i in items],
            }
        )
    )
    assert attest.load(path) == items
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "attestations": [items[0].to_dict(), items[0].to_dict()],
            }
        )
    )
    with pytest.raises(ValueError, match="one attestation"):
        attest.load(path)
    path.write_text(json.dumps({"format_version": 2, "attestations": []}))
    with pytest.raises(ValueError, match="format_version"):
        attest.load(path)


def go(spec, attestors, report=None):
    return run.identify(
        spec,
        observational(1200, 1),
        observational(1200, 2),
        BUDGET,
        attestors=attestors,
        certified=GATES,
        seed=0,
        attestation_report=report,
    )


def test_the_tier_rises_above_unverified_only_with_verified_attestations():
    spec = block()
    verified = apply(everything())
    rejected = apply(everything(), key="other")
    up = go(spec, verified.accepted, verified.to_dict())
    down = go(spec, rejected.accepted, rejected.to_dict())
    assert up.documents["causal_claims.json"]["tier"] in (
        "attested",
        "confirmed",
    )
    claims = down.documents["causal_claims.json"]
    assert claims["tier"] == "unverified"
    assert "no named attestor" in " ".join(claims["reasons"])
    assert claims["attestations"]["accepted"] == []
    assert claims["attestations"]["rejected"]
    accepted = up.documents["causal_claims.json"]["attestations"]["accepted"]
    assert {a["assumption"] for a in accepted} == set(NEEDED)
