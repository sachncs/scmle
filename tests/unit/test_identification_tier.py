"""The tier rule: each condition, each cap, and the held-out-only world."""

import dataclasses

import pytest

from scmle.causal.identification import tier
from scmle.contracts.qualification import Tier

NEEDED = ("positivity", "no_interference")
GATES = {"overlap", "coverage_aipw", "battery_nco", "sensitivity_bound"}


def facts(**changes) -> tier.Inputs:
    values = {
        "identified": True,
        "needed": NEEDED,
        "attestors": dict.fromkeys(NEEDED, "A. Analyst, 2026-11-04"),
        "confirmatory": True,
        "sound": True,
        "gates": ("overlap",),
        "interval": "coverage_aipw",
        "ran": True,
        "passed": True,
        "negative": True,
        "gate": "battery_nco",
        "replicated": True,
        "holds": True,
        "sensitivity": "sensitivity_bound",
        "unit": False,
        "instrument": False,
        "certified": GATES,
    }
    values.update(changes)
    return tier.Inputs(**values)


def test_all_five_conditions_with_certificates_confirm():
    out = tier.decide(facts())
    assert out.tier == Tier.CONFIRMED and all(out.conditions.values())
    assert out.reasons == () and out.action == ""


def test_not_identified_has_no_tier_and_refuses():
    out = tier.decide(facts(identified=False))
    assert out.tier is None and out.status == tier.NOT_IDENTIFIED
    assert out.action == "refuse_claim"
    assert out.to_dict()["tier"] is None


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"negative": False}, "no negative control"),
        ({"ran": False}, "no negative control"),
        ({"replicated": None}, "fewer than two strata"),
        ({"replicated": False}, "disagree"),
        ({"holds": None}, "no sensitivity"),
        ({"holds": False}, "sensitivity gate failed"),
        ({"unit": True}, "nothing to hold out"),
        ({"instrument": True}, "weak instrument"),
        ({"checked": False}, "unchecked attestation"),
        ({"confirmatory": False}, "did not use C"),
        ({"certified": GATES - {"overlap"}}, "design gate"),
        ({"certified": GATES - {"coverage_aipw"}}, "design gate"),
        ({"certified": GATES - {"battery_nco"}}, "battery has no passing"),
        (
            {"certified": GATES - {"sensitivity_bound"}},
            "sensitivity analysis has no",
        ),
    ],
)
def test_each_missing_condition_caps_the_tier_at_attested(changes, reason):
    out = tier.decide(facts(**changes))
    assert out.tier == Tier.ATTESTED
    assert any(reason in r for r in out.reasons), out.reasons
    assert not all(out.conditions.values())


@pytest.mark.parametrize(
    "changes",
    [
        {"attestors": {"positivity": "A"}},
        {"attestors": {"positivity": " ", "no_interference": "B"}},
        {"sound": False},
        {"passed": False},
    ],
)
def test_missing_attestation_failed_design_or_rejection_is_unverified(changes):
    # Enforces website/src/pages/docs/causal-decision.astro:64.
    out = tier.decide(facts(**changes))
    assert out.tier == Tier.UNVERIFIED and out.action == "downgrade_tier"


def test_a_sensitivity_gate_is_not_needed_when_the_strategy_has_none():
    out = tier.decide(facts(sensitivity=None, holds=True))
    assert out.tier == Tier.CONFIRMED


def test_the_held_out_estimate_alone_never_confirms():
    # significant estimate on C, confounded by an unobserved variable, but no
    # negative control, no strata, no sensitivity analysis declared
    # Enforces docs/WHAT_SCMLE_CANNOT_DO.md:126.
    # Enforces docs/WHEN_THINGS_GO_WRONG.md:368.
    # Enforces website/src/learn/what-scmle-cannot-do.html:27-28.
    # Enforces website/src/learn/when-things-go-wrong.html:168.
    # Enforces website/src/pages/docs/causal-decision.astro:64.
    out = tier.decide(
        facts(
            ran=False,
            negative=False,
            replicated=None,
            holds=None,
        )
    )
    assert out.tier != Tier.CONFIRMED
    assert out.conditions["estimated_on_c"]


def test_the_tier_is_a_function_of_the_facts_only():
    one, two = facts(), dataclasses.replace(facts())
    assert tier.decide(one) == tier.decide(two)
