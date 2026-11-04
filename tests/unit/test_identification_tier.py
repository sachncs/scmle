"""The tier rule: each condition, each cap, and the held-out-only world."""

import dataclasses

import pytest

from scmle.contracts.qualification import Tier
from scmle.data.identification import tier

NEEDED = ("positivity", "no_interference")
GATES = {"overlap", "coverage_aipw", "battery_nco", "sensitivity_bound"}


def facts(**changes) -> tier.TierInputs:
    values = {
        "identified": True,
        "needed": NEEDED,
        "attestors": dict.fromkeys(NEEDED, "A. Analyst, 2026-11-04"),
        "estimated_on_c": True,
        "design_passed": True,
        "design_gates": ("overlap",),
        "interval_gate": "coverage_aipw",
        "battery_ran": True,
        "battery_passed": True,
        "negative_control_declared": True,
        "battery_gate": "battery_nco",
        "replication_passed": True,
        "sensitivity_holds": True,
        "sensitivity_gate": "sensitivity_bound",
        "single_unit": False,
        "weak_instrument": False,
        "certified": GATES,
    }
    values.update(changes)
    return tier.TierInputs(**values)


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
        ({"negative_control_declared": False}, "no negative control"),
        ({"battery_ran": False}, "no negative control"),
        ({"replication_passed": None}, "fewer than two strata"),
        ({"replication_passed": False}, "disagree"),
        ({"sensitivity_holds": None}, "no sensitivity"),
        ({"sensitivity_holds": False}, "sensitivity gate failed"),
        ({"single_unit": True}, "nothing to hold out"),
        ({"weak_instrument": True}, "weak instrument"),
        ({"estimated_on_c": False}, "did not use C"),
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
        {"design_passed": False},
        {"battery_passed": False},
    ],
)
def test_missing_attestation_failed_design_or_rejection_is_unverified(changes):
    out = tier.decide(facts(**changes))
    assert out.tier == Tier.UNVERIFIED and out.action == "downgrade_tier"


def test_a_sensitivity_gate_is_not_needed_when_the_strategy_has_none():
    out = tier.decide(facts(sensitivity_gate=None, sensitivity_holds=True))
    assert out.tier == Tier.CONFIRMED


def test_the_held_out_estimate_alone_never_confirms():
    # significant estimate on C, confounded by an unobserved variable, but no
    # negative control, no strata, no sensitivity analysis declared
    out = tier.decide(
        facts(
            battery_ran=False,
            negative_control_declared=False,
            replication_passed=None,
            sensitivity_holds=None,
        )
    )
    assert out.tier != Tier.CONFIRMED
    assert out.conditions["estimated_on_c"]


def test_the_tier_is_a_function_of_the_facts_only():
    one, two = facts(), dataclasses.replace(facts())
    assert tier.decide(one) == tier.decide(two)
