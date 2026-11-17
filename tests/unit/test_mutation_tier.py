"""The identification tier's shortfall when no sensitivity analysis ran."""

import types

from scmle.causal.identification import tier


def test_a_missing_sensitivity_analysis_is_the_only_shortfall() -> None:
    facts = types.SimpleNamespace(
        battery_ran=True,
        negative_control_declared=True,
        battery_gate="battery",
        replication_passed=True,
        sensitivity_holds=None,
    )
    found = tier.evidence_shortfalls(facts, {}, {"battery"})
    assert found == ["no sensitivity analysis was declared or possible"]
