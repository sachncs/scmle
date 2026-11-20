"""The identification tier's shortfall when no sensitivity analysis ran."""

import types

from scmle.causal.identification import tier


def test_a_missing_sensitivity_analysis_is_the_only_shortfall() -> None:
    facts = types.SimpleNamespace(
        ran=True,
        negative=True,
        gate="battery",
        replicated=True,
        holds=None,
    )
    found = tier.diagnose(facts, {}, {"battery"})
    assert found == ["no sensitivity analysis was declared or possible"]
