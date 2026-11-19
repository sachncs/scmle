"""The charter's decision block: validation and round trips."""

from fractions import Fraction

import pytest

import scmle.assurance.charter.decision


def plan(**overrides):
    values = {
        "unit": "user",
        "primary": "conversion",
        "assignment": 0.5,
        "delta": 0.005,
        "tolerance": 0.01,
        "limit": 1000,
        "promotion": Fraction(1, 50),
        "rollback": Fraction(1, 100),
        "guardrails": (
            scmle.assurance.charter.decision.Guardrail(
                "latency", margin=20.0, low=0, high=500
            ),
        ),
    }
    values.update(overrides)
    return scmle.assurance.charter.decision.Experiment(**values)


def test_the_full_block_round_trips_through_its_record():
    layer = scmle.assurance.charter.decision.Layer(
        model=scmle.assurance.charter.decision.Policy(
            unit="EUR per case",
            cost=2.0,
            gates=scmle.assurance.charter.decision.Gates(
                clip=20.0, penalty=0.01
            ),
        ),
        fairness=(
            scmle.assurance.charter.decision.Constraint(
                "sex", "equal_opportunity", 0.05, "group_thresholds", "legal"
            ),
        ),
        rollout=plan(),
    )
    assert (
        scmle.assurance.charter.decision.Layer.from_dict(layer.to_dict())
        == layer
    )
    assert "rollout" in layer.to_dict()
    bare = scmle.assurance.charter.decision.Layer(
        scmle.assurance.charter.decision.Policy("EUR")
    )
    assert "rollout" not in bare.to_dict()
    assert (
        scmle.assurance.charter.decision.Layer.from_dict(bare.to_dict()) == bare
    )


def test_group_thresholds_need_an_attestation():
    with pytest.raises(ValueError, match="attestation"):
        scmle.assurance.charter.decision.Constraint(
            "sex", "equal_opportunity", 0.05, "group_thresholds"
        )
    scmle.assurance.charter.decision.Constraint(
        "sex", "equal_opportunity", 0.05, "reweighing"
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda: scmle.assurance.charter.decision.Constraint(
            "s", "unknown", 0.1, "reweighing"
        ),
        lambda: scmle.assurance.charter.decision.Constraint(
            "s", "equalized_odds", 0.0, "reweighing"
        ),
        lambda: scmle.assurance.charter.decision.Constraint(
            "s", "equalized_odds", 0.1, "magic"
        ),
        lambda: scmle.assurance.charter.decision.Constraint(
            "s", "equalized_odds", 0.1, "reweighing", minimum=1
        ),
        lambda: scmle.assurance.charter.decision.Policy(
            "u", baselines=("treat_all", "treat_all")
        ),
        lambda: scmle.assurance.charter.decision.Policy(
            "u", baselines=("best",)
        ),
        lambda: scmle.assurance.charter.decision.Policy("u", calibration=1.0),
        lambda: scmle.assurance.charter.decision.Gates(clip=0.5),
        lambda: scmle.assurance.charter.decision.Gates(propensity=0.0),
        lambda: scmle.assurance.charter.decision.Gates(penalty=-1.0),
        lambda: scmle.assurance.charter.decision.Guardrail("g", margin=0.0),
        lambda: scmle.assurance.charter.decision.Guardrail(
            "g", margin=1.0, low=1.0, high=1.0
        ),
        lambda: plan(assignment=1.0),
        lambda: plan(delta=2.0),
        lambda: plan(promotion=Fraction(1, 1)),
        lambda: plan(limit=1),
        lambda: plan(
            guardrails=(
                scmle.assurance.charter.decision.Guardrail(
                    "conversion", margin=0.1
                ),
            )
        ),
        lambda: plan(srm=0.0),
    ],
)
def test_invalid_blocks_are_refused(build):
    with pytest.raises(ValueError):
        build()


def test_the_plan_builds_its_budget_and_rejects_unknown_fields():
    rollout = plan(horizon=5)
    budget = rollout.budget()
    assert budget.promotion == Fraction(1, 50) and budget.horizon == 5
    record = rollout.to_dict()
    with pytest.raises(ValueError, match="unexpected"):
        scmle.assurance.charter.decision.Experiment.from_dict(
            {**record, "x": 1}
        )
    with pytest.raises(ValueError, match="missing"):
        scmle.assurance.charter.decision.Experiment.from_dict(
            {k: v for k, v in record.items() if k != "salt"}
        )


# --- the block inside the charter ---------------------------------------


def with_cost_model():
    from scmle.assurance.framing import costs

    return costs.Tariff(
        0.0,
        100.0,
        5.0,
        0.0,
        0.02,
        0.15,
        (costs.Basis.HUMAN,) * 4,
        costs.Basis.HUMAN,
    )


def layer():
    return scmle.assurance.charter.decision.Layer(
        scmle.assurance.charter.decision.Policy("EUR per case"),
        rollout=plan(),
    )


def test_a_charter_without_the_layer_keeps_its_hash_and_format():
    # Enforces website/src/pages/docs/api-stability.astro:51.
    # Enforces website/src/pages/docs/qualification.astro:67.
    from tests.unit.test_charter_contracts import make_charter
    from tests.unit.test_charter_rules import FROZEN_HASH

    charter = make_charter()
    assert charter.decision is None
    assert charter.digest == FROZEN_HASH
    assert "decision" not in charter.to_dict()
    assert charter.format_version == 2


def test_the_layer_changes_the_hash_round_trips_and_needs_format_five():
    # Enforces website/src/pages/docs/api-stability.astro:75.
    # Enforces website/src/pages/docs/causal-decision.astro:69.
    import json

    from scmle.assurance.charter import contracts
    from tests.unit.test_charter_contracts import make_charter

    base = make_charter(costs=with_cost_model())
    charter = make_charter(costs=with_cost_model(), decision=layer())
    assert charter.digest != base.digest
    assert charter.format_version == 5
    record = json.loads(json.dumps(charter.to_dict()))
    assert contracts.Charter.from_dict(record) == charter
    other = make_charter(
        costs=with_cost_model(),
        decision=scmle.assurance.charter.decision.Layer(
            scmle.assurance.charter.decision.Policy(
                "EUR per case", calibration=0.05
            ),
            rollout=plan(),
        ),
    )
    assert other.digest != charter.digest


def test_a_layer_without_a_cost_model_is_refused():
    from tests.unit.test_charter_contracts import make_charter

    with pytest.raises(ValueError, match="needs costs"):
        make_charter(decision=layer())


def test_a_layer_in_an_older_format_is_refused(tmp_path):
    # Enforces website/src/pages/docs/api-stability.astro:51.
    import json

    from scmle.assurance.charter import sealing
    from scmle.contracts import errors
    from tests.unit.test_charter_contracts import make_charter

    charter = make_charter(costs=with_cost_model(), decision=layer())
    wrapper = sealing.envelope(charter, "k")
    assert wrapper["format"] == 5
    wrapper["format"] = 4
    path = tmp_path / "charter.json"
    path.write_text(json.dumps(wrapper))
    with pytest.raises(errors.ConfigError, match="decision"):
        sealing.read(path, signing_key="k")
