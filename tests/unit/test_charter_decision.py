"""The charter's decision block: validation and round trips."""

from fractions import Fraction

import pytest

from scmle.data.charter import decision as d


def plan(**overrides):
    values = {
        "unit": "user",
        "primary": "conversion",
        "assignment_prob": 0.5,
        "delta_min": 0.005,
        "tolerance": 0.01,
        "n_max": 1000,
        "alpha_promote": Fraction(1, 50),
        "alpha_rollback": Fraction(1, 100),
        "guardrails": (d.Guardrail("latency", margin=20.0, low=0, high=500),),
    }
    values.update(overrides)
    return d.RolloutPlan(**values)


def test_the_full_block_round_trips_through_its_record():
    layer = d.DecisionLayer(
        model=d.DecisionModel(
            unit="EUR per case",
            treatment_cost=2.0,
            off_policy=d.OffPolicyRules(clip=20.0, penalty=0.01),
        ),
        fairness=(
            d.FairnessConstraint(
                "sex", "equal_opportunity", 0.05, "group_thresholds", "legal"
            ),
        ),
        rollout=plan(),
    )
    assert d.DecisionLayer.from_dict(layer.to_dict()) == layer
    assert "rollout" in layer.to_dict()
    bare = d.DecisionLayer(d.DecisionModel("EUR"))
    assert "rollout" not in bare.to_dict()
    assert d.DecisionLayer.from_dict(bare.to_dict()) == bare


def test_group_thresholds_need_an_attestation():
    with pytest.raises(ValueError, match="attestation"):
        d.FairnessConstraint(
            "sex", "equal_opportunity", 0.05, "group_thresholds"
        )
    d.FairnessConstraint("sex", "equal_opportunity", 0.05, "reweighing")


@pytest.mark.parametrize(
    "build",
    [
        lambda: d.FairnessConstraint("s", "unknown", 0.1, "reweighing"),
        lambda: d.FairnessConstraint("s", "equalized_odds", 0.0, "reweighing"),
        lambda: d.FairnessConstraint("s", "equalized_odds", 0.1, "magic"),
        lambda: d.FairnessConstraint(
            "s", "equalized_odds", 0.1, "reweighing", min_group=1
        ),
        lambda: d.DecisionModel("u", baselines=("treat_all", "treat_all")),
        lambda: d.DecisionModel("u", baselines=("best",)),
        lambda: d.DecisionModel("u", calibration_level=1.0),
        lambda: d.OffPolicyRules(clip=0.5),
        lambda: d.OffPolicyRules(min_propensity=0.0),
        lambda: d.OffPolicyRules(penalty=-1.0),
        lambda: d.Guardrail("g", margin=0.0),
        lambda: d.Guardrail("g", margin=1.0, low=1.0, high=1.0),
        lambda: plan(assignment_prob=1.0),
        lambda: plan(delta_min=2.0),
        lambda: plan(alpha_promote=Fraction(1, 1)),
        lambda: plan(n_max=1),
        lambda: plan(guardrails=(d.Guardrail("conversion", margin=0.1),)),
        lambda: plan(srm_level=0.0),
    ],
)
def test_invalid_blocks_are_refused(build):
    with pytest.raises(ValueError):
        build()


def test_the_plan_builds_its_budget_and_rejects_unknown_fields():
    rollout = plan(horizon=5)
    budget = rollout.budget()
    assert budget.alpha_promote == Fraction(1, 50) and budget.horizon == 5
    record = rollout.to_dict()
    with pytest.raises(ValueError, match="unexpected"):
        d.RolloutPlan.from_dict({**record, "x": 1})
    with pytest.raises(ValueError, match="missing"):
        d.RolloutPlan.from_dict(
            {k: v for k, v in record.items() if k != "salt"}
        )


# --- the block inside the charter ---------------------------------------


def with_cost_model():
    from scmle.data.framing import costs

    return costs.DecisionCostModel(
        0.0,
        100.0,
        5.0,
        0.0,
        0.02,
        0.15,
        (costs.CostSource.HUMAN,) * 4,
        costs.CostSource.HUMAN,
    )


def layer():
    return d.DecisionLayer(
        d.DecisionModel("EUR per case"),
        rollout=plan(),
    )


def test_a_charter_without_the_layer_keeps_its_hash_and_format():
    from tests.unit.test_charter_contracts import make_charter
    from tests.unit.test_charter_rules import FROZEN_HASH

    charter = make_charter()
    assert charter.decision is None
    assert charter.charter_hash == FROZEN_HASH
    assert "decision" not in charter.to_dict()
    assert charter.format_version == 2


def test_the_layer_changes_the_hash_round_trips_and_needs_format_five():
    import json

    from scmle.data.charter import contracts
    from tests.unit.test_charter_contracts import make_charter

    base = make_charter(cost_model=with_cost_model())
    charter = make_charter(cost_model=with_cost_model(), decision=layer())
    assert charter.charter_hash != base.charter_hash
    assert charter.format_version == 5
    record = json.loads(json.dumps(charter.to_dict()))
    assert contracts.AnalysisCharter.from_dict(record) == charter
    other = make_charter(
        cost_model=with_cost_model(),
        decision=d.DecisionLayer(
            d.DecisionModel("EUR per case", calibration_level=0.05),
            rollout=plan(),
        ),
    )
    assert other.charter_hash != charter.charter_hash


def test_a_layer_without_a_cost_model_is_refused():
    from tests.unit.test_charter_contracts import make_charter

    with pytest.raises(ValueError, match="needs a cost_model"):
        make_charter(decision=layer())


def test_a_layer_in_an_older_format_is_refused(tmp_path):
    import json

    from scmle.contracts import errors
    from scmle.data.charter import sealing
    from tests.unit.test_charter_contracts import make_charter

    charter = make_charter(cost_model=with_cost_model(), decision=layer())
    wrapper = sealing.envelope(charter, "k")
    assert wrapper["format"] == 5
    wrapper["format"] = 4
    path = tmp_path / "charter.json"
    path.write_text(json.dumps(wrapper))
    with pytest.raises(errors.ConfigError, match="decision"):
        sealing.read_verified(path, signing_key="k")
