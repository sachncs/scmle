"""The charter's decision block: validation and round trips."""

from fractions import Fraction

import pytest

import scmle.assurance.charter.decision


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
            treatment_cost=2.0,
            off_policy=scmle.assurance.charter.decision.Gates(
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
            "s", "equalized_odds", 0.1, "reweighing", min_group=1
        ),
        lambda: scmle.assurance.charter.decision.Policy(
            "u", baselines=("treat_all", "treat_all")
        ),
        lambda: scmle.assurance.charter.decision.Policy(
            "u", baselines=("best",)
        ),
        lambda: scmle.assurance.charter.decision.Policy(
            "u", calibration_level=1.0
        ),
        lambda: scmle.assurance.charter.decision.Gates(clip=0.5),
        lambda: scmle.assurance.charter.decision.Gates(min_propensity=0.0),
        lambda: scmle.assurance.charter.decision.Gates(penalty=-1.0),
        lambda: scmle.assurance.charter.decision.Guardrail("g", margin=0.0),
        lambda: scmle.assurance.charter.decision.Guardrail(
            "g", margin=1.0, low=1.0, high=1.0
        ),
        lambda: plan(assignment_prob=1.0),
        lambda: plan(delta_min=2.0),
        lambda: plan(alpha_promote=Fraction(1, 1)),
        lambda: plan(n_max=1),
        lambda: plan(
            guardrails=(
                scmle.assurance.charter.decision.Guardrail(
                    "conversion", margin=0.1
                ),
            )
        ),
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
    from tests.unit.test_charter_contracts import make_charter
    from tests.unit.test_charter_rules import FROZEN_HASH

    charter = make_charter()
    assert charter.decision is None
    assert charter.charter_hash == FROZEN_HASH
    assert "decision" not in charter.to_dict()
    assert charter.format_version == 2


def test_the_layer_changes_the_hash_round_trips_and_needs_format_five():
    import json

    from scmle.assurance.charter import contracts
    from tests.unit.test_charter_contracts import make_charter

    base = make_charter(cost_model=with_cost_model())
    charter = make_charter(cost_model=with_cost_model(), decision=layer())
    assert charter.charter_hash != base.charter_hash
    assert charter.format_version == 5
    record = json.loads(json.dumps(charter.to_dict()))
    assert contracts.AnalysisCharter.from_dict(record) == charter
    other = make_charter(
        cost_model=with_cost_model(),
        decision=scmle.assurance.charter.decision.Layer(
            scmle.assurance.charter.decision.Policy(
                "EUR per case", calibration_level=0.05
            ),
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

    from scmle.assurance.charter import sealing
    from scmle.contracts import errors
    from tests.unit.test_charter_contracts import make_charter

    charter = make_charter(cost_model=with_cost_model(), decision=layer())
    wrapper = sealing.envelope(charter, "k")
    assert wrapper["format"] == 5
    wrapper["format"] = 4
    path = tmp_path / "charter.json"
    path.write_text(json.dumps(wrapper))
    with pytest.raises(errors.ConfigError, match="decision"):
        sealing.read_verified(path, signing_key="k")
