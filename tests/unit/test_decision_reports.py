"""The F11 report builders: policy value, fairness and the rollout design."""

from fractions import Fraction

import numpy as np
import pandas as pd

from scmle.core import decision_reports
from scmle.data.charter import decision as decision_module
from scmle.data.framing import costs
from scmle.decision import rollout
from tests.unit.test_charter_contracts import make_charter


def layer(fairness=(), plan=None):
    return decision_module.DecisionLayer(
        decision_module.DecisionModel("EUR"), fairness, plan
    )


def plan():
    return decision_module.RolloutPlan(
        unit="user",
        primary="conv",
        assignment_prob=0.5,
        delta_min=0.05,
        tolerance=0.02,
        n_max=500,
        alpha_promote=Fraction(1, 20),
        alpha_rollback=Fraction(1, 20),
    )


def charter(**kwargs):
    human = (costs.CostSource.HUMAN,) * 4
    model = costs.DecisionCostModel(
        0.0, 2.0, 1.0, 0.0, 0.1, 0.5, human, costs.CostSource.HUMAN
    )
    return make_charter(cost_model=model, decision=layer(**kwargs))


def test_without_a_randomized_log_the_value_is_not_assessable():
    found = decision_reports.policy_value(
        layer(), np.ones(5), np.ones(5), None, None
    )
    assert found["status"] == decision_reports.NOT_ASSESSABLE
    assert "never estimated" in found["reason"]


def test_bad_propensities_are_refused_not_clipped():
    found = decision_reports.policy_value(
        layer(),
        np.ones(4),
        np.ones(4),
        np.ones(4),
        np.array([0.5, 1, 0.5, 0.5]),
    )
    assert found["status"] == decision_reports.NOT_ASSESSABLE


def test_a_randomized_log_gives_a_value_near_the_truth():
    rng = np.random.default_rng(0)
    n = 4000
    logged = (rng.random(n) < 0.5).astype(float)
    outcome = (rng.random(n) < 0.3 + 0.2 * logged).astype(float)
    acts = np.ones(n)  # treat everyone: true value 0.5
    found = decision_reports.policy_value(
        layer(), acts, outcome, logged, np.full(n, 0.5)
    )
    assert found["status"] == "issued"
    low, high = found["interval"]
    assert low < 0.5 < high


def test_fairness_reports_each_constraint_and_missing_columns():
    rng = np.random.default_rng(1)
    n = 600
    frame = pd.DataFrame({"g": rng.choice(["a", "b"], n)})
    outcome = (rng.random(n) < 0.5).astype(float)
    acts = (rng.random(n) < 0.5).astype(float)
    items = (
        decision_module.FairnessConstraint(
            "g", "demographic_parity", 0.3, "reweighing"
        ),
        decision_module.FairnessConstraint(
            "absent", "demographic_parity", 0.3, "reweighing"
        ),
    )
    found = decision_reports.fairness(layer(items), frame, outcome, acts, 0.05)
    assert found[0]["attribute"] == "g" and found[0]["status"] in {
        "satisfied",
        "not_satisfied",
    }
    assert found[1]["status"] == decision_reports.NOT_ASSESSABLE


def test_the_rollout_design_has_a_hashed_log_and_no_assignments():
    built = decision_reports.rollout_design(charter(plan=plan()), 0.4)
    record, log = built
    assert record["ran"] is False and record["fixed_horizon_rows"] > 0
    assert record["expected_sample_number"] is not None
    assert log.verify() and len(log.rows) == 1
    assert log.rows[0]["body"]["plan_hash"] == record["plan_hash"]
    assert isinstance(log, rollout.DecisionLog)


def test_no_rollout_plan_means_no_design_and_no_room_means_no_size():
    assert decision_reports.rollout_design(charter(), 0.4) is None
    record, ignored = decision_reports.rollout_design(
        charter(plan=plan()), 0.99
    )
    assert record["fixed_horizon_rows"] is None
