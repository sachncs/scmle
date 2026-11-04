"""Snapshots of the decision layer on fixed inputs (mutation pins).

See `tests/unit/golden.py` and `tests/unit/MUTANTS_decision.md`. Regenerate
with SCMLE_WRITE_GOLDEN=1 only after a reviewed change of behaviour.
"""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from scmle.data.charter import decision as charter
from scmle.data.framing import costs
from scmle.decision import anytime
from scmle.decision import constraints
from scmle.decision import design
from scmle.decision import netbenefit
from scmle.decision import ope
from scmle.decision import rollout
from scmle.decision import threshold
from tests.unit import golden

FIXTURE = Path(__file__).parents[1] / "fixtures" / "decision_golden.json"


def plan(**overrides):
    values = {
        "unit": "user",
        "primary": "conversion",
        "assignment_prob": 0.5,
        "delta_min": 0.01,
        "tolerance": 0.01,
        "n_max": 6000,
        "alpha_promote": Fraction(1, 50),
        "alpha_rollback": Fraction(1, 100),
        "guardrails": (charter.Guardrail("bounce", margin=0.05),),
    }
    values.update(overrides)
    return charter.RolloutPlan(**values)


def traffic(n, seed, lift=0.04, bounce_lift=0.0, base=0.10):
    rng = np.random.default_rng(seed)
    arm = (rng.random(n) < 0.5).astype(float)
    primary = (rng.random(n) < base + lift * arm).astype(float)
    bounce = (rng.random(n) < 0.2 + bounce_lift * arm).astype(float)
    return arm, primary, {"bounce": bounce}


def cost_model():
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


def score_world(n, seed):
    rng = np.random.default_rng(seed)
    p = rng.beta(1, 6, n)
    y = (rng.random(n) < p).astype(float)
    return p, y


def fairness(criterion="equal_opportunity", gap=0.05):
    return charter.FairnessConstraint(
        "sex", criterion, gap, "group_thresholds", "counsel"
    )


def fair_world(n, seed, shift=0.6):
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.3).astype(float)
    centre = y * 1.5 - groups * shift * y
    score = 1 / (1 + np.exp(-(rng.normal(centre - 0.75, 1.0))))
    return score, y, groups


def bandit(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 1))
    mu1 = np.clip(0.5 + 0.25 * np.tanh(x[:, 0]), 0.05, 0.95)
    behaviour = np.column_stack([1 - mu1, mu1])
    action = (rng.random(n) < mu1).astype(int)
    mean = np.column_stack([0.3 + 0 * x[:, 0], 0.3 + 0.2 * (x[:, 0] > 0)])
    reward = (rng.random(n) < mean[np.arange(n), action]).astype(float)
    treat = (x[:, 0] > 0).astype(float)
    target = np.column_stack([1 - treat, treat])
    log = ope.LoggedBandit(action, reward, behaviour, target, mean)
    return x, log


def case_anytime():
    stream = anytime.Stream(0.0, 1.0, 0.4)
    values = np.random.default_rng(1).random(40) * 0.5 + 0.3
    process = anytime.EProcess(stream)
    process.update(values[:25])
    process.update(values[25:])
    arm = (np.arange(40) % 2).astype(float)
    return {
        "grid": anytime.bet_grid(0.4, 5),
        "wealth": anytime.log_wealth(values[:6], 0.4, 4),
        "paths": anytime.log_wealth_paths(values[:8], stream, 4),
        "process": process.log_e,
        "max": process.max_log_e,
        "bound": anytime.confidence_bound(values, stream, 0.05, 6, 8),
        "pseudo": anytime.pseudo_outcomes(arm, values, 0.25),
        "pseudo_c": anytime.pseudo_outcomes(arm, values, 0.25, values * 0.9),
        "center": anytime.running_center(values),
        "positive": anytime.positive_level(0.05),
        "centered": anytime.centered_paths(arm, values, 0.5, 0.0),
        "centered_less": anytime.centered_paths(
            arm, values, 0.5, 0.1, "less", -1.0
        ),
        "streams": [
            anytime.centered_stream(0.5, 0.0),
            anytime.difference_stream(0.5, 0.1, "less"),
            anytime.bound_stream(2.0, 0.5),
        ],
    }


def case_design():
    ids = [f"user-{i}" for i in range(500)]
    return {
        "assign": design.assign(ids, "salt", 0.3)[:60],
        "rows": design.fixed_horizon_rows(0.10, 0.02, alpha=0.05, power=0.8),
        "rows_de": design.fixed_horizon_rows(
            0.10, 0.02, alpha=0.05, power=0.9, design_effect=1.5
        ),
        "srm_p": [
            design.srm_p_value(k, 1000, 0.5) for k in (480, 500, 530, 560)
        ],
        "srm_bf": design.srm_log_bayes_factor(
            [10, 60, 130], [20, 100, 200], 0.5
        ),
        "srm_rej": [
            design.srm_rejects(5200, 10000, 0.5, 0.001),
            design.srm_rejects(5600, 10000, 0.5, 0.001),
        ],
        "stopping": design.simulate_stopping(
            plan(n_max=3000), 0.10, 0.05, 6, 3
        ),
    }


def case_rollout():
    arm, primary, guards = traffic(6000, 0)
    run = rollout.Rollout(plan(), "e1", 0.02)
    decisions = [
        run.observe(
            arm[:2000], primary[:2000], {"bounce": guards["bounce"][:2000]}
        ),
        run.observe(arm, primary, guards),
    ]
    arm2, primary2, guards2 = traffic(6000, 3, bounce_lift=0.08)
    harm = rollout.Rollout(plan(), "e2", 0.02).observe(arm2, primary2, guards2)
    monitor = rollout.RollbackMonitor(0.01, 0.05)
    rng = np.random.default_rng(5)
    outcomes = [
        monitor.observe(
            (rng.random(200) < 0.3).astype(float),
            (rng.random(200) < 0.45).astype(float),
        )
        for ignored in range(4)
    ]
    return {
        "decisions": decisions,
        "harm": harm,
        "monitor": outcomes,
        "hash": rollout.row_hash({"a": 1, "b": [1, 2]}),
        "stream": rollout.metric_stream(plan(), None),
        "stream_guard": rollout.metric_stream(
            plan(), charter.Guardrail("bounce", margin=0.05)
        ),
    }


def case_threshold():
    p, y = score_world(2000, 0)
    model = cost_model()
    return {
        "bayes": threshold.bayes_threshold(5, 100),
        "pl": threshold.probabilities_and_labels([0.1, 0.9], [0, 1]),
        "check": threshold.calibration_check(p, y),
        "check_bins": threshold.calibration_check(p, y, 0.05, 5),
        "recal": threshold.fit_recalibrator(p**0.6, y),
        "cross": threshold.cross_fitted(p**0.6, y, 1)[:20],
        "choice": threshold.choose_threshold(p, y, model, replicates=60),
        "choice_skew": threshold.choose_threshold(
            np.clip(p**0.5, 0, 1), y, model, replicates=60
        ),
    }


def case_constraints():
    score, y, groups = fair_world(3000, 0)
    acts = (score > 0.5).astype(float)
    result = {}
    for name in ("equal_opportunity", "demographic_parity"):
        constraint = fairness(name)
        result[name] = {
            "test": constraints.satisfaction_test(y, acts, groups, constraint),
            "fit": constraints.fit_group_thresholds(
                score, y, groups, constraint, 0.05
            ),
        }
    result["wilson"] = constraints.wilson(30, 100, 1.96)
    result["newcombe"] = constraints.newcombe(40, 100, 20, 100, 0.9)
    result["base"] = constraints.base_rates(y, groups)
    result["feasible"] = constraints.check_feasibility(
        [fairness(), fairness("demographic_parity")], y, groups
    )
    result["price"] = constraints.price_of_constraint(
        y, acts, (score > 0.6).astype(float), 0.05, replicates=50, seed=2
    )
    return result


def case_ope():
    x, log = bandit(1500, 0)
    out = {}
    for name in ope.ESTIMATORS:
        out[name] = ope.estimate_value(log, name)
        out[name + "_clip"] = ope.estimate_value(log, name, clip=3.0)
    baseline = np.column_stack([np.ones(1500), np.zeros(1500)])
    out["diff"] = ope.difference_value(log, baseline)
    out["weights"] = ope.weights(log, 2.0)
    out["terms"] = ope.terms(log, "dr", None)
    out["support"] = ope.support_violations(log, 0.4)
    out["mean"] = ope.mean_interval(np.arange(10.0), None, 0.9)
    out["mean_c"] = ope.mean_interval(
        np.arange(12.0), np.repeat(np.arange(4), 3), 0.9
    )
    out["policy"] = ope.fit_threshold_policy(log, x[:, 0], penalty=0.01)
    model = ope.fit_outcome_model(x, log.actions, log.rewards, 2)
    out["model"] = repr(type(model).__name__)
    return out


def case_netbenefit():
    p, y = score_world(1500, 1)
    acts = (p > 0.1).astype(float)
    return {
        "hash": netbenefit.rule_hash("rule", 0.1),
        "band": netbenefit.net_benefit_band(
            y, acts, (0.02, 0.15), replicates=60
        ),
        "band_inc": netbenefit.net_benefit_band(
            y,
            acts,
            (0.02, 0.15),
            incumbent=(p > 0.2).astype(float),
            clusters=np.arange(1500) // 5,
            replicates=60,
        ),
        "span": netbenefit.useful_span(
            np.array([0.1, 0.2, 0.3]), np.array([False, True, True])
        ),
        "factorize": netbenefit.factorize(["a", "b", "a"]),
    }


CASES = {
    "anytime": case_anytime,
    "design": case_design,
    "rollout": case_rollout,
    "threshold": case_threshold,
    "constraints": case_constraints,
    "ope": case_ope,
    "netbenefit": case_netbenefit,
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_decision_layer_reproduces_its_recorded_outputs(name):
    expected, actual = golden.check(FIXTURE, name, CASES[name])
    assert actual == expected
