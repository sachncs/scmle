"""The decision layer's report files for a charter run (F11).

Built once from the locked test set after the frozen rule exists, inside
`FinalEvaluationStage`:

* `policy_value.json`: off-policy value of the frozen rule from a randomized
  log in the test rows (`data.logged_action_column` holds the logged 0/1
  action and `data.propensity_column` the recorded probability that action 1
  was taken).
  The estimator is self-normalised IPW with the charter's clip, support and
  effective-sample-size gates; the reward is the 0/1 outcome indicator of the
  positive class. Without both columns the file says `not_assessable`; no
  propensity is ever estimated here.
* the fairness satisfaction test for every declared constraint, once, on the
  same rows (`satisfaction_test`, upper bound of the gap against `max_gap`).
* `rollout_design.json`: the plan, its hash, the fixed-horizon sample size for
  the planned improvement, the expected sample number of the anytime-valid test
  by simulation, and the level the decision budget would assign to the first
  experiment. No experiment ran, so there is no assignment or decision to log.
* `rollout_decisions.jsonl`: the hash-chained decision log, holding one row,
  the registration of the plan. Experiment assignments and e-process
  snapshots are written to it by whoever runs the rollout, not by this run.
* `decision_report.json`: the cost model hash, the threshold and net benefit
  already computed, the three results above and what each does not show.

Costs, margins and propensities are inputs. Nothing here is gated on a file
that did not pass its own declared gates.
"""

from collections.abc import Mapping
import hashlib
from typing import Any

import numpy as np
import pandas as pd

from scmle.contracts import signing
from scmle.data.charter import contracts as charter_contracts
from scmle.data.charter import decision as charter_decision
from scmle.decision import constraints
from scmle.decision import design
from scmle.decision import ope
from scmle.decision import rollout

NOT_ASSESSABLE = "not_assessable"
ESTIMATOR = "snips"
SIMULATION_RUNS = 200
FIRST_EXPERIMENT = "design-1"


def policy_value(
    layer: charter_decision.DecisionLayer,
    actions: np.ndarray,
    outcome: np.ndarray,
    logged: np.ndarray | None,
    propensity: np.ndarray | None,
) -> dict[str, Any]:
    """Off-policy value of the frozen rule, or why there is none."""
    if logged is None or propensity is None:
        return {
            "status": NOT_ASSESSABLE,
            "reason": "no randomized log: data.logged_action_column and "
            "data.propensity_column are both needed; propensities are never "
            "estimated here",
        }
    p = np.asarray(propensity, dtype=float)
    a = np.asarray(logged, dtype=float)
    if (
        not np.isfinite(p).all()
        or ((p <= 0) | (p >= 1)).any()
        or not np.isin(a, (0.0, 1.0)).all()
    ):
        return {
            "status": NOT_ASSESSABLE,
            "reason": "propensities must lie strictly between 0 and 1 and the "
            "logged action must be 0 or 1",
        }
    behaviour = np.column_stack([1.0 - p, p])
    target = np.column_stack([1.0 - actions, actions])
    log = ope.LoggedBandit(
        a.astype(np.intp), np.asarray(outcome, dtype=float), behaviour, target
    )
    rules = layer.model.off_policy
    found = ope.estimate_value(
        log,
        ESTIMATOR,
        clip=rules.clip,
        min_propensity=rules.min_propensity,
        ess_floor=rules.ess_floor,
    )
    return {
        **found.to_dict(),
        "rows": log.rows,
        "reward": "1 if the positive class occurred",
        "assumptions": [
            "the log was randomized with the recorded probabilities",
            "the rule was frozen before these outcomes were read",
        ],
    }


def fairness(
    layer: charter_decision.DecisionLayer,
    features: pd.DataFrame,
    outcome: np.ndarray,
    actions: np.ndarray,
    alpha: float,
) -> list[dict[str, Any]]:
    """The satisfaction test for each declared constraint."""
    out = []
    for item in layer.fairness:
        if item.attribute not in features.columns:
            out.append(
                {
                    "attribute": item.attribute,
                    "status": NOT_ASSESSABLE,
                    "reason": "the attribute is not a column of the test rows",
                }
            )
            continue
        found = constraints.satisfaction_test(
            outcome, actions, features[item.attribute].to_numpy(), item, alpha
        )
        out.append({"attribute": item.attribute, **found.to_dict()})
    return out


def rollout_design(
    charter: charter_contracts.AnalysisCharter, base_rate: float
) -> tuple[dict[str, Any], rollout.DecisionLog] | None:
    """The rollout plan's design record and its one-row decision log."""
    layer = charter.decision
    plan = layer.rollout if layer is not None else None
    if layer is None or plan is None:
        return None
    plan_hash = hashlib.sha256(
        signing.canonical_payload(plan.to_dict())
    ).hexdigest()
    budget = plan.budget()
    level = float(plan.alpha_promote) / 2.0
    record: dict[str, Any] = {
        "plan": plan.to_dict(),
        "plan_hash": plan_hash,
        "base_rate": base_rate,
        "first_experiment_level": level,
        "budget": budget.to_dict(),
        "ran": False,
        "note": "design only: no experiment has run, so there is no "
        "assignment or decision; the first level shown is illustrative",
    }
    if base_rate > 0.0 and base_rate + plan.delta_min < 1.0:
        record["fixed_horizon_rows"] = design.fixed_horizon_rows(
            base_rate,
            plan.delta_min,
            alpha=float(plan.alpha_promote),
            power=charter.power,
        )
        record["expected_sample_number"] = design.simulate_stopping(
            plan, base_rate, plan.delta_min, SIMULATION_RUNS, charter.seed
        )
    else:
        record["fixed_horizon_rows"] = None
        record["expected_sample_number"] = None
        record["note"] += (
            "; the base rate leaves no room for delta_min, so no sample size "
            "was computed"
        )
    log = rollout.DecisionLog()
    log.append(
        "registered",
        {"plan_hash": plan_hash, "charter_hash": charter.charter_hash},
    )
    return record, log


def decision_report(
    charter: charter_contracts.AnalysisCharter,
    decision: Mapping[str, Any] | None,
    net_benefit: Mapping[str, Any] | None,
    value: Mapping[str, Any],
    satisfaction: list[dict[str, Any]],
    design_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The summary file."""
    model = charter.cost_model
    return {
        "schema_version": 1,
        "cost_model_hash": None if model is None else model.model_hash,
        "unit": None
        if charter.decision is None
        else charter.decision.model.unit,
        "threshold": None if decision is None else decision.get("threshold"),
        "net_benefit": None if net_benefit is None else dict(net_benefit),
        "policy_value": {
            "status": value.get("status"),
            "file": "policy_value.json",
        },
        "fairness": satisfaction,
        "rollout": None
        if design_record is None
        else {"plan_hash": design_record["plan_hash"], "ran": False},
        "limits": [
            "costs, margins and propensities are inputs, not findings",
            "the rule and the fairness bounds were fixed before these rows "
            "were read; each test ran once",
            "legal admissibility of group thresholds is an attestation",
        ],
    }
