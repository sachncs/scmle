"""The decision layer's report files for a charter run.

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
* `rollout_design.json`: the rollout plan, its hash, the fixed-horizon
  sample size for the planned improvement, the expected sample number of
  the anytime-valid test by simulation, and the level the decision budget
  would assign to the first experiment. No experiment ran, so there is no
  assignment or decision to log.
* `rollout_decisions.jsonl`: the hash-chained decision log, holding one row,
  the registration of the rollout plan. Experiment assignments and e-process
  snapshots are written to it by whoever runs the rollout, not by this run.
* `decision_threshold.json` and `net_benefit.json`: the frozen threshold rule
  and the net benefit of the rule on the locked rows, as their own files.
* `decision_report.json`: the cost model hash, the threshold and net benefit
  already computed, the three results above and what each does not show.

The hashes of the four files that carry a result (threshold, net benefit,
policy value, rollout design) are appended to the custody log as one
`decision_files` event right after they are written, so `claims verify` can
tell an edited, removed or added file from the registration. A file that does
not exist (no rollout plan, say) is registered as absent.

Costs, margins and propensities are inputs. Nothing here is gated on a file
that did not pass its own declared gates.
"""

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy
import pandas

from scmle.contracts import signing
from scmle.data.charter import contracts
from scmle.data.charter import decision as charter_decision
from scmle.data.ledger import ledger
from scmle.decision import constraints
from scmle.decision import design
from scmle.decision import ope
from scmle.decision import rollout

NOT_ASSESSABLE = "not_assessable"
ESTIMATOR = "snips"
SIMULATION_RUNS = 200
FIRST_EXPERIMENT = "design-1"
THRESHOLD_FILE = "decision_threshold.json"
NET_BENEFIT_FILE = "net_benefit.json"
POLICY_FILE = "policy_value.json"
DESIGN_FILE = "rollout_design.json"
DECISION_FILES = (THRESHOLD_FILE, NET_BENEFIT_FILE, POLICY_FILE, DESIGN_FILE)
REGISTRATION_EVENT = "decision_files"


def file_digest(path: Path) -> str | None:
    """Hash of a JSON file's content (`ledger.digest`), None if absent.

    Args:
        path: The JSON file.

    Returns:
        The content digest, or None if the file does not exist.

    Raises:
        ValueError: If the file is not a JSON object.
    """
    if not path.exists():
        return None
    body = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return ledger.digest(body)


def file_digests(folder: str | Path) -> dict[str, str | None]:
    """The hash of each of the four decision files, None where absent.

    Args:
        folder: The directory holding the decision files.

    Returns:
        A mapping from decision file name to its content digest, or None
        for a file that does not exist.
    """
    return {name: file_digest(Path(folder) / name) for name in DECISION_FILES}


def policy_value(
    layer: charter_decision.DecisionLayer,
    actions: numpy.ndarray,
    outcome: numpy.ndarray,
    logged: numpy.ndarray | None,
    propensity: numpy.ndarray | None,
) -> dict[str, Any]:
    """Off-policy value of the frozen rule, or why there is none.

    Args:
        layer: The charter's decision layer, whose off-policy rules apply.
        actions: The probability that the rule takes action 1, one per row.
        outcome: The observed outcome, one per row.
        logged: The action taken in the log (0 or 1), or None if there is no
            randomized log.
        propensity: The probability with which the logged action was taken,
            or None.

    Returns:
        The estimate (a refused one, without value or interval, when the
        support or effective-sample-size gate fails) with its row count,
        reward definition and assumptions, or a record whose status is not
        assessable and says why when the log is missing or its propensities
        and actions are invalid.
    """
    if logged is None or propensity is None:
        return {
            "status": NOT_ASSESSABLE,
            "reason": "no randomized log: data.logged_action_column and "
            "data.propensity_column are both needed; propensities are never "
            "estimated here",
        }
    p = numpy.asarray(propensity, dtype=float)
    a = numpy.asarray(logged, dtype=float)
    if (
        not numpy.isfinite(p).all()
        or ((p <= 0) | (p >= 1)).any()
        or not numpy.isin(a, (0.0, 1.0)).all()
    ):
        return {
            "status": NOT_ASSESSABLE,
            "reason": "propensities must lie strictly between 0 and 1 and the "
            "logged action must be 0 or 1",
        }
    behaviour = numpy.column_stack([1.0 - p, p])
    target = numpy.column_stack([1.0 - actions, actions])
    log = ope.LoggedBandit(
        a.astype(numpy.intp),
        numpy.asarray(outcome, dtype=float),
        behaviour,
        target,
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
    features: pandas.DataFrame,
    outcome: numpy.ndarray,
    actions: numpy.ndarray,
    alpha: float,
) -> list[dict[str, Any]]:
    """The satisfaction test for each declared constraint.

    Args:
        layer: The charter's decision layer, listing the constraints.
        features: The test rows, holding each constrained attribute.
        outcome: The observed outcome, one per row.
        actions: The rule's actions (0 or 1), one per row.
        alpha: Significance level of each satisfaction test.

    Returns:
        One record per constraint, in declaration order: the test result, or
        a not-assessable status if the attribute is not a column.
    """
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
    charter: contracts.AnalysisCharter, base_rate: float
) -> tuple[dict[str, Any], rollout.DecisionLog] | None:
    """The rollout plan's design record and its one-row decision log.

    Args:
        charter: The charter, which may declare a rollout plan.
        base_rate: The baseline rate of the positive outcome.

    Returns:
        The design record (plan hash, first experiment level, budget and,
        where the base rate leaves room for the minimum effect, the fixed
        horizon and expected sample number) with a decision log holding the
        registration, or None if the charter has no rollout plan.
    """
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
    charter: contracts.AnalysisCharter,
    decision: Mapping[str, Any] | None,
    net_benefit: Mapping[str, Any] | None,
    value: Mapping[str, Any],
    satisfaction: list[dict[str, Any]],
    design_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The summary file.

    Args:
        charter: The charter, supplying the cost model, unit and plan.
        decision: The decision record, or None.
        net_benefit: The net-benefit record, or None.
        value: The off-policy value record.
        satisfaction: The fairness records.
        design_record: The rollout design record, or None.

    Returns:
        The summary document: the cost model hash, unit, threshold, the net
        benefit record, the status of the policy value (with its file name),
        the fairness records, the rollout plan hash (marked as not run) and
        the limits of the analysis.
    """
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
