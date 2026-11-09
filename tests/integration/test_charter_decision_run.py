"""A charter with a decision layer chooses the threshold from its cost model."""

import json
import pathlib

import support

from scmle.data.charter import sealing
import scmle.data.charter.decision
from scmle.data.framing import costs
from tests.unit.test_charter_contracts import make_charter

KEY = "decision-key"


def charter_with_costs(tmp_path, c_fp=1.0, c_fn=2.0):
    human = (costs.CostSource.HUMAN,) * 4
    model = costs.DecisionCostModel(
        0.0, c_fn, c_fp, 0.0, 0.05, 0.9, human, costs.CostSource.HUMAN
    )
    layer = scmle.data.charter.decision.DecisionLayer(
        scmle.data.charter.decision.DecisionModel("EUR per case")
    )
    path = tmp_path / "charter.json"
    sealing.write_charter(
        path,
        make_charter(min_effect=0.5, k_max=3, cost_model=model, decision=layer),
        signing_key=KEY,
    )
    return path, model


def evaluation(result):
    folder = pathlib.Path(result.run_dir)
    for name in ("evaluation.json", "diagnostics/evaluation.json"):
        if (folder / name).exists():
            return json.loads((folder / name).read_text())
    for candidate in folder.rglob("*.json"):
        data = json.loads(candidate.read_text())
        if isinstance(data, dict) and "decision" in data and "n_dev" in data:
            return data
    raise AssertionError("no evaluation artifact found")


def test_the_charter_cost_model_sets_the_threshold(binary_frame, tmp_path):
    path, model = charter_with_costs(tmp_path)
    result = support.make_project(
        binary_frame,
        "y",
        time_budget=300,
        output_dir=tmp_path / "out",
        charter=path,
    ).run()
    document = evaluation(result)
    decision = document["decision"]
    assert decision["objective"] == "chartered_cost"
    assert decision["cost_model_hash"] == model.model_hash
    detail = decision["f11"]
    assert detail["path"] in ("bayes", "recalibrated", "grid")
    assert detail["rule_hash"]
    if detail["path"] == "bayes":
        assert abs(decision["threshold"] - model.threshold) < 1e-12
    benefit = document["net_benefit"]
    assert benefit["claim"] in ("useful", "not_useful_in_range")
    assert benefit["rule_hash"] == detail["rule_hash"]
    assert benefit["tier"] in ("attested", "unverified")
    assert benefit["replicates"] >= 500
    assert benefit["thresholds"][0] == model.t_lo


def test_a_run_without_a_decision_layer_has_no_net_benefit(
    binary_frame, tmp_path
):
    result = support.make_project(
        binary_frame, "y", time_budget=300, output_dir=tmp_path / "out"
    ).run()
    assert evaluation(result)["net_benefit"] is None
