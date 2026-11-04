"""A charter run with a decision block writes the F11 report files."""

from fractions import Fraction
import json
import pathlib

import numpy as np
import pandas as pd
import support

from scmle.data.charter import decision as decision_module
from scmle.data.charter import sealing
from scmle.data.framing import costs
from scmle.decision import rollout
from tests.unit.test_charter_contracts import make_charter

KEY = "decision-reports-key"


def frame():
    rng = np.random.default_rng(0)
    rows = 600
    x = rng.normal(size=rows)
    table = pd.DataFrame(
        {
            "x": x,
            "z": rng.normal(size=rows),
            "grp": rng.choice(["a", "b"], rows),
            "treat": (rng.random(rows) < 0.5).astype(float),
            "prop": np.full(rows, 0.5),
        }
    )
    table["y"] = (x + rng.normal(scale=0.7, size=rows) > 0).astype(int)
    return table


def charter_file(tmp_path, plan=True):
    human = (costs.CostSource.HUMAN,) * 4
    model = costs.DecisionCostModel(
        0.0, 2.0, 1.0, 0.0, 0.05, 0.9, human, costs.CostSource.HUMAN
    )
    rollout_plan = decision_module.RolloutPlan(
        unit="user",
        primary="conv",
        assignment_prob=0.5,
        delta_min=0.05,
        tolerance=0.02,
        n_max=400,
        alpha_promote=Fraction(1, 20),
        alpha_rollback=Fraction(1, 20),
    )
    layer = decision_module.DecisionLayer(
        decision_module.DecisionModel("EUR per case"),
        (
            decision_module.FairnessConstraint(
                "grp", "demographic_parity", 0.4, "reweighing", min_group=20
            ),
        ),
        rollout_plan if plan else None,
    )
    path = tmp_path / "charter.json"
    sealing.write_charter(
        path,
        make_charter(min_effect=0.5, k_max=3, cost_model=model, decision=layer),
        signing_key=KEY,
    )
    return path


def run(tmp_path, **kwargs):
    return support.make_project(
        frame(),
        "y",
        time_budget=300,
        output_dir=tmp_path / "out",
        charter=charter_file(tmp_path, kwargs.pop("plan", True)),
        **kwargs,
    ).run()


def read(result, name):
    return json.loads((pathlib.Path(result.run_dir) / name).read_text())


def test_the_decision_files_are_written_with_a_randomized_log(tmp_path):
    result = run(
        tmp_path, logged_action_column="treat", propensity_column="prop"
    )
    folder = pathlib.Path(result.run_dir)

    value = read(result, "policy_value.json")
    assert value["status"] == "issued" and value["rows"] > 0
    report = read(result, "decision_report.json")
    assert report["fairness"][0]["attribute"] == "grp"
    assert report["policy_value"]["file"] == "policy_value.json"
    assert report["rollout"]["ran"] is False
    design = read(result, "rollout_design.json")
    assert design["plan_hash"] == report["rollout"]["plan_hash"]
    log = rollout.DecisionLog.read(folder / "rollout_decisions.jsonl")
    assert log.verify() and len(log.rows) == 1
    profile = read(result, "dataset_profile.json")["feature_plan"]
    assert "prop" not in profile["numeric"] + profile["categorical"]


def test_without_a_log_the_value_is_not_assessable_and_no_plan_no_design(
    tmp_path,
):
    result = run(tmp_path, plan=False)
    folder = pathlib.Path(result.run_dir)

    assert read(result, "policy_value.json")["status"] == "not_assessable"
    assert not (folder / "rollout_design.json").exists()
    assert not (folder / "rollout_decisions.jsonl").exists()
    assert read(result, "decision_report.json")["rollout"] is None
