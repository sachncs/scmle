"""A causal charter run: the identification stage and its files."""

import dataclasses
import json
import pathlib

import numpy as np
import pandas as pd
import support

from scmle.data.charter import causal
from scmle.data.charter import sealing
from tests.unit.test_charter_contracts import make_charter

KEY = "identification-key"
ASSUMPTIONS = (
    "unconfoundedness_given:x1,x2",
    "positivity",
    "no_interference",
)


def causal_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = 600
    x1, x2 = rng.normal(size=rows), rng.normal(size=rows)
    t = (rng.uniform(size=rows) < 1 / (1 + np.exp(-0.5 * x1))).astype(int)
    y = 2.0 * t + x1 - 0.5 * x2 + rng.normal(size=rows)
    return pd.DataFrame({"x1": x1, "x2": x2, "t": t, "y": y})


def charter_file(tmp_path, assumptions) -> pathlib.Path:
    base = make_charter(min_effect=0.5, k_max=3)
    estimand = dataclasses.replace(
        base.estimand,
        kind="causal",
        causal=causal.CausalSpec(
            treatment="t",
            outcome="y",
            estimand="ATE",
            assumptions=assumptions,
            strategies=("backdoor_aipw",),
        ),
    )
    path = tmp_path / "charter.json"
    sealing.write_charter(
        path, make_charter(estimand=estimand, min_effect=0.5, k_max=3), KEY
    )
    return path


def run(tmp_path, assumptions):
    return support.make_project(
        causal_frame(),
        "y",
        time_budget=300,
        output_dir=tmp_path / "out",
        charter=charter_file(tmp_path, assumptions),
    ).run()


def read(result, name):
    return json.loads((pathlib.Path(result.run_dir) / name).read_text())


def test_an_identified_charter_writes_the_plan_and_an_unverified_claim(
    tmp_path,
):
    result = run(tmp_path, ASSUMPTIONS)
    plan = read(result, "identification_plan.json")
    report = read(result, "identification_report.json")
    claims = read(result, "causal_claims.json")
    assert plan["primary"] == "backdoor_aipw"
    assert report["estimate"] is None
    assert report["design"]["overlap_on_x"]["n"] > 0
    assert claims["tier"] == "unverified"
    assert "opened once" in claims["reasons"][0]
    assert set(claims["hashes"]) == {
        "identification_plan.json",
        "identification_report.json",
        "falsification_report.json",
        "sensitivity_report.json",
    }


def test_a_charter_no_strategy_identifies_is_refused_without_an_estimate(
    tmp_path,
):
    result = run(tmp_path, ("positivity",))
    claims = read(result, "causal_claims.json")
    report = read(result, "identification_report.json")
    assert claims["status"] == "not_identified" and claims["tier"] is None
    assert report["estimate"] is None and report["action"] == "refuse_claim"
    assert result.evaluation["test_set_opened"] is True
