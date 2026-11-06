"""A causal charter run: the identification stage and its files."""

import dataclasses
import json
import pathlib

import numpy as np
import pandas as pd
import pytest
import support

from scmle import cli
from scmle.artifacts import claims_audit
from scmle.contracts import errors
from scmle.core import causal_claim
from scmle.data.admissibility import dag as dag_module
from scmle.data.charter import causal
from scmle.data.charter import sealing
from scmle.data.ledger import ledger as ledger_module
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


def charter_file(tmp_path, assumptions, dag=None, subgroups=()) -> pathlib.Path:
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
            dag=dag,
            subgroups=subgroups,
        ),
    )
    path = tmp_path / "charter.json"
    sealing.write_charter(
        path, make_charter(estimand=estimand, min_effect=0.5, k_max=3), KEY
    )
    return path


def run(tmp_path, assumptions, dag=None, subgroups=(), frame=None):
    return support.make_project(
        causal_frame() if frame is None else frame,
        "y",
        time_budget=300,
        output_dir=tmp_path / "out",
        charter=charter_file(tmp_path, assumptions, dag, subgroups),
    ).run()


def read(result, name):
    return json.loads((pathlib.Path(result.run_dir) / name).read_text())


def test_an_identified_charter_estimates_on_the_single_opening_of_c(tmp_path):
    result = run(tmp_path, ASSUMPTIONS)
    plan = read(result, "identification_plan.json")
    report = read(result, "identification_report.json")
    claims = read(result, "causal_claims.json")
    assert plan["primary"] == "backdoor_aipw"
    estimate = report["estimate"]
    assert estimate["low"] < 2.0 < estimate["high"]
    assert abs(estimate["point"] - 2.0) < 0.6
    assert claims["status"] == "identified"
    assert claims["tier"] != "confirmed"
    assert "no named attestor" in " ".join(claims["reasons"])
    assert set(claims["hashes"]) == {
        "identification_plan.json",
        "identification_report.json",
        "falsification_report.json",
        "sensitivity_report.json",
    }
    folder = pathlib.Path(result.run_dir)
    log = (folder / "custody_log.jsonl").read_text()
    assert log.count('"event":"open"') + log.count('"event": "open"') == 1
    ledger = ledger_module.Ledger.load(folder / ledger_module.FILE_NAME)
    registered = ledger.claims["causal_primary_theta_zero"]
    assert registered["kind"] == "causal_primary"
    sealed = ledger.confirmation["attachments"]
    assert set(sealed) == set(causal_claim.DOCUMENTS)
    for name, expected in sealed.items():
        assert ledger_module.digest(read(result, name)) == expected
    assert ledger.confirmation["claims"] == {
        "causal_primary_theta_zero": {
            "kind": "causal_primary",
            "sha256": registered["sha256"],
            "status": "identified",
        }
    }


def test_claims_prints_the_identification_status_and_verifies(tmp_path, capsys):
    result = run(tmp_path, ASSUMPTIONS)

    assert cli.main(["claims", "list", str(result.run_dir)]) == 0
    printed = capsys.readouterr().out
    assert "registered causal_primary" in printed
    assert "identification=identified" in printed
    verified = claims_audit.verify(result.run_dir)
    names = {f.name: f.status for f in verified.findings}
    assert names["registered claims: files hash to the sealed ledger"] == "pass"
    assert verified.ok

    folder = pathlib.Path(result.run_dir)
    report = json.loads((folder / "identification_report.json").read_text())
    report["estimate"]["point"] = 0.0
    (folder / "identification_report.json").write_text(json.dumps(report))
    assert not claims_audit.verify(result.run_dir).ok


def test_a_declared_dag_is_loaded_checked_and_lifts_the_adjustment_cap(
    tmp_path,
):
    graph = dag_module.Dag(
        ("t", "x1", "x2", "y"),
        (("t", "y"), ("x1", "t"), ("x1", "y"), ("x2", "y")),
        (),
        "test",
        "none",
    )
    path = tmp_path / "dag.json"
    path.write_text(json.dumps(graph.artifact()))
    result = run(tmp_path, ASSUMPTIONS, dag=str(path))

    report = read(result, "identification_report.json")
    assert report["dag"]["dag_hash"] == graph.digest()
    assert report["dag"]["falsification"]["ran"] is True
    assert report["dag"]["adjustment_members_refused"] == {}
    plan = read(result, "identification_plan.json")
    assert plan["primary"] == "backdoor_aipw"
    table = read(result, "admissibility_table.json")
    assert table["kind"] == "causal" and table["dag_hash"] == graph.digest()
    claims = read(result, "causal_claims.json")
    assert "adjustment set is an unchecked attestation" not in " ".join(
        read(result, "identification_report.json")["notes"]
    )
    assert claims["status"] == "identified"


def test_an_adjustment_member_that_is_a_descendant_makes_backdoor_ineligible(
    tmp_path,
):
    graph = dag_module.Dag(
        ("t", "x1", "x2", "y"),
        (("t", "x2"), ("t", "y"), ("x1", "t"), ("x1", "y")),
        (),
        "test",
        "none",
    )
    path = tmp_path / "dag.json"
    path.write_text(json.dumps(graph.artifact()))
    result = run(tmp_path, ASSUMPTIONS, dag=str(path))

    claims = read(result, "causal_claims.json")
    report = read(result, "identification_report.json")
    assert claims["status"] == "not_identified"
    assert "x2" in report["dag"]["adjustment_members_refused"]


def test_a_dag_that_cannot_be_loaded_is_a_config_error(tmp_path):
    path = tmp_path / "dag.json"
    path.write_text("{}")
    with pytest.raises(errors.ConfigError, match="declared DAG"):
        run(tmp_path, ASSUMPTIONS, dag=str(path))


def test_a_charter_no_strategy_identifies_is_refused_without_an_estimate(
    tmp_path,
):
    result = run(tmp_path, ("positivity",))
    claims = read(result, "causal_claims.json")
    report = read(result, "identification_report.json")
    assert claims["status"] == "not_identified" and claims["tier"] is None
    assert report["estimate"] is None and report["action"] == "refuse_claim"
    assert result.evaluation["test_set_opened"] is True


def test_subgroup_hypotheses_are_registered_and_reported_in_the_causal_files(
    tmp_path,
):
    frame = causal_frame()
    frame["seg"] = np.where(frame["x1"] > 0, "hi", "lo")
    plain = run(tmp_path / "plain", ASSUMPTIONS, frame=frame)
    split = run(
        tmp_path / "split",
        ASSUMPTIONS,
        subgroups=(causal.Subgroup("seg", ("hi", "lo")),),
        frame=frame,
    )
    report = read(split, "identification_report.json")["subgroups"]
    assert report["rule"] == "holm" and report["family_size"] == 2
    assert {r["value"] for r in report["hypotheses"]} == {"hi", "lo"}
    assert report["rejected"] == ["seg=hi", "seg=lo"]
    claims = read(split, "causal_claims.json")
    assert claims["subgroups"]["family_size"] == 2
    assert "subgroups" not in read(plain, "causal_claims.json")
    registered = {}
    for name, result in (("plain", plain), ("split", split)):
        folder = pathlib.Path(result.run_dir)
        ledger = ledger_module.Ledger.load(folder / ledger_module.FILE_NAME)
        registered[name] = ledger.claims["causal_primary_theta_zero"]["sha256"]
    assert registered["plain"] != registered["split"]
    assert claims_audit.verify(split.run_dir).ok
