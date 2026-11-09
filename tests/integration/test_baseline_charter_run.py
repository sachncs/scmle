"""The framing baseline claim in a charter run: frozen on X, tested once on C."""

import json
import pathlib

import numpy
import pandas
import pytest
import support

from scmle import cli
from scmle.artifacts import claims_audit
from scmle.contracts import errors
from scmle.data.charter import sealing
from scmle.data.framing import costs
from scmle.data.ledger import ledger as ledger_module
from tests.unit.test_charter_contracts import make_charter

KEY = "baseline-key"


def frame(seed=0, rows=1100, groups=140, selective=False):
    rng = numpy.random.default_rng(seed)
    x1, x2 = rng.normal(size=rows), rng.normal(size=rows)
    p = 1 / (1 + numpy.exp(-(1.5 * x1 + 0.5 * x2 - 1.0)))
    y = (rng.uniform(size=rows) < p).astype(float)
    current = (rng.uniform(size=rows) < 0.5).astype(float)
    table = pandas.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "entity": rng.integers(0, groups, rows).astype(str),
            "current": current,
            "y": y,
        }
    )
    if selective:
        hidden = (current == 0) & (rng.uniform(size=rows) < 0.6)
        table.loc[hidden, "y"] = numpy.nan
    return table


def charter_file(tmp_path, sources=None):
    kinds = sources or (costs.CostSource.HUMAN,) * 4
    model = costs.DecisionCostModel(
        0.0, 1.0, 1.0, 0.0, 0.35, 0.5, kinds, costs.CostSource.HUMAN
    )
    path = tmp_path / "charter.json"
    sealing.write_charter(
        path,
        make_charter(min_effect=0.5, k_max=3, cost_model=model),
        signing_key=KEY,
    )
    return path


def run(table, tmp_path, charter, **kwargs):
    return support.make_project(
        table,
        "y",
        time_budget=300,
        output_dir=tmp_path / "out",
        charter=charter,
        group_column="entity",
        baseline_column="current",
        **kwargs,
    ).run()


def read(result, name):
    return json.loads((pathlib.Path(result.run_dir) / name).read_text())


def test_the_claim_is_registered_by_hash_and_tested_once_on_c(tmp_path):
    result = run(frame(), tmp_path, charter_file(tmp_path))

    report = read(result, "baseline_report.json")
    assert report["status"] in {"issued", "not_shown"}
    assert len(report["endpoints"]) == 2
    claims = read(result, "confirmed_claims.json")
    entry = claims["registered_claims"]["baseline_vs_current_process"]
    assert entry["kind"] == "baseline"
    assert entry["sha256"] == report["registration_hash"]
    ledger = ledger_module.Ledger.load(
        pathlib.Path(result.run_dir) / ledger_module.FILE_NAME
    )
    assert (
        ledger.claims["baseline_vs_current_process"]["sha256"]
        == (report["registration_hash"])
    )
    sealed = ledger.confirmation["attachments"]
    assert set(sealed) == {"baseline_report.json"}
    log = (pathlib.Path(result.run_dir) / "custody_log.jsonl").read_text()
    assert log.count('"event":"open"') + log.count('"event": "open"') == 1


def test_the_current_decision_is_never_a_model_feature(tmp_path):
    result = run(frame(), tmp_path, charter_file(tmp_path))

    profile = read(result, "dataset_profile.json")
    plan = profile["feature_plan"]
    assert "current" not in plan["numeric"] + plan["categorical"]


def test_costs_from_data_cap_the_claim_at_unverified(tmp_path):
    kinds = (costs.CostSource.DATA,) * 4
    result = run(frame(), tmp_path, charter_file(tmp_path, kinds))

    report = read(result, "baseline_report.json")
    assert report["claim"]["tier"] == "unverified"
    why = read(result, "framing_provenance.json")
    assert why["tier_ceiling"] == "unverified"
    assert why["cost_entries_not_from_a_person"]


def test_selective_labels_on_x_stop_the_comparison(tmp_path):
    result = run(
        frame(selective=True),
        tmp_path,
        charter_file(tmp_path),
        unlabeled="ignore",
    )

    report = read(result, "baseline_report.json")
    assert report["status"] == "not_issued"
    assert report["selective_labels"]["flagged"] is True
    assert report["claim"]["tier"] == "unverified"


def test_the_column_needs_a_cost_model(tmp_path):
    path = tmp_path / "charter.json"
    sealing.write_charter(
        path, make_charter(min_effect=0.5, k_max=3), signing_key=KEY
    )
    with pytest.raises(errors.ConfigError, match="cost_model"):
        run(frame(), tmp_path, path)


def test_a_charter_run_without_the_column_is_unchanged(tmp_path):
    result = support.make_project(
        frame(),
        "y",
        time_budget=300,
        output_dir=tmp_path / "out",
        charter=charter_file(tmp_path),
        group_column="entity",
    ).run()

    folder = pathlib.Path(result.run_dir)
    assert not (folder / "baseline_report.json").exists()
    assert "registered_claims" not in read(result, "confirmed_claims.json")


def test_claims_lists_and_verifies_the_registered_claim(tmp_path, capsys):
    result = run(frame(), tmp_path, charter_file(tmp_path))

    rows = claims_audit.list_claims(result.run_dir)
    registered = [r for r in rows if r.get("kind") == "baseline"]
    assert len(registered) == 1
    assert registered[0]["status"] in {"issued", "not_shown"}
    verified = claims_audit.verify(result.run_dir)
    names = {f.name: f.status for f in verified.findings}
    assert names["registered claims: files hash to the sealed ledger"] == "pass"
    assert verified.ok
    assert cli.main(["claims", "list", str(result.run_dir)]) == 0
    assert "registered baseline" in capsys.readouterr().out

    folder = pathlib.Path(result.run_dir)
    report = json.loads((folder / "baseline_report.json").read_text())
    report["issued"] = not report["issued"]
    (folder / "baseline_report.json").write_text(json.dumps(report))
    assert not claims_audit.verify(result.run_dir).ok


def kinds(result):
    lines = (pathlib.Path(result.run_dir) / "events.jsonl").read_text()
    return [json.loads(line)["kind"] for line in lines.splitlines()]


def test_the_selected_model_is_the_subject_of_the_claim(tmp_path):
    result = run(
        frame(),
        tmp_path,
        charter_file(tmp_path),
        baseline_subject="selected_model",
    )

    report = read(result, "baseline_report.json")
    assert report["status"] in {"issued", "not_shown"}
    assert len(report["endpoints"]) == 2
    subject = report["rule"]
    assert subject["kind"] == "selected_model"
    assert subject["model"] == result.selected_model
    assert report["reference_rule"]["kind"] == "standardised_logistic"
    order = kinds(result)
    assert order.index("selection.frozen") < order.index("ledger.completed")
    assert order.index("ledger.completed") < order.index("test.opened")
    claims = read(result, "confirmed_claims.json")
    entry = claims["registered_claims"]["baseline_vs_current_process"]
    assert entry["sha256"] == report["registration_hash"]
    assert subject["selection_sha256"]
    log = (pathlib.Path(result.run_dir) / "custody_log.jsonl").read_text()
    assert log.count('"event":"open"') + log.count('"event": "open"') == 1


def test_the_default_subject_is_still_the_frozen_rule(tmp_path):
    result = run(frame(), tmp_path, charter_file(tmp_path))

    report = read(result, "baseline_report.json")
    assert report["rule"]["kind"] == "standardised_logistic"
    order = kinds(result)
    assert order.index("ledger.completed") < order.index("selection.frozen")
