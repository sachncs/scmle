"""A run registers its coverage claim before the final rows are read."""

import json
import pathlib

import support
from test_decision_reports_run import run as charter_run

from scmle.artifacts import claims_audit
from scmle.data.ledger import ledger as ledger_module
from scmle.evaluation import coverage_run


def read(folder, name):
    return json.loads((folder / name).read_text())


def kinds(folder):
    lines = (folder / "events.jsonl").read_text().splitlines()
    return [json.loads(line)["kind"] for line in lines]


def test_the_claim_is_hashed_and_registered_before_the_test_rows_open(
    tmp_path,
):
    result = charter_run(tmp_path, plan=False)
    folder = pathlib.Path(result.run_dir)

    claim = read(folder, "coverage_claim.json")
    report = read(folder, "coverage_report.json")
    sha = ledger_module.digest(claim)
    assert report["claim_sha256"] == sha == coverage_run.claim_hash(claim)
    order = kinds(folder)
    assert order.index("coverage.registered") < order.index("test.opened")
    assert order.index("test.opened") < order.index("coverage.reported")
    log = (folder / "custody_log.jsonl").read_text()
    assert '"coverage_claim_registered"' in log and sha in log
    found = {
        f.name: f
        for f in claims_audit.verify(folder).findings
        if f.name.startswith("coverage claim")
    }
    assert (
        found["coverage claim: registered before the rows were read"].status
        == "pass"
    )


def test_a_run_without_a_charter_writes_the_claim_too(tmp_path):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    frame = pd.DataFrame({"x": x, "y": 2 * x + rng.normal(size=400)})
    result = support.make_project(
        frame, "y", time_budget=120, output_dir=tmp_path / "out"
    ).run()
    folder = pathlib.Path(result.run_dir)

    claim = read(folder, "coverage_claim.json")
    assert claim["task"] == "regression"
    assert read(folder, "coverage_report.json")["claim_sha256"] == (
        coverage_run.claim_hash(claim)
    )
    order = kinds(folder)
    assert order.index("coverage.registered") < order.index("test.opened")


def test_declared_slices_are_in_the_claim_and_described_without_flags(tmp_path):
    from scmle.data.charter import uncertainty

    result = charter_run(
        tmp_path,
        plan=False,
        uncertainty=uncertainty.UncertaintySpec(None, ("grp", "z", "ghost")),
    )
    folder = pathlib.Path(result.run_dir)

    claim = read(folder, "coverage_claim.json")
    assert claim["slice_columns"] == ["grp", "z", "ghost"]
    report = read(folder, "coverage_report.json")
    method = next(iter(report["methods"].values()))
    level = method["0.9"]["declared_slices"]
    assert level["ghost"]["status"] == "not_assessable"
    assert level["grp"]["descriptive"] is True
    assert {s["slice"] for s in level["grp"]["slices"]} <= {"a", "b"}
    assert {s["slice"] for s in level["z"]["slices"]} <= {
        "band 1",
        "band 2",
        "band 3",
        "band 4",
    }
    assert '"flagged"' not in json.dumps(level)
    assert "by_slice" not in method["0.9"]
