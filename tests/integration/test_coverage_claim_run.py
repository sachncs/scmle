"""A run registers its coverage claim before the final rows are read."""

import json
import pathlib

import support
from test_decision_reports_run import run

from scmle.artifacts import claims_audit
from scmle.data.ledger import ledger


def read(folder, name):
    return json.loads((folder / name).read_text())


def kinds(folder):
    lines = (folder / "events.jsonl").read_text().splitlines()
    return [json.loads(line)["kind"] for line in lines]


def test_the_claim_is_hashed_and_registered_before_the_test_rows_open(
    tmp_path,
):
    result = run(tmp_path, plan=False)
    folder = pathlib.Path(result.run_dir)

    claim = read(folder, "coverage_claim.json")
    report = read(folder, "coverage_report.json")
    sha = ledger.digest(claim)
    assert report["claim_sha256"] == sha == ledger.digest(claim)
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
    import numpy
    import pandas

    rng = numpy.random.default_rng(0)
    x = rng.normal(size=400)
    frame = pandas.DataFrame({"x": x, "y": 2 * x + rng.normal(size=400)})
    result = support.make_project(
        frame, "y", time_budget=120, output_dir=tmp_path / "out"
    ).run()
    folder = pathlib.Path(result.run_dir)

    claim = read(folder, "coverage_claim.json")
    assert claim["task"] == "regression"
    assert read(folder, "coverage_report.json")["claim_sha256"] == (
        ledger.digest(claim)
    )
    order = kinds(folder)
    assert order.index("coverage.registered") < order.index("test.opened")


def test_declared_slices_are_in_the_claim_and_flagged_by_the_margin(tmp_path):
    from scmle.data.charter import uncertainty

    result = run(
        tmp_path,
        plan=False,
        uncertainty=uncertainty.Uncertainty(None, ("grp", "z", "ghost")),
    )
    folder = pathlib.Path(result.run_dir)

    claim = read(folder, "coverage_claim.json")
    assert claim["slice_columns"] == ["grp", "z", "ghost"]
    report = read(folder, "coverage_report.json")
    method = next(iter(report["methods"].values()))
    level = method["0.9"]["declared_slices"]
    assert level["ghost"]["status"] == "not_assessable"
    assert {s["slice"] for s in level["grp"]["slices"]} <= {"a", "b"}
    assert {s["slice"] for s in level["z"]["slices"]} <= {
        "band 1",
        "band 2",
        "band 3",
        "band 4",
    }
    assert level["grp"]["flag_gate"]["status"] in ("gating", "descriptive")
    assert all("flag" in s for s in level["grp"]["slices"])
    assert "by_slice" not in method["0.9"]


def confirmation_run(tmp_path):
    from scmle.data.charter import uncertainty

    return run(
        tmp_path,
        plan=False,
        uncertainty=uncertainty.Uncertainty(None, (), True),
    )


def finding(folder, name):
    return next(
        f for f in claims_audit.verify(folder).findings if f.name == name
    )


ON_C = "coverage claim on C: registered, recomputed, status matches"


def test_the_coverage_claim_is_tested_once_on_c_after_the_model_is_frozen(
    tmp_path,
):
    result = confirmation_run(tmp_path)
    folder = pathlib.Path(result.run_dir)

    document = read(folder, "coverage_confirmation.json")
    assert document["claim_id"] == "coverage_nominal"
    assert document["status"] in ("confirmed", "unverified")
    assert document["rows"] > 0 and document["alpha"] == 0.05
    claim = read(folder, "coverage_claim.json")
    assert document["coverage_claim_sha256"] == ledger.digest(claim)
    registered = read(folder, "confirmed_claims.json")["registered_claims"]
    assert registered["coverage_nominal"]["status"] == document["status"]
    assert registered["coverage_nominal"]["kind"] == "coverage"
    order = kinds(folder)
    assert order.count("ledger.completed") == 1
    assert order.index("ledger.deferred") < order.index("coverage.registered")
    assert order.index("coverage.registered") < order.index("ledger.completed")
    assert finding(folder, ON_C).status == "pass"
    assert finding(folder, "registered claims: files hash to the sealed ledger")


def test_an_edited_confirmation_document_fails_the_verification(tmp_path):
    result = confirmation_run(tmp_path)
    folder = pathlib.Path(result.run_dir)
    path = folder / "coverage_confirmation.json"
    document = json.loads(path.read_text())
    document["covered"] = document["rows"]
    path.write_text(json.dumps(document))

    assert finding(folder, ON_C).status == "fail"


def test_without_the_flag_c_is_not_held_back_for_coverage(tmp_path):
    result = run(tmp_path, plan=False)
    folder = pathlib.Path(result.run_dir)

    assert not (folder / "coverage_confirmation.json").exists()
    assert "ledger.deferred" not in kinds(folder)
