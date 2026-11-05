"""The Phase 15 evidence generator's local scenarios run and pass."""

import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load():
    path = ROOT / "scripts" / "make_phase15_evidence.py"
    spec = importlib.util.spec_from_file_location("phase15_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_scenario_generators_pass_and_write_their_files(tmp_path):
    module = load()
    assert all(module.conformance_reports(tmp_path).values())
    assert module.interrupted_job(tmp_path)
    assert all(module.workflow_scenarios(tmp_path).values())
    assert module.lineage(tmp_path)
    assert not any(module.emitted(tmp_path).values())
    for name in (
        "object-store-conformance.json",
        "interrupted-job.json",
        "duplicate-run.json",
        "partial-failure.json",
        "missed-schedule.json",
        "lineage-trace.json",
        "step-functions.json",
        "cloud-workflows.json",
    ):
        assert json.loads((tmp_path / name).read_text())


def test_the_committed_evidence_says_what_was_not_run():
    summary = json.loads(
        (ROOT / "plan" / "evidence" / "phase-15" / "summary.json").read_text()
    )
    assert summary["local_checks_passed"] is True
    assert any("AWS or GCP account" in item for item in summary["not_run"])
    assert any("reproducibility-tolerance" in i for i in summary["not_run"])
