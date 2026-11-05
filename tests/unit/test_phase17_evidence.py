"""The Phase 17 evidence generator's local scenarios run and pass."""

import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load():
    path = ROOT / "scripts" / "make_phase17_evidence.py"
    spec = importlib.util.spec_from_file_location("phase17_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_failure_drift_and_degradation_scenarios_pass(tmp_path):
    module = load()
    ops = module.observability_scenarios(tmp_path)
    assert ops["passed"]
    assert (
        ops["injected_failure"]["classification"]["verdict"] == "infrastructure"
    )
    drift = ops["controlled_drift"]
    assert drift["drift_only_before_labels"]["verdict"] == "data_quality"
    assert drift["after_delayed_labels"]["verdict"] == "mixed"
    assert ops["every_log_line_has_trace_and_version"]
    assert ops["exporters"]["payload_problems"] == 0
    assert (tmp_path / "metrics.prom").read_text().startswith("# HELP")
    continuous = module.continuous_scenarios(tmp_path)
    assert continuous["passed"]
    assert continuous["alarm_gates_retrain"] is False
    assert continuous["hook_calls_before_reply"] == 0
    assert all(r["refused"] for r in continuous["refusals"].values())
    assert continuous["replacement"]["done"]
    claim = module.production_claim(tmp_path)
    assert not claim["committed_folders_allow_the_claim"]
    assert claim["checker_self_test"]["every_seeded_missing_record_refused"]
    assert claim["claim_made"] is False
    for name in (
        "observability.json",
        "continuous-evaluation.json",
        "exporter-payloads.json",
        "production-claim.json",
    ):
        assert json.loads((tmp_path / name).read_text())


def test_the_committed_evidence_says_what_was_not_run():
    folder = ROOT / "plan" / "evidence" / "phase-17" / "local"
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["local_checks_passed"] is True
    assert summary["production_claim_made"] is False
    not_run = " ".join(summary["not_run"])
    for word in ("AWS or GCP", "pager", "cloud halves", "reviewer's sign-off"):
        assert word in not_run
    recovery = json.loads((folder / "recovery.json").read_text())
    assert recovery["all_local_passed"] is True
    for provider in ("aws", "gcp"):
        assert recovery["cloud"][provider]["status"] == "not_run"
    ran = [r for r in recovery["results"] if r["status"] != "not_run"]
    assert len(ran) == 11
    for result in ran:
        assert result["detail"]["processes_still_alive_after_close"] == 0
    for provider in ("aws", "gcp"):
        folder_files = {
            p.name
            for p in (
                ROOT / "plan" / "evidence" / "phase-17" / provider
            ).iterdir()
        }
        assert folder_files == {"README.md"}
