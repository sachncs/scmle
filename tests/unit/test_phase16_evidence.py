"""The Phase 16 evidence generator's local scenarios run and pass."""

import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load():
    path = ROOT / "scripts" / "make_phase16_evidence.py"
    spec = importlib.util.spec_from_file_location("phase16_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_scenario_generators_pass_and_write_their_files(tmp_path):
    module = load()
    policy = module.policy_check(tmp_path)
    assert policy["committed_tree_clean"] and policy["seeded_all_caught"]
    assert module.tool_availability(tmp_path) is not None
    assert not module.destroy_orphans(tmp_path)["aws leftover bucket"]
    live = module.live_checks(tmp_path)
    assert live["recorded_all_pass"]
    assert live["every_seeded_failure_fails_a_check"]
    gates = module.gate_decisions(tmp_path)
    assert gates["staging_allowed"] and gates["production_allowed"]
    assert gates["every_seeded_gate_refused"]
    verdicts = module.reconciles(tmp_path)
    for steps in verdicts.values():
        assert steps["rollback to v1"]
        assert not steps["seeded: alias moved around the runtime"]
    assert module.hosted_controls(tmp_path)
    for name in (
        "policy-check.json",
        "terraform-tool.json",
        "destroy-orphans.json",
        "live-checks.json",
        "gate-decisions.json",
        "reconcile-reports.json",
        "hosted-controls.json",
    ):
        assert json.loads((tmp_path / name).read_text())


def test_the_committed_evidence_says_what_was_not_run():
    summary = json.loads(
        (ROOT / "plan" / "evidence" / "phase-16" / "summary.json").read_text()
    )
    assert summary["local_checks_passed"] is True
    not_run = " ".join(summary["not_run"])
    for word in ("terraform", "AWS or GCP account", "protected", "recorded"):
        assert word in not_run
    assert summary["checks"]["terraform"]["tool_installed"] in (True, False)
