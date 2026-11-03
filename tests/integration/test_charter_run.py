"""A run governed by an analysis charter: custody, artifacts and limits."""

import json
import pathlib

import pytest
import support

from scmle.contracts import errors
from scmle.data.charter import sealing
from scmle.data.custody import manifest as manifest_module
from scmle.data.custody import sealed
from tests.unit.test_charter_contracts import make_charter

KEY = "integration-key"


def write_charter(tmp_path, **overrides):
    values = {"min_effect": 0.5, "k_max": 3}
    values.update(overrides)
    path = tmp_path / "charter.json"
    sealing.write_charter(path, make_charter(**values), signing_key=KEY)
    return path


def run(frame, tmp_path, charter=None, **kwargs):
    return support.make_project(
        frame,
        "y",
        time_budget=300,
        output_dir=tmp_path / "out",
        charter=charter,
        **kwargs,
    ).run()


def events(run_dir):
    lines = (pathlib.Path(run_dir) / "events.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def event(run_dir, kind):
    return next(e for e in events(run_dir) if e["kind"] == kind)


def test_a_charter_run_seals_c_even_in_cooperative_mode(binary_frame, tmp_path):
    result = run(binary_frame, tmp_path, write_charter(tmp_path))

    custody = event(result.run_dir, "custody.sealed")["data"]
    assert custody["isolation"] == {
        "C": "vault process",
        "final_test": "vault process",
    }
    assert custody["confirmation_rows"] > 0
    assert custody["scope"] == "confirmatory"
    created = event(result.run_dir, "split.created")["data"]
    assert created["test_set_isolation"] == "separate vault process"


def test_the_charter_artifacts_are_written_and_verifiable(
    binary_frame, tmp_path
):
    result = run(binary_frame, tmp_path, write_charter(tmp_path))
    folder = pathlib.Path(result.run_dir)

    for name in (
        "analysis_charter.json",
        "split_manifest.json",
        "error_budget.json",
        "feasibility.json",
        "custody_log.jsonl",
    ):
        assert (folder / name).exists(), name
    loaded = manifest_module.load_manifest(
        json.loads((folder / "split_manifest.json").read_text())
    )
    assert len(loaded.sha256) == 64
    assert sealing.read_verified(
        folder / "analysis_charter.json", signing_key=KEY
    ).signed
    sealed.OpenLog.verify_file(folder / "custody_log.jsonl")


def test_profiling_sees_exploration_rows_only(binary_frame, tmp_path):
    plain = run(binary_frame, tmp_path / "plain")
    governed = run(binary_frame, tmp_path, write_charter(tmp_path))

    def profiled(result):
        folder = pathlib.Path(result.run_dir)
        report = json.loads((folder / "curation_report.json").read_text())
        return report["profiled_rows"]

    manifest = json.loads(
        (pathlib.Path(governed.run_dir) / "split_manifest.json").read_text()
    )
    assert profiled(governed) == len(manifest["exploration_ids"])
    assert profiled(governed) < profiled(plain)
    assert set(manifest["final_test_ids"]) == set(
        json.loads(
            (
                pathlib.Path(plain.run_dir) / "validation_manifest.json"
            ).read_text()
        )["test_ids"]
    )


def test_the_final_test_still_opens_once_and_scores(binary_frame, tmp_path):
    result = run(binary_frame, tmp_path, write_charter(tmp_path))
    assert result.evaluation["test_set_opened"] is True
    assert result.evaluation["n_test"] > 0


def test_a_run_without_a_charter_is_unchanged(binary_frame, tmp_path):
    result = run(binary_frame, tmp_path)
    folder = pathlib.Path(result.run_dir)
    assert not (folder / "split_manifest.json").exists()
    assert not (folder / "custody_log.jsonl").exists()
    created = event(result.run_dir, "split.created")["data"]
    assert created["test_set_isolation"] == "in-process protocol"
    assert all(e["kind"] != "custody.sealed" for e in events(result.run_dir))


def test_a_tampered_charter_stops_the_run_before_profiling(
    binary_frame, tmp_path
):
    path = write_charter(tmp_path)
    record = json.loads(path.read_text())
    record["charter"]["question"] = "A different question?"
    path.write_text(json.dumps(record))

    with pytest.raises(errors.ArtifactError, match="hash"):
        run(binary_frame, tmp_path, path)
    runs = list((tmp_path / "out").glob("run_*"))
    assert runs and not (runs[0] / "dataset_profile.json").exists()


def test_too_few_rows_run_unconfirmed_instead_of_failing(tmp_path):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {"a": rng.normal(size=120), "y": (rng.random(120) < 0.4).astype(int)}
    )
    result = run(frame, tmp_path, write_charter(tmp_path))
    custody = event(result.run_dir, "custody.sealed")["data"]
    assert custody["confirmation_rows"] == 0
    assert custody["scope"] == "exploratory_unconfirmed"
    assert custody["isolation"] == {"final_test": "vault process"}


def test_a_missing_charter_file_is_a_config_error(binary_frame, tmp_path):
    with pytest.raises(errors.ConfigError):
        run(binary_frame, tmp_path, tmp_path / "nope.json")


def test_a_charter_run_writes_the_ledger_control_and_claims(
    binary_frame, tmp_path
):
    result = run(binary_frame, tmp_path, write_charter(tmp_path))
    folder = pathlib.Path(result.run_dir)
    for name in (
        "hypothesis_ledger.jsonl",
        "negative_control.json",
        "confirmed_claims.json",
    ):
        assert (folder / name).exists(), name
    done = event(result.run_dir, "ledger.completed")["data"]
    claims = json.loads((folder / "confirmed_claims.json").read_text())
    assert done["ledger_head"] == claims["ledger_head"]
    from scmle.data.ledger import ledger as ledger_module

    ledger = ledger_module.Ledger.load(
        folder / "hypothesis_ledger.jsonl", expected_head=done["ledger_head"]
    )
    # Only numeric features are candidates; the categorical one is skipped.
    assert {h.spec.column for h in ledger.hypotheses.values()} == {"a", "b"}
    assert claims["outcome"] == done["outcome"]
    log = sealed.OpenLog.verify_file(folder / "custody_log.jsonl")
    events_c = [e["event"] for e in log.entries if e["set"] == "C"]
    if claims["outcome"] == "confirmed":
        assert events_c.count("open") == 1
        assert "selection_frozen" in events_c
        assert events_c.index("selection_frozen") < events_c.index("open")
    else:
        assert "open" not in events_c


def test_a_run_without_a_charter_has_no_ledger(binary_frame, tmp_path):
    result = run(binary_frame, tmp_path)
    folder = pathlib.Path(result.run_dir)
    assert not (folder / "hypothesis_ledger.jsonl").exists()
    assert all(e["kind"] != "ledger.completed" for e in events(result.run_dir))


def leaky_frame(binary_frame):
    frame = binary_frame.copy()
    frame["echo"] = frame["y"] * 1.0
    return frame


def test_a_charter_run_writes_the_admissibility_artifacts(
    binary_frame, tmp_path
):
    result = run(binary_frame, tmp_path, write_charter(tmp_path))
    folder = pathlib.Path(result.run_dir)

    table = json.loads((folder / "admissibility_table.json").read_text())
    assert table["kind"] == "predictive" and table["mode"] == "permissive"
    assert {d["name"] for d in table["decisions"]} == {"a", "b", "c"}
    assert all(d["admission"] == "admit_flagged" for d in table["decisions"])
    copied = json.loads((folder / "detector_certificates.json").read_text())
    assert copied["certificates"]
    done = event(result.run_dir, "admissibility.completed")["data"]
    assert done["quarantined"] == {}
    assert set(done["detector_status"]) >= {"f5-sentinel", "f5-missingness"}


def test_leakage_suspicion_is_logged_but_descriptive_until_certified(
    binary_frame, tmp_path
):
    result = run(leaky_frame(binary_frame), tmp_path, write_charter(tmp_path))
    folder = pathlib.Path(result.run_dir)

    done = event(result.run_dir, "admissibility.completed")["data"]
    assert done["leakage_gate"] == "descriptive"
    assert "echo" in done["suspects"] and done["quarantined"] == {}
    rows = [
        json.loads(line)
        for line in (folder / "hypothesis_ledger.jsonl")
        .read_text()
        .splitlines()
    ]
    diagnostics = [r for r in rows if r["kind"] == "diagnostic"]
    assert diagnostics
    assert {r["body"]["check"]["set_id"] for r in diagnostics} == {"X"}
    plan = json.loads((folder / "dataset_profile.json").read_text())
    assert "echo" in plan["feature_plan"]["numeric"]


def test_a_gating_certificate_drops_the_quarantined_field_from_the_plan(
    binary_frame, tmp_path, monkeypatch
):
    from scmle.data.admissibility import qualify
    from scmle.data.detectors import certificates

    monkeypatch.setattr(
        qualify.leakage,
        "gate_status",
        lambda *args, **kwargs: certificates.GateStatus.GATING,
    )
    result = run(leaky_frame(binary_frame), tmp_path, write_charter(tmp_path))
    folder = pathlib.Path(result.run_dir)

    plan = json.loads((folder / "dataset_profile.json").read_text())[
        "feature_plan"
    ]
    assert "echo" not in plan["numeric"]
    assert "quarantined by admissibility" in plan["dropped"]["echo"]
    table = json.loads((folder / "admissibility_table.json").read_text())
    decisions = {d["name"]: d["admission"] for d in table["decisions"]}
    assert decisions["echo"] == "quarantine"


def test_a_run_without_a_charter_writes_no_admissibility_artifacts(
    binary_frame, tmp_path
):
    result = run(binary_frame, tmp_path)
    folder = pathlib.Path(result.run_dir)

    assert not (folder / "admissibility_table.json").exists()
    assert not (folder / "detector_certificates.json").exists()
    assert all(
        e["kind"] != "admissibility.completed" for e in events(result.run_dir)
    )
