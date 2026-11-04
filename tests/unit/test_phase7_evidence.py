"""The committed Phase 7 evidence artifacts still verify."""

import importlib.util
import json
import pathlib

import pytest

from scmle.contracts import errors
from scmle.data.custody import manifest as manifest_module
from scmle.data.custody import sealed
from tests.unit import test_custody_golden as golden

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "plan" / "evidence" / "phase-7"


def generator():
    spec = importlib.util.spec_from_file_location(
        "make_phase7_evidence", ROOT / "scripts" / "make_phase7_evidence.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


KEY = generator().DEMONSTRATION_KEY


def test_the_sample_run_verifies_and_matches_its_recorded_result():
    folder = EVIDENCE / "signed-sample-run"
    steps = generator().verify_run(folder)
    recorded = json.loads((folder / "verification.json").read_text())
    assert steps == recorded
    assert steps["charter_signature"] == "valid"


def test_the_sample_run_log_has_a_mac_on_every_entry():
    log = sealed.OpenLog.verify_file(
        EVIDENCE / "signed-sample-run" / "custody_log.jsonl", KEY
    )
    assert log.entries and all("mac" in e for e in log.entries)


def test_a_wrong_key_does_not_verify_the_sample_run():
    with pytest.raises(errors.ArtifactError, match="MAC"):
        sealed.OpenLog.verify_file(
            EVIDENCE / "signed-sample-run" / "custody_log.jsonl", "other"
        )


def test_the_attack_log_matches_the_recorded_attempts():
    folder = EVIDENCE / "access-attempt-log"
    log = sealed.OpenLog.verify_file(folder / "custody_log.jsonl", KEY)
    attempts = json.loads((folder / "attempts.json").read_text())
    refused = [a for a in attempts if a["outcome"].startswith("refused")]
    released = [a for a in attempts if a["outcome"] == "released"]
    events = [e["event"] for e in log.entries]
    assert events.count("denied") == len(refused) == 4
    assert events.count("open") == len(released) == 1
    assert "bound" in events


def test_every_manifest_verifies_and_the_fixtures_match_the_pins():
    folder = EVIDENCE / "split-manifests"
    for name in ("run.json", "grouped-fixture.json", "temporal-fixture.json"):
        record = json.loads((folder / name).read_text())
        loaded = manifest_module.CustodyManifest.from_dict(record)
        assert loaded.row_roots
    grouped = json.loads((folder / "grouped-fixture.json").read_text())
    temporal = json.loads((folder / "temporal-fixture.json").read_text())
    assert grouped["sha256"] == golden.GROUPED_SHA256
    assert temporal["sha256"] == golden.TEMPORAL_SHA256
