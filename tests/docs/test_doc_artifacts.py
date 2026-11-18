"""Guarantees of docs/ARTIFACTS.md that no other test enforced."""

import datetime
import json
import pathlib

import pandas
import pytest

from benchmarks import reliability
from scmle.artifacts import bundle
from scmle.assurance.custody import sealed
from scmle.contracts import errors
from scmle.data.splitting.splitters import lockbox
from scmle.deploy import packaging
from scmle.deploy import parity
from scmle.deploy import vuln_gate

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_FILES = (
    "pipeline.joblib",
    "schema.json",
    "reference.json",
    "conformal.json",
    "novelty.joblib",
    "model_manifest.json",
)
STEPS = (
    "python -m scmle.deploy.context check .",
    "python -m scmle.deploy.base_image resolve",
    "docker buildx build",
    "pip-audit --requirement",
    "python -m scmle.deploy.vuln_gate scan.json --policy vuln-policy.json",
    "python -m scmle.deploy.attest image",
)


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    """The run directory of one small, quick binary run.

    Returns:
        The path of the run directory.
    """
    directory = tmp_path_factory.mktemp("artifacts")
    project = reliability.fast_project(
        reliability.binary_data(200), "y", directory, time_budget=30
    )
    return pathlib.Path(project.run().run_dir)


@pytest.fixture(scope="module")
def service(tmp_path_factory):
    """A service folder exported from the handwritten reference model.

    Returns:
        The path of the service folder.
    """
    golden = parity.goldens(ROOT / "deploy" / "local" / "golden")
    target = tmp_path_factory.mktemp("service") / "service"
    packaging.scaffold(
        parity.handwrite(golden.spec),
        target,
        signing_key="artifacts-doc-signing-key",
    )
    return target


def read(path):
    """Parses a JSON file.

    Args:
        path: The file.

    Returns:
        The parsed document.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_decision_log_holds_included_rejected_and_deferred_choices(folder):
    """Enforces docs/ARTIFACTS.md:23."""
    plan = read(folder / "diagnostics" / "decision_log.json")["plan"]
    assert {d["action"] for d in plan} == {"include", "reject", "defer"}
    assert all(d["technique"] and d["reason"] for d in plan)


def test_without_criteria_the_release_evaluation_blocks(folder):
    """Enforces docs/ARTIFACTS.md:125."""
    record = read(folder / "release_evaluation.json")
    assert record["status"] == "policy_not_declared"
    assert record["blocked"] is True and record["passed"] is False
    assert "not invented from observed metrics" in record["blockers"][0]


def test_the_model_folder_is_checksummed_and_unsigned_needs_trust(folder):
    """Enforces docs/ARTIFACTS.md:108."""
    model = folder / "model"
    for name in MODEL_FILES:
        assert (model / name).is_file(), name
    files = read(model / "model_manifest.json")["files"]
    assert set(MODEL_FILES) - {"model_manifest.json"} <= set(files)
    with pytest.raises(errors.ArtifactError):
        bundle.Bundle.load(model)
    assert bundle.Bundle.load(model, trusted=True).info.files


def test_the_custody_log_records_seal_issue_open_and_refusal():
    """Enforces docs/ARTIFACTS.md:38."""
    log = sealed.Journal()
    rows = lockbox.Resident(
        pandas.DataFrame({"a": [123456.789, 2.0, 3.0]}),
        pandas.Series([0, 1, 0]),
        (10, 11, 12),
    )
    locker = sealed.Locker("C", rows, log)
    locker.commit("c", "l")
    token = locker.mint()
    locker.open(token, "confirmation", claims_hash="c", ledger_hash="l")
    with pytest.raises(errors.LeakageError):
        locker.open(token, "exploration", claims_hash="c", ledger_hash="l")
    events = [e["event"] for e in log.entries]
    assert events == ["sealed", "bound", "token_issued", "open", "denied"]
    for earlier, later in zip(log.entries, log.entries[1:], strict=False):
        assert later["prev"] == earlier["hash"]
    log.verify()
    assert "123456.789" not in json.dumps(log.entries)


def test_build_sh_scans_resolves_builds_scans_and_attests_in_order(service):
    """Enforces docs/ARTIFACTS.md:136."""
    script = (service / "build.sh").read_text(encoding="utf-8")
    assert "set -euo pipefail" in script
    places = [script.index(step) for step in STEPS]
    assert places == sorted(places)
    ignored = (service / ".dockerignore").read_text(encoding="utf-8")
    assert ".env" in ignored.split()


def test_the_shipped_gate_policy_blocks_high_with_one_dated_waiver(service):
    """Enforces docs/ARTIFACTS.md:140."""
    policy = read(service / "vuln-policy.json")
    assert policy == vuln_gate.DEFAULT_POLICY
    assert policy["fail_at_or_above"] == "HIGH"
    assert policy["exceptions"] == []
    (waiver,) = policy["unscanned_exceptions"]
    assert waiver["name"] == "scmle" and waiver["expires"]
    today = datetime.date(2026, 11, 5)

    def report(severity):
        finding = {
            "VulnerabilityID": "CVE-1",
            "PkgName": "openssl",
            "InstalledVersion": "3.0",
            "Severity": severity,
        }
        package = {"Name": "openssl", "Version": "3.0"}
        return {
            "Results": [{"Packages": [package], "Vulnerabilities": [finding]}]
        }

    assert not vuln_gate.evaluate(report("HIGH"), policy, today=today)["passed"]
    assert vuln_gate.evaluate(report("MEDIUM"), policy, today=today)["passed"]
