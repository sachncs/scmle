"""`scmle deploy`: live checks, the gate and record verification."""

import dataclasses
import json

import pytest

from scmle import cli
from scmle.deploy import livetests
from scmle.deploy import provenance
from tests import support_release as sr


@pytest.fixture
def release(tmp_path):
    """A registry, an approval and every evidence file, all consistent."""
    root = tmp_path / "release"
    bundle = root / "registry" / "churn" / "v3"
    bundle.mkdir(parents=True)
    (bundle / "model_manifest.json").write_text(
        json.dumps({"files": {"model.pkl": "ab" * 32}, "model_version": "3"}),
        encoding="utf-8",
    )
    version = dataclasses.asdict(sr.version())
    (root / "registry" / "registry.json").write_text(
        json.dumps(
            {"models": {"churn": {"versions": {"3": version}, "aliases": {}}}}
        ),
        encoding="utf-8",
    )
    digest = provenance.bundle_digest(bundle)
    (root / "approval.json").write_text(
        json.dumps(sr.approval().to_dict()), encoding="utf-8"
    )
    (root / "scan.json").write_text(
        json.dumps(
            {"dependencies": [{"name": "numpy", "version": "2.0", "vulns": []}]}
        ),
        encoding="utf-8",
    )
    checks = sr.run_live(sr.recorded(digest), model_digest=digest)
    (root / "checks.json").write_text(
        json.dumps([c.to_dict() for c in checks]), encoding="utf-8"
    )
    (root / "observation.json").write_text(
        json.dumps(sr.seen().to_dict()), encoding="utf-8"
    )
    (root / "key").write_text(sr.KEY + "\n", encoding="utf-8")
    manifest = {
        "environment": "staging",
        "provider": "aws",
        "source_commit": sr.COMMIT,
        "registry": "registry",
        "model": "churn",
        "version": 3,
        "container_digest": sr.CONTAINER,
        "config_digest": sr.CONFIG,
        "requested_by": "author",
        "approver": "release-manager",
        "approved_at": sr.NOW,
        "approval": "approval.json",
        "scan_report": "scan.json",
        "checks": "checks.json",
        "observation": "observation.json",
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, manifest, digest


def write_manifest(root, manifest):
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def gate(root, stage="certify", *extra):
    return cli.main(
        [
            "deploy",
            "gate",
            stage,
            str(root / "manifest.json"),
            "--key-file",
            str(root / "key"),
            *extra,
        ]
    )


def test_a_complete_release_is_certified_and_the_record_verifies(
    release, capsys
):
    root, ignored, digest = release
    assert gate(root, "admit") == cli.EXIT_OK
    capsys.readouterr()
    output = root / "record.json"
    assert gate(root, "certify", "--output", str(output)) == cli.EXIT_OK
    decision = json.loads(capsys.readouterr().out)
    assert decision["allowed"] is True
    assert decision["record"]["model_digest"] == digest
    code = cli.main(
        [
            "deploy",
            "verify-record",
            str(output),
            "--key-file",
            str(root / "key"),
        ]
    )
    assert code == cli.EXIT_OK
    stored = json.loads(output.read_text())
    stored["approver"] = "someone-else"
    output.write_text(json.dumps(stored), encoding="utf-8")
    code = cli.main(
        [
            "deploy",
            "verify-record",
            str(output),
            "--key-file",
            str(root / "key"),
        ]
    )
    assert code == cli.EXIT_FAILED


@pytest.mark.parametrize(
    "drop", ["approval", "scan_report", "checks", "observation"]
)
def test_evidence_that_is_not_supplied_blocks_the_release(
    release, capsys, drop
):
    root, manifest, ignored = release
    del manifest[drop]
    write_manifest(root, manifest)
    output = root / "record.json"
    assert gate(root, "certify", "--output", str(output)) == cli.EXIT_BLOCKED
    assert json.loads(capsys.readouterr().out)["allowed"] is False
    assert not output.exists()


def test_an_unsigned_model_blocks_the_release(release):
    root = release[0]
    path = root / "registry" / "registry.json"
    index = json.loads(path.read_text())
    index["models"]["churn"]["versions"]["3"]["signed"] = False
    path.write_text(json.dumps(index), encoding="utf-8")
    assert gate(root, "admit") == cli.EXIT_BLOCKED


def test_a_scan_with_a_blocking_finding_blocks_the_release(release):
    root = release[0]
    (root / "scan.json").write_text(
        json.dumps(
            {
                "dependencies": [
                    {
                        "name": "numpy",
                        "version": "1.0",
                        "vulns": [{"id": "PYSEC-1", "aliases": []}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert gate(root, "admit") == cli.EXIT_BLOCKED


def test_an_empty_scan_blocks_the_release(release):
    root = release[0]
    (root / "scan.json").write_text('{"dependencies": []}', encoding="utf-8")
    assert gate(root, "admit") == cli.EXIT_BLOCKED


def test_a_named_file_that_is_missing_is_a_failure_not_a_pass(release):
    root = release[0]
    (root / "approval.json").unlink()
    assert (
        cli.main(
            [
                "deploy",
                "gate",
                "admit",
                str(root / "manifest.json"),
                "--key-file",
                str(root / "key"),
            ]
        )
        == cli.EXIT_FAILED
    )


def test_a_manifest_missing_required_fields_is_a_failure(release):
    root, manifest, ignored = release
    del manifest["approver"]
    write_manifest(root, manifest)
    assert gate(root, "admit") == cli.EXIT_FAILED


def test_production_with_the_staging_record_is_certified(release, capsys):
    root, manifest, digest = release
    output = root / "staging-record.json"
    assert gate(root, "certify", "--output", str(output)) == cli.EXIT_OK
    capsys.readouterr()
    production = sr.run_live(
        sr.recorded(digest), environment="production", model_digest=digest
    )
    (root / "checks.json").write_text(
        json.dumps([c.to_dict() for c in production]), encoding="utf-8"
    )
    manifest.update(
        environment="production", staging_record="staging-record.json"
    )
    write_manifest(root, manifest)
    assert gate(root, "certify") == cli.EXIT_OK
    capsys.readouterr()
    del manifest["staging_record"]
    write_manifest(root, manifest)
    assert gate(root, "certify") == cli.EXIT_BLOCKED


def test_live_writes_results_and_fails_when_a_check_fails(
    release, monkeypatch, capsys
):
    root, ignored, digest = release
    (root / "token").write_text("good-token\n", encoding="utf-8")
    document = sr.recorded(digest)
    monkeypatch.setattr(
        livetests, "http_fetch", lambda url, timeout: sr.fetch_from(document)
    )
    base = [
        "deploy",
        "live",
        "--url",
        document["base_url"],
        "--token-file",
        str(root / "token"),
        "--environment",
        "staging",
        "--commit",
        sr.COMMIT,
        "--container-digest",
        sr.CONTAINER,
        "--model-digest",
        digest,
        "--output",
        str(root / "live.json"),
    ]
    assert cli.main(base) == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out) == {
        "health": True,
        "schema": True,
        "inference": True,
        "security": True,
    }
    saved = json.loads((root / "live.json").read_text())
    assert [item["name"] for item in saved] == list(livetests.CHECK_NAMES)
    for item in document["exchanges"]:
        if item["path"] == "/predict" and item["auth"] == "none":
            item["status"] = 200
    assert cli.main(base) == cli.EXIT_FAILED
    assert json.loads(capsys.readouterr().out)["security"] is False


def test_an_empty_key_file_is_a_configuration_error(release):
    root = release[0]
    (root / "key").write_text("\n", encoding="utf-8")
    assert gate(root, "admit") == cli.EXIT_FAILED
