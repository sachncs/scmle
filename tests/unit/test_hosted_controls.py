"""Tests for credential-free hosted CI and Pages verification."""

import hashlib
import io
import json
from pathlib import Path
import shutil

import pytest
import yaml

from scripts import verify_hosted_controls


def test_current_workflows_have_static_hosted_controls():
    """The repository's workflow contract is self-verifying."""
    root = Path(__file__).parents[2]

    evidence = verify_hosted_controls.workflow_evidence(root)

    assert evidence["static_controls"]["passed"] is True
    assert evidence["static_controls"]["failures"] == []
    assert evidence["static_controls"]["action_references"] > 0


def test_unpinned_action_is_a_static_failure(tmp_path):
    """A workflow action must be pinned to an immutable commit."""
    workflow_root = tmp_path / ".github" / "workflows"
    workflow_root.mkdir(parents=True)
    for name in verify_hosted_controls.WORKFLOW_NAMES:
        (workflow_root / name).write_text(
            "name: placeholder\n", encoding="utf-8"
        )
    (workflow_root / "ci.yml").write_text(
        "- uses: actions/checkout@v4\n", encoding="utf-8"
    )

    failures = verify_hosted_controls.workflow_failures(tmp_path)

    assert any("not pinned" in failure for failure in failures)


class FakeResponse:
    """Minimal response object for a deterministic Pages probe."""

    status = 200

    def __init__(self, body: bytes):
        self.body = io.BytesIO(body)

    def __enter__(self):
        return self

    def __exit__(self, *arguments):
        self.body.close()

    def read(self):
        return self.body.read()


def test_public_pages_probe_records_route_digests(monkeypatch):
    """The live probe is public GET-only and records reproducible digests."""
    requested = []

    def open_url(request, timeout):
        requested.append((request.full_url, timeout))
        return FakeResponse(b"<html>verified</html>")

    monkeypatch.setattr(
        verify_hosted_controls.urllib.request, "urlopen", open_url
    )
    routes = ("/", "/docs/")

    evidence = verify_hosted_controls.public_page_evidence(
        "https://example.test/scmle", routes=routes, timeout=3
    )

    assert evidence["passed"] is True
    assert [item["route"] for item in evidence["routes"]] == list(routes)
    assert (
        evidence["routes"][0]["sha256"]
        == hashlib.sha256(b"<html>verified</html>").hexdigest()
    )
    assert requested[0][0].endswith("/scmle/")


def test_public_pages_probe_requires_https():
    """The hosted verification cannot silently downgrade to HTTP."""
    with pytest.raises(ValueError, match="HTTPS"):
        verify_hosted_controls.public_page_evidence("http://example.test")


def test_main_writes_static_evidence(tmp_path, capsys):
    """The command emits machine-readable evidence without a network call."""
    root = Path(__file__).parents[2]
    output = tmp_path / "hosted-controls.json"

    assert (
        verify_hosted_controls.main(
            ["--root", str(root), "--output", str(output)]
        )
        == 0
    )
    record = json.loads(output.read_text(encoding="utf-8"))

    assert record["passed"] is True
    assert json.loads(capsys.readouterr().out)["passed"] is True


ROOT = Path(__file__).parents[2]


def copy_workflows(tmp_path):
    target = tmp_path / ".github" / "workflows"
    shutil.copytree(ROOT / ".github" / "workflows", target)
    site = tmp_path / "website"
    site.mkdir()
    shutil.copy(ROOT / "website" / "astro.config.mjs", site)
    return target


def failures_after(tmp_path, name, old, new):
    target = copy_workflows(tmp_path)
    path = target / name
    text = path.read_text(encoding="utf-8")
    assert old in text, old
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return verify_hosted_controls.workflow_failures(tmp_path)


def test_the_cloud_workflows_are_valid_yaml_and_guarded():
    for name in ("cloud-release.yml", "cloud-deploy.yml"):
        document = yaml.safe_load(
            (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        )
        assert document["permissions"] == {"contents": "read"}
    release = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "cloud-release.yml").read_text()
    )
    triggers = release[True] if True in release else release["on"]
    assert list(triggers) == ["workflow_dispatch"]
    jobs = release["jobs"]
    assert set(jobs) == {"guard", "build-scan", "staging", "production"}
    for name, job in jobs.items():
        if name != "guard":
            assert "needs.guard.outputs.enabled == 'true'" in job["if"], name
    assert jobs["production"]["needs"] == ["guard", "staging"]
    deploy = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "cloud-deploy.yml").read_text()
    )
    job = deploy["jobs"]["deploy"]
    assert job["environment"] == "${{ inputs.environment }}"
    assert job["permissions"]["id-token"] == "write"


def test_no_workflow_holds_a_long_lived_cloud_credential():
    for path in (ROOT / ".github" / "workflows").glob("*.yml"):
        text = path.read_text(encoding="utf-8")
        for marker in verify_hosted_controls.LONG_LIVED_CREDENTIALS:
            assert marker not in text, (path.name, marker)


def test_a_long_lived_key_in_a_workflow_is_a_failure(tmp_path):
    found = failures_after(
        tmp_path,
        "cloud-deploy.yml",
        "role-duration-seconds: 3600",
        "aws-access-key-id: AKIAXXXXXXXXXXXXXXXX",
    )
    assert any("long-lived credential" in item for item in found)


def test_an_unguarded_cloud_job_is_a_failure(tmp_path):
    found = failures_after(
        tmp_path,
        "cloud-release.yml",
        "    needs: [guard, staging]\n    if: needs.guard.outputs.enabled == 'true'\n",
        "    needs: [guard, staging]\n",
    )
    assert any("must test the guard" in item for item in found)


def test_a_push_trigger_on_the_cloud_release_is_a_failure(tmp_path):
    found = failures_after(
        tmp_path,
        "cloud-release.yml",
        "on:\n  workflow_dispatch:",
        "on:\n  push:\n    branches: [master]\n  workflow_dispatch:",
    )
    assert any("push:" in item for item in found)


def test_removing_the_environment_from_the_deploy_job_is_a_failure(tmp_path):
    found = failures_after(
        tmp_path,
        "cloud-deploy.yml",
        "    environment: ${{ inputs.environment }}\n",
        "",
    )
    assert any("must run in an environment" in item for item in found)


def test_dropping_a_gate_from_the_deploy_job_is_a_failure(tmp_path):
    found = failures_after(
        tmp_path, "cloud-deploy.yml", "scmle deploy gate certify", "echo"
    )
    assert any("scmle deploy gate certify" in item for item in found)


def test_a_cloud_deploy_id_token_at_the_top_level_is_a_failure(tmp_path):
    found = failures_after(
        tmp_path,
        "cloud-deploy.yml",
        "permissions:\n  contents: read\n\njobs:",
        "permissions:\n  contents: read\n  id-token: write\n\njobs:",
    )
    assert any("id-token at the top" in item for item in found)


def test_unpinned_third_party_actions_in_the_cloud_workflows_are_failures(
    tmp_path,
):
    found = failures_after(
        tmp_path,
        "cloud-deploy.yml",
        "google-github-actions/auth@7c6bc770dae815cd3e89ee6cdf493a5fab2cc093",
        "google-github-actions/auth@v3",
    )
    assert any("not pinned" in item for item in found)
