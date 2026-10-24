"""Tests for credential-free hosted CI and Pages verification."""

import hashlib
import io
import json
from pathlib import Path

import pytest

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
