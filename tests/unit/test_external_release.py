"""Tests for no-secret public release reproducibility checks."""

import json

from scripts import verify_external_release


def make_artifacts(tmp_path):
    """Create deterministic wheel and source-distribution fixtures."""
    wheel = tmp_path / "scmle-0.1.0b1-py3-none-any.whl"
    source = tmp_path / "scmle-0.1.0b1.tar.gz"
    wheel.write_bytes(b"wheel-bytes")
    source.write_bytes(b"source-bytes")
    return wheel, source


def test_external_release_matches_public_github_and_pypi_digests(
    monkeypatch, tmp_path
):
    """Published assets must match the exact local artifacts."""
    wheel, source = make_artifacts(tmp_path)
    records = verify_external_release.artifact_records(tmp_path)
    digests = {record["name"]: record["sha256"] for record in records}
    release_url = (
        "https://api.example/repos/sachncs/scmle/releases/tags/v0.1.0b1"
    )
    commit_url = "https://api.example/repos/sachncs/scmle/commits/v0.1.0b1"
    wheel_url = "https://downloads.example/wheel"
    source_url = "https://downloads.example/source"
    responses = {
        release_url: {
            "tag_name": "v0.1.0b1",
            "assets": [
                {"name": wheel.name, "browser_download_url": wheel_url},
                {"name": source.name, "browser_download_url": source_url},
            ],
        },
        commit_url: {"sha": "a" * 40},
        "https://pypi.example/scmle/0.1.0b1/json": {
            "info": {"version": "0.1.0b1"},
            "urls": [
                {
                    "filename": wheel.name,
                    "digests": {"sha256": digests[wheel.name]},
                },
                {
                    "filename": source.name,
                    "digests": {"sha256": digests[source.name]},
                },
            ],
        },
    }
    bytes_responses = {
        wheel_url: wheel.read_bytes(),
        source_url: source.read_bytes(),
    }

    def fake_json(url, timeout=30):
        return responses[url]

    def fake_bytes(url, timeout=30):
        return bytes_responses[url]

    monkeypatch.setattr(verify_external_release, "request_json", fake_json)
    monkeypatch.setattr(verify_external_release, "request_bytes", fake_bytes)

    evidence = verify_external_release.verify_external_release(
        "sachncs/scmle",
        "v0.1.0b1",
        "a" * 40,
        tmp_path,
        "scmle",
        github_api="https://api.example",
        pypi_api="https://pypi.example",
    )

    assert evidence["passed"] is True
    assert evidence["failures"] == []
    assert [record["name"] for record in evidence["local_artifacts"]] == [
        wheel.name,
        source.name,
    ]


def test_external_release_rejects_published_digest_mismatch(
    monkeypatch, tmp_path
):
    """A changed public asset fails instead of becoming release evidence."""
    wheel, source = make_artifacts(tmp_path)
    records = verify_external_release.artifact_records(tmp_path)
    release_url = (
        "https://api.example/repos/sachncs/scmle/releases/tags/v0.1.0b1"
    )
    commit_url = "https://api.example/repos/sachncs/scmle/commits/v0.1.0b1"
    pypi_url = "https://pypi.example/scmle/0.1.0b1/json"
    responses = {
        release_url: {
            "tag_name": "v0.1.0b1",
            "assets": [
                {
                    "name": wheel.name,
                    "browser_download_url": "https://downloads.example/wheel",
                },
                {
                    "name": source.name,
                    "browser_download_url": "https://downloads.example/source",
                },
            ],
        },
        commit_url: {"sha": "b" * 40},
        pypi_url: {
            "info": {"version": "0.1.0b1"},
            "urls": [
                {
                    "filename": record["name"],
                    "digests": {"sha256": record["sha256"]},
                }
                for record in records
            ],
        },
    }

    def fake_json(url, timeout=30):
        return responses[url]

    def fake_bytes(url, timeout=30):
        return (
            b"changed-public-bytes" if "wheel" in url else source.read_bytes()
        )

    monkeypatch.setattr(verify_external_release, "request_json", fake_json)
    monkeypatch.setattr(verify_external_release, "request_bytes", fake_bytes)

    evidence = verify_external_release.verify_external_release(
        "sachncs/scmle",
        "v0.1.0b1",
        "a" * 40,
        tmp_path,
        "scmle",
        github_api="https://api.example",
        pypi_api="https://pypi.example",
    )

    assert evidence["passed"] is False
    assert any("commit mismatch" in item for item in evidence["failures"])
    assert any("asset digest mismatch" in item for item in evidence["failures"])


def test_external_release_output_is_canonical_json(
    monkeypatch, tmp_path, capsys
):
    """The command output can be retained as external release evidence."""
    wheel, source = make_artifacts(tmp_path)
    records = verify_external_release.artifact_records(tmp_path)
    digest = {item["name"]: item["sha256"] for item in records}
    wheel_url = "https://downloads.example/wheel"
    source_url = "https://downloads.example/source"
    github_api = "https://api.example"
    pypi_api = "https://pypi.example"
    responses = {
        f"{github_api}/repos/a/b/releases/tags/v0.1.0b1": {
            "tag_name": "v0.1.0b1",
            "assets": [
                {"name": wheel.name, "browser_download_url": wheel_url},
                {"name": source.name, "browser_download_url": source_url},
            ],
        },
        f"{github_api}/repos/a/b/commits/v0.1.0b1": {"sha": "c" * 40},
        f"{pypi_api}/package/0.1.0b1/json": {
            "info": {"version": "0.1.0b1"},
            "urls": [
                {
                    "filename": wheel.name,
                    "digests": {"sha256": digest[wheel.name]},
                },
                {
                    "filename": source.name,
                    "digests": {"sha256": digest[source.name]},
                },
            ],
        },
    }

    def fake_json(url, timeout=30):
        return responses[url]

    def fake_bytes(url, timeout=30):
        return wheel.read_bytes() if "wheel" in url else source.read_bytes()

    monkeypatch.setattr(verify_external_release, "request_json", fake_json)
    monkeypatch.setattr(verify_external_release, "request_bytes", fake_bytes)

    output = tmp_path / "external-release.json"
    assert (
        verify_external_release.main(
            [
                "--repository",
                "a/b",
                "--tag",
                "v0.1.0b1",
                "--source-commit",
                "c" * 40,
                "--artifact-dir",
                str(tmp_path),
                "--pypi-project",
                "package",
                "--github-api",
                github_api,
                "--pypi-api",
                pypi_api,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True
    assert json.loads(capsys.readouterr().out)["passed"] is True
