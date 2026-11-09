#!/usr/bin/env python3
"""Verify public release bytes against locally built artifacts.

The verifier uses unauthenticated GitHub and PyPI APIs. It binds the public
tag to an exact commit and compares every published wheel and source archive
with the local artifact digest. It never treats a local build, a workflow
definition, or a private release endpoint as proof of publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
from urllib.parse import quote
from urllib.request import Request
from urllib.request import urlopen


def artifact_records(directory: Path) -> list[dict[str, Any]]:
    """Return deterministic digests for the wheel and source archives.

    Args:
        directory: Folder holding the built artifacts.

    Returns:
        One record per `.whl` or `.tar.gz` file, sorted by name, with its
        name, SHA-256 and size in bytes.

    Raises:
        ValueError: If the folder holds no wheel or source archive.
    """
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and (path.name.endswith(".whl") or path.name.endswith(".tar.gz"))
    )
    if not paths:
        raise ValueError(f"no wheel or source archive found in {directory}")
    records: list[dict[str, Any]] = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(
            {"name": path.name, "sha256": digest, "bytes": path.stat().st_size}
        )
    return records


def request_bytes(url: str, timeout: float = 30.0) -> bytes:
    """Fetch a public release response without credentials.

    Args:
        url: The URL to fetch.
        timeout: Seconds to wait for the response.

    Returns:
        The response body.
    """
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "scmle-release-verifier/1",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def request_json(url: str, timeout: float = 30.0) -> dict[str, Any]:
    """Fetch and decode one public JSON response.

    Args:
        url: The URL to fetch.
        timeout: Seconds to wait for the response.

    Returns:
        The decoded JSON object.

    Raises:
        ValueError: If the response is not a JSON object.
    """
    value = json.loads(request_bytes(url, timeout=timeout))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object from {url}")
    return value


def retry_json(
    url: str, attempts: int, delay: float, timeout: float
) -> dict[str, Any]:
    """Retry an eventually consistent public endpoint.

    Args:
        url: The URL to fetch.
        attempts: Maximum number of tries, at least 1.
        delay: Seconds to wait between tries.
        timeout: Seconds to wait for each response.

    Returns:
        The first JSON object successfully fetched. After the last failed
        try the error of that try (`OSError` or `ValueError`) is re-raised.

    Raises:
        ValueError: If `attempts` is less than 1.
    """
    if attempts < 1:
        raise ValueError("attempts must be positive")
    failure: Exception | None = None
    for attempt in range(attempts):
        try:
            return request_json(url, timeout=timeout)
        except (OSError, ValueError) as error:
            failure = error
            if attempt + 1 < attempts and delay:
                time.sleep(delay)
    assert failure is not None
    raise failure


def verify_external_release(
    repository: str,
    tag: str,
    source_commit: str,
    artifact_directory: Path,
    pypi_project: str,
    github_api: str = "https://api.github.com",
    pypi_api: str = "https://pypi.org/pypi",
    attempts: int = 1,
    delay: float = 0.0,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Verify GitHub Release and PyPI publication against local artifacts.

    Args:
        repository: GitHub repository as `owner/name`.
        tag: Release tag, starting with `v`; the PyPI version is the tag
            without the `v`.
        source_commit: The 40-character commit SHA the tag must name.
        artifact_directory: Folder with the locally built artifacts.
        pypi_project: Project name on PyPI.
        github_api: Base URL of the GitHub API.
        pypi_api: Base URL of the PyPI JSON API.
        attempts: Tries per endpoint.
        delay: Seconds between tries.
        timeout: Seconds to wait for each response.

    Returns:
        The evidence: the local artifact digests, the digests found on the
        GitHub release and on PyPI, the failures and `passed`, true only when
        there are no failures and both publications carry exactly the local
        artifacts.

    Raises:
        ValueError: If the repository, tag or commit has the wrong form (a
            missing artifact or an unusable response also raises
            `ValueError` from the helpers).
    """
    if not repository or "/" not in repository:
        raise ValueError("repository must be in owner/name form")
    if not tag.startswith("v"):
        raise ValueError("tag must start with v")
    if not source_commit or len(source_commit) != 40:
        raise ValueError("source_commit must be a 40-character commit SHA")
    local = artifact_records(artifact_directory)
    names = {record["name"] for record in local}
    encoded_repository = quote(repository, safe="/")
    encoded_tag = quote(tag, safe="")
    release_url = (
        f"{github_api.rstrip('/')}/repos/{encoded_repository}/releases/tags/"
        f"{encoded_tag}"
    )
    commit_url = (
        f"{github_api.rstrip('/')}/repos/{encoded_repository}/commits/"
        f"{encoded_tag}"
    )
    release = retry_json(release_url, attempts, delay, timeout)
    commit = retry_json(commit_url, attempts, delay, timeout)
    failures: list[str] = []
    if release.get("tag_name") != tag:
        failures.append(f"GitHub tag mismatch: {release.get('tag_name')!r}")
    if commit.get("sha") != source_commit:
        failures.append(
            f"GitHub commit mismatch: {commit.get('sha')!r} != {source_commit}"
        )
    assets = {
        str(asset.get("name")): asset
        for asset in release.get("assets", [])
        if isinstance(asset, dict) and asset.get("name")
    }
    github_records: list[dict[str, Any]] = []
    for record in local:
        asset = assets.get(record["name"])
        if asset is None:
            failures.append(
                f"GitHub release is missing asset: {record['name']}"
            )
            continue
        download_url = asset.get("browser_download_url")
        if not isinstance(download_url, str) or not download_url:
            failures.append(f"GitHub asset has no public URL: {record['name']}")
            continue
        published = hashlib.sha256(
            request_bytes(download_url, timeout=timeout)
        ).hexdigest()
        github_records.append({"name": record["name"], "sha256": published})
        if published != record["sha256"]:
            failures.append(f"GitHub asset digest mismatch: {record['name']}")

    pypi_url = (
        f"{pypi_api.rstrip('/')}/{quote(pypi_project, safe='')}/"
        f"{quote(tag[1:], safe='')}/json"
    )
    pypi = retry_json(pypi_url, attempts, delay, timeout)
    if pypi.get("info", {}).get("version") != tag[1:]:
        failures.append(
            f"PyPI version mismatch: {pypi.get('info', {}).get('version')!r}"
        )
    pypi_files = {
        str(item.get("filename")): item
        for item in pypi.get("urls", [])
        if isinstance(item, dict) and item.get("filename")
    }
    pypi_records: list[dict[str, Any]] = []
    for record in local:
        file_record = pypi_files.get(record["name"])
        if file_record is None:
            failures.append(f"PyPI is missing file: {record['name']}")
            continue
        digest = file_record.get("digests", {}).get("sha256")
        pypi_records.append({"name": record["name"], "sha256": digest})
        if digest != record["sha256"]:
            failures.append(f"PyPI digest mismatch: {record['name']}")
    return {
        "schema_version": 1,
        "repository": repository,
        "tag": tag,
        "source_commit": source_commit,
        "local_artifacts": local,
        "github": {"release_url": release_url, "assets": github_records},
        "pypi": {"url": pypi_url, "files": pypi_records},
        "passed": not failures
        and names == {record["name"] for record in github_records}
        and names == {record["name"] for record in pypi_records},
        "failures": failures,
    }


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse public release verification arguments.

    Args:
        arguments: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        The namespace with the repository, tag, commit, artifact folder,
        endpoints, retry settings and output path.
    """
    parser = argparse.ArgumentParser(
        description="Verify public GitHub and PyPI release digests."
    )
    parser.add_argument("--repository", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--pypi-project", default="scmle")
    parser.add_argument("--github-api", default="https://api.github.com")
    parser.add_argument("--pypi-api", default="https://pypi.org/pypi")
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--retry-delay", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Run public release checks and return a release-safe exit status.

    Prints the evidence and writes it to `--output` when given.

    Args:
        arguments: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        0 when the evidence passed; 1 when it did not; 2 when the
        verification could not run (an I/O error or invalid input).
    """
    options = parse_arguments(arguments)
    try:
        evidence = verify_external_release(
            options.repository,
            options.tag,
            options.source_commit,
            options.artifact_dir,
            options.pypi_project,
            github_api=options.github_api,
            pypi_api=options.pypi_api,
            attempts=options.attempts,
            delay=options.retry_delay,
            timeout=options.timeout,
        )
        serialized = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
        if options.output:
            options.output.parent.mkdir(parents=True, exist_ok=True)
            options.output.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"external release verification failed: {error}", file=sys.stderr)
        return 2
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
