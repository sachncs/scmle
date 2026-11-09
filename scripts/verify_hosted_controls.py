#!/usr/bin/env python3
"""Verify hosted CI and GitHub Pages controls without credentials.

The static checks validate the repository's workflow contract. When a public
Pages URL is supplied, the live check performs only unauthenticated GET
requests and records the response digests. A successful static check is not
hosted-run evidence; a successful live check is not evidence for
container, kernel, or broker semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlparse
import urllib.request

LONG_LIVED_CREDENTIALS = (
    "aws-access-key-id",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "credentials_json",
    "GOOGLE_CREDENTIALS",
    "service_account_key",
)
ACTION_REFERENCE = re.compile(
    r"^\s*(?:-\s+)?uses:\s+(?P<action>[^\s@]+)@(?P<revision>[^\s#]+)"
)
SHA256_REVISION = re.compile(r"^[0-9a-f]{40}$")
WORKFLOW_NAMES = (
    "ci.yml",
    "full-suite.yml",
    "pages.yml",
    "release.yml",
    "site.yml",
)
PAGES_ROUTES = (
    "/",
    "/docs/",
    "/docs/quickstart/",
    "/docs/data/",
    "/docs/modalities/",
    "/docs/capabilities/",
    "/docs/modeling/",
    "/docs/evidence/",
    "/docs/workflows/",
    "/docs/api/",
    "/docs/architecture/",
    "/docs/cards/",
    "/docs/operations/",
    "/docs/security/",
    "/docs/recovery/",
    "/docs/migrations/",
    "/docs/troubleshooting/",
)


def read_text(path: Path) -> str:
    """Reads one UTF-8 control file with a useful error.

    Args:
        path: The file.

    Returns:
        The file's text.

    Raises:
        ValueError: If the file cannot be read.
    """
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error}") from error


def action_references(content: str) -> list[dict[str, str]]:
    """Returns every GitHub Action reference in a workflow.

    Args:
        content: The workflow text.

    Returns:
        One record per `uses:` line with the action, the revision and the
        line number; a workflow of this repository (`./`) is skipped, and a
        line that cannot be parsed is recorded with revision `unparseable`.
    """
    references: list[dict[str, str]] = []
    for line_number, line in enumerate(content.splitlines(), 1):
        if "uses:" not in line:
            continue
        match = ACTION_REFERENCE.match(line)
        if match is None and re.match(r"^\s*(?:-\s+)?uses:\s+\./", line):
            continue  # a workflow of this repository is pinned by the commit
        if match is None:
            references.append(
                {
                    "action": line.strip(),
                    "revision": "unparseable",
                    "line": str(line_number),
                }
            )
            continue
        references.append(
            {
                "action": match.group("action"),
                "revision": match.group("revision"),
                "line": str(line_number),
            }
        )
    return references


def workflow_failures(root: Path) -> list[str]:
    """Returns static workflow-control failures for a repository root.

    Checks that each expected workflow exists, every action is pinned to a
    40-character commit, `pull_request_target` and long-lived cloud
    credentials are absent, each workflow carries its required control
    markers, and the site configuration is present.

    Args:
        root: The repository root.

    Returns:
        One message per failure; empty when all controls hold.
    """
    workflow_root = root / ".github" / "workflows"
    failures: list[str] = []
    contents: dict[str, str] = {}
    for name in WORKFLOW_NAMES:
        path = workflow_root / name
        if not path.is_file():
            failures.append(f"missing workflow: {path}")
            continue
        try:
            contents[name] = read_text(path)
        except ValueError as error:
            failures.append(str(error))

    for name, content in contents.items():
        for reference in action_references(content):
            if not SHA256_REVISION.fullmatch(reference["revision"]):
                failures.append(
                    f"{name}:{reference['line']} action is not pinned to a "
                    f"40-character commit SHA: {reference['action']}"
                )
        if "pull_request_target:" in content:
            failures.append(f"{name} uses pull_request_target")

    for name, content in contents.items():
        for marker in LONG_LIVED_CREDENTIALS:
            if marker in content:
                failures.append(
                    f"{name} refers to a long-lived credential: {marker}"
                )

    required_markers = {
        "ci.yml": (
            "name: ci",
            "permissions:\n  contents: read",
            "scripts/check_pytest_skips.py",
            "scripts/verify_hosted_controls.py",
            "scripts/verify_release_artifacts.sh",
            "-W error::ResourceWarning",
        ),
        "site.yml": (
            "name: site",
            "configure-pages@",
            "npm audit --omit=dev --audit-level=high",
            "npm run check:a11y",
            "npm run check:links",
            "scripts/verify_hosted_controls.py",
        ),
        "pages.yml": (
            "name: Deploy documentation site",
            "configure-pages@",
            "upload-pages-artifact@",
            "deploy-pages@",
            "pages: write",
            "id-token: write",
            "environment:",
            "name: github-pages",
            "needs: deploy",
            "scripts/verify_hosted_controls.py",
            "curl --fail",
        ),
        "full-suite.yml": (
            "name: full-suite",
            "scripts/check_test_stability.py",
            "exact-SHA clean-clone rehearsal",
        ),
        "release.yml": (
            "name: release",
            'tags: ["v*"]',
            "scripts/verify_release_artifacts.sh",
            "scripts/verify_external_release.py",
            "twine check dist/*",
            "id-token: write",
            "contents: write",
        ),
    }
    for name, markers in required_markers.items():
        content = contents.get(name, "")
        for marker in markers:
            if marker not in content:
                failures.append(f"{name} is missing required control: {marker}")

    astro_config = root / "website" / "astro.config.mjs"
    if astro_config.is_file():
        astro_content = read_text(astro_config)
        for marker in (
            'site: "https://sachncs.github.io/scmle"',
            'base: "/scmle"',
        ):
            if marker not in astro_content:
                failures.append(
                    f"website/astro.config.mjs is missing: {marker}"
                )
    else:
        failures.append(f"missing site configuration: {astro_config}")
    return failures


def workflow_evidence(root: Path) -> dict[str, Any]:
    """Builds deterministic evidence for the static hosted-control check.

    Args:
        root: The repository root.

    Returns:
        The schema version and the static controls: workflow files, number of
        action references, `passed` and the failures.
    """
    workflow_root = root / ".github" / "workflows"
    references = 0
    for name in WORKFLOW_NAMES:
        path = workflow_root / name
        if path.is_file():
            references += len(action_references(read_text(path)))
    failures = workflow_failures(root)
    return {
        "schema_version": 1,
        "static_controls": {
            "workflow_files": list(WORKFLOW_NAMES),
            "action_references": references,
            "passed": not failures,
            "failures": failures,
        },
    }


def public_page_evidence(
    pages_url: str,
    routes: tuple[str, ...] = PAGES_ROUTES,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """GET every required public route and return digest-bound evidence.

    Args:
        pages_url: Absolute HTTPS base URL of the published site.
        routes: Route paths to request.
        timeout: Seconds to wait for each response.

    Returns:
        For each route that answered, its URL, status, SHA-256 and size;
        the failures (a request error, a status outside 200-299 or an empty
        body); and `passed`, true only without failures and with every route
        answered.

    Raises:
        ValueError: If the URL is not an absolute HTTPS URL.
    """
    parsed = urlparse(pages_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("pages URL must be an absolute HTTPS URL")
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    base = pages_url.rstrip("/")
    for route in routes:
        url = f"{base}{route}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "scmle-release-verifier/1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = response.status
                body = response.read()
        except (OSError, HTTPError) as error:
            failures.append(f"{route}: {error}")
            continue
        digest = hashlib.sha256(body).hexdigest()
        record = {
            "route": route,
            "url": url,
            "status": status,
            "sha256": digest,
            "bytes": len(body),
        }
        records.append(record)
        if status < 200 or status >= 300:
            failures.append(f"{route}: HTTP {status}")
        if not body:
            failures.append(f"{route}: empty response")
    return {
        "url": pages_url,
        "routes": records,
        "passed": not failures and len(records) == len(routes),
        "failures": failures,
    }


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parses the hosted-control verifier arguments.

    Args:
        arguments: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        The namespace with the repository root, the optional Pages URL, the
        timeout and the output path.
    """
    parser = argparse.ArgumentParser(
        description="Verify hosted CI and public GitHub Pages controls."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--pages-url")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Runs static checks and, optionally, the unauthenticated Pages probe.

    Prints the evidence and writes it to `--output` when given.

    Args:
        arguments: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        0 when every section passed; 1 when one did not; 2 when the
        verification could not run (an I/O or HTTP error or invalid input).
    """
    options = parse_arguments(arguments)
    try:
        evidence = workflow_evidence(options.root.resolve())
        if options.pages_url:
            evidence["public_pages"] = public_page_evidence(
                options.pages_url, timeout=options.timeout
            )
        evidence["passed"] = all(
            section.get("passed", False)
            for section in evidence.values()
            if isinstance(section, dict) and "passed" in section
        )
        serialized = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
        if options.output:
            options.output.parent.mkdir(parents=True, exist_ok=True)
            options.output.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
    except (OSError, ValueError, HTTPError) as error:
        print(f"hosted-control verification failed: {error}", file=sys.stderr)
        return 2
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
