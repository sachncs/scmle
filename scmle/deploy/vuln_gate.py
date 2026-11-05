r"""Fail a build when a vulnerability scan finds something it may not ship.

The gate reads a scanner report (pip-audit JSON or Trivy JSON), applies the
policy file and exits 0 only when nothing blocking remains. It fails closed:
an unreadable report, a report from a scan that looked at nothing, a finding
without a severity (pip-audit gives none) and an expired exception all fail.

    pip-audit --format json --requirement requirements.lock -o scan.json
    python -m scmle.deploy.vuln_gate scan.json \
        --policy deploy/vuln-policy.json --sbom sbom.cdx.json

The gate does not scan. Which scanner ran, and when, is whatever the report
says; the gate prints it. A passing gate says the report is clean under the
policy, not that the software has no vulnerabilities.
"""

import argparse
import datetime
import json
from pathlib import Path
import sys
from typing import Any

SEVERITIES = ("UNKNOWN", "LOW", "MEDIUM", "HIGH", "CRITICAL")
DEFAULT_POLICY: dict[str, Any] = {
    "fail_at_or_above": "HIGH",
    "exceptions": [],
    "unscanned_exceptions": [
        {
            "name": "scmle",
            "reason": "the framework is installed from the project's own "
            "build, not from an index a scanner can query; its dependencies "
            "are scanned",
            "expires": "2027-06-30",
        }
    ],
}


class GateError(Exception):
    """The report or the policy cannot be trusted."""


def load_json(path: Path) -> Any:
    """Read JSON or raise a `GateError`."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateError(f"cannot read {path}: {error}") from error


def parse_pip_audit(report: dict[str, Any]) -> tuple[list[dict], set[str], str]:
    """Return findings, scanned `name==version` pairs and the scanner label."""
    dependencies = report.get("dependencies")
    if not isinstance(dependencies, list):
        raise GateError("pip-audit report has no 'dependencies' list")
    findings = []
    scanned = set()
    for dependency in dependencies:
        name = str(dependency.get("name", "")).lower().replace("_", "-")
        version = str(dependency.get("version", ""))
        if "skip_reason" not in dependency:
            scanned.add(f"{name}=={version}")
        for vulnerability in dependency.get("vulns", []):
            findings.append(
                {
                    "id": vulnerability.get("id", "unknown"),
                    "aliases": vulnerability.get("aliases", []),
                    "package": f"{name}=={version}",
                    "severity": None,
                    "fixed": vulnerability.get("fix_versions", []),
                }
            )
    return findings, scanned, "pip-audit"


def parse_trivy(report: dict[str, Any]) -> tuple[list[dict], set[str], str]:
    """Return findings, scanned packages and the scanner label."""
    results = report.get("Results")
    if not isinstance(results, list):
        raise GateError("Trivy report has no 'Results' list")
    findings = []
    scanned = set()
    for result in results:
        for package in result.get("Packages", []) or []:
            scanned.add(
                f"{str(package.get('Name', '')).lower()}=="
                f"{package.get('Version', '')}"
            )
        for item in result.get("Vulnerabilities", []) or []:
            findings.append(
                {
                    "id": item.get("VulnerabilityID", "unknown"),
                    "aliases": [],
                    "package": f"{item.get('PkgName', '')}=="
                    f"{item.get('InstalledVersion', '')}",
                    "severity": str(item.get("Severity", "UNKNOWN")).upper(),
                    "fixed": [item["FixedVersion"]]
                    if item.get("FixedVersion")
                    else [],
                }
            )
    return findings, scanned, "trivy"


def parse_report(report: Any) -> tuple[list[dict], set[str], str]:
    """Detect the scanner format and parse it."""
    if isinstance(report, dict) and "dependencies" in report:
        return parse_pip_audit(report)
    if isinstance(report, dict) and "Results" in report:
        return parse_trivy(report)
    raise GateError("unrecognised scanner report (expected pip-audit or Trivy)")


def sbom_components(sbom: dict[str, Any]) -> set[str]:
    """Return the `name==version` pairs an SBOM lists as libraries."""
    return {
        f"{item['name']}=={item['version']}"
        for item in sbom.get("components", [])
        if item.get("type") == "library"
    }


def active_entries(
    items: list[dict[str, Any]], key: str, today: datetime.date
) -> dict[str, str]:
    """Return the unexpired entries of an exception list by `key`."""
    active = {}
    for item in items:
        for field in (key, "reason", "expires"):
            if not str(item.get(field, "")).strip():
                raise GateError(f"exception lacks {field!r}: {item}")
        try:
            expires = datetime.date.fromisoformat(item["expires"])
        except ValueError as error:
            raise GateError(f"bad expiry in {item[key]}") from error
        if expires >= today:
            active[item[key]] = item["reason"]
    return active


def check_policy(
    policy: dict[str, Any], today: datetime.date
) -> dict[str, str]:
    """Validate the policy; return active exceptions by finding id."""
    threshold = policy.get("fail_at_or_above")
    if threshold not in SEVERITIES:
        raise GateError(f"fail_at_or_above must be one of {SEVERITIES}")
    active_entries(policy.get("unscanned_exceptions", []), "name", today)
    return active_entries(policy.get("exceptions", []), "id", today)


def evaluate(
    report: Any,
    policy: dict[str, Any],
    sbom: dict[str, Any] | None = None,
    today: datetime.date | None = None,
) -> dict[str, Any]:
    """Return the verdict for one report."""
    today = today or datetime.datetime.now(datetime.UTC).date()
    active = check_policy(policy, today)
    findings, scanned, scanner = parse_report(report)
    if not scanned and not findings:
        raise GateError(
            "the scan looked at no packages: an empty scan is not a pass"
        )
    threshold = SEVERITIES.index(policy["fail_at_or_above"])
    blocking, excepted = [], []
    for finding in findings:
        identifiers = {finding["id"], *finding["aliases"]}
        if identifiers & set(active):
            excepted.append(finding)
            continue
        severity = finding["severity"]
        rank = SEVERITIES.index(severity) if severity in SEVERITIES else 0
        # A finding with no severity cannot be judged below the bar.
        if severity is None or rank >= threshold:
            blocking.append(finding)
    uncovered: list[str] = []
    if sbom is not None:
        if not scanned:
            raise GateError(
                "cannot check SBOM coverage: the report lists no packages"
            )
        waived = set(
            active_entries(
                policy.get("unscanned_exceptions", []), "name", today
            )
        )
        uncovered = sorted(
            item
            for item in sbom_components(sbom) - scanned
            if item.split("==")[0] not in waived
        )
    passed = not blocking and not uncovered
    return {
        "passed": passed,
        "scanner": scanner,
        "packages_scanned": len(scanned),
        "blocking": blocking,
        "excepted": excepted,
        "sbom_components_not_scanned": uncovered,
        "threshold": policy["fail_at_or_above"],
    }


def main(arguments: list[str] | None = None) -> int:
    """Run the gate; print the verdict as JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("report", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--sbom", type=Path)
    options = parser.parse_args(arguments)
    try:
        policy = load_json(options.policy) if options.policy else DEFAULT_POLICY
        sbom = load_json(options.sbom) if options.sbom else None
        verdict = evaluate(load_json(options.report), policy, sbom)
    except GateError as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
