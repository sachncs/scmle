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
    """Reads JSON or raises a `GateError`.

    Args:
        path: JSON file.

    Returns:
        The parsed JSON value.

    Raises:
        GateError: If the file cannot be read or parsed.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateError(f"cannot read {path}: {error}") from error


def parse_pip_audit(report: dict[str, Any]) -> tuple[list[dict], set[str], str]:
    """Parses a pip-audit report.

    Args:
        report: The parsed pip-audit JSON.

    Returns:
        The findings (with no severity), the scanned `name==version` pairs
        (dependencies that were not skipped) and the scanner label.

    Raises:
        GateError: If the report has no `dependencies` list.
    """
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
    """Parses a Trivy report.

    Args:
        report: The parsed Trivy JSON.

    Returns:
        The findings (with upper-cased severity), the scanned `name==version`
        pairs and the scanner label.

    Raises:
        GateError: If the report has no `Results` list.
    """
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
    """Detects the scanner format and parses it.

    Args:
        report: The parsed scanner JSON.

    Returns:
        The findings, the scanned `name==version` pairs and the scanner label.

    Raises:
        GateError: If the report is neither pip-audit nor Trivy format.
    """
    if isinstance(report, dict) and "dependencies" in report:
        return parse_pip_audit(report)
    if isinstance(report, dict) and "Results" in report:
        return parse_trivy(report)
    raise GateError("unrecognised scanner report (expected pip-audit or Trivy)")


def sbom_components(sbom: dict[str, Any]) -> set[str]:
    """Lists the libraries of an SBOM.

    Args:
        sbom: A CycloneDX document.

    Returns:
        The `name==version` pairs of its components of type `library`.
    """
    return {
        f"{item['name']}=={item['version']}"
        for item in sbom.get("components", [])
        if item.get("type") == "library"
    }


def active_entries(
    items: list[dict[str, Any]], key: str, today: datetime.date
) -> dict[str, str]:
    """Returns the unexpired entries of an exception list by `key`.

    Args:
        items: Exception entries, each with `key`, `reason` and `expires` (an
            ISO date).
        key: The field that identifies an entry.
        today: The date entries are compared with; an entry expiring today is
            still active.

    Returns:
        A mapping from each active entry's `key` value to its reason.

    Raises:
        GateError: If an entry lacks `key`, `reason` or `expires`, or its expiry
            is not an ISO date.
    """
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
    """Validates the policy.

    Args:
        policy: The policy, with `fail_at_or_above` and optional `exceptions`
            and `unscanned_exceptions`.
        today: The date exceptions are compared with.

    Returns:
        The active exceptions, keyed by finding id.

    Raises:
        GateError: If the threshold is not one of `SEVERITIES`. A malformed
            exception entry is refused with the same error.
    """
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
    """Evaluates one report against the policy.

    A finding blocks if its id or an alias has no active exception and its
    severity is unknown (pip-audit gives none) or at or above the threshold.
    With an SBOM, a library that the scan did not cover blocks unless waived.

    Args:
        report: The parsed scanner report.
        policy: The policy.
        sbom: A CycloneDX document whose libraries must be covered by the scan;
            coverage is not checked if None.
        today: Date for exception expiry (default: today, UTC).

    Returns:
        The verdict: `passed`, `scanner`, `packages_scanned`, the `blocking` and
        `excepted` findings, `sbom_components_not_scanned` and the `threshold`.

    Raises:
        GateError: If the scan looked at no packages, or an SBOM was given but
            the report lists no packages. An unreadable policy or report is
            refused with the same error.
    """
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
    """Runs the gate and prints the verdict as JSON.

    Args:
        arguments: Command-line arguments; None reads `sys.argv`.

    Returns:
        0 if the gate passes, 1 if it finds something blocking, 2 if the report
        or policy cannot be trusted.
    """
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
