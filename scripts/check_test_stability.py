"""Run pytest repeatedly and enforce an explicit flake quarantine policy.

The policy is intentionally small and evidence-first. A test may be
quarantined only with an owner, a reason, and an expiry date. Unlisted
failures, errors, skips, changed test inventories, and expired quarantines
fail the command. A quarantined failure is visible in the output as
``passed_with_quarantine`` rather than being presented as a clean pass.

Usage:
    python scripts/check_test_stability.py --runs 2 --output stability.json \
        tests/unit/test_api_surface.py
"""

import argparse
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import datetime
import enum
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
from typing import Self
import xml.etree.ElementTree as ElementTree


class TestStatus(enum.StrEnum):
    """Outcome reported for one collected test case."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


STRICT_PYTEST_ARGUMENTS = ("-W", "error::ResourceWarning")
"""Pytest options shared by every repeated stability attempt."""


def parse_date(value: str) -> datetime.date:
    """Parse an ISO-8601 quarantine expiry date."""
    try:
        return datetime.date.fromisoformat(value)
    except ValueError as error:
        raise ValueError("expiry must be an ISO-8601 date") from error


@dataclasses.dataclass(frozen=True, slots=True)
class QuarantineEntry:
    """Owned, time-bounded explanation for one known unstable test."""

    test_id: str
    owner: str
    reason: str
    expires_at: str

    def __post_init__(self) -> None:
        """Reject unreviewable quarantine records."""
        for name in ("test_id", "owner", "reason", "expires_at"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        parse_date(self.expires_at)

    @property
    def expired(self) -> bool:
        """Whether the quarantine is no longer valid today."""
        return parse_date(self.expires_at) < datetime.date.today()

    def to_dict(self) -> dict[str, str]:
        """Return JSON-native quarantine evidence."""
        return {
            "owner": self.owner,
            "reason": self.reason,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, test_id: str, record: Mapping[str, Any]) -> Self:
        """Build a quarantine entry from policy JSON."""
        return cls(
            test_id=test_id,
            owner=str(record["owner"]),
            reason=str(record["reason"]),
            expires_at=str(record["expires_at"]),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class StabilityRun:
    """Results from one pytest attempt."""

    attempt: int
    returncode: int
    tests: dict[str, TestStatus]

    def __post_init__(self) -> None:
        """Validate attempt identity and normalize status values."""
        if self.attempt < 1:
            raise ValueError("attempt must be positive")
        normalized = {
            str(test_id): (
                status if isinstance(status, TestStatus) else TestStatus(status)
            )
            for test_id, status in self.tests.items()
        }
        if not normalized:
            raise ValueError("stability run must contain test results")
        object.__setattr__(self, "tests", normalized)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-native run evidence."""
        return {
            "attempt": self.attempt,
            "returncode": self.returncode,
            "tests": {
                test_id: status.value
                for test_id, status in sorted(self.tests.items())
            },
        }


@dataclasses.dataclass(frozen=True, slots=True)
class StabilityReport:
    """Fail-closed stability decision with all observed test outcomes."""

    status: str
    runs: tuple[StabilityRun, ...]
    unquarantined_failures: tuple[str, ...]
    quarantined_failures: tuple[str, ...]
    expired_quarantines: tuple[str, ...]
    inventory_changes: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Whether the policy permits a clean stability result."""
        return not self.unquarantined_failures and not self.expired_quarantines

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-native stability evidence."""
        return {
            "schema_version": 1,
            "status": self.status,
            "passed": self.passed,
            "runs": [run.to_dict() for run in self.runs],
            "unquarantined_failures": list(self.unquarantined_failures),
            "quarantined_failures": list(self.quarantined_failures),
            "expired_quarantines": list(self.expired_quarantines),
            "inventory_changes": list(self.inventory_changes),
        }


class StabilityPolicy:
    """Evaluate repeated test results against quarantine entries."""

    def __init__(self, entries: Mapping[str, QuarantineEntry]) -> None:
        """Create a policy from a normalized test-id mapping."""
        self.entries = dict(entries)

    def evaluate(self, runs: Sequence[StabilityRun]) -> StabilityReport:
        """Return a fail-closed decision for repeated test results."""
        normalized_runs = tuple(runs)
        if len(normalized_runs) < 2:
            raise ValueError("stability policy requires at least two runs")
        inventory = set(normalized_runs[0].tests)
        inventory_changes = sorted(
            {
                f"attempt {run.attempt}: test inventory differs"
                for run in normalized_runs[1:]
                if set(run.tests) != inventory
            }
        )
        failures: set[str] = set()
        quarantined_failures: set[str] = set()
        for run in normalized_runs:
            if run.returncode != 0:
                failures.add(
                    f"attempt {run.attempt}: pytest returned {run.returncode}"
                )
            for test_id, outcome in run.tests.items():
                if outcome in {
                    TestStatus.FAILED,
                    TestStatus.ERROR,
                    TestStatus.SKIPPED,
                }:
                    if (
                        test_id in self.entries
                        and not self.entries[test_id].expired
                    ):
                        quarantined_failures.add(test_id)
                    else:
                        failures.add(test_id)
        expired = sorted(
            test_id for test_id, entry in self.entries.items() if entry.expired
        )
        if inventory_changes:
            failures.update(inventory_changes)
        verdict = (
            "failed"
            if failures or expired
            else "passed_with_quarantine"
            if quarantined_failures
            else "passed"
        )
        return StabilityReport(
            status=verdict,
            runs=normalized_runs,
            unquarantined_failures=tuple(sorted(failures)),
            quarantined_failures=tuple(sorted(quarantined_failures)),
            expired_quarantines=tuple(expired),
            inventory_changes=tuple(inventory_changes),
        )


def load_policy(path: Path) -> StabilityPolicy:
    """Load and validate a quarantine policy file."""
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != 1:
        raise ValueError("quarantine policy schema_version must be 1")
    tests = record.get("tests")
    if not isinstance(tests, dict):
        raise ValueError("quarantine policy tests must be an object")
    entries = {
        str(test_id): QuarantineEntry.from_dict(str(test_id), value)
        for test_id, value in tests.items()
    }
    return StabilityPolicy(entries)


def parse_junit(path: Path) -> dict[str, TestStatus]:
    """Parse pytest JUnit XML into stable test identifiers and statuses."""
    root = ElementTree.parse(path).getroot()
    results: dict[str, TestStatus] = {}
    for case in root.iter("testcase"):
        classname = case.attrib.get("classname", "")
        name = case.attrib.get("name", "")
        test_id = f"{classname}::{name}" if classname else name
        if case.find("error") is not None:
            status = TestStatus.ERROR
        elif case.find("failure") is not None:
            status = TestStatus.FAILED
        elif case.find("skipped") is not None:
            status = TestStatus.SKIPPED
        else:
            status = TestStatus.PASSED
        if not test_id.strip():
            raise ValueError("JUnit testcase has no stable identifier")
        results[test_id] = status
    if not results:
        raise ValueError("JUnit report contains no testcases")
    return results


def run_pytest(
    arguments: Sequence[str], attempt: int, directory: Path
) -> StabilityRun:
    """Run pytest once and parse its durable JUnit result."""
    report_path = directory / f"attempt-{attempt}.xml"
    command = [
        sys.executable,
        "-m",
        "pytest",
        *STRICT_PYTEST_ARGUMENTS,
        *arguments,
        "--junitxml",
        str(report_path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if not report_path.exists():
        raise RuntimeError(
            f"pytest attempt {attempt} produced no JUnit report: "
            f"{completed.stderr[-500:]}"
        )
    return StabilityRun(
        attempt=attempt,
        returncode=completed.returncode,
        tests=parse_junit(report_path),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the stability policy command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--quarantine", default="tests/quarantine.json")
    parser.add_argument("--output", default="test-stability.json")
    parser.add_argument("tests", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(argv)
    if arguments.runs < 2:
        parser.error("--runs must be at least 2")
    test_arguments = tuple(arguments.tests) or ("tests/unit",)
    if test_arguments and test_arguments[0] == "--":
        test_arguments = test_arguments[1:]
    policy = load_policy(Path(arguments.quarantine))
    with tempfile.TemporaryDirectory(prefix="scmle-test-stability-") as root:
        runs = tuple(
            run_pytest(test_arguments, attempt, Path(root))
            for attempt in range(1, arguments.runs + 1)
        )
    report = policy.evaluate(runs)
    Path(arguments.output).write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
