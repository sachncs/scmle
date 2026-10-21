"""Tests for repeated test execution and quarantine policy evidence."""

import datetime
import json
from pathlib import Path

import pytest

from scripts import check_test_stability

TEST_ID = "tests/unit/test_example.py::test_example"


def run(attempt: int, status: check_test_stability.TestStatus):
    """Build a deterministic stability run."""
    return check_test_stability.StabilityRun(
        attempt=attempt,
        returncode=0,
        tests={TEST_ID: status},
    )


def entry(expires_at: str) -> check_test_stability.QuarantineEntry:
    """Build a valid quarantine entry."""
    return check_test_stability.QuarantineEntry(
        test_id=TEST_ID,
        owner="ml-platform",
        reason="upstream dependency timing",
        expires_at=expires_at,
    )


def test_clean_repeated_runs_pass():
    """Two complete green inventories produce a clean verdict."""
    report = check_test_stability.StabilityPolicy({}).evaluate(
        (
            run(1, check_test_stability.TestStatus.PASSED),
            run(2, check_test_stability.TestStatus.PASSED),
        )
    )

    assert report.status == "passed"
    assert report.passed


def test_unlisted_failure_and_skip_fail_closed():
    """Failures and skips cannot disappear without an explicit policy entry."""
    report = check_test_stability.StabilityPolicy({}).evaluate(
        (
            run(1, check_test_stability.TestStatus.FAILED),
            run(2, check_test_stability.TestStatus.SKIPPED),
        )
    )

    assert not report.passed
    assert report.status == "failed"
    assert report.unquarantined_failures == (
        "tests/unit/test_example.py::test_example",
    )


def test_nonzero_pytest_returncode_fails_even_with_green_cases():
    """Runner-level failures cannot be hidden by a green JUnit testcase."""
    first = check_test_stability.StabilityRun(
        attempt=1,
        returncode=2,
        tests={TEST_ID: check_test_stability.TestStatus.PASSED},
    )
    second = run(2, check_test_stability.TestStatus.PASSED)

    report = check_test_stability.StabilityPolicy({}).evaluate((first, second))

    assert not report.passed
    assert report.unquarantined_failures == ("attempt 1: pytest returned 2",)


def test_active_quarantine_is_visible_but_not_silent():
    """An active quarantine produces a distinct, reviewable status."""
    quarantine = entry("2999-12-31")
    policy = check_test_stability.StabilityPolicy(
        {quarantine.test_id: quarantine}
    )
    report = policy.evaluate(
        (
            run(1, check_test_stability.TestStatus.FAILED),
            run(2, check_test_stability.TestStatus.PASSED),
        )
    )

    assert report.passed
    assert report.status == "passed_with_quarantine"
    assert report.quarantined_failures == (quarantine.test_id,)


def test_expired_quarantine_fails_even_if_test_is_not_collected():
    """Stale exceptions cannot remain in policy indefinitely."""
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    quarantine = entry(yesterday.isoformat())
    policy = check_test_stability.StabilityPolicy(
        {quarantine.test_id: quarantine}
    )

    report = policy.evaluate(
        (
            run(1, check_test_stability.TestStatus.PASSED),
            run(2, check_test_stability.TestStatus.PASSED),
        )
    )

    assert not report.passed
    assert report.expired_quarantines == (quarantine.test_id,)


def test_policy_and_junit_parser_are_deterministic(tmp_path: Path):
    """Policy JSON and JUnit output round-trip through stable identifiers."""
    quarantine = entry("2999-12-31")
    policy_path = tmp_path / "quarantine.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tests": {quarantine.test_id: quarantine.to_dict()},
            }
        ),
        encoding="utf-8",
    )
    loaded = check_test_stability.load_policy(policy_path)
    junit_path = tmp_path / "results.xml"
    junit_path.write_text(
        '<testsuite><testcase classname="tests.unit.test_example" '
        'name="test_example" /></testsuite>',
        encoding="utf-8",
    )

    assert loaded.entries[quarantine.test_id].owner == "ml-platform"
    assert check_test_stability.parse_junit(junit_path) == {
        "tests.unit.test_example::test_example": check_test_stability.TestStatus.PASSED
    }


def test_quarantine_entry_requires_valid_expiry():
    """Malformed policy records are rejected at the policy boundary."""
    with pytest.raises(ValueError, match="ISO-8601"):
        entry("tomorrow")


def test_stability_attempts_enforce_strict_resource_warnings():
    """Repeated runs use the same warning gate as release suites."""
    assert check_test_stability.STRICT_PYTEST_ARGUMENTS == (
        "-W",
        "error::ResourceWarning",
    )
