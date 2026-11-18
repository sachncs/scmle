"""The outcome of a release evaluation and its readiness summary."""

from collections.abc import Mapping
import dataclasses
import json
import types
from typing import Any
from typing import ClassVar

from scmle.contracts import errors
from scmle.contracts import records
from scmle.evaluation.release.finding import exhaustive
from scmle.evaluation.release.finding import Finding
from scmle.evaluation.release.finding import Kind
from scmle.evaluation.release.finding import Readiness
from scmle.evaluation.release.finding import tallies
from scmle.evaluation.release.finding import Tally


def recount(step: records.Migration) -> records.Migration:
    """Builds the step that moves every kind's counts in a by-kind mapping.

    Args:
        step: The counts' own step.

    Returns:
        The step over a mapping of kind to counts.
    """
    return records.Migration(
        change=lambda layout: {
            kind: step.upgrade(dict(counts)) for kind, counts in layout.items()
        },
        revert=lambda layout: {
            kind: step.downgrade(dict(counts))
            for kind, counts in layout.items()
        },
    )


KINDS = recount(Tally.migrations[0])


@dataclasses.dataclass(frozen=True, slots=True)
class Scorecard:
    """Machine-readable readiness summary derived from release findings.

    The summary intentionally contains counts rather than duplicating the
    findings. Callers can use it for release dashboards and automation, then
    retain ``Verdict`` as the detailed evidence-backed record.

    Attributes:
        task: Task type the policy applies to.
        policy_schema_version: Schema version of the evaluated policy.
        status: Blocked when any blocking finding failed, otherwise ready.
        total_findings: Number of findings.
        passed_findings: Number of passed findings.
        failed_findings: Number of failed findings.
        blocking_findings: Number of findings that are blocking gates.
        blocking_failures: Number of failed blocking findings.
        non_blocking_failures: Number of failed non-blocking findings.
        counts_by_kind: Counts for every finding kind, keyed by the kind's
            value; stored as a read-only mapping.
        migrations: The written layout's version steps, oldest first;
            version 2 writes `policy`, `findings.{total, passed, failed,
            blocking}`, `blockers`, `warnings` and `kinds` (each kind's
            counts at its own version 2).
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            {
                "policy_schema_version": "policy",
                "blocking_failures": "blockers",
                "non_blocking_failures": "warnings",
                "counts_by_kind": "kinds",
            },
            parts={"kinds": KINDS},
            groups={
                "findings": {
                    "total_findings": "total",
                    "passed_findings": "passed",
                    "failed_findings": "failed",
                    "blocking_findings": "blocking",
                }
            },
        ),
    )

    task: str
    policy_schema_version: int
    status: Readiness
    total_findings: int
    passed_findings: int
    failed_findings: int
    blocking_findings: int
    blocking_failures: int
    non_blocking_failures: int
    counts_by_kind: Mapping[str, Tally]

    def __post_init__(self) -> None:
        """Validates summary counts, status, and kind coverage.

        Raises:
            ValueError: If `task` is not a string or is blank,
                `policy_schema_version` is not a non-negative integer, a
                count is negative, the counts are
                inconsistent with one another, `status` does not match
                `blocking_failures`, `counts_by_kind` does not cover every
                finding kind, or its totals differ from the aggregate counts.
            TypeError: If `status` is not a `Readiness`, a count is not
                an integer, or `counts_by_kind` is not a mapping of strings
                to `Tally`.
        """
        task: object = self.task  # may be a non-string at runtime
        if not isinstance(task, str) or not task.strip():
            raise errors.InputError("task must not be empty")
        schema_version: object = self.policy_schema_version  # may be non-int
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version < 0
        ):
            raise errors.InputError(
                "policy_schema_version must be non-negative"
            )
        if not isinstance(self.status, Readiness):
            raise errors.KindError("status must be Readiness")
        values = {
            "total_findings": self.total_findings,
            "passed_findings": self.passed_findings,
            "failed_findings": self.failed_findings,
            "blocking_findings": self.blocking_findings,
            "blocking_failures": self.blocking_failures,
            "non_blocking_failures": self.non_blocking_failures,
        }
        tallies(values)
        if self.total_findings != self.passed_findings + self.failed_findings:
            raise errors.InputError(
                "total_findings must equal passed plus failed"
            )
        if self.failed_findings != (
            self.blocking_failures + self.non_blocking_failures
        ):
            raise errors.InputError(
                "failed_findings must equal blocking plus non-blocking failures"
            )
        if self.blocking_failures > self.blocking_findings:
            raise errors.InputError(
                "blocking_failures must not exceed blocking_findings"
            )
        expected_status = (
            Readiness.BLOCKED if self.blocking_failures else Readiness.READY
        )
        if self.status is not expected_status:
            raise errors.InputError("status does not match blocking_failures")
        exhaustive(self.counts_by_kind)
        object.__setattr__(
            self,
            "counts_by_kind",
            types.MappingProxyType(dict(self.counts_by_kind)),
        )
        combined = Tally(
            total=self.total_findings,
            passed=self.passed_findings,
            failed=self.failed_findings,
            blocking=self.blocking_findings,
            blocking_failures=self.blocking_failures,
        )
        by_kind = list(self.counts_by_kind.values())
        if combined != Tally(
            total=sum(item.total for item in by_kind),
            passed=sum(item.passed for item in by_kind),
            failed=sum(item.failed for item in by_kind),
            blocking=sum(item.blocking for item in by_kind),
            blocking_failures=sum(item.blocking_failures for item in by_kind),
        ):
            raise errors.InputError(
                "aggregate counts do not match counts_by_kind"
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns deterministic JSON-compatible readiness data.

        Returns:
            A mapping of the summary's fields, with the status as its value
            and the per-kind counts as mappings in sorted kind order, in
            the current layout.
        """
        legacy = {
            "task": self.task,
            "policy_schema_version": self.policy_schema_version,
            "status": self.status.value,
            "total_findings": self.total_findings,
            "passed_findings": self.passed_findings,
            "failed_findings": self.failed_findings,
            "blocking_findings": self.blocking_findings,
            "blocking_failures": self.blocking_failures,
            "non_blocking_failures": self.non_blocking_failures,
            "counts_by_kind": {
                kind: records.downgrade(
                    Tally.migrations, self.counts_by_kind[kind].to_dict()
                )
                for kind in sorted(self.counts_by_kind)
            },
        }
        return records.upgrade(self.migrations, legacy, records.ORIGIN)

    def canonize(self) -> str:
        """Returns stable JSON suitable for release automation.

        Returns:
            The `to_dict` record as compact JSON with sorted keys and ASCII
            only.
        """
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Verdict:
    """Immutable outcome of evaluating one policy and its supplied evidence.

    Attributes:
        task: Task type the policy applies to.
        policy_schema_version: Schema version of the evaluated policy.
        passed: Whether no blocking finding failed.
        findings: Every finding, in the order produced.
        blocking_reasons: The reasons of the failed blocking findings, in
            the order of `findings`.
        migrations: The written layout's version steps, oldest first;
            version 2 writes `policy_schema_version` as `policy`,
            `blocking_reasons` as `blockers` and the readiness summary at
            its version 2.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            {
                "policy_schema_version": "policy",
                "blocking_reasons": "blockers",
            },
            parts={"readiness": Scorecard.migrations[0]},
        ),
    )

    task: str
    policy_schema_version: int
    passed: bool
    findings: tuple[Finding, ...]
    blocking_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        """Ensures the aggregate outcome agrees with its findings.

        Raises:
            ValueError: If `task` is not a string or is blank,
                `policy_schema_version` is negative, `blocking_reasons`
                differs from the reasons of the failed blocking findings,
                or `passed` disagrees with them.
            TypeError: If `passed` is not a bool, `findings` is not a tuple
                of `Finding`, or `blocking_reasons` is not a tuple.
        """
        task: object = self.task  # may be a non-string at runtime
        if not isinstance(task, str) or not task.strip():
            raise errors.InputError("task must not be empty")
        if self.policy_schema_version < 0:
            raise errors.InputError(
                "policy_schema_version must not be negative"
            )
        if not isinstance(self.passed, bool):
            raise errors.KindError("passed must be bool")
        if not isinstance(self.findings, tuple):
            raise errors.KindError("findings must be a tuple")
        if not all(isinstance(item, Finding) for item in self.findings):
            raise errors.KindError("findings must contain Finding values")
        if not isinstance(self.blocking_reasons, tuple):
            raise errors.KindError("blocking_reasons must be a tuple")
        expected_reasons = tuple(
            item.reason
            for item in self.findings
            if not item.passed and item.blocking
        )
        if self.blocking_reasons != expected_reasons:
            raise errors.InputError("blocking_reasons do not match findings")
        if self.passed != (not expected_reasons):
            raise errors.InputError("passed does not match blocking findings")

    @property
    def blocked(self) -> bool:
        """Whether at least one failed blocking finding prevents release."""
        return not self.passed

    def to_dict(self) -> dict[str, Any]:
        """Returns a deterministic JSON-compatible release outcome.

        Returns:
            A mapping with the task, schema version, pass and blocked
            flags, blocking reasons, findings as mappings, and the
            readiness summary as a mapping.
        """
        readiness = self.tally()
        legacy = {
            "task": self.task,
            "policy_schema_version": self.policy_schema_version,
            "passed": self.passed,
            "blocked": self.blocked,
            "blocking_reasons": list(self.blocking_reasons),
            "findings": [item.to_dict() for item in self.findings],
            "readiness": records.downgrade(
                readiness.migrations, readiness.to_dict()
            ),
        }
        return records.upgrade(self.migrations, legacy, records.ORIGIN)

    def canonize(self) -> str:
        """Returns stable JSON suitable for release evidence storage.

        Returns:
            The `to_dict` record as compact JSON with sorted keys and ASCII
            only.
        """
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def tally(self) -> Scorecard:
        """Returns deterministic readiness counts derived from findings.

        Returns:
            The summary, with overall counts and counts for every finding
            kind.
        """
        counts: dict[str, Tally] = {}
        for kind in Kind:
            matching = tuple(
                item for item in self.findings if item.kind is kind
            )
            counts[kind.value] = Tally(
                total=len(matching),
                passed=sum(item.passed for item in matching),
                failed=sum(not item.passed for item in matching),
                blocking=sum(item.blocking for item in matching),
                blocking_failures=sum(
                    not item.passed and item.blocking for item in matching
                ),
            )
        passed = sum(item.passed for item in self.findings)
        failed = len(self.findings) - passed
        blocking = sum(item.blocking for item in self.findings)
        blocking_failures = sum(
            not item.passed and item.blocking for item in self.findings
        )
        return Scorecard(
            task=self.task,
            policy_schema_version=self.policy_schema_version,
            status=(
                Readiness.BLOCKED if blocking_failures else Readiness.READY
            ),
            total_findings=len(self.findings),
            passed_findings=passed,
            failed_findings=failed,
            blocking_findings=blocking,
            blocking_failures=blocking_failures,
            non_blocking_failures=failed - blocking_failures,
            counts_by_kind=counts,
        )
