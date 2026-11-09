"""Evidence-backed release evaluation for declared task policies.

This module evaluates release evidence; it does not calculate metrics or
diagnostics. A caller supplies the outputs produced by those algorithms and
the evaluator applies the thresholds and evidence references declared by an
``Criteria``. Unknown inputs and incomplete evidence fail closed.
"""

from collections.abc import Mapping
import dataclasses
import enum
import json
import math
from numbers import Real
import types
from typing import Any

from scmle.contracts import evaluation
from scmle.contracts import evidence


class FindingKind(enum.StrEnum):
    """Category of one release finding."""

    DIAGNOSTIC = "diagnostic"
    EVIDENCE = "evidence"
    EXPLANATION = "explanation"
    FAIRNESS = "fairness"
    INPUT = "input"
    METRIC = "metric"
    RELIABILITY = "reliability"
    SECURITY = "security"


class ReleaseStatus(enum.StrEnum):
    """Machine-readable release readiness status."""

    READY = "ready"
    BLOCKED = "blocked"


def finite_real(value: Any) -> float | None:
    """Reads a finite real number.

    Args:
        value: Any value.

    Returns:
        `value` as a float when it is a real number that is neither a bool
        nor infinite nor NaN, otherwise None.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def normalize_measure(field_name: str, value: Any) -> float | str | None:
    """Validates an observed or expected value of a finding.

    Args:
        field_name: Name of the field, used in error messages.
        value: The candidate value.

    Returns:
        A real number as a float, text or None unchanged.

    Raises:
        TypeError: If `value` is a bool, or is neither a number, text nor
            None.
        ValueError: If `value` is a number that is not finite.
    """
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must not be bool")
    if isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{field_name} must be finite")
        return numeric
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{field_name} must be numeric, text, or None")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class ReleaseFinding:
    """One immutable pass or fail finding from release evaluation.

    Attributes:
        kind: Gate or input category that produced the finding.
        name: Stable metric, diagnostic, fairness metric, or evidence name.
        passed: Whether the supplied result satisfied this finding.
        blocking: Whether a failed finding blocks release. A passed finding
            may still have ``blocking=True`` because the declared gate is a
            blocking gate.
        reason: Human-readable, deterministic explanation of the outcome.
        observed: Value observed from the supplied report, when applicable.
        expected: Threshold or expected state, when applicable.
        evidence: Evidence-index references required for this finding.
    """

    kind: FindingKind
    name: str
    passed: bool
    blocking: bool
    reason: str
    observed: float | str | None = None
    expected: float | str | None = None
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validates the immutable finding shape.

        Numeric `observed` and `expected` values are converted to floats.

        Raises:
            TypeError: If `kind` is not a `FindingKind`, `passed` or
                `blocking` is not a bool, `evidence` is not a tuple,
                `observed` or `expected` is a bool, or is neither a number,
                text nor None.
            ValueError: If `name` or `reason` is not a string or is blank,
                an evidence item is not a string, is blank or is repeated,
                or a numeric `observed` or `expected` is not finite.
        """
        if not isinstance(self.kind, FindingKind):
            raise TypeError("kind must be FindingKind")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must not be empty")
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be bool")
        if not isinstance(self.blocking, bool):
            raise TypeError("blocking must be bool")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must not be empty")
        if not isinstance(self.evidence, tuple):
            raise TypeError("evidence must be a tuple of strings")
        if any(
            not isinstance(item, str) or not item.strip()
            for item in self.evidence
        ):
            raise ValueError("evidence must contain non-empty strings")
        if len(self.evidence) != len(set(self.evidence)):
            raise ValueError("evidence must not contain duplicates")
        for field_name in ("observed", "expected"):
            object.__setattr__(
                self,
                field_name,
                normalize_measure(field_name, getattr(self, field_name)),
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns a deterministic JSON-compatible finding.

        Returns:
            A mapping of the finding's fields. `observed` and `expected`
            appear only when they are not None.
        """
        record: dict[str, Any] = {
            "kind": self.kind.value,
            "name": self.name,
            "passed": self.passed,
            "blocking": self.blocking,
            "reason": self.reason,
            "evidence": list(self.evidence),
        }
        if self.observed is not None:
            record["observed"] = self.observed
        if self.expected is not None:
            record["expected"] = self.expected
        return record


@dataclasses.dataclass(frozen=True, slots=True)
class FindingCounts:
    """Validated aggregate counts for a set of release findings.

    Attributes:
        total: Number of findings.
        passed: Number of passed findings.
        failed: Number of failed findings.
        blocking: Number of findings that are blocking gates.
        blocking_failures: Number of failed blocking findings.
    """

    total: int
    passed: int
    failed: int
    blocking: int
    blocking_failures: int

    def __post_init__(self) -> None:
        """Validates count relationships before exposing the aggregate.

        Raises:
            TypeError: If a count is not an integer (bools are rejected).
            ValueError: If a count is negative, `total` differs from
                `passed` plus `failed`, `blocking` exceeds `total`, or
                `blocking_failures` exceeds `failed` or `blocking`.
        """
        values = {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "blocking": self.blocking,
            "blocking_failures": self.blocking_failures,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a non-negative integer")
            if value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.total != self.passed + self.failed:
            raise ValueError("total must equal passed plus failed")
        if self.blocking > self.total:
            raise ValueError("blocking must not exceed total")
        if self.blocking_failures > self.failed:
            raise ValueError("blocking_failures must not exceed failed")
        if self.blocking_failures > self.blocking:
            raise ValueError("blocking_failures must not exceed blocking")

    def to_dict(self) -> dict[str, int]:
        """Returns deterministic JSON-compatible aggregate counts.

        Returns:
            A mapping from each count's name to its value.
        """
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "blocking": self.blocking,
            "blocking_failures": self.blocking_failures,
        }


def check_count_values(values: Mapping[str, Any]) -> None:
    """Checks that every named count is a non-negative integer.

    Args:
        values: The counts, keyed by field name.

    Raises:
        TypeError: If a count is a bool or not an integer.
        ValueError: If a count is negative.
    """
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a non-negative integer")
        if value < 0:
            raise ValueError(f"{name} must be a non-negative integer")


def check_kind_coverage(counts_by_kind: Any) -> None:
    """Checks that the counts cover every finding kind and only those.

    Args:
        counts_by_kind: The candidate mapping from kind value to counts.

    Raises:
        TypeError: If it is not a mapping of strings to `FindingCounts`.
        ValueError: If its keys are not exactly the finding kinds.
    """
    if not isinstance(counts_by_kind, Mapping):
        raise TypeError("counts_by_kind must be a mapping")
    expected_kinds = {kind.value for kind in FindingKind}
    if set(counts_by_kind) != expected_kinds:
        raise ValueError("counts_by_kind must contain every finding kind")
    if not all(
        isinstance(kind, str) and isinstance(counts, FindingCounts)
        for kind, counts in counts_by_kind.items()
    ):
        raise TypeError("counts_by_kind must map strings to FindingCounts")


@dataclasses.dataclass(frozen=True, slots=True)
class ReleaseReadinessSummary:
    """Machine-readable readiness summary derived from release findings.

    The summary intentionally contains counts rather than duplicating the
    findings. Callers can use it for release dashboards and automation, then
    retain ``ReleaseEvaluation`` as the detailed evidence-backed record.

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
    """

    task: str
    policy_schema_version: int
    status: ReleaseStatus
    total_findings: int
    passed_findings: int
    failed_findings: int
    blocking_findings: int
    blocking_failures: int
    non_blocking_failures: int
    counts_by_kind: Mapping[str, FindingCounts]

    def __post_init__(self) -> None:
        """Validates summary counts, status, and kind coverage.

        Raises:
            ValueError: If `task` is not a string or is blank,
                `policy_schema_version` is not a non-negative integer, a
                count is negative, the counts are
                inconsistent with one another, `status` does not match
                `blocking_failures`, `counts_by_kind` does not cover every
                finding kind, or its totals differ from the aggregate counts.
            TypeError: If `status` is not a `ReleaseStatus`, a count is not
                an integer, or `counts_by_kind` is not a mapping of strings
                to `FindingCounts`.
        """
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("task must not be empty")
        if (
            isinstance(self.policy_schema_version, bool)
            or not isinstance(self.policy_schema_version, int)
            or self.policy_schema_version < 0
        ):
            raise ValueError("policy_schema_version must be non-negative")
        if not isinstance(self.status, ReleaseStatus):
            raise TypeError("status must be ReleaseStatus")
        values = {
            "total_findings": self.total_findings,
            "passed_findings": self.passed_findings,
            "failed_findings": self.failed_findings,
            "blocking_findings": self.blocking_findings,
            "blocking_failures": self.blocking_failures,
            "non_blocking_failures": self.non_blocking_failures,
        }
        check_count_values(values)
        if self.total_findings != self.passed_findings + self.failed_findings:
            raise ValueError("total_findings must equal passed plus failed")
        if self.failed_findings != (
            self.blocking_failures + self.non_blocking_failures
        ):
            raise ValueError(
                "failed_findings must equal blocking plus non-blocking failures"
            )
        if self.blocking_failures > self.blocking_findings:
            raise ValueError(
                "blocking_failures must not exceed blocking_findings"
            )
        expected_status = (
            ReleaseStatus.BLOCKED
            if self.blocking_failures
            else ReleaseStatus.READY
        )
        if self.status is not expected_status:
            raise ValueError("status does not match blocking_failures")
        check_kind_coverage(self.counts_by_kind)
        object.__setattr__(
            self,
            "counts_by_kind",
            types.MappingProxyType(dict(self.counts_by_kind)),
        )
        combined = FindingCounts(
            total=self.total_findings,
            passed=self.passed_findings,
            failed=self.failed_findings,
            blocking=self.blocking_findings,
            blocking_failures=self.blocking_failures,
        )
        by_kind = list(self.counts_by_kind.values())
        if combined != FindingCounts(
            total=sum(item.total for item in by_kind),
            passed=sum(item.passed for item in by_kind),
            failed=sum(item.failed for item in by_kind),
            blocking=sum(item.blocking for item in by_kind),
            blocking_failures=sum(item.blocking_failures for item in by_kind),
        ):
            raise ValueError("aggregate counts do not match counts_by_kind")

    def to_dict(self) -> dict[str, Any]:
        """Returns deterministic JSON-compatible readiness data.

        Returns:
            A mapping of the summary's fields, with the status as its value
            and the per-kind counts as mappings in sorted kind order.
        """
        return {
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
                kind: self.counts_by_kind[kind].to_dict()
                for kind in sorted(self.counts_by_kind)
            },
        }

    def canonical_json(self) -> str:
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
class ReleaseEvaluation:
    """Immutable outcome of evaluating one policy and its supplied evidence.

    Attributes:
        task: Task type the policy applies to.
        policy_schema_version: Schema version of the evaluated policy.
        passed: Whether no blocking finding failed.
        findings: Every finding, in the order produced.
        blocking_reasons: The reasons of the failed blocking findings, in
            the order of `findings`.
    """

    task: str
    policy_schema_version: int
    passed: bool
    findings: tuple[ReleaseFinding, ...]
    blocking_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        """Ensures the aggregate outcome agrees with its findings.

        Raises:
            ValueError: If `task` is not a string or is blank,
                `policy_schema_version` is negative, `blocking_reasons`
                differs from the reasons of the failed blocking findings,
                or `passed` disagrees with them.
            TypeError: If `passed` is not a bool, `findings` is not a tuple
                of `ReleaseFinding`, or `blocking_reasons` is not a tuple.
        """
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("task must not be empty")
        if self.policy_schema_version < 0:
            raise ValueError("policy_schema_version must not be negative")
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be bool")
        if not isinstance(self.findings, tuple):
            raise TypeError("findings must be a tuple")
        if not all(isinstance(item, ReleaseFinding) for item in self.findings):
            raise TypeError("findings must contain ReleaseFinding values")
        if not isinstance(self.blocking_reasons, tuple):
            raise TypeError("blocking_reasons must be a tuple")
        expected_reasons = tuple(
            item.reason
            for item in self.findings
            if not item.passed and item.blocking
        )
        if self.blocking_reasons != expected_reasons:
            raise ValueError("blocking_reasons do not match findings")
        if self.passed != (not expected_reasons):
            raise ValueError("passed does not match blocking findings")

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
        return {
            "task": self.task,
            "policy_schema_version": self.policy_schema_version,
            "passed": self.passed,
            "blocked": self.blocked,
            "blocking_reasons": list(self.blocking_reasons),
            "findings": [item.to_dict() for item in self.findings],
            "readiness": self.readiness_summary().to_dict(),
        }

    def canonical_json(self) -> str:
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

    def readiness_summary(self) -> ReleaseReadinessSummary:
        """Returns deterministic readiness counts derived from findings.

        Returns:
            The summary, with overall counts and counts for every finding
            kind.
        """
        counts: dict[str, FindingCounts] = {}
        for kind in FindingKind:
            matching = tuple(
                item for item in self.findings if item.kind is kind
            )
            counts[kind.value] = FindingCounts(
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
        return ReleaseReadinessSummary(
            task=self.task,
            policy_schema_version=self.policy_schema_version,
            status=(
                ReleaseStatus.BLOCKED
                if blocking_failures
                else ReleaseStatus.READY
            ),
            total_findings=len(self.findings),
            passed_findings=passed,
            failed_findings=failed,
            blocking_findings=blocking,
            blocking_failures=blocking_failures,
            non_blocking_failures=failed - blocking_failures,
            counts_by_kind=counts,
        )


def required_evidence_references(
    policy: evaluation.Criteria,
) -> tuple[str, ...]:
    """Returns every evidence reference declared by a policy in sorted order.

    References are collected from the release policy, metric declarations,
    diagnostic gates, fairness gates, and explanation policy. Duplicate
    references are evaluated once because one accepted evidence record can
    support several declarations.

    Args:
        policy: The evaluation policy to read the declarations from.

    Returns:
        The unique references, sorted.
    """
    references = set(policy.release.required_evidence)
    for metric in policy.metrics:
        references.update(metric.evidence)
    for diagnostic_gate in policy.release.diagnostic_gates:
        references.update(diagnostic_gate.required_evidence)
    for fairness_gate in policy.release.fairness_gates:
        references.update(fairness_gate.required_evidence)
    references.update(policy.explanations.required_evidence)
    return tuple(sorted(references))


def fairness_metric_values(report: Any, metric: str) -> tuple[float, ...]:
    """Finds finite values for one fairness metric in a nested report.

    Fairness reports may be keyed by sensitive column, or may be wrapped in a
    named report entry. The traversal accepts JSON-like mappings and lists and
    ignores unrelated scalar fields. A non-numeric metric value is not treated
    as a passing value, and a mapping or list stored under `metric` itself is
    not searched further.

    Args:
        report: A nested JSON-like report of mappings, lists and tuples.
        metric: Key under which the metric value is stored.

    Returns:
        The finite numeric values stored under `metric` anywhere in the
        report, sorted ascending. Boolean, non-numeric and non-finite
        values are skipped.
    """
    values: list[float] = []
    pending: list[Any] = [report]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if isinstance(current, Mapping):
            identity = id(current)
            if identity in visited:
                continue
            visited.add(identity)
            for key in sorted(current, key=str, reverse=True):
                value = current[key]
                if key == metric:
                    numeric = finite_real(value)
                    if numeric is not None:
                        values.append(numeric)
                elif isinstance(value, (Mapping, list, tuple)):
                    pending.append(value)
        elif isinstance(current, (list, tuple)):
            identity = id(current)
            if identity in visited:
                continue
            visited.add(identity)
            pending.extend(reversed(current))
    return tuple(sorted(values))


def explanation_quality_value(report: Any, field_name: str) -> float | None:
    """Returns one finite quality score from an explanation report.

    Args:
        report: An explanation report with a `quality` mapping.
        field_name: Key of the score inside `quality`.

    Returns:
        The score as a float, or None if the report, its `quality` entry or
        the score is not of the expected form, or the score is not a number
        in [0, 1].
    """
    if not isinstance(report, Mapping):
        return None
    quality = report.get("quality")
    if not isinstance(quality, Mapping):
        return None
    value = quality.get(field_name)
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def apply_reliability_gate(
    outcome: ReleaseEvaluation, reliability_plan: Any
) -> ReleaseEvaluation:
    """Blocks release until the persisted fault plan has verified evidence.

    A model may satisfy its statistical success criteria while still being
    unsafe to release. The fault-injection plan is therefore an independent
    release gate. Only an explicitly completed plan with
    ``execution_status=passed`` and ``release_blocked=false`` can pass this
    gate; malformed or incomplete evidence fails closed.

    Args:
        outcome: The evaluation to extend.
        reliability_plan: The persisted fault-injection plan, as a mapping.

    Returns:
        `outcome` unchanged when the gate passes; otherwise a blocked
        evaluation with an added failed blocking finding.

    Raises:
        TypeError: If `outcome` is not a `ReleaseEvaluation`.
    """
    if not isinstance(outcome, ReleaseEvaluation):
        raise TypeError("outcome must be ReleaseEvaluation")
    passed = (
        isinstance(reliability_plan, Mapping)
        and reliability_plan.get("execution_status") == "passed"
        and reliability_plan.get("release_blocked") is False
    )
    if passed:
        return outcome
    finding = ReleaseFinding(
        FindingKind.RELIABILITY,
        "fault_injection",
        passed=False,
        blocking=True,
        reason=(
            "fault-injection recovery evidence is not complete; release blocked"
        ),
        observed=(
            str(reliability_plan.get("execution_status"))
            if isinstance(reliability_plan, Mapping)
            else "missing"
        ),
        expected="execution_status=passed and release_blocked=false",
        evidence=("fault_injection_plan.json",),
    )
    findings = tuple(
        sorted(
            (*outcome.findings, finding),
            key=lambda item: (item.kind.value, item.name, item.reason),
        )
    )
    blocking_reasons = tuple(
        item.reason for item in findings if not item.passed and item.blocking
    )
    return ReleaseEvaluation(
        task=outcome.task,
        policy_schema_version=outcome.policy_schema_version,
        passed=False,
        findings=findings,
        blocking_reasons=blocking_reasons,
    )


def apply_attestation_gate(
    outcome: ReleaseEvaluation,
    attestation_record: Any,
    verified: bool | None = None,
) -> ReleaseEvaluation:
    """Blocks release until packaged evidence has a valid signed attestation.

    Args:
        outcome: The evaluation to extend.
        attestation_record: The attestation, as a mapping holding a
            `signature` with an `hmac-sha256` algorithm and a non-blank
            value.
        verified: Result of verifying the signature; None when it was not
            checked, which is treated as acceptable.

    Returns:
        `outcome` unchanged when the signature is present and not known to
        be invalid; otherwise a blocked evaluation with an added failed
        blocking finding.

    Raises:
        TypeError: If `outcome` is not a `ReleaseEvaluation`.
    """
    if not isinstance(outcome, ReleaseEvaluation):
        raise TypeError("outcome must be ReleaseEvaluation")
    signature = (
        attestation_record.get("signature")
        if isinstance(attestation_record, Mapping)
        else None
    )
    passed = (
        isinstance(signature, Mapping)
        and signature.get("algorithm") == "hmac-sha256"
        and isinstance(signature.get("value"), str)
        and bool(signature["value"].strip())
        and (verified is None or verified)
    )
    if passed:
        return outcome
    finding = ReleaseFinding(
        FindingKind.SECURITY,
        "release_attestation",
        passed=False,
        blocking=True,
        reason="release attestation is missing or invalid; release blocked",
        observed=(
            "invalid"
            if signature and verified is False
            else "signed"
            if signature
            else "unsigned"
        ),
        expected="verified hmac-sha256 attestation signature",
        evidence=("release_attestation.json",),
    )
    findings = tuple(
        sorted(
            (*outcome.findings, finding),
            key=lambda item: (item.kind.value, item.name, item.reason),
        )
    )
    blocking_reasons = tuple(
        item.reason for item in findings if not item.passed and item.blocking
    )
    return ReleaseEvaluation(
        task=outcome.task,
        policy_schema_version=outcome.policy_schema_version,
        passed=False,
        findings=findings,
        blocking_reasons=blocking_reasons,
    )


def apply_evidence_index_gate(
    outcome: ReleaseEvaluation,
    index: evidence.Index | None,
    verified: bool,
) -> ReleaseEvaluation:
    """Blocks release until the persisted evidence index is signed.

    Args:
        outcome: The evaluation to extend.
        index: The persisted evidence index, or None if missing.
        verified: Whether the index signature was verified.

    Returns:
        `outcome` unchanged when the index is signed and verified;
        otherwise a blocked evaluation with an added failed blocking
        finding.

    Raises:
        TypeError: If `outcome` is not a `ReleaseEvaluation`.
    """
    if not isinstance(outcome, ReleaseEvaluation):
        raise TypeError("outcome must be ReleaseEvaluation")
    passed = (
        isinstance(index, evidence.Index)
        and index.signature is not None
        and verified
    )
    if passed:
        return outcome
    finding = ReleaseFinding(
        FindingKind.SECURITY,
        "release_evidence",
        passed=False,
        blocking=True,
        reason=(
            "release evidence index is missing or has an invalid signature; "
            "release blocked"
        ),
        observed=(
            "invalid"
            if isinstance(index, evidence.Index)
            and index.signature is not None
            and not verified
            else "unsigned"
        ),
        expected="verified hmac-sha256 evidence index signature",
        evidence=("release_evidence.json",),
    )
    findings = tuple(
        sorted(
            (*outcome.findings, finding),
            key=lambda item: (item.kind.value, item.name, item.reason),
        )
    )
    blocking_reasons = tuple(
        item.reason for item in findings if not item.passed and item.blocking
    )
    return ReleaseEvaluation(
        task=outcome.task,
        policy_schema_version=outcome.policy_schema_version,
        passed=False,
        findings=findings,
        blocking_reasons=blocking_reasons,
    )


def blocked_input_finding(name: str, reason: str) -> ReleaseFinding:
    """Builds the blocking finding for an input the evaluator cannot read.

    Args:
        name: The input's name.
        reason: Why the input is unsupported.

    Returns:
        A failed, blocking input finding.
    """
    return ReleaseFinding(
        FindingKind.INPUT,
        name,
        passed=False,
        blocking=True,
        reason=reason,
    )


def metric_gate_finding(
    metric_gate: evaluation.MetricGate,
    declared_metrics: set[str],
    metric_values: Mapping[str, Any],
) -> ReleaseFinding:
    """Applies one metric gate to the supplied metric values.

    Args:
        metric_gate: The declared gate for one metric.
        declared_metrics: Names of the metrics the policy declares.
        metric_values: Computed metric values keyed by metric name.

    Returns:
        The finding for the gate: failed and blocking when the metric is
        undeclared, missing, not a finite number, or misses its threshold.
    """
    name = metric_gate.metric
    threshold = metric_gate.threshold
    if name not in declared_metrics:
        return ReleaseFinding(
            FindingKind.METRIC,
            name,
            passed=False,
            blocking=True,
            reason=f"metric '{name}' is unsupported or undeclared",
            expected=threshold,
        )
    if name not in metric_values:
        return ReleaseFinding(
            FindingKind.METRIC,
            name,
            passed=False,
            blocking=True,
            reason=f"metric '{name}' is missing",
            expected=threshold,
        )
    value = metric_values[name]
    if isinstance(value, bool) or not isinstance(value, Real):
        return ReleaseFinding(
            FindingKind.METRIC,
            name,
            passed=False,
            blocking=True,
            reason=f"metric '{name}' has an unsupported value",
            expected=threshold,
        )
    observed = float(value)
    if not math.isfinite(observed):
        return ReleaseFinding(
            FindingKind.METRIC,
            name,
            passed=False,
            blocking=True,
            reason=f"metric '{name}' is non-finite",
            observed=observed,
            expected=threshold,
        )
    if metric_gate.direction is evaluation.MetricDirection.MAXIMIZE:
        passed = observed >= threshold
        operator = ">="
    elif metric_gate.direction is evaluation.MetricDirection.MINIMIZE:
        passed = observed <= threshold
        operator = "<="
    else:
        passed = False
        operator = "?"
    result = "passed" if passed else "failed"
    return ReleaseFinding(
        FindingKind.METRIC,
        name,
        passed,
        blocking=True,
        reason=f"metric '{name}' {result}: {observed:g} "
        f"{operator} {threshold:g}",
        observed=observed,
        expected=threshold,
    )


def diagnostic_verdict(report: Mapping[str, Any]) -> Any:
    """Reads the pass verdict a diagnostic report carries.

    Args:
        report: A non-empty diagnostic report.

    Returns:
        The value of the first of `passed`, `pass`, `ok` and `failed` found
        (negated for `failed`); otherwise the `status` read as a pass when it
        is one of pass, passed, ok or success; None when there is no verdict.
    """
    verdict = None
    for field_name in ("passed", "pass", "ok", "failed"):
        if field_name in report:
            verdict = report[field_name]
            if field_name == "failed":
                verdict = not verdict
            break
    if "status" in report and verdict is None:
        verdict = report["status"]
        if isinstance(verdict, str):
            verdict = verdict.lower() in {
                "pass",
                "passed",
                "ok",
                "success",
            }
    return verdict


def diagnostic_gate_finding(
    diagnostic_gate: evaluation.DiagnosticGate,
    diagnostic_reports: Mapping[str, Any],
) -> ReleaseFinding:
    """Applies one diagnostic gate to the supplied diagnostic reports.

    Args:
        diagnostic_gate: The declared gate for one diagnostic.
        diagnostic_reports: Diagnostic reports keyed by diagnostic name.

    Returns:
        The finding for the gate. Blocking when the gate's action blocks,
        and always blocking when the report is unsupported or its verdict is
        not a bool.
    """
    name = diagnostic_gate.name
    report = diagnostic_reports.get(name)
    blocking = diagnostic_gate.action is evaluation.DiagnosticAction.BLOCK
    if report is None:
        return ReleaseFinding(
            FindingKind.DIAGNOSTIC,
            name,
            passed=False,
            blocking=blocking,
            reason=f"diagnostic '{name}' is missing",
            expected="report without error",
            evidence=diagnostic_gate.required_evidence,
        )
    if not isinstance(report, Mapping) or not report:
        return ReleaseFinding(
            FindingKind.DIAGNOSTIC,
            name,
            passed=False,
            blocking=True,
            reason=f"diagnostic '{name}' has an unsupported report",
            expected="non-empty mapping",
            evidence=diagnostic_gate.required_evidence,
        )
    failed = False
    reason = f"diagnostic '{name}' is present"
    if report.get("error") not in (None, "", False):
        failed = True
        reason = f"diagnostic '{name}' reported an error"
    elif report.get("available") is False:
        failed = True
        reason = f"diagnostic '{name}' is unavailable"
    else:
        verdict = diagnostic_verdict(report)
        if verdict is not None and not isinstance(verdict, bool):
            return ReleaseFinding(
                FindingKind.DIAGNOSTIC,
                name,
                passed=False,
                blocking=True,
                reason=f"diagnostic '{name}' has an unsupported verdict",
                expected="boolean pass verdict",
                evidence=diagnostic_gate.required_evidence,
            )
        if verdict is False:
            failed = True
            reason = f"diagnostic '{name}' reported failure"
    return ReleaseFinding(
        FindingKind.DIAGNOSTIC,
        name,
        not failed,
        blocking,
        reason,
        expected="report without error",
        evidence=diagnostic_gate.required_evidence,
    )


def explanation_findings(
    policy: evaluation.Criteria, diagnostic_reports: Mapping[str, Any]
) -> list[ReleaseFinding]:
    """Applies the required explanation quality thresholds.

    Args:
        policy: The declared evaluation policy.
        diagnostic_reports: Diagnostic reports keyed by diagnostic name.

    Returns:
        One finding per required method and required quality measure.

    Raises:
        ValueError: If a required quality measure has no threshold.
    """
    quality_requirements = (
        (
            "stability_score",
            policy.explanations.require_stability,
            policy.explanations.stability_threshold,
            "stability",
        ),
        (
            "fidelity_score",
            policy.explanations.require_fidelity,
            policy.explanations.fidelity_threshold,
            "fidelity",
        ),
    )
    findings: list[ReleaseFinding] = []
    for method in policy.explanations.required_methods:
        report = diagnostic_reports.get(method)
        for field_name, required, threshold, label in quality_requirements:
            if not required:
                continue
            name = f"{method}.{label}"
            if threshold is None:
                raise ValueError(
                    f"explanation '{method}' has no {label} threshold"
                )
            score = explanation_quality_value(report, field_name)
            if score is None:
                findings.append(
                    ReleaseFinding(
                        FindingKind.EXPLANATION,
                        name,
                        passed=False,
                        blocking=True,
                        reason=(
                            f"explanation '{method}' has no valid {label} score"
                        ),
                        expected=threshold,
                    )
                )
                continue
            passed = score >= threshold
            result = "passed" if passed else "failed"
            findings.append(
                ReleaseFinding(
                    FindingKind.EXPLANATION,
                    name,
                    passed,
                    blocking=True,
                    reason=f"explanation '{method}' {label} {result}: "
                    f"{score:g} >= {threshold:g}",
                    observed=score,
                    expected=threshold,
                )
            )
    return findings


def fairness_gate_finding(
    fairness_gate: evaluation.FairnessGate, fairness_reports: Mapping[str, Any]
) -> ReleaseFinding:
    """Applies one fairness gate to the supplied fairness reports.

    Args:
        fairness_gate: The declared gate for one fairness metric.
        fairness_reports: Fairness reports keyed by report name, or one
            direct fairness report.

    Returns:
        The finding for the gate: failed and blocking when the metric is
        missing or unsupported, the gate has no bound, or the worst value
        breaks the bound.
    """
    metric = fairness_gate.metric
    if metric in fairness_reports:
        reports = (fairness_reports,)
    else:
        reports = tuple(
            fairness_reports[name] for name in sorted(fairness_reports, key=str)
        )
    values = tuple(
        value
        for report in reports
        for value in fairness_metric_values(report, metric)
    )
    if not values:
        return ReleaseFinding(
            FindingKind.FAIRNESS,
            metric,
            passed=False,
            blocking=True,
            reason=f"fairness metric '{metric}' is missing or unsupported",
            expected=fairness_gate.maximum
            if fairness_gate.maximum is not None
            else fairness_gate.minimum,
            evidence=fairness_gate.required_evidence,
        )
    expected: float | None
    if fairness_gate.maximum is not None:
        observed = max(values)
        passed = observed <= fairness_gate.maximum
        expected = fairness_gate.maximum
        operator = "<="
    elif fairness_gate.minimum is not None:
        observed = min(values)
        passed = observed >= fairness_gate.minimum
        expected = fairness_gate.minimum
        operator = ">="
    else:
        return ReleaseFinding(
            FindingKind.FAIRNESS,
            metric,
            passed=False,
            blocking=True,
            reason=f"fairness metric '{metric}' has no bound",
            evidence=fairness_gate.required_evidence,
        )
    result = "passed" if passed else "failed"
    return ReleaseFinding(
        FindingKind.FAIRNESS,
        metric,
        passed,
        blocking=True,
        reason=(
            f"fairness metric '{metric}' {result}: "
            f"worst value {observed:g} {operator} {expected:g}"
        ),
        observed=observed,
        expected=expected,
        evidence=fairness_gate.required_evidence,
    )


def evidence_reference_finding(
    reference: str, evidence_index: evidence.Index | None
) -> ReleaseFinding:
    """Checks one evidence reference the policy requires.

    Args:
        reference: The required evidence reference.
        evidence_index: Index to look the reference up in; any other type
            counts as missing evidence.

    Returns:
        A passed finding when the record passed and was accepted, otherwise
        a failed blocking finding.
    """
    record = (
        evidence_index.records.get(reference)
        if isinstance(evidence_index, evidence.Index)
        else None
    )
    observed_value: float | str | None
    if record is None:
        reason = f"required evidence '{reference}' is missing"
        observed_value = None
    elif (
        record.result is evidence.Outcome.PASS
        and record.decision is evidence.Disposition.ACCEPTED
    ):
        reason = f"required evidence '{reference}' is accepted"
        observed_value = "pass:accepted"
    else:
        reason = (
            f"required evidence '{reference}' is not accepted: "
            f"{record.result.value}:{record.decision.value}"
        )
        observed_value = f"{record.result.value}:{record.decision.value}"
    return ReleaseFinding(
        FindingKind.EVIDENCE,
        reference,
        record is not None
        and record.result is evidence.Outcome.PASS
        and record.decision is evidence.Disposition.ACCEPTED,
        blocking=True,
        reason=reason,
        observed=observed_value,
        expected="pass:accepted",
        evidence=(reference,),
    )


def evaluate_release(
    policy: evaluation.Criteria,
    metric_values: Mapping[str, Any],
    diagnostic_reports: Mapping[str, Any],
    fairness_reports: Mapping[str, Any],
    evidence_index: evidence.Index | None,
) -> ReleaseEvaluation:
    """Evaluates supplied release outputs against a declared policy.

    Args:
        policy: Complete task-aware metric, diagnostic, fairness, explanation,
            and release-gate declaration.
        metric_values: Computed metric values keyed by metric name.
        diagnostic_reports: Diagnostic reports keyed by diagnostic name.
        fairness_reports: Fairness reports keyed by report name or a direct
            JSON-like fairness report.
        evidence_index: Evidence index to look the references declared in
            ``policy`` up in; None or any other type blocks release when the
            policy declares any reference.

    Returns:
        An immutable evaluation. A `policy` that is not an
        ``Criteria`` yields a blocked evaluation with task
        ``"unknown"`` and one input finding. Otherwise a warning diagnostic
        may produce a failed finding without blocking. Unsupported inputs,
        failed blocking gates, and missing or unaccepted evidence always
        block. A required explanation quality measure with no threshold in
        the policy raises `ValueError`.
    """
    if not isinstance(policy, evaluation.Criteria):
        finding = blocked_input_finding(
            "policy", "unsupported evaluation policy"
        )
        return ReleaseEvaluation(
            "unknown",
            0,
            passed=False,
            findings=(finding,),
            blocking_reasons=(finding.reason,),
        )

    findings: list[ReleaseFinding] = []

    if not isinstance(metric_values, Mapping):
        metric_values = {}
        findings.append(
            blocked_input_finding(
                "metric_values",
                "metric values are unsupported; expected a mapping",
            )
        )
    if not isinstance(diagnostic_reports, Mapping):
        diagnostic_reports = {}
        findings.append(
            blocked_input_finding(
                "diagnostic_reports",
                "diagnostic reports are unsupported; expected a mapping",
            )
        )
    if not isinstance(fairness_reports, Mapping):
        fairness_reports = {}
        findings.append(
            blocked_input_finding(
                "fairness_reports",
                "fairness reports are unsupported; expected a mapping",
            )
        )

    declared_metrics = {metric.name for metric in policy.metrics}
    findings.extend(
        blocked_input_finding(
            str(name),
            f"metric '{name}' is not declared by the evaluation policy",
        )
        for name in sorted(metric_values, key=str)
        if name not in declared_metrics
    )
    findings.extend(
        metric_gate_finding(metric_gate, declared_metrics, metric_values)
        for metric_gate in policy.release.metric_gates
    )
    findings.extend(
        diagnostic_gate_finding(diagnostic_gate, diagnostic_reports)
        for diagnostic_gate in policy.release.diagnostic_gates
    )
    findings.extend(explanation_findings(policy, diagnostic_reports))
    findings.extend(
        fairness_gate_finding(fairness_gate, fairness_reports)
        for fairness_gate in policy.release.fairness_gates
    )

    references = required_evidence_references(policy)
    findings.extend(
        evidence_reference_finding(reference, evidence_index)
        for reference in references
    )
    if references and not isinstance(evidence_index, evidence.Index):
        findings.append(
            ReleaseFinding(
                FindingKind.INPUT,
                "evidence_index",
                passed=False,
                blocking=True,
                reason="evidence index is unsupported or missing",
                expected="Index",
            )
        )

    findings.sort(key=lambda item: (item.kind.value, item.name, item.reason))
    immutable_findings = tuple(findings)
    blocking_reasons = tuple(
        item.reason
        for item in immutable_findings
        if not item.passed and item.blocking
    )
    return ReleaseEvaluation(
        task=policy.task.value,
        policy_schema_version=policy.schema_version,
        passed=not blocking_reasons,
        findings=immutable_findings,
        blocking_reasons=blocking_reasons,
    )
