"""Evidence-backed release evaluation for declared task policies.

This module evaluates release evidence; it does not calculate metrics or
diagnostics. A caller supplies the outputs produced by those algorithms and
the evaluator applies the thresholds and evidence references declared by an
``EvaluationPolicy``. Unknown inputs and incomplete evidence fail closed.
"""

from collections.abc import Mapping
import dataclasses
import enum
import json
import math
from numbers import Real
from typing import Any

from scml.contracts import evaluation
from scml.contracts import evidence


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
        """Validate the immutable finding shape."""
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
            value = getattr(self, field_name)
            if isinstance(value, bool):
                raise TypeError(f"{field_name} must not be bool")
            if isinstance(value, Real):
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError(f"{field_name} must be finite")
                object.__setattr__(self, field_name, numeric)
            elif value is not None and not isinstance(value, str):
                raise TypeError(f"{field_name} must be numeric, text, or None")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible finding."""
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
class ReleaseEvaluation:
    """Immutable outcome of evaluating one policy and its supplied evidence."""

    task: str
    policy_schema_version: int
    passed: bool
    findings: tuple[ReleaseFinding, ...]
    blocking_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        """Ensure the aggregate outcome agrees with its findings."""
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
        """Return a deterministic JSON-compatible release outcome."""
        return {
            "task": self.task,
            "policy_schema_version": self.policy_schema_version,
            "passed": self.passed,
            "blocked": self.blocked,
            "blocking_reasons": list(self.blocking_reasons),
            "findings": [item.to_dict() for item in self.findings],
        }

    def canonical_json(self) -> str:
        """Return stable JSON suitable for release evidence storage."""
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


def required_evidence_references(
    policy: evaluation.EvaluationPolicy,
) -> tuple[str, ...]:
    """Return every evidence reference declared by a policy in sorted order.

    References are collected from the release policy, metric declarations,
    diagnostic gates, fairness gates, and explanation policy. Duplicate
    references are evaluated once because one accepted evidence record can
    support several declarations.
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
    """Find finite values for one fairness metric in a nested report.

    Fairness reports may be keyed by sensitive column, or may be wrapped in a
    named report entry. The traversal accepts JSON-like mappings and lists and
    ignores unrelated scalar fields. A non-numeric metric value is not treated
    as a passing value.
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
                    if isinstance(value, bool) or not isinstance(value, Real):
                        continue
                    numeric = float(value)
                    if math.isfinite(numeric):
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
    """Return one finite quality score from an explanation report."""
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
    """Block release until the persisted fault plan has verified evidence.

    A model may satisfy its statistical success criteria while still being
    unsafe to release. The fault-injection plan is therefore an independent
    release gate. Only an explicitly completed plan with
    ``execution_status=passed`` and ``release_blocked=false`` can pass this
    gate; malformed or planned evidence fails closed.
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
        False,
        True,
        "fault-injection recovery evidence is not complete; release blocked",
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
    """Block release until packaged evidence has a valid signed attestation."""
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
        False,
        True,
        "release attestation is missing or invalid; release blocked",
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


def evaluate_release(
    policy: evaluation.EvaluationPolicy,
    metric_values: Mapping[str, Any],
    diagnostic_reports: Mapping[str, Any],
    fairness_reports: Mapping[str, Any],
    evidence_index: evidence.EvidenceIndex | None,
) -> ReleaseEvaluation:
    """Evaluate supplied release outputs against a declared policy.

    Args:
        policy: Complete task-aware metric, diagnostic, fairness, explanation,
            and release-gate declaration.
        metric_values: Computed metric values keyed by metric name.
        diagnostic_reports: Diagnostic reports keyed by diagnostic name.
        fairness_reports: Fairness reports keyed by report name or a direct
            JSON-like fairness report.
        evidence_index: Accepted release evidence records keyed by the
            references declared in ``policy``.

    Returns:
        An immutable evaluation. A warning diagnostic may produce a failed
        finding without blocking. Unsupported inputs, failed blocking gates,
        and missing or unaccepted evidence always block.
    """
    if not isinstance(policy, evaluation.EvaluationPolicy):
        finding = ReleaseFinding(
            kind=FindingKind.INPUT,
            name="policy",
            passed=False,
            blocking=True,
            reason="unsupported evaluation policy",
        )
        return ReleaseEvaluation(
            "unknown", 0, False, (finding,), (finding.reason,)
        )

    findings: list[ReleaseFinding] = []

    if not isinstance(metric_values, Mapping):
        metric_values = {}
        findings.append(
            ReleaseFinding(
                FindingKind.INPUT,
                "metric_values",
                False,
                True,
                "metric values are unsupported; expected a mapping",
            )
        )
    if not isinstance(diagnostic_reports, Mapping):
        diagnostic_reports = {}
        findings.append(
            ReleaseFinding(
                FindingKind.INPUT,
                "diagnostic_reports",
                False,
                True,
                "diagnostic reports are unsupported; expected a mapping",
            )
        )
    if not isinstance(fairness_reports, Mapping):
        fairness_reports = {}
        findings.append(
            ReleaseFinding(
                FindingKind.INPUT,
                "fairness_reports",
                False,
                True,
                "fairness reports are unsupported; expected a mapping",
            )
        )

    declared_metrics = {metric.name for metric in policy.metrics}
    for name in sorted(metric_values, key=str):
        if name not in declared_metrics:
            findings.append(
                ReleaseFinding(
                    FindingKind.INPUT,
                    str(name),
                    False,
                    True,
                    f"metric '{name}' is not declared by the evaluation policy",
                )
            )

    for metric_gate in policy.release.metric_gates:
        if metric_gate.metric not in declared_metrics:
            findings.append(
                ReleaseFinding(
                    FindingKind.METRIC,
                    metric_gate.metric,
                    False,
                    True,
                    f"metric '{metric_gate.metric}' is unsupported or "
                    "undeclared",
                    expected=metric_gate.threshold,
                )
            )
            continue
        if metric_gate.metric not in metric_values:
            findings.append(
                ReleaseFinding(
                    FindingKind.METRIC,
                    metric_gate.metric,
                    False,
                    True,
                    f"metric '{metric_gate.metric}' is missing",
                    expected=metric_gate.threshold,
                )
            )
            continue
        value = metric_values[metric_gate.metric]
        if isinstance(value, bool) or not isinstance(value, Real):
            findings.append(
                ReleaseFinding(
                    FindingKind.METRIC,
                    metric_gate.metric,
                    False,
                    True,
                    f"metric '{metric_gate.metric}' has an unsupported value",
                    expected=metric_gate.threshold,
                )
            )
            continue
        observed = float(value)
        if not math.isfinite(observed):
            findings.append(
                ReleaseFinding(
                    FindingKind.METRIC,
                    metric_gate.metric,
                    False,
                    True,
                    f"metric '{metric_gate.metric}' is non-finite",
                    observed=observed,
                    expected=metric_gate.threshold,
                )
            )
            continue
        if metric_gate.direction is evaluation.MetricDirection.MAXIMIZE:
            passed = observed >= metric_gate.threshold
            operator = ">="
        elif metric_gate.direction is evaluation.MetricDirection.MINIMIZE:
            passed = observed <= metric_gate.threshold
            operator = "<="
        else:
            passed = False
            operator = "?"
        result = "passed" if passed else "failed"
        findings.append(
            ReleaseFinding(
                FindingKind.METRIC,
                metric_gate.metric,
                passed,
                True,
                f"metric '{metric_gate.metric}' {result}: {observed:g} "
                f"{operator} {metric_gate.threshold:g}",
                observed=observed,
                expected=metric_gate.threshold,
            )
        )

    for diagnostic_gate in policy.release.diagnostic_gates:
        report = diagnostic_reports.get(diagnostic_gate.name)
        blocking = diagnostic_gate.action is evaluation.DiagnosticAction.BLOCK
        if report is None:
            findings.append(
                ReleaseFinding(
                    FindingKind.DIAGNOSTIC,
                    diagnostic_gate.name,
                    False,
                    blocking,
                    f"diagnostic '{diagnostic_gate.name}' is missing",
                    expected="report without error",
                    evidence=diagnostic_gate.required_evidence,
                )
            )
            continue
        if not isinstance(report, Mapping) or not report:
            findings.append(
                ReleaseFinding(
                    FindingKind.DIAGNOSTIC,
                    diagnostic_gate.name,
                    False,
                    True,
                    f"diagnostic '{diagnostic_gate.name}' has an "
                    "unsupported report",
                    expected="non-empty mapping",
                    evidence=diagnostic_gate.required_evidence,
                )
            )
            continue
        failed = False
        reason = f"diagnostic '{diagnostic_gate.name}' is present"
        if report.get("error") not in (None, "", False):
            failed = True
            reason = f"diagnostic '{diagnostic_gate.name}' reported an error"
        elif report.get("available") is False:
            failed = True
            reason = f"diagnostic '{diagnostic_gate.name}' is unavailable"
        else:
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
            if verdict is not None and not isinstance(verdict, bool):
                findings.append(
                    ReleaseFinding(
                        FindingKind.DIAGNOSTIC,
                        diagnostic_gate.name,
                        False,
                        True,
                        f"diagnostic '{diagnostic_gate.name}' has an "
                        "unsupported verdict",
                        expected="boolean pass verdict",
                        evidence=diagnostic_gate.required_evidence,
                    )
                )
                continue
            if verdict is False:
                failed = True
                reason = f"diagnostic '{diagnostic_gate.name}' reported failure"
        findings.append(
            ReleaseFinding(
                FindingKind.DIAGNOSTIC,
                diagnostic_gate.name,
                not failed,
                blocking,
                reason,
                expected="report without error",
                evidence=diagnostic_gate.required_evidence,
            )
        )

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
                        False,
                        True,
                        f"explanation '{method}' has no valid {label} score",
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
                    True,
                    f"explanation '{method}' {label} {result}: "
                    f"{score:g} >= {threshold:g}",
                    observed=score,
                    expected=threshold,
                )
            )

    if policy.release.fairness_gates:
        if not isinstance(fairness_reports, Mapping):
            fairness_reports = {}
        for fairness_gate in policy.release.fairness_gates:
            if fairness_gate.metric in fairness_reports:
                reports = (fairness_reports,)
            else:
                reports = tuple(
                    fairness_reports[name]
                    for name in sorted(fairness_reports, key=str)
                )
            values = tuple(
                value
                for report in reports
                for value in fairness_metric_values(
                    report, fairness_gate.metric
                )
            )
            if not values:
                findings.append(
                    ReleaseFinding(
                        FindingKind.FAIRNESS,
                        fairness_gate.metric,
                        False,
                        True,
                        f"fairness metric '{fairness_gate.metric}' is "
                        "missing or "
                        "unsupported",
                        expected=fairness_gate.maximum
                        if fairness_gate.maximum is not None
                        else fairness_gate.minimum,
                        evidence=fairness_gate.required_evidence,
                    )
                )
                continue
            if fairness_gate.maximum is not None:
                observed = max(values)
                passed = observed <= fairness_gate.maximum
                expected: float | None = fairness_gate.maximum
                operator = "<="
            elif fairness_gate.minimum is not None:
                observed = min(values)
                passed = observed >= fairness_gate.minimum
                expected = fairness_gate.minimum
                operator = ">="
            else:
                findings.append(
                    ReleaseFinding(
                        FindingKind.FAIRNESS,
                        fairness_gate.metric,
                        False,
                        True,
                        f"fairness metric '{fairness_gate.metric}' has no "
                        "bound",
                        evidence=fairness_gate.required_evidence,
                    )
                )
                continue
            result = "passed" if passed else "failed"
            findings.append(
                ReleaseFinding(
                    FindingKind.FAIRNESS,
                    fairness_gate.metric,
                    passed,
                    True,
                    f"fairness metric '{fairness_gate.metric}' {result}: "
                    f"worst value {observed:g} {operator} {expected:g}",
                    observed=observed,
                    expected=expected,
                    evidence=fairness_gate.required_evidence,
                )
            )

    references = required_evidence_references(policy)
    for reference in references:
        record = (
            evidence_index.find(reference)
            if isinstance(evidence_index, evidence.EvidenceIndex)
            else None
        )
        observed_value: float | str | None
        if record is None:
            reason = f"required evidence '{reference}' is missing"
            observed_value = None
        elif (
            record.result is evidence.EvidenceResult.PASS
            and record.decision is evidence.EvidenceDecision.ACCEPTED
        ):
            reason = f"required evidence '{reference}' is accepted"
            observed_value = "pass:accepted"
        else:
            reason = (
                f"required evidence '{reference}' is not accepted: "
                f"{record.result.value}:{record.decision.value}"
            )
            observed_value = f"{record.result.value}:{record.decision.value}"
        findings.append(
            ReleaseFinding(
                FindingKind.EVIDENCE,
                reference,
                record is not None
                and record.result is evidence.EvidenceResult.PASS
                and record.decision is evidence.EvidenceDecision.ACCEPTED,
                True,
                reason,
                observed=observed_value,
                expected="pass:accepted",
                evidence=(reference,),
            )
        )
    if references and not isinstance(evidence_index, evidence.EvidenceIndex):
        findings.append(
            ReleaseFinding(
                FindingKind.INPUT,
                "evidence_index",
                False,
                True,
                "evidence index is unsupported or missing",
                expected="EvidenceIndex",
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
