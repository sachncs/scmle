"""Applies a declared policy's gates to the supplied release outputs."""

from collections.abc import Mapping
import math
from numbers import Real
from typing import Any

from scmle.contracts import errors
from scmle.contracts import evaluation
from scmle.contracts import evidence
from scmle.evaluation.release.finding import Finding
from scmle.evaluation.release.finding import Kind
from scmle.evaluation.release.finding import parse
from scmle.evaluation.release.verdict import Verdict

# The quality record's score keys before version 2 (explainability.quality).
SCORES = {"stability": "stability_score", "fidelity": "fidelity_score"}


def citations(
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


def disparities(report: Any, metric: str) -> tuple[float, ...]:
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
                    numeric = parse(value)
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


def grade(report: Any, field_name: str) -> float | None:
    """Returns one finite quality score from an explanation report.

    Args:
        report: An explanation report with a `quality` mapping.
        field_name: Key of the score inside `quality`; a quality record
            written before version 2 holds it under its old key (see
            `SCORES`).

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
    value = quality.get(field_name, quality.get(SCORES.get(field_name, "")))
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def unreadable(name: str, reason: str) -> Finding:
    """Builds the blocking finding for an input the evaluator cannot read.

    Args:
        name: The input's name.
        reason: Why the input is unsupported.

    Returns:
        A failed, blocking input finding.
    """
    return Finding(
        Kind.INPUT,
        name,
        passed=False,
        blocking=True,
        reason=reason,
    )


def judge(
    metric_gate: evaluation.Hurdle,
    declared_metrics: set[str],
    metric_values: Mapping[str, Any],
) -> Finding:
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
        return Finding(
            Kind.METRIC,
            name,
            passed=False,
            blocking=True,
            reason=f"metric '{name}' is unsupported or undeclared",
            expected=threshold,
        )
    if name not in metric_values:
        return Finding(
            Kind.METRIC,
            name,
            passed=False,
            blocking=True,
            reason=f"metric '{name}' is missing",
            expected=threshold,
        )
    value = metric_values[name]
    if isinstance(value, bool) or not isinstance(value, Real):
        return Finding(
            Kind.METRIC,
            name,
            passed=False,
            blocking=True,
            reason=f"metric '{name}' has an unsupported value",
            expected=threshold,
        )
    observed = float(value)
    if not math.isfinite(observed):
        return Finding(
            Kind.METRIC,
            name,
            passed=False,
            blocking=True,
            reason=f"metric '{name}' is non-finite",
            observed=observed,
            expected=threshold,
        )
    direction: object = metric_gate.direction  # may be a raw value at runtime
    if direction is evaluation.Orientation.MAXIMIZE:
        passed = observed >= threshold
        operator = ">="
    elif direction is evaluation.Orientation.MINIMIZE:
        passed = observed <= threshold
        operator = "<="
    else:
        passed = False
        operator = "?"
    result = "passed" if passed else "failed"
    return Finding(
        Kind.METRIC,
        name,
        passed,
        blocking=True,
        reason=f"metric '{name}' {result}: {observed:g} "
        f"{operator} {threshold:g}",
        observed=observed,
        expected=threshold,
    )


def ruling(report: Mapping[str, Any]) -> Any:
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


def adjudge(
    diagnostic_gate: evaluation.Requirement,
    diagnostic_reports: Mapping[str, Any],
) -> Finding:
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
    blocking = diagnostic_gate.action is evaluation.Remedy.BLOCK
    if report is None:
        return Finding(
            Kind.DIAGNOSTIC,
            name,
            passed=False,
            blocking=blocking,
            reason=f"diagnostic '{name}' is missing",
            expected="report without error",
            evidence=diagnostic_gate.required_evidence,
        )
    if not isinstance(report, Mapping) or not report:
        return Finding(
            Kind.DIAGNOSTIC,
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
        verdict = ruling(report)
        if verdict is not None and not isinstance(verdict, bool):
            return Finding(
                Kind.DIAGNOSTIC,
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
        elif verdict is None and diagnostic_gate.decisive:
            failed = True
            reason = f"diagnostic '{name}' reported no verdict"

    return Finding(
        Kind.DIAGNOSTIC,
        name,
        not failed,
        blocking,
        reason,
        expected="report without error",
        evidence=diagnostic_gate.required_evidence,
    )


def explain(
    policy: evaluation.Criteria, diagnostic_reports: Mapping[str, Any]
) -> list[Finding]:
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
            "stability",
            policy.explanations.require_stability,
            policy.explanations.stability_threshold,
            "stability",
        ),
        (
            "fidelity",
            policy.explanations.require_fidelity,
            policy.explanations.fidelity_threshold,
            "fidelity",
        ),
    )
    findings: list[Finding] = []
    for method in policy.explanations.required_methods:
        report = diagnostic_reports.get(method)
        for field_name, required, threshold, label in quality_requirements:
            if not required:
                continue
            name = f"{method}.{label}"
            if threshold is None:
                raise errors.InputError(
                    f"explanation '{method}' has no {label} threshold"
                )
            score = grade(report, field_name)
            if score is None:
                findings.append(
                    Finding(
                        Kind.EXPLANATION,
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
                Finding(
                    Kind.EXPLANATION,
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


def audit(
    fairness_gate: evaluation.Equity, fairness_reports: Mapping[str, Any]
) -> Finding:
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
        value for report in reports for value in disparities(report, metric)
    )
    if not values:
        return Finding(
            Kind.FAIRNESS,
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
        return Finding(
            Kind.FAIRNESS,
            metric,
            passed=False,
            blocking=True,
            reason=f"fairness metric '{metric}' has no bound",
            evidence=fairness_gate.required_evidence,
        )
    result = "passed" if passed else "failed"
    return Finding(
        Kind.FAIRNESS,
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


def referenced(
    reference: str, evidence_index: evidence.Index | None
) -> Finding:
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
    return Finding(
        Kind.EVIDENCE,
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


def screen(
    policy: evaluation.Criteria,
    metric_values: Mapping[str, Any],
    diagnostic_reports: Mapping[str, Any],
    fairness_reports: Mapping[str, Any],
) -> list[Finding]:
    """Applies the policy's metric, diagnostic and fairness gates.

    An input that is not a mapping is read as empty and adds a blocking input
    finding.

    Args:
        policy: The declared evaluation policy.
        metric_values: Computed metric values keyed by metric name.
        diagnostic_reports: Diagnostic reports keyed by diagnostic name.
        fairness_reports: Fairness reports keyed by report name or a direct
            JSON-like fairness report.

    Returns:
        The input findings, one finding per undeclared metric, then the
        findings of the metric, diagnostic, explanation and fairness gates.
    """
    findings: list[Finding] = []

    supplied_metrics: object = metric_values  # callers may pass any object
    if not isinstance(supplied_metrics, Mapping):
        metric_values = {}
        findings.append(
            unreadable(
                "metric_values",
                "metric values are unsupported; expected a mapping",
            )
        )
    supplied_diagnostics: object = diagnostic_reports  # may be any object
    if not isinstance(supplied_diagnostics, Mapping):
        diagnostic_reports = {}
        findings.append(
            unreadable(
                "diagnostic_reports",
                "diagnostic reports are unsupported; expected a mapping",
            )
        )
    supplied_fairness: object = fairness_reports  # callers may pass any object
    if not isinstance(supplied_fairness, Mapping):
        fairness_reports = {}
        findings.append(
            unreadable(
                "fairness_reports",
                "fairness reports are unsupported; expected a mapping",
            )
        )
    declared_metrics = {metric.name for metric in policy.metrics}
    findings.extend(
        unreadable(
            str(name),
            f"metric '{name}' is not declared by the evaluation policy",
        )
        for name in sorted(metric_values, key=str)
        if name not in declared_metrics
    )
    findings.extend(
        judge(metric_gate, declared_metrics, metric_values)
        for metric_gate in policy.release.metric_gates
    )
    findings.extend(
        adjudge(diagnostic_gate, diagnostic_reports)
        for diagnostic_gate in policy.release.diagnostic_gates
    )
    findings.extend(explain(policy, diagnostic_reports))
    findings.extend(
        audit(fairness_gate, fairness_reports)
        for fairness_gate in policy.release.fairness_gates
    )
    return findings


def attest(
    policy: evaluation.Criteria, evidence_index: evidence.Index | None
) -> list[Finding]:
    """Checks every evidence reference the policy declares.

    Args:
        policy: The declared evaluation policy.
        evidence_index: Index to look the references up in; None or any other
            type fails every reference.

    Returns:
        One finding per reference, plus a blocking input finding when there
        is a reference and `evidence_index` is not an index.
    """
    findings: list[Finding] = []
    references = citations(policy)
    findings.extend(
        referenced(reference, evidence_index) for reference in references
    )
    if references and not isinstance(evidence_index, evidence.Index):
        findings.append(
            Finding(
                Kind.INPUT,
                "evidence_index",
                passed=False,
                blocking=True,
                reason="evidence index is unsupported or missing",
                expected="Index",
            )
        )
    return findings


def assess(
    policy: evaluation.Criteria,
    metric_values: Mapping[str, Any],
    diagnostic_reports: Mapping[str, Any],
    fairness_reports: Mapping[str, Any],
    evidence_index: evidence.Index | None,
) -> Verdict:
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
    supplied_policy: object = policy  # callers may pass any object
    if not isinstance(supplied_policy, evaluation.Criteria):
        finding = unreadable("policy", "unsupported evaluation policy")
        return Verdict(
            "unknown",
            0,
            passed=False,
            findings=(finding,),
            reasons=(finding.reason,),
        )

    findings = screen(
        policy, metric_values, diagnostic_reports, fairness_reports
    )
    findings.extend(attest(policy, evidence_index))

    findings.sort(key=lambda item: (item.kind.value, item.name, item.reason))
    immutable_findings = tuple(findings)
    blocking_reasons = tuple(
        item.reason
        for item in immutable_findings
        if not item.passed and item.blocking
    )
    return Verdict(
        task=policy.task.value,
        policy=policy.schema_version,
        passed=not blocking_reasons,
        findings=immutable_findings,
        reasons=blocking_reasons,
    )
