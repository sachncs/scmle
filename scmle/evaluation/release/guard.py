"""Independent release gates that block a verdict on missing evidence."""

from collections.abc import Mapping
from typing import Any

from scmle.contracts import errors
from scmle.contracts import evidence
from scmle.evaluation.release.finding import Finding
from scmle.evaluation.release.finding import Kind
from scmle.evaluation.release.verdict import Verdict


def faults(outcome: Verdict, reliability_plan: Any) -> Verdict:
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
        TypeError: If `outcome` is not a `Verdict`.
    """
    if not isinstance(outcome, Verdict):
        raise errors.KindError("outcome must be Verdict")
    passed = (
        isinstance(reliability_plan, Mapping)
        and reliability_plan.get("execution_status") == "passed"
        and reliability_plan.get("release_blocked") is False
    )
    if passed:
        return outcome
    finding = Finding(
        Kind.RELIABILITY,
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
    return Verdict(
        task=outcome.task,
        policy_schema_version=outcome.policy_schema_version,
        passed=False,
        findings=findings,
        blocking_reasons=blocking_reasons,
    )


def certify(
    outcome: Verdict,
    attestation_record: Any,
    verified: bool,
) -> Verdict:
    """Blocks release until packaged evidence has a valid signed attestation.

    Args:
        outcome: The evaluation to extend.
        attestation_record: The attestation, as a mapping holding a
            `signature` with an `hmac-sha256` algorithm and a non-blank
            value.
        verified: Whether the signature was checked and is valid. A
            signature that was not checked does not pass (ERR-4).

    Returns:
        `outcome` unchanged when the signature is present and verified;
        otherwise a blocked evaluation with an added failed blocking
        finding.

    Raises:
        TypeError: If `outcome` is not a `Verdict`.
    """
    if not isinstance(outcome, Verdict):
        raise errors.KindError("outcome must be Verdict")
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
        and verified is True
    )
    if passed:
        return outcome
    finding = Finding(
        Kind.SECURITY,
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
    return Verdict(
        task=outcome.task,
        policy_schema_version=outcome.policy_schema_version,
        passed=False,
        findings=findings,
        blocking_reasons=blocking_reasons,
    )


def countersign(
    outcome: Verdict,
    index: evidence.Index | None,
    verified: bool,
) -> Verdict:
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
        TypeError: If `outcome` is not a `Verdict`.
    """
    if not isinstance(outcome, Verdict):
        raise errors.KindError("outcome must be Verdict")
    passed = (
        isinstance(index, evidence.Index)
        and index.signature is not None
        and verified
    )
    if passed:
        return outcome
    finding = Finding(
        Kind.SECURITY,
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
    return Verdict(
        task=outcome.task,
        policy_schema_version=outcome.policy_schema_version,
        passed=False,
        findings=findings,
        blocking_reasons=blocking_reasons,
    )
