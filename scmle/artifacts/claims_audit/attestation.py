"""Checks of the attestations a causal claim rests on."""

from pathlib import Path
from typing import Any

from scmle.artifacts.claims_audit import files
from scmle.artifacts.claims_audit import verdict
from scmle.assurance.charter import contracts
from scmle.causal.identification import attest
from scmle.contracts import errors
from scmle.contracts import signing


def assess(
    item: dict[str, Any],
    report: dict[str, Any],
    charter: Any,
    allowed: dict[str, set[str]],
    signing_key: str | bytes | None,
) -> tuple[Any, list[str], bool]:
    """Checks one accepted attestation.

    Args:
        item: The attestation record.
        report: The attestation report, which names the evaluation day.
        charter: The charter the attestation must be scoped to.
        allowed: The assumptions each named attestor may attest.
        signing_key: Key for the signature, or None to use the environment
            key.

    Returns:
        The parsed attestation (None if malformed), a message for each
        problem, and whether the signature could not be checked for lack of
        a key.
    """
    try:
        given = attest.Attestation.from_dict(item)
    except (ValueError, TypeError) as error:
        # handled: returned as failed finding/problem/refusal with error text
        return None, [f"malformed attestation: {error}"], False
    problems = []
    if given.assumption not in allowed.get(given.attestor, set()):
        problems.append(f"{given.assumption}: {given.attestor} not permitted")
    if given.scope != charter.charter_hash:
        problems.append(f"{given.assumption}: scope is not this charter")
    day = str(attest.Applied.KEYS.upgrade(report).get("evaluated", ""))
    try:
        valid = given.date <= day <= given.expires
    except TypeError:
        # handled: returned as failed finding/problem/refusal with error text
        valid = False
    if not valid:
        problems.append(f"{given.assumption}: not valid on {day!r}")
    unchecked = False
    if signing_key is None and not signing.keying(None):
        unchecked = True
    elif not given.verify(signing_key):
        problems.append(f"{given.assumption}: signature does not verify")
    return given, problems, unchecked


def vet(
    report: dict[str, Any],
    charter: Any,
    allowed: dict[str, set[str]],
    signing_key: str | bytes | None,
) -> tuple[dict[str, str], list[str], bool]:
    """Checks every accepted attestation of a report.

    Args:
        report: The attestation report.
        charter: The charter the attestations must be scoped to.
        allowed: The assumptions each named attestor may attest.
        signing_key: Key for the signatures, or None to use the environment
            key.

    Returns:
        The attestor by assumption for each well-formed attestation, a
        message for each problem, and whether any signature could not be
        checked for lack of a key.
    """
    accepted: dict[str, str] = {}
    problems: list[str] = []
    unchecked = False
    for item in report.get("accepted", []):
        given, found, skipped = assess(
            item, report, charter, allowed, signing_key
        )
        problems.extend(found)
        unchecked = unchecked or skipped
        if given is not None:
            accepted[given.assumption] = given.attestor
    return accepted, problems, unchecked


def vouch(run_dir: Path, signing_key: str | bytes | None) -> verdict.Finding:
    """Checks that a claim above `unverified` rests on verified attestors.

    Args:
        run_dir: The run directory.
        signing_key: Key for the attestation signatures, or None to use the
            environment key.

    Returns:
        A finding: skipped without a causal claims file, a charter file or a
        charter causal block. Failing if the charter cannot be read, a tier
        above `unverified` has no named attestors or no accepted attestation,
        an assumption's named attestor has no accepted attestation, or an
        accepted attestation is malformed, not permitted, scoped to another
        charter, outside its validity window or has an invalid signature.
        Otherwise skipped if no key is available to check the signatures, else
        passing.
    """
    name = "causal attestations verify"
    claims = files.read_optional(run_dir / files.CAUSAL_FILE)
    record = files.read_optional(run_dir / files.CHARTER_FILE)
    if claims is None or record is None:
        return verdict.Finding(
            name, verdict.SKIPPED, "no causal claims or charter file"
        )
    try:
        charter = contracts.parse(record.get("charter", record))
    except errors.ConfigError as error:
        # handled: returned as failed finding/problem/refusal with error text
        return verdict.Finding(name, verdict.FAIL, str(error))
    block = charter.estimand.causal
    if block is None:
        return verdict.Finding(
            name, verdict.SKIPPED, "the charter has no causal block"
        )
    report = claims.get("attestations") or {}
    raised = claims.get("tier") in ("attested", "confirmed")
    problems = []
    if raised and not block.attestors:
        problems.append(
            "tier above unverified but the charter names no attestors"
        )
    allowed = {a.name: set(a.assumptions) for a in block.attestors}
    accepted, found, unchecked = vet(report, charter, allowed, signing_key)
    problems.extend(found)
    for assumption, named in (claims.get("assumptions") or {}).items():
        if named and accepted.get(assumption) != named:
            problems.append(
                f"{assumption}: {named} has no accepted attestation"
            )
    if raised and not accepted:
        problems.append("tier above unverified with no accepted attestation")
    if problems:
        return verdict.Finding(name, verdict.FAIL, "; ".join(problems))
    if unchecked:
        return verdict.Finding(
            name, verdict.SKIPPED, "no signing key to check the signatures"
        )
    return verdict.Finding(
        name,
        verdict.PASS,
        f"{len(accepted)} attestation(s), none required"
        if not accepted
        else f"{len(accepted)} attestation(s) verified",
    )
