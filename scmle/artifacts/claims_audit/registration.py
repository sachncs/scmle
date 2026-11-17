"""Checks of the claims registered by hash and the negative control."""

import json
from pathlib import Path
from typing import Any
from typing import cast

from scmle.artifacts.claims_audit import files
from scmle.artifacts.claims_audit import verdict
from scmle.assurance import coverage_confirm
from scmle.assurance import coverage_run
from scmle.assurance.ledger import confirm
import scmle.assurance.ledger.ledger


def orphaned(
    name: str, document: dict[str, Any], registered: dict[str, Any] | None
) -> verdict.Finding | None:
    """Judges a coverage confirmation whose registration decides the check.

    Args:
        name: The name of the finding.
        document: The coverage confirmation document.
        registered: The ledger's registration of the claim, or None.

    Returns:
        A failing finding if the claim is registered but not evaluated, or
        evaluated but not registered; a skipped one if the claim was neither;
        otherwise None.
    """
    if document.get("status") == confirm.NOT_EVALUATED:
        if registered is not None:
            return verdict.Finding(
                name,
                verdict.FAIL,
                "a registered claim has no evaluation document",
            )
        return verdict.Finding(
            name, verdict.SKIPPED, "the claim could not be tested on C"
        )
    if registered is None:
        return verdict.Finding(
            name, verdict.FAIL, "the claim is not registered in the ledger"
        )
    return None


def reconcile(
    document: dict[str, Any],
    redone: dict[str, Any],
    ledger: scmle.assurance.ledger.ledger.Ledger,
) -> list[str]:
    """Checks the stated coverage status against the test and the ledger.

    Args:
        document: The coverage confirmation document.
        redone: The recomputed decision for the claim.
        ledger: The run's ledger.

    Returns:
        A message for each status that does not follow from the test or
        differs from the sealed ledger row; empty if both agree.
    """
    problems = []
    gate = document.get("task_gate")
    expected = redone["status"]
    if isinstance(gate, dict) and "task" in gate:
        expected = coverage_confirm.capped(str(gate["task"]), expected)
    if expected != document.get("status"):
        problems.append("the status does not follow from the test")
    sealed_row = (ledger.confirmation or {}).get("claims", {})
    stated = (sealed_row.get(coverage_confirm.CLAIM_ID) or {}).get("status")
    if stated != document.get("status"):
        problems.append("the status differs from the sealed ledger row")
    return problems


def reconfirm(
    run_dir: Path, ledger: scmle.assurance.ledger.ledger.Ledger
) -> verdict.Finding | None:
    """Checks the coverage claim on `C`: registered, recomputed and consistent.

    Args:
        run_dir: The run directory.
        ledger: The run's ledger.

    Returns:
        A finding, or None when the run holds no `coverage_confirmation.json`.
        It is skipped when the document says the claim was not evaluated and the
        ledger has no registration for it. It fails if the claim is registered
        but not evaluated, is evaluated but not registered, the document is
        malformed, the registration or claim hash differs, the p-value or
        status does not recompute, or the status differs from the sealed ledger
        row.
    """
    name = "coverage claim on C: registered, recomputed, status matches"
    document = files.read_optional(run_dir / coverage_confirm.DOCUMENT)
    if document is None:
        return None
    registered = ledger.claims.get(coverage_confirm.CLAIM_ID)
    early = orphaned(name, document, registered)
    if early is not None:
        return early
    problems = []
    claim = files.read_optional(run_dir / coverage_run.CLAIM_FILE)
    try:
        claim_hash = scmle.assurance.ledger.ledger.digest(claim or {})
        choice = document["primary"]
        body = coverage_confirm.prefreeze(claim_hash, choice)
        if document.get("coverage_claim_sha256") != claim_hash:
            problems.append("the document names another coverage claim")
        # `orphaned` returned None: it is registered.
        registration = cast("dict[str, Any]", registered)
        if scmle.assurance.ledger.ledger.digest(body) != registration["sha256"]:
            problems.append("the registration hash differs from the ledger's")
        redone = coverage_confirm.decide(
            int(document["covered"]), int(document["rows"]), choice
        )
    except (KeyError, TypeError, ValueError):
        # handled: returned as failed finding/problem/refusal with error text
        return verdict.Finding(name, verdict.FAIL, "the document is malformed")
    if abs(redone["p_value"] - float(document["p_value"])) > verdict.TOLERANCE:
        problems.append("the p-value does not recompute from the counts")
    problems.extend(reconcile(document, redone, ledger))
    if problems:
        return verdict.Finding(name, verdict.FAIL, "; ".join(problems))
    return verdict.Finding(
        name,
        verdict.PASS,
        f"{document['covered']} of {document['rows']} covered; "
        f"status {document['status']}",
    )


def misnamed(
    run_dir: Path, ledger: scmle.assurance.ledger.ledger.Ledger
) -> list[str]:
    """Checks that the baseline report names the registered baseline claim.

    Args:
        run_dir: The run directory.
        ledger: The run's ledger.

    Returns:
        A message for each baseline claim whose registration the report does
        not name; empty if there is no report or all match.
    """
    baseline_doc = files.read_optional(run_dir / files.BASELINE_FILE)
    return [
        f"{claim_id}: report names another registration"
        for claim_id, item in ledger.claims.items()
        if item["kind"] == "baseline"
        and baseline_doc is not None
        and baseline_doc.get("registration_hash") != item["sha256"]
    ]


def check_registered(
    run_dir: Path,
    document: dict[str, Any],
    ledger: scmle.assurance.ledger.ledger.Ledger,
) -> verdict.Finding:
    """Checks that registered claims are stated at the seal and match.

    Args:
        run_dir: The run directory.
        document: The claims document.
        ledger: The run's ledger.

    Returns:
        A passing finding if the registrations, attachment hashes and baseline
        report agree with the sealed ledger; otherwise a failing one listing the
        problems.
    """
    name = "registered claims: files hash to the sealed ledger"
    sealed_row = ledger.confirmation or {}
    attachments = sealed_row.get("attachments", {})
    problems = []
    if document.get("registered_claims") != sealed_row.get("claims"):
        problems.append("claims file and confirmation row differ")
    for claim_id, item in ledger.claims.items():
        stated = (sealed_row.get("claims") or {}).get(claim_id)
        if stated is None or stated.get("sha256") != item["sha256"]:
            problems.append(f"{claim_id}: registration hash differs")
    for file_name, expected in sorted(attachments.items()):
        found = files.read_optional(run_dir / file_name)
        if found is None:
            problems.append(f"{file_name}: missing")
        elif scmle.assurance.ledger.ledger.digest(found) != expected:
            problems.append(f"{file_name}: differs from the sealed hash")
    if document.get("attachments") != attachments:
        problems.append("attachment hashes differ from the claims file")
    problems.extend(misnamed(run_dir, ledger))
    if problems:
        return verdict.Finding(name, verdict.FAIL, "; ".join(problems))
    return verdict.Finding(
        name,
        verdict.PASS,
        f"{len(ledger.claims)} claim(s), {len(attachments)} file(s) bound",
    )


def witness(
    run_dir: Path, ledger: scmle.assurance.ledger.ledger.Ledger
) -> verdict.Finding:
    """Checks that the negative-control file is the one the ledger recorded.

    Args:
        run_dir: The run directory.
        ledger: The run's ledger.

    Returns:
        A finding: skipped without a control file or control row, passing if the
        file's hash equals the ledger's, otherwise failing.
    """
    name = "negative control matches the ledger"
    path = run_dir / files.CONTROL_FILE
    if not path.exists() or ledger.control is None:
        return verdict.Finding(
            name, verdict.SKIPPED, "no control file or no control row"
        )
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        # handled: returned as failed finding/problem/refusal with error text
        return verdict.Finding(
            name, verdict.FAIL, f"{files.CONTROL_FILE} is not JSON"
        )
    if scmle.assurance.ledger.ledger.digest(body) == ledger.control.get(
        "sha256"
    ):
        return verdict.Finding(name, verdict.PASS, "hash equal")
    return verdict.Finding(
        name, verdict.FAIL, "the file differs from the ledger's control row"
    )
