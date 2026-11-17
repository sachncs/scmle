"""Checks against the custody log: `C` opened, decision files, coverage."""

import json
from pathlib import Path
from typing import Any

from scmle.artifacts.claims_audit import files
from scmle.artifacts.claims_audit import verdict
from scmle.assurance import coverage_run
from scmle.assurance.custody import sealed
from scmle.assurance.ledger import confirm
import scmle.assurance.ledger.ledger
from scmle.contracts import errors
from scmle.decision import reports


def openings(
    name: str, document: dict[str, Any], opened: int
) -> verdict.Finding | None:
    """Judges how often `C` was opened, when that alone decides the check.

    Args:
        name: The name of the finding.
        document: The claims document.
        opened: The number of open events for `C` in the custody log.

    Returns:
        A failing finding if `C` was opened more than once or a confirmed
        outcome has no single open event, a passing one if `C` was never
        opened in a non-confirmed run, otherwise None.
    """
    if opened > 1:
        return verdict.Finding(
            name, verdict.FAIL, f"C was opened {opened} times"
        )
    if document.get("outcome") == confirm.CONFIRMED and opened != 1:
        return verdict.Finding(
            name, verdict.FAIL, "outcome is confirmed but C has no open event"
        )
    if opened == 0 and document.get("outcome") != confirm.CONFIRMED:
        return verdict.Finding(
            name, verdict.PASS, f"C not opened (outcome {document['outcome']})"
        )
    return None


def frozen_selection_problem(
    entries: list[dict[str, Any]],
    names: list[str],
    document: dict[str, Any],
) -> str | None:
    """Checks the `selection_frozen` event against the claims document.

    Args:
        entries: The custody log entries for `C`.
        names: The event names of those entries, in order.
        document: The claims document.

    Returns:
        What is wrong with the frozen selection, or None if nothing is.
    """
    frozen = [e for e in entries if e["event"] == "selection_frozen"]
    if len(frozen) != 1 or names.index("selection_frozen") > names.index(
        "open"
    ):
        return "no single selection_frozen before the open"
    detail = frozen[0].get("detail", {})
    if detail.get("survivors_sha256") != document.get("survivors_sha256"):
        return "frozen survivor hash differs from the claims"
    record = document.get("stage_two")
    if (
        record is not None
        and detail.get("family_sha256") != record["family_sha256"]
    ):
        return "frozen family hash differs from the claims"
    return None


def custodial(
    run_dir: Path,
    document: dict[str, Any],
    signing_key: str | bytes | None,
) -> verdict.Finding:
    """Checks the custody log: selection frozen first, `C` opened once.

    Args:
        run_dir: The run directory.
        document: The claims document.
        signing_key: Key for the custody log MACs, or None.

    Returns:
        A finding: skipped if there is no custody log, failing if the log is
        invalid, `C` was opened more than once, a confirmed outcome has no open
        event, the selection was not frozen exactly once before the open, or the
        frozen hashes differ from the claims; otherwise passing (including a
        non-confirmed run in which `C` was never opened).
    """
    name = "custody log: selection frozen before C opened, C opened once"
    path = run_dir / files.CUSTODY_FILE
    if not path.exists():
        return verdict.Finding(
            name, verdict.SKIPPED, f"{files.CUSTODY_FILE} is not in the run"
        )
    try:
        log = sealed.Journal(path, signing_key)
    except errors.InputError:
        # ERR-20: an unusable signing key (empty) is the caller's input
        # error and propagates, as its ValueError did before the error
        # types were narrowed; it is not a finding about the run.
        raise
    except errors.ScmleError as error:
        # handled: returned as failed finding/problem/refusal with error text
        return verdict.Finding(name, verdict.FAIL, str(error))
    entries = [e for e in log.entries if e.get("set") == "C"]
    names = [e["event"] for e in entries]
    opened = names.count("open")
    early = openings(name, document, opened)
    if early is not None:
        return early
    problem = frozen_selection_problem(entries, names, document)
    if problem is not None:
        return verdict.Finding(name, verdict.FAIL, problem)
    return verdict.Finding(
        name, verdict.PASS, f"{len(log.entries)} entries, chain verified"
    )


def tampered(run_dir: Path, registered: dict[str, Any]) -> list[str]:
    """Compares each decision file with the hash the custody log registered.

    Args:
        run_dir: The run directory.
        registered: Registered file digests by file name.

    Returns:
        One message per file that is unreadable, unregistered, missing or
        changed; empty if all match.
    """
    problems = []
    for file_name in reports.DECISION_FILES:
        expected = registered.get(file_name)
        try:
            found = reports.file_digest(run_dir / file_name)
        except ValueError as error:
            # handled: returned as failed finding/problem/refusal with error
            # text
            problems.append(f"{file_name}: unreadable ({error})")
            continue
        if expected is None and found is not None:
            problems.append(f"{file_name}: not registered")
        elif expected is not None and found is None:
            problems.append(f"{file_name}: missing")
        elif found != expected:
            problems.append(f"{file_name}: differs from the registered hash")
    return problems


def check_decision_files(
    run_dir: Path, signing_key: str | bytes | None
) -> verdict.Finding | None:
    """Checks that decision files hash to what the custody log registered.

    Args:
        run_dir: The run directory.
        signing_key: Key for the custody log MACs, or None.

    Returns:
        A finding, or None when the run registered no decision files and holds
        none. It is skipped for an older run that holds files without a
        registration, and failing for an invalid log, a double registration, or
        a file that is unregistered, missing, unreadable or changed.
    """
    name = "decision files: hashes match the custody log"
    path = run_dir / files.CUSTODY_FILE
    entries: list[dict[str, Any]] = []
    if path.exists():
        try:
            log = sealed.Journal(path, signing_key)
        except errors.InputError:
            # ERR-20: an unusable signing key (empty) is the caller's input
            # error and propagates, as its ValueError did before the error
            # types were narrowed; it is not a finding about the run.
            raise
        except errors.ScmleError as error:
            # handled: returned as failed finding/problem/refusal with error
            # text
            return verdict.Finding(name, verdict.FAIL, str(error))
        entries = [
            e
            for e in log.entries
            if e.get("event") == reports.REGISTRATION_EVENT
        ]
    present = [n for n in reports.DECISION_FILES if (run_dir / n).exists()]
    if not entries:
        if not present:
            return None
        return verdict.Finding(
            name,
            verdict.SKIPPED,
            "the run registered no decision file hashes (older run)",
        )
    if len(entries) > 1:
        return verdict.Finding(
            name, verdict.FAIL, "the decision files were registered twice"
        )
    registered = entries[0].get("detail", {}).get("files", {})
    problems = tampered(run_dir, registered)
    if problems:
        return verdict.Finding(name, verdict.FAIL, "; ".join(problems))
    count = sum(1 for v in registered.values() if v is not None)
    return verdict.Finding(
        name, verdict.PASS, f"{count} file(s) hash to the registration"
    )


def coverage_report_problems(
    report_path: Path, claim_hash: str | None
) -> list[str]:
    """Checks that a coverage report names the registered claim.

    Args:
        report_path: Where the coverage report would be.
        claim_hash: The hash of the claim, or None if the claim is unreadable.

    Returns:
        A message if the report exists and is unreadable or names another
        claim; else empty.
    """
    if not report_path.exists():
        return []
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (
        OSError,
        ValueError,
    ) as error:  # handled: returned as a problem naming the error
        return [f"{coverage_run.REPORT_FILE}: unreadable ({error})"]
    if claim_hash is not None and report.get("claim_sha256") != claim_hash:
        return [f"{coverage_run.REPORT_FILE}: names another claim"]
    return []


def corroborate(
    run_dir: Path, signing_key: str | bytes | None
) -> verdict.Finding | None:
    """Checks that the coverage claim hashes to its registration.

    Args:
        run_dir: The run directory.
        signing_key: Key for the custody log MACs, or None.

    Returns:
        A finding, or None when the run holds neither the claim nor a coverage
        report. It is skipped without a custody log, and failing if the log is
        invalid, the claim is unreadable, not registered exactly once, differs
        from the registered hash, or the report does not name the claim's hash.
    """
    name = "coverage claim: registered before the rows were read"
    claim_path = run_dir / coverage_run.CLAIM_FILE
    report_path = run_dir / coverage_run.REPORT_FILE
    if not claim_path.exists() and not report_path.exists():
        return None
    log_path = run_dir / files.CUSTODY_FILE
    problems = []
    try:
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
        found = scmle.assurance.ledger.ledger.digest(claim)
    except (OSError, ValueError, TypeError):
        # handled: returned as failed finding/problem/refusal with error text
        claim, found = None, None
        problems.append(f"{coverage_run.CLAIM_FILE}: missing or unreadable")
    if log_path.exists():
        try:
            log = sealed.Journal(log_path, signing_key)
        except errors.InputError:
            # ERR-20: an unusable signing key (empty) is the caller's input
            # error and propagates, as its ValueError did before the error
            # types were narrowed; it is not a finding about the run.
            raise
        except errors.ScmleError as error:
            # handled: returned as failed finding/problem/refusal with error
            # text
            return verdict.Finding(name, verdict.FAIL, str(error))
        entries = [
            e
            for e in log.entries
            if e.get("event") == coverage_run.REGISTRATION_EVENT
        ]
        if len(entries) != 1:
            problems.append(
                f"{len(entries)} registrations in the custody log"
                + (" (no registration)" if not entries else "")
            )
        elif claim is not None and entries[0]["detail"].get("sha256") != found:
            problems.append(
                f"{coverage_run.CLAIM_FILE}: differs from the registered hash"
            )
    else:
        return verdict.Finding(
            name, verdict.SKIPPED, f"{files.CUSTODY_FILE} is not in the run"
        )
    problems.extend(coverage_report_problems(report_path, found))
    if problems:
        return verdict.Finding(name, verdict.FAIL, "; ".join(problems))
    return verdict.Finding(
        name, verdict.PASS, "claim hash registered; the report names it"
    )
