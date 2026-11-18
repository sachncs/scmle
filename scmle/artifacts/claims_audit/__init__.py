"""Reading and verifying the claims of a charter run (`scmle claims`).

A charter run leaves `hypothesis_ledger.jsonl` (hash-chained, replayable),
`confirmed_claims.json`, `negative_control.json`, `analysis_charter.json` and
`custody_log.jsonl`. This module lists and shows the claims and checks that
the files agree with each other, from the files alone:

* the ledger chain verifies and every recorded decision replays;
* the claims file names the ledger head, the frozen survivor hash, the outcome
  and the confirmed set that the sealed ledger holds;
* the multiplicity rule, replayed on the p-values of `C`, gives the recorded
  adjusted p-values and tiers;
* the withdrawal accounting (family and its hash) is consistent, no
  withdrawn survivor was tested on `C`, and the custody log recorded the
  family hash before `C` opened, and `C` opened at most once;
* the claims registered by hash (the baseline comparison and the primary
  causal hypothesis) are stated by the confirmation row, and the report
  files they produced hash to the values the sealed ledger holds;
* the four decision files (threshold, net benefit, policy value,
  rollout design) hash to what the custody log registered when they were
  written; an edited, removed or added file fails with its name;
* the coverage claim (levels, methods, rule) hashes to what the custody log
  registered before the final rows were released, and the coverage report
  names that hash;
* the coverage claim on `C` (charter format 11) was registered before `C`
  opened (its registration hash is the ledger's), its test recomputes from the
  stored counts and its status is the ledger's;
* a causal claim above `unverified` rests on attestors the charter names:
  every accepted attestation is by a listed attestor for that assumption,
  carries this charter's hash, was valid on the run date and (given the
  signing key) has a signature that verifies;
* the charter file verifies, matches the ledger's charter hash and, when it
  fixes rules, the ledger and the claims used them.

It proves consistency of the artifacts, not that they were produced by the
run they describe: someone who can rewrite every file consistently can pass
it, and the custody log is tamper-evident, not adversary-proof.
Statistics are not recomputed from data, which the run directory does not hold.

This package lists, shows and verifies; the checks live in `seal` (the
claims file and the charter against the sealed ledger), `custody` (the
custody log), `registration` (the claims registered by hash and the negative
control) and `attestation` (the attestors of a causal claim), and `files` and
`verdict` hold what they share.
"""

from pathlib import Path
from typing import Any

from scmle.artifacts.claims_audit.attestation import assess
from scmle.artifacts.claims_audit.attestation import vet
from scmle.artifacts.claims_audit.attestation import vouch
from scmle.artifacts.claims_audit.custody import check_decision_files
from scmle.artifacts.claims_audit.custody import corroborate
from scmle.artifacts.claims_audit.custody import coverage_report_problems
from scmle.artifacts.claims_audit.custody import custodial
from scmle.artifacts.claims_audit.custody import frozen_selection_problem
from scmle.artifacts.claims_audit.custody import openings
from scmle.artifacts.claims_audit.custody import tampered
from scmle.artifacts.claims_audit.files import BASELINE_FILE
from scmle.artifacts.claims_audit.files import CAUSAL_FILE
from scmle.artifacts.claims_audit.files import CHARTER_FILE
from scmle.artifacts.claims_audit.files import CONTROL_FILE
from scmle.artifacts.claims_audit.files import CUSTODY_FILE
from scmle.artifacts.claims_audit.files import migrate
from scmle.artifacts.claims_audit.files import read_optional
from scmle.artifacts.claims_audit.files import recall
from scmle.artifacts.claims_audit.files import rehydrate
from scmle.artifacts.claims_audit.registration import check_registered
from scmle.artifacts.claims_audit.registration import misnamed
from scmle.artifacts.claims_audit.registration import orphaned
from scmle.artifacts.claims_audit.registration import reconcile
from scmle.artifacts.claims_audit.registration import reconfirm
from scmle.artifacts.claims_audit.registration import witness
from scmle.artifacts.claims_audit.seal import authenticate
from scmle.artifacts.claims_audit.seal import check_ledger_head
from scmle.artifacts.claims_audit.seal import compare
from scmle.artifacts.claims_audit.seal import crosscheck
from scmle.artifacts.claims_audit.seal import rederive
from scmle.artifacts.claims_audit.seal import stage_two_record_problems
from scmle.artifacts.claims_audit.seal import tally
from scmle.artifacts.claims_audit.seal import upheld
from scmle.artifacts.claims_audit.verdict import FAIL
from scmle.artifacts.claims_audit.verdict import Finding
from scmle.artifacts.claims_audit.verdict import PASS
from scmle.artifacts.claims_audit.verdict import SKIPPED
from scmle.artifacts.claims_audit.verdict import TOLERANCE
from scmle.artifacts.claims_audit.verdict import Verification
import scmle.assurance.ledger.ledger
from scmle.contracts import errors
from scmle.contracts import qualification

__all__ = [
    "BASELINE_FILE",
    "CAUSAL_FILE",
    "CHARTER_FILE",
    "CONTROL_FILE",
    "CUSTODY_FILE",
    "FAIL",
    "PASS",
    "SKIPPED",
    "TOLERANCE",
    "Finding",
    "Verification",
    "assess",
    "authenticate",
    "check_decision_files",
    "check_ledger_head",
    "check_registered",
    "compare",
    "corroborate",
    "coverage_report_problems",
    "crosscheck",
    "custodial",
    "enrolled",
    "frozen_selection_problem",
    "inspect",
    "inventory",
    "locate",
    "misnamed",
    "openings",
    "orphaned",
    "read_optional",
    "recall",
    "reconcile",
    "reconfirm",
    "rederive",
    "rehydrate",
    "stage_two_record_problems",
    "tally",
    "tampered",
    "upheld",
    "verify",
    "vet",
    "vouch",
    "witness",
]


def locate(
    ledger: scmle.assurance.ledger.ledger.Ledger, claim_id: str
) -> str | None:
    """Finds the feature a claim's hypothesis tests.

    Args:
        ledger: The run's ledger.
        claim_id: The claim identifier.

    Returns:
        The column name, or None if the ledger has no such hypothesis.
    """
    hypothesis = ledger.hypotheses.get(claim_id)
    return None if hypothesis is None else hypothesis.spec.column


def inventory(run_dir: str | Path) -> list[dict[str, Any]]:
    """Lists the claims of a run.

    Args:
        run_dir: The run directory.

    Returns:
        One row per claim: id, feature, tier, adjusted p-value, effect and
        whether it was withdrawn, followed by the claims registered by hash.
    """
    document = recall(run_dir)
    ledger = rehydrate(run_dir)
    stage_two = document.get("second") or {}
    withdrawn = stage_two.get("withdrawn", {})
    listed = [
        {
            "claim_id": claim["identifier"],
            "feature": locate(ledger, claim["identifier"]),
            "tier": claim["tier"],
            "adjusted_p_value": claim["adjusted_p_value"],
            "effect": claim["effect"],
            "withdrawn": claim["identifier"] in withdrawn,
        }
        for claim in (
            migrate(qualification.Claim, c) for c in document["claims"]
        )
    ]
    return listed + enrolled(run_dir, ledger)


def enrolled(
    run_dir: str | Path, ledger: scmle.assurance.ledger.ledger.Ledger
) -> list[dict[str, Any]]:
    """Lists one row per claim registered by hash (baseline, causal primary).

    The tier and identification detail come from the baseline or causal report
    file in the run directory, and the status from the sealed confirmation row.
    A missing or unreadable report file leaves the tier `unverified` and the
    identification None.

    Args:
        run_dir: The run directory.
        ledger: The run's ledger.

    Returns:
        One row per registered claim, with its kind, tier (`unverified` if no
        report states one), status and identification detail.
    """
    folder = Path(run_dir)
    stated = (ledger.confirmation or {}).get("claims", {})
    rows = []
    for claim_id, item in ledger.claims.items():
        record = stated.get(claim_id, {})
        tier: str | None = None
        detail: str | None = None
        if item["kind"] == "baseline":
            report = read_optional(folder / BASELINE_FILE)
            tier = ((report or {}).get("claim") or {}).get("tier")
            detail = (report or {}).get("status")
        elif item["kind"] == "causal_primary":
            report = read_optional(folder / CAUSAL_FILE)
            tier = (report or {}).get("tier")
            detail = (report or {}).get("status")
        rows.append(
            {
                "claim_id": claim_id,
                "feature": None,
                "kind": item["kind"],
                "tier": tier or "unverified",
                "status": record.get("status", "not_stated"),
                "identification": detail,
                "adjusted_p_value": None,
                "effect": None,
                "withdrawn": False,
            }
        )
    return rows


def inspect(run_dir: str | Path, claim_id: str) -> dict[str, Any]:
    """Shows everything the run recorded about one claim.

    Args:
        run_dir: The run directory.
        claim_id: The claim identifier.

    Returns:
        For a claim registered by hash, its registration, status and the run
        outcome. For another claim, the claim record, its hypothesis, the
        exploration result, the check on `C`, any withdrawal reason and the run
        outcome.

    Raises:
        ArtifactError: If the claim is unknown or is not in the ledger.
    """
    document = recall(run_dir)
    ledger = rehydrate(run_dir)
    if claim_id in ledger.claims:
        return {
            "registered": ledger.claims[claim_id],
            "status": (document.get("registered_claims") or {}).get(claim_id),
            "outcome": document["outcome"],
        }
    claims = {
        c["identifier"]: c
        for c in (
            migrate(qualification.Claim, row) for row in document["claims"]
        )
    }
    claim = claims.get(claim_id)
    if claim is None:
        known = ", ".join(claims)
        raise errors.ArtifactError(
            f"no claim {claim_id!r}; the run has: {known or 'none'}"
        )
    if claim_id not in ledger.hypotheses or claim_id not in ledger.results:
        raise errors.ArtifactError(
            f"claim {claim_id!r} is not in the sealed ledger"
        )
    hypothesis = ledger.hypotheses[claim_id]
    result = ledger.results[claim_id]
    stage_two = document.get("second") or {}
    return {
        "claim": claim,
        "hypothesis": hypothesis.spec.to_dict(),
        "exploration": {
            "decision": result.decision,
            "level": str(result.level),
            "check": None if result.check is None else result.check.to_dict(),
        },
        "confirmation_check": next(
            (
                check
                for check in (
                    migrate(qualification.Check, c) for c in document["checks"]
                )
                if check["identifier"] == f"{claim_id}:C"
            ),
            None,
        ),
        "withdrawn": stage_two.get("withdrawn", {}).get(claim_id),
        "outcome": document["outcome"],
    }


def verify(
    run_dir: str | Path, signing_key: str | bytes | None = None
) -> Verification:
    """Checks that the artifacts of a charter run agree with each other.

    Args:
        run_dir: The run directory.
        signing_key: Key for the charter signature, the custody log MACs and
            the attestation signatures, or None.

    Returns:
        Every step with its status; `ok` is false if any step failed. The result
        holds a single failing step if the ledger cannot be loaded, and stops
        after the claims file if that cannot be read.
    """
    folder = Path(run_dir)
    chain = "ledger chain verifies and decisions replay"
    findings: list[Finding] = []
    try:
        ledger = scmle.assurance.ledger.ledger.Ledger.load(
            folder / scmle.assurance.ledger.ledger.FILE_NAME
        )
    except scmle.assurance.ledger.ledger.LedgerError as error:
        # handled: returned as failed finding/problem/refusal with error text
        return Verification(str(folder), (Finding(chain, FAIL, str(error)),))
    findings.append(
        Finding(
            chain, PASS, f"{len(ledger.rows)} rows, head {ledger.head[:12]}"
        )
    )
    try:
        document = recall(folder)
    except errors.ArtifactError as error:
        # handled: returned as failed finding/problem/refusal with error text
        findings.append(Finding("claims file reads", FAIL, str(error)))
        return Verification(str(folder), tuple(findings))
    findings.append(check_ledger_head(document, ledger))
    findings.extend(authenticate(folder, ledger, signing_key))
    findings.append(crosscheck(document, ledger))
    findings.append(rederive(document))
    findings.append(tally(document, ledger))
    findings.append(custodial(folder, document, signing_key))
    if ledger.claims:
        findings.append(check_registered(folder, document, ledger))
    findings.append(witness(folder, ledger))
    findings.append(vouch(folder, signing_key))
    decided = check_decision_files(folder, signing_key)
    if decided is not None:
        findings.append(decided)
    measured = corroborate(folder, signing_key)
    if measured is not None:
        findings.append(measured)
    on_c = reconfirm(folder, ledger)
    if on_c is not None:
        findings.append(on_c)
    return Verification(str(folder), tuple(findings))
