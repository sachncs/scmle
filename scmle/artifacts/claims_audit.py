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
* the second-stage accounting (family and its hash) is consistent, no
  withdrawn survivor was tested on `C`, and the custody log recorded the
  family hash before `C` opened, and `C` opened at most once;
* the claims registered by hash (the F8 baseline comparison, the F10 primary
  causal hypothesis) are stated by the confirmation row, and the report
  files they produced hash to the values the sealed ledger holds;
* the four decision files of F11 (threshold, net benefit, policy value,
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
it, and the custody log is tamper-evident, not adversary-proof (canon P6).
Statistics are not recomputed from data, which the run directory does not hold.
"""

import dataclasses
from fractions import Fraction
import json
import math
from pathlib import Path
from typing import Any

from scmle.contracts import errors
from scmle.contracts import signing
from scmle.core import decision_reports
from scmle.data.charter import contracts as charter_contracts
from scmle.data.charter import sealing as charter_sealing
from scmle.data.custody import sealed
from scmle.data.identification import attest
from scmle.data.ledger import budget as budget_module
from scmle.data.ledger import confirm
from scmle.data.ledger import ledger as ledger_module
from scmle.data.ledger import multiplicity
from scmle.evaluation import coverage_confirm
from scmle.evaluation import coverage_run

CHARTER_FILE = "analysis_charter.json"
CUSTODY_FILE = "custody_log.jsonl"
CONTROL_FILE = "negative_control.json"
BASELINE_FILE = "baseline_report.json"
CAUSAL_FILE = "causal_claims.json"
PASS = "pass"  # nosec B105
FAIL = "fail"
SKIPPED = "skipped"
TOLERANCE = 1e-12


@dataclasses.dataclass(frozen=True, slots=True)
class Finding:
    """One verification step.

    Attributes:
        name: What was checked.
        status: `pass`, `fail` or `skipped` (the input does not exist or
            does not apply; a skip never counts as a pass).
        detail: One sentence.
    """

    name: str
    status: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        """JSON-ready record."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class Verification:
    """The result of `verify`.

    Attributes:
        run_dir: The directory verified.
        findings: Every step, in order.
    """

    run_dir: str
    findings: tuple[Finding, ...]

    @property
    def ok(self) -> bool:
        """True when no step failed."""
        return all(f.status != FAIL for f in self.findings)

    @property
    def verified(self) -> int:
        """Number of steps that passed."""
        return sum(1 for f in self.findings if f.status == PASS)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return {
            "run_dir": self.run_dir,
            "ok": self.ok,
            "passed": self.verified,
            "findings": [f.to_dict() for f in self.findings],
        }


def read_claims(run_dir: str | Path) -> dict[str, Any]:
    """Reads ``confirmed_claims.json`` of a run.

    Raises:
        ArtifactError: If the file is missing or is not a JSON object.
    """
    path = Path(run_dir) / confirm.CLAIMS_FILE
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise errors.ArtifactError(f"cannot read {path}: {error}") from error
    if not isinstance(document, dict) or "claims" not in document:
        raise errors.ArtifactError(f"{path} is not a claims document")
    return document


def read_ledger(run_dir: str | Path) -> ledger_module.Ledger:
    """Loads the run's ledger (verifying its chain and replaying it).

    Raises:
        ArtifactError: If the ledger is missing or fails verification.
    """
    try:
        return ledger_module.Ledger.load(
            Path(run_dir) / ledger_module.FILE_NAME
        )
    except ledger_module.LedgerError as error:
        raise errors.ArtifactError(str(error)) from error


def column_of(ledger: ledger_module.Ledger, claim_id: str) -> str | None:
    """The feature a claim's hypothesis tests."""
    hypothesis = ledger.hypotheses.get(claim_id)
    return None if hypothesis is None else hypothesis.spec.column


def list_claims(run_dir: str | Path) -> list[dict[str, Any]]:
    """One row per claim: id, feature, tier, adjusted p-value, effect."""
    document = read_claims(run_dir)
    ledger = read_ledger(run_dir)
    stage_two = document.get("stage_two") or {}
    withdrawn = stage_two.get("withdrawn", {})
    listed = [
        {
            "claim_id": claim["claim_id"],
            "feature": column_of(ledger, claim["claim_id"]),
            "tier": claim["tier"],
            "adjusted_p_value": claim["adjusted_p_value"],
            "effect": claim["effect"],
            "withdrawn": claim["claim_id"] in withdrawn,
        }
        for claim in document["claims"]
    ]
    return listed + registered_rows(run_dir, ledger)


def registered_rows(
    run_dir: str | Path, ledger: ledger_module.Ledger
) -> list[dict[str, Any]]:
    """One row per claim registered by hash (baseline, causal primary).

    The tier and status come from the report files the sealed ledger binds;
    a file that is missing is reported as such rather than guessed.
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


def read_optional(path: Path) -> dict[str, Any] | None:
    """A JSON object file, or None when it is missing or unreadable."""
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def show_claim(run_dir: str | Path, claim_id: str) -> dict[str, Any]:
    """Everything the run recorded about one claim.

    Raises:
        ArtifactError: If the claim is unknown.
    """
    document = read_claims(run_dir)
    ledger = read_ledger(run_dir)
    if claim_id in ledger.claims:
        return {
            "registered": ledger.claims[claim_id],
            "status": (document.get("registered_claims") or {}).get(claim_id),
            "outcome": document["outcome"],
        }
    claim = next(
        (c for c in document["claims"] if c["claim_id"] == claim_id), None
    )
    if claim is None:
        known = ", ".join(c["claim_id"] for c in document["claims"])
        raise errors.ArtifactError(
            f"no claim {claim_id!r}; the run has: {known or 'none'}"
        )
    hypothesis = ledger.hypotheses[claim_id]
    result = ledger.results[claim_id]
    stage_two = document.get("stage_two") or {}
    return {
        "claim": claim,
        "hypothesis": hypothesis.spec.to_dict(),
        "exploration": {
            "decision": result.decision,
            "level": str(result.level),
            "check": None if result.check is None else result.check.to_dict(),
        },
        "confirmation_check": next(
            (c for c in document["checks"] if c["check_id"] == f"{claim_id}:C"),
            None,
        ),
        "withdrawn": stage_two.get("withdrawn", {}).get(claim_id),
        "outcome": document["outcome"],
    }


def step(name: str, status: str, detail: str) -> Finding:
    """A finding."""
    return Finding(name, status, detail)


def check_ledger_head(
    document: dict[str, Any], ledger: ledger_module.Ledger
) -> Finding:
    """The claims file names the ledger's current head."""
    name = "claims file names the ledger head"
    if document.get("ledger_head") == ledger.head:
        return step(name, PASS, f"head {ledger.head[:12]}")
    return step(
        name, FAIL, "confirmed_claims.json names a different ledger head"
    )


def check_charter(
    run_dir: Path,
    ledger: ledger_module.Ledger,
    signing_key: str | bytes | None,
) -> list[Finding]:
    """The charter file verifies and is the ledger's charter."""
    path = run_dir / CHARTER_FILE
    name = "charter verifies and matches the ledger"
    if not path.exists():
        return [step(name, SKIPPED, f"{CHARTER_FILE} is not in the run")]
    try:
        loaded = charter_sealing.read_verified(path, signing_key=signing_key)
    except errors.ScmleError as error:
        return [step(name, FAIL, str(error))]
    if loaded.charter_hash != ledger.header.get("charter_hash"):
        return [step(name, FAIL, "the ledger names another charter hash")]
    how = "signature verified" if loaded.signed else "hash only, not signed"
    findings = [step(name, PASS, f"{loaded.charter_hash[:12]} ({how})")]
    rules = loaded.charter.rules
    rule_step = "chartered rules were used"
    if rules is None:
        findings.append(step(rule_step, SKIPPED, "the charter fixes no rules"))
    else:
        used = ledger.header["rule"]["rule"]
        confirmed_with = (ledger.confirmation or {}).get("rule")
        if used != rules.ledger_rule or confirmed_with not in (
            None,
            rules.confirmation_rule,
        ):
            findings.append(
                step(
                    rule_step,
                    FAIL,
                    f"ledger rule {used!r} / confirmation rule "
                    f"{confirmed_with!r} differ from the charter's "
                    f"{rules.ledger_rule!r} / {rules.confirmation_rule!r}",
                )
            )
        else:
            findings.append(
                step(rule_step, PASS, f"{used} and {rules.confirmation_rule}")
            )
    return findings


def check_claims_against_ledger(
    document: dict[str, Any], ledger: ledger_module.Ledger
) -> Finding:
    """Survivors, outcome and the confirmed set agree with the sealed ledger."""
    name = "claims agree with the sealed ledger"
    sealed_row = ledger.confirmation
    survivors = ledger.survivor_set
    if sealed_row is None or survivors is None:
        return step(name, FAIL, "the ledger is not sealed")
    problems = []
    if document.get("survivors_sha256") != survivors.sha256:
        problems.append("survivor hash")
    if document.get("outcome") != sealed_row.get("outcome"):
        problems.append("outcome")
    if document.get("rule") != sealed_row.get("rule"):
        problems.append("rule")
    if document.get("alpha_confirm") != sealed_row.get("alpha_confirm"):
        problems.append("alpha_confirm")
    ids = [c["claim_id"] for c in document["claims"]]
    if document.get("outcome") == confirm.NO_SURVIVORS:
        ok_ids = ids == []
    else:
        ok_ids = ids == list(survivors.hypothesis_ids)
    if not ok_ids:
        problems.append("claim ids differ from the frozen survivors")
    confirmed = [c["claim_id"] for c in document["claims"] if is_confirmed(c)]
    if confirmed != list(sealed_row.get("confirmed", [])):
        problems.append("confirmed set")
    if document.get("stage_two") != sealed_row.get("stage_two"):
        problems.append("second-stage accounting")
    if problems:
        return step(
            name, FAIL, "differs from the ledger: " + ", ".join(problems)
        )
    return step(name, PASS, f"{len(confirmed)} confirmed of {len(ids)} claims")


def is_confirmed(claim: dict[str, Any]) -> bool:
    """Whether a claim record has tier `confirmed`."""
    return claim.get("tier") == "confirmed"


def check_replay(document: dict[str, Any]) -> Finding:
    """The multiplicity rule, replayed on the p-values of `C`."""
    name = "confirmation replays from the p-values on C"
    if document.get("outcome") != confirm.CONFIRMED:
        return step(name, SKIPPED, f"outcome is {document.get('outcome')!r}")
    by_check = {c["check_id"]: c for c in document["checks"]}
    tested = [c for c in document["claims"] if f"{c['claim_id']}:C" in by_check]
    try:
        values = [by_check[f"{c['claim_id']}:C"]["p_value"] for c in tested]
        adjustment = multiplicity.adjust(
            values,
            budget_module.MultiplicityRule(document["rule"]),
            Fraction(document["alpha_confirm"]),
        )
    except (ValueError, KeyError) as error:
        return step(name, FAIL, f"cannot replay: {error}")
    for index, claim in enumerate(tested):
        recorded = claim["adjusted_p_value"]
        if recorded is None or not math.isclose(
            recorded, adjustment.adjusted[index], abs_tol=TOLERANCE
        ):
            return step(name, FAIL, f"{claim['claim_id']}: adjusted p differs")
        if is_confirmed(claim) != adjustment.rejected[index]:
            return step(name, FAIL, f"{claim['claim_id']}: tier differs")
    return step(
        name, PASS, f"{len(tested)} hypotheses, rule {document['rule']}"
    )


def check_stage_two(
    document: dict[str, Any], ledger: ledger_module.Ledger
) -> Finding:
    """The family accounting is consistent and withdrawn ids were not tested."""
    name = "second-stage accounting is consistent"
    record = document.get("stage_two")
    if record is None:
        return step(name, SKIPPED, "no withdrawal rule ran")
    survivors = ledger.survivor_set
    if survivors is None:
        return step(name, FAIL, "the ledger has no frozen survivors")
    withdrawn = set(record["withdrawn"])
    family = [i for i in survivors.hypothesis_ids if i not in withdrawn]
    expected = ledger_module.digest(
        {"survivors_sha256": survivors.sha256, "family": family}
    )
    problems = []
    if not withdrawn <= set(survivors.hypothesis_ids):
        problems.append("a withdrawn id is not a survivor")
    if record["family"] != family:
        problems.append("family is not survivors minus withdrawn")
    if record["family_sha256"] != expected:
        problems.append("family hash")
    if record["survivors_sha256"] != survivors.sha256:
        problems.append("survivor hash")
    tested = {c["check_id"].split(":")[0] for c in document["checks"]}
    if tested & withdrawn:
        problems.append("a withdrawn survivor was tested on C")
    if tested - set(family):
        problems.append("a check on C is outside the family")
    for claim in document["claims"]:
        if claim["claim_id"] in withdrawn and is_confirmed(claim):
            problems.append("a withdrawn survivor is confirmed")
    if problems:
        return step(name, FAIL, "; ".join(problems))
    return step(
        name,
        PASS,
        f"{len(family)} tested, {len(withdrawn)} withdrawn before C opened",
    )


def check_custody(
    run_dir: Path,
    document: dict[str, Any],
    signing_key: str | bytes | None,
) -> Finding:
    """The custody log verifies, froze the selection first and opened C once."""
    name = "custody log: selection frozen before C opened, C opened once"
    path = run_dir / CUSTODY_FILE
    if not path.exists():
        return step(name, SKIPPED, f"{CUSTODY_FILE} is not in the run")
    try:
        log = sealed.OpenLog(path, signing_key)
    except errors.ScmleError as error:
        return step(name, FAIL, str(error))
    entries = [e for e in log.entries if e.get("set") == "C"]
    names = [e["event"] for e in entries]
    opened = names.count("open")
    if opened > 1:
        return step(name, FAIL, f"C was opened {opened} times")
    if document.get("outcome") == confirm.CONFIRMED and opened != 1:
        return step(name, FAIL, "outcome is confirmed but C has no open event")
    if opened == 0 and document.get("outcome") != confirm.CONFIRMED:
        return step(name, PASS, f"C not opened (outcome {document['outcome']})")
    frozen = [e for e in entries if e["event"] == "selection_frozen"]
    if len(frozen) != 1 or names.index("selection_frozen") > names.index(
        "open"
    ):
        return step(name, FAIL, "no single selection_frozen before the open")
    detail = frozen[0].get("detail", {})
    if detail.get("survivors_sha256") != document.get("survivors_sha256"):
        return step(name, FAIL, "frozen survivor hash differs from the claims")
    record = document.get("stage_two")
    if (
        record is not None
        and detail.get("family_sha256") != record["family_sha256"]
    ):
        return step(name, FAIL, "frozen family hash differs from the claims")
    return step(name, PASS, f"{len(log.entries)} entries, chain verified")


def check_decision_files(
    run_dir: Path, signing_key: str | bytes | None
) -> Finding | None:
    """The four decision files hash to what the custody log registered.

    None when the run registered no decision files and holds none.
    """
    name = "decision files: hashes match the custody log"
    path = run_dir / CUSTODY_FILE
    entries: list[dict[str, Any]] = []
    if path.exists():
        try:
            log = sealed.OpenLog(path, signing_key)
        except errors.ScmleError as error:
            return step(name, FAIL, str(error))
        entries = [
            e
            for e in log.entries
            if e.get("event") == decision_reports.REGISTRATION_EVENT
        ]
    present = [
        n for n in decision_reports.DECISION_FILES if (run_dir / n).exists()
    ]
    if not entries:
        if not present:
            return None
        return step(
            name,
            SKIPPED,
            "the run registered no decision file hashes (older run)",
        )
    if len(entries) > 1:
        return step(name, FAIL, "the decision files were registered twice")
    registered = entries[0].get("detail", {}).get("files", {})
    problems = []
    for file_name in decision_reports.DECISION_FILES:
        expected = registered.get(file_name)
        try:
            found = decision_reports.file_digest(run_dir / file_name)
        except ValueError as error:
            problems.append(f"{file_name}: unreadable ({error})")
            continue
        if expected is None and found is not None:
            problems.append(f"{file_name}: not registered")
        elif expected is not None and found is None:
            problems.append(f"{file_name}: missing")
        elif found != expected:
            problems.append(f"{file_name}: differs from the registered hash")
    if problems:
        return step(name, FAIL, "; ".join(problems))
    count = sum(1 for v in registered.values() if v is not None)
    return step(name, PASS, f"{count} file(s) hash to the registration")


def check_coverage_claim(
    run_dir: Path, signing_key: str | bytes | None
) -> Finding | None:
    """The coverage claim hashes to its registration and the report names it.

    None when the run holds neither the claim nor a coverage report.
    """
    name = "coverage claim: registered before the rows were read"
    claim_path = run_dir / coverage_run.CLAIM_FILE
    report_path = run_dir / coverage_run.REPORT_FILE
    if not claim_path.exists() and not report_path.exists():
        return None
    log_path = run_dir / CUSTODY_FILE
    problems = []
    try:
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
        found = coverage_run.claim_hash(claim)
    except (OSError, ValueError, TypeError):
        claim, found = None, None
        problems.append(f"{coverage_run.CLAIM_FILE}: missing or unreadable")
    if log_path.exists():
        try:
            log = sealed.OpenLog(log_path, signing_key)
        except errors.ScmleError as error:
            return step(name, FAIL, str(error))
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
        return step(name, SKIPPED, f"{CUSTODY_FILE} is not in the run")
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            report = {}
        if found is not None and report.get("claim_sha256") != found:
            problems.append(f"{coverage_run.REPORT_FILE}: names another claim")
    if problems:
        return step(name, FAIL, "; ".join(problems))
    return step(name, PASS, "claim hash registered; the report names it")


def check_coverage_confirmation(
    run_dir: Path, ledger: ledger_module.Ledger
) -> Finding | None:
    """The coverage claim on `C`: registered, recomputed and consistent.

    None when the run holds no `coverage_confirmation.json`.
    """
    name = "coverage claim on C: registered, recomputed, status matches"
    document = read_optional(run_dir / coverage_confirm.DOCUMENT)
    if document is None:
        return None
    registered = ledger.claims.get(coverage_confirm.CLAIM_ID)
    if document.get("status") == confirm.NOT_EVALUATED:
        if registered is not None:
            return step(
                name, FAIL, "a registered claim has no evaluation document"
            )
        return step(name, SKIPPED, "the claim could not be tested on C")
    if registered is None:
        return step(name, FAIL, "the claim is not registered in the ledger")
    problems = []
    claim = read_optional(run_dir / coverage_run.CLAIM_FILE)
    try:
        claim_hash = coverage_run.claim_hash(claim or {})
        choice = document["primary"]
        body = coverage_confirm.registration_body(claim_hash, choice)
        if document.get("coverage_claim_sha256") != claim_hash:
            problems.append("the document names another coverage claim")
        if ledger_module.digest(body) != registered["sha256"]:
            problems.append("the registration hash differs from the ledger's")
        redone = coverage_confirm.decide(
            int(document["covered"]), int(document["rows"]), choice
        )
    except (KeyError, TypeError, ValueError):
        return step(name, FAIL, "the document is malformed")
    if abs(redone["p_value"] - float(document["p_value"])) > TOLERANCE:
        problems.append("the p-value does not recompute from the counts")
    if redone["status"] != document.get("status"):
        problems.append("the status does not follow from the test")
    sealed_row = (ledger.confirmation or {}).get("claims", {})
    stated = (sealed_row.get(coverage_confirm.CLAIM_ID) or {}).get("status")
    if stated != document.get("status"):
        problems.append("the status differs from the sealed ledger row")
    if problems:
        return step(name, FAIL, "; ".join(problems))
    return step(
        name,
        PASS,
        f"{document['covered']} of {document['rows']} covered; "
        f"status {document['status']}",
    )


def check_registered(
    run_dir: Path, document: dict[str, Any], ledger: ledger_module.Ledger
) -> Finding:
    """Registered claims are stated at the seal and their files match it."""
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
        found = read_optional(run_dir / file_name)
        if found is None:
            problems.append(f"{file_name}: missing")
        elif ledger_module.digest(found) != expected:
            problems.append(f"{file_name}: differs from the sealed hash")
    if document.get("attachments") != attachments:
        problems.append("attachment hashes differ from the claims file")
    baseline_doc = read_optional(run_dir / BASELINE_FILE)
    for claim_id, item in ledger.claims.items():
        if (
            item["kind"] == "baseline"
            and baseline_doc is not None
            and baseline_doc.get("registration_hash") != item["sha256"]
        ):
            problems.append(f"{claim_id}: report names another registration")
    if problems:
        return step(name, FAIL, "; ".join(problems))
    return step(
        name,
        PASS,
        f"{len(ledger.claims)} claim(s), {len(attachments)} file(s) bound",
    )


def check_control(run_dir: Path, ledger: ledger_module.Ledger) -> Finding:
    """The negative-control file is the one the ledger recorded."""
    name = "negative control matches the ledger"
    path = run_dir / CONTROL_FILE
    if not path.exists() or ledger.control is None:
        return step(name, SKIPPED, "no control file or no control row")
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return step(name, FAIL, f"{CONTROL_FILE} is not JSON")
    if ledger_module.digest(body) == ledger.control.get("sha256"):
        return step(name, PASS, "hash equal")
    return step(name, FAIL, "the file differs from the ledger's control row")


def check_attestations(
    run_dir: Path, signing_key: str | bytes | None
) -> Finding:
    """A claim above `unverified` rests on verified, charter-named attestors."""
    name = "causal attestations verify"
    claims = read_optional(run_dir / CAUSAL_FILE)
    record = read_optional(run_dir / CHARTER_FILE)
    if claims is None or record is None:
        return step(name, SKIPPED, "no causal claims or charter file")
    try:
        charter = charter_contracts.charter_from_record(
            record.get("charter", record)
        )
    except errors.ConfigError as error:
        return step(name, FAIL, str(error))
    block = charter.estimand.causal
    if block is None:
        return step(name, SKIPPED, "the charter has no causal block")
    report = claims.get("attestations") or {}
    raised = claims.get("tier") in ("attested", "confirmed")
    problems = []
    if raised and not block.attestors:
        problems.append(
            "tier above unverified but the charter names no attestors"
        )
    allowed = {a.name: set(a.assumptions) for a in block.attestors}
    accepted = {}
    unchecked = False
    for item in report.get("accepted", []):
        try:
            given = attest.Attestation.from_dict(item)
        except (ValueError, TypeError) as error:
            problems.append(f"malformed attestation: {error}")
            continue
        accepted[given.assumption] = given.attestor
        if given.assumption not in allowed.get(given.attestor, set()):
            problems.append(
                f"{given.assumption}: {given.attestor} not permitted"
            )
        if given.scope != charter.charter_hash:
            problems.append(f"{given.assumption}: scope is not this charter")
        day = str(report.get("evaluated_on", ""))
        try:
            valid = given.date <= day <= given.expires
        except TypeError:
            valid = False
        if not valid:
            problems.append(f"{given.assumption}: not valid on {day!r}")
        if signing_key is None and not signing.resolve_key(None):
            unchecked = True
        elif not given.verify(signing_key):
            problems.append(f"{given.assumption}: signature does not verify")
    for assumption, named in (claims.get("assumptions") or {}).items():
        if named and accepted.get(assumption) != named:
            problems.append(
                f"{assumption}: {named} has no accepted attestation"
            )
    if raised and not accepted:
        problems.append("tier above unverified with no accepted attestation")
    if problems:
        return step(name, FAIL, "; ".join(problems))
    if unchecked:
        return step(name, SKIPPED, "no signing key to check the signatures")
    return step(
        name,
        PASS,
        f"{len(accepted)} attestation(s), none required"
        if not accepted
        else f"{len(accepted)} attestation(s) verified",
    )


def verify(
    run_dir: str | Path, signing_key: str | bytes | None = None
) -> Verification:
    """Checks that the artifacts of a charter run agree with each other.

    Args:
        run_dir: The run directory.
        signing_key: Key for the charter signature and the custody log MACs.

    Returns:
        Every step with its status; `ok` is false if any step failed.
    """
    folder = Path(run_dir)
    chain = "ledger chain verifies and decisions replay"
    findings: list[Finding] = []
    try:
        ledger = ledger_module.Ledger.load(folder / ledger_module.FILE_NAME)
    except ledger_module.LedgerError as error:
        return Verification(str(folder), (step(chain, FAIL, str(error)),))
    findings.append(
        step(chain, PASS, f"{len(ledger.rows)} rows, head {ledger.head[:12]}")
    )
    try:
        document = read_claims(folder)
    except errors.ArtifactError as error:
        findings.append(step("claims file reads", FAIL, str(error)))
        return Verification(str(folder), tuple(findings))
    findings.append(check_ledger_head(document, ledger))
    findings.extend(check_charter(folder, ledger, signing_key))
    findings.append(check_claims_against_ledger(document, ledger))
    findings.append(check_replay(document))
    findings.append(check_stage_two(document, ledger))
    findings.append(check_custody(folder, document, signing_key))
    if ledger.claims:
        findings.append(check_registered(folder, document, ledger))
    findings.append(check_control(folder, ledger))
    findings.append(check_attestations(folder, signing_key))
    decided = check_decision_files(folder, signing_key)
    if decided is not None:
        findings.append(decided)
    measured = check_coverage_claim(folder, signing_key)
    if measured is not None:
        findings.append(measured)
    on_c = check_coverage_confirmation(folder, ledger)
    if on_c is not None:
        findings.append(on_c)
    return Verification(str(folder), tuple(findings))
