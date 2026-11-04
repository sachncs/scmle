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
from scmle.data.charter import sealing as charter_sealing
from scmle.data.custody import sealed
from scmle.data.ledger import budget as budget_module
from scmle.data.ledger import confirm
from scmle.data.ledger import ledger as ledger_module
from scmle.data.ledger import multiplicity

CHARTER_FILE = "analysis_charter.json"
CUSTODY_FILE = "custody_log.jsonl"
CONTROL_FILE = "negative_control.json"
BASELINE_FILE = "baseline_report.json"
CAUSAL_FILE = "causal_claims.json"
PASS = "pass"
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
    return Verification(str(folder), tuple(findings))
