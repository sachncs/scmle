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
from scmle.data.charter import contracts
from scmle.data.charter import sealing
from scmle.data.custody import sealed
from scmle.data.identification import attest
from scmle.data.ledger import budget
from scmle.data.ledger import confirm
from scmle.data.ledger import multiplicity
import scmle.data.ledger.ledger
from scmle.evaluation import coverage_confirm
from scmle.evaluation import coverage_run

CHARTER_FILE = "analysis_charter.json"
CUSTODY_FILE = "custody_log.jsonl"
CONTROL_FILE = "negative_control.json"
BASELINE_FILE = "baseline_report.json"
CAUSAL_FILE = "causal_claims.json"
PASS = "pass"  # noqa: S105 -- a verdict label, not a credential # nosec B105
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
        """Builds a JSON-ready record.

        Returns:
            A dict with `name`, `status` and `detail`.
        """
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
        """The number of steps that passed."""
        return sum(1 for f in self.findings if f.status == PASS)

    def to_dict(self) -> dict[str, Any]:
        """Builds a JSON-ready record.

        Returns:
            A dict with the directory, `ok`, the pass count and the findings.
        """
        return {
            "run_dir": self.run_dir,
            "ok": self.ok,
            "passed": self.verified,
            "findings": [f.to_dict() for f in self.findings],
        }


def read_claims(run_dir: str | Path) -> dict[str, Any]:
    """Reads ``confirmed_claims.json`` of a run.

    Args:
        run_dir: The run directory.

    Returns:
        The parsed claims document.

    Raises:
        ArtifactError: If the file is missing or unreadable, or is not a JSON
            object with a `claims` key.
    """
    path = Path(run_dir) / confirm.CLAIMS_FILE
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise errors.ArtifactError(f"cannot read {path}: {error}") from error
    if not isinstance(document, dict) or "claims" not in document:
        raise errors.ArtifactError(f"{path} is not a claims document")
    return document


def read_ledger(run_dir: str | Path) -> scmle.data.ledger.ledger.Ledger:
    """Loads the run's ledger (verifying its chain and replaying it).

    Args:
        run_dir: The run directory.

    Returns:
        The loaded ledger.

    Raises:
        ArtifactError: If the ledger is missing or fails verification.
    """
    try:
        return scmle.data.ledger.ledger.Ledger.load(
            Path(run_dir) / scmle.data.ledger.ledger.FILE_NAME
        )
    except scmle.data.ledger.ledger.LedgerError as error:
        raise errors.ArtifactError(str(error)) from error


def column_of(
    ledger: scmle.data.ledger.ledger.Ledger, claim_id: str
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


def list_claims(run_dir: str | Path) -> list[dict[str, Any]]:
    """Lists the claims of a run.

    Args:
        run_dir: The run directory.

    Returns:
        One row per claim: id, feature, tier, adjusted p-value, effect and
        whether it was withdrawn, followed by the claims registered by hash.
    """
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
    run_dir: str | Path, ledger: scmle.data.ledger.ledger.Ledger
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


def read_optional(path: Path) -> dict[str, Any] | None:
    """Reads a JSON object file.

    Args:
        path: The file.

    Returns:
        The object, or None when the file is missing, unreadable or not a JSON
        object.
    """
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def show_claim(run_dir: str | Path, claim_id: str) -> dict[str, Any]:
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
    if claim_id not in ledger.hypotheses or claim_id not in ledger.results:
        raise errors.ArtifactError(
            f"claim {claim_id!r} is not in the sealed ledger"
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


def check_ledger_head(
    document: dict[str, Any], ledger: scmle.data.ledger.ledger.Ledger
) -> Finding:
    """Checks that the claims file names the ledger's current head.

    Args:
        document: The claims document.
        ledger: The run's ledger.

    Returns:
        A passing finding if the heads agree, otherwise a failing one.
    """
    name = "claims file names the ledger head"
    if document.get("ledger_head") == ledger.head:
        return Finding(name, PASS, f"head {ledger.head[:12]}")
    return Finding(
        name, FAIL, "confirmed_claims.json names a different ledger head"
    )


def check_charter(
    run_dir: Path,
    ledger: scmle.data.ledger.ledger.Ledger,
    signing_key: str | bytes | None,
) -> list[Finding]:
    """Checks that the charter file verifies and is the ledger's charter.

    Args:
        run_dir: The run directory.
        ledger: The run's ledger.
        signing_key: Key for the charter signature, or None.

    Returns:
        A single finding for the charter if the file is absent (skipped), does
        not verify or names another hash (failing). Otherwise two findings: the
        passing charter finding, then one for the chartered rules (skipped if
        the charter fixes none, failing if the ledger or confirmation used
        other rules, else passing).
    """
    path = run_dir / CHARTER_FILE
    name = "charter verifies and matches the ledger"
    if not path.exists():
        return [Finding(name, SKIPPED, f"{CHARTER_FILE} is not in the run")]
    try:
        loaded = sealing.read_verified(path, signing_key=signing_key)
    except errors.ScmleError as error:
        return [Finding(name, FAIL, str(error))]
    if loaded.charter_hash != ledger.header.get("charter_hash"):
        return [Finding(name, FAIL, "the ledger names another charter hash")]
    how = "signature verified" if loaded.signed else "hash only, not signed"
    findings = [Finding(name, PASS, f"{loaded.charter_hash[:12]} ({how})")]
    rules = loaded.charter.rules
    rule_step = "chartered rules were used"
    if rules is None:
        findings.append(
            Finding(rule_step, SKIPPED, "the charter fixes no rules")
        )
    else:
        used = ledger.header["rule"]["rule"]
        confirmed_with = (ledger.confirmation or {}).get("rule")
        if used != rules.ledger_rule or confirmed_with not in (
            None,
            rules.confirmation_rule,
        ):
            findings.append(
                Finding(
                    rule_step,
                    FAIL,
                    f"ledger rule {used!r} / confirmation rule "
                    f"{confirmed_with!r} differ from the charter's "
                    f"{rules.ledger_rule!r} / {rules.confirmation_rule!r}",
                )
            )
        else:
            findings.append(
                Finding(
                    rule_step, PASS, f"{used} and {rules.confirmation_rule}"
                )
            )
    return findings


def ledger_header_mismatches(
    document: dict[str, Any],
    sealed_row: dict[str, Any],
    survivors: Any,
) -> list[str]:
    """Names the header fields of a claims document that differ from the seal.

    Args:
        document: The claims document.
        sealed_row: The ledger's confirmation row.
        survivors: The ledger's frozen survivor set.

    Returns:
        The names of the differing fields, in a fixed order; empty if none.
    """
    problems = []
    if document.get("survivors_sha256") != survivors.sha256:
        problems.append("survivor hash")
    if document.get("outcome") != sealed_row.get("outcome"):
        problems.append("outcome")
    if document.get("rule") != sealed_row.get("rule"):
        problems.append("rule")
    if document.get("alpha_confirm") != sealed_row.get("alpha_confirm"):
        problems.append("alpha_confirm")
    return problems


def check_claims_against_ledger(
    document: dict[str, Any], ledger: scmle.data.ledger.ledger.Ledger
) -> Finding:
    """Checks that survivors, outcome and confirmed set agree with the ledger.

    Args:
        document: The claims document.
        ledger: The run's ledger.

    Returns:
        A passing finding if the claims file agrees with the sealed ledger;
        otherwise a failing one listing what differs (also failing if the ledger
        is not sealed).
    """
    name = "claims agree with the sealed ledger"
    sealed_row = ledger.confirmation
    survivors = ledger.survivor_set
    if sealed_row is None or survivors is None:
        return Finding(name, FAIL, "the ledger is not sealed")
    problems = ledger_header_mismatches(document, sealed_row, survivors)
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
        return Finding(
            name, FAIL, "differs from the ledger: " + ", ".join(problems)
        )
    return Finding(
        name, PASS, f"{len(confirmed)} confirmed of {len(ids)} claims"
    )


def is_confirmed(claim: dict[str, Any]) -> bool:
    """Reports whether a claim record has tier `confirmed`.

    Args:
        claim: A claim record.

    Returns:
        True if its tier is `confirmed`.
    """
    return claim.get("tier") == "confirmed"


def check_replay(document: dict[str, Any]) -> Finding:
    """Replays the multiplicity rule on the p-values of `C`.

    Args:
        document: The claims document.

    Returns:
        A passing finding if the recorded adjusted p-values and tiers are
        reproduced; skipped if the outcome is not `confirmed`; failing if they
        differ or the replay is impossible.
    """
    name = "confirmation replays from the p-values on C"
    if document.get("outcome") != confirm.CONFIRMED:
        return Finding(name, SKIPPED, f"outcome is {document.get('outcome')!r}")
    try:
        by_check = {c["check_id"]: c for c in document["checks"]}
        tested = [
            c for c in document["claims"] if f"{c['claim_id']}:C" in by_check
        ]
        values = [by_check[f"{c['claim_id']}:C"]["p_value"] for c in tested]
        adjustment = multiplicity.adjust(
            values,
            budget.Correction(document["rule"]),
            Fraction(document["alpha_confirm"]),
        )
        for index, claim in enumerate(tested):
            recorded = claim["adjusted_p_value"]
            if recorded is None or not math.isclose(
                recorded, adjustment.adjusted[index], abs_tol=TOLERANCE
            ):
                return Finding(
                    name, FAIL, f"{claim['claim_id']}: adjusted p differs"
                )
            if is_confirmed(claim) != adjustment.rejected[index]:
                return Finding(name, FAIL, f"{claim['claim_id']}: tier differs")
    except (
        ValueError,
        KeyError,
        TypeError,
        ZeroDivisionError,
    ) as error:
        return Finding(name, FAIL, f"cannot replay: {error!r}")
    return Finding(
        name, PASS, f"{len(tested)} hypotheses, rule {document['rule']}"
    )


def stage_two_record_problems(
    record: dict[str, Any],
    survivors: Any,
    withdrawn: set[str],
    family: list[str],
    expected: str,
) -> list[str]:
    """Checks a second-stage record against the frozen survivors.

    Args:
        record: The claims document's `stage_two` record.
        survivors: The ledger's frozen survivor set.
        withdrawn: The withdrawn hypothesis ids.
        family: The survivors that remain after the withdrawal.
        expected: The digest the family and survivors must hash to.

    Returns:
        What is inconsistent, in a fixed order; empty if nothing is.
    """
    problems = []
    if not withdrawn <= set(survivors.hypothesis_ids):
        problems.append("a withdrawn id is not a survivor")
    if record["family"] != family:
        problems.append("family is not survivors minus withdrawn")
    if record["family_sha256"] != expected:
        problems.append("family hash")
    if record["survivors_sha256"] != survivors.sha256:
        problems.append("survivor hash")
    return problems


def check_stage_two(
    document: dict[str, Any], ledger: scmle.data.ledger.ledger.Ledger
) -> Finding:
    """Checks the withdrawal accounting and that withdrawn ids were not tested.

    Args:
        document: The claims document.
        ledger: The run's ledger.

    Returns:
        A finding: skipped if no withdrawal rule ran, passing if the family, its
        hash and the checks on `C` are consistent with the frozen survivors and
        no withdrawn survivor is confirmed, otherwise failing with the
        problems (also failing if the ledger has no frozen survivors).
    """
    name = "second-stage accounting is consistent"
    record = document.get("stage_two")
    if record is None:
        return Finding(name, SKIPPED, "no withdrawal rule ran")
    survivors = ledger.survivor_set
    if survivors is None:
        return Finding(name, FAIL, "the ledger has no frozen survivors")
    withdrawn = set(record["withdrawn"])
    family = [i for i in survivors.hypothesis_ids if i not in withdrawn]
    expected = scmle.data.ledger.ledger.digest(
        {"survivors_sha256": survivors.sha256, "family": family}
    )
    problems = stage_two_record_problems(
        record, survivors, withdrawn, family, expected
    )
    tested = {c["check_id"].split(":")[0] for c in document["checks"]}
    if tested & withdrawn:
        problems.append("a withdrawn survivor was tested on C")
    if tested - set(family):
        problems.append("a check on C is outside the family")
    problems.extend(
        "a withdrawn survivor is confirmed"
        for claim in document["claims"]
        if claim["claim_id"] in withdrawn and is_confirmed(claim)
    )
    if problems:
        return Finding(name, FAIL, "; ".join(problems))
    return Finding(
        name,
        PASS,
        f"{len(family)} tested, {len(withdrawn)} withdrawn before C opened",
    )


def custody_open_finding(
    name: str, document: dict[str, Any], opened: int
) -> Finding | None:
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
        return Finding(name, FAIL, f"C was opened {opened} times")
    if document.get("outcome") == confirm.CONFIRMED and opened != 1:
        return Finding(
            name, FAIL, "outcome is confirmed but C has no open event"
        )
    if opened == 0 and document.get("outcome") != confirm.CONFIRMED:
        return Finding(
            name, PASS, f"C not opened (outcome {document['outcome']})"
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


def check_custody(
    run_dir: Path,
    document: dict[str, Any],
    signing_key: str | bytes | None,
) -> Finding:
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
    path = run_dir / CUSTODY_FILE
    if not path.exists():
        return Finding(name, SKIPPED, f"{CUSTODY_FILE} is not in the run")
    try:
        log = sealed.OpenLog(path, signing_key)
    except errors.ScmleError as error:
        return Finding(name, FAIL, str(error))
    entries = [e for e in log.entries if e.get("set") == "C"]
    names = [e["event"] for e in entries]
    opened = names.count("open")
    early = custody_open_finding(name, document, opened)
    if early is not None:
        return early
    problem = frozen_selection_problem(entries, names, document)
    if problem is not None:
        return Finding(name, FAIL, problem)
    return Finding(name, PASS, f"{len(log.entries)} entries, chain verified")


def decision_file_problems(
    run_dir: Path, registered: dict[str, Any]
) -> list[str]:
    """Compares each decision file with the hash the custody log registered.

    Args:
        run_dir: The run directory.
        registered: Registered file digests by file name.

    Returns:
        One message per file that is unreadable, unregistered, missing or
        changed; empty if all match.
    """
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
    return problems


def check_decision_files(
    run_dir: Path, signing_key: str | bytes | None
) -> Finding | None:
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
    path = run_dir / CUSTODY_FILE
    entries: list[dict[str, Any]] = []
    if path.exists():
        try:
            log = sealed.OpenLog(path, signing_key)
        except errors.ScmleError as error:
            return Finding(name, FAIL, str(error))
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
        return Finding(
            name,
            SKIPPED,
            "the run registered no decision file hashes (older run)",
        )
    if len(entries) > 1:
        return Finding(name, FAIL, "the decision files were registered twice")
    registered = entries[0].get("detail", {}).get("files", {})
    problems = decision_file_problems(run_dir, registered)
    if problems:
        return Finding(name, FAIL, "; ".join(problems))
    count = sum(1 for v in registered.values() if v is not None)
    return Finding(name, PASS, f"{count} file(s) hash to the registration")


def coverage_report_problems(
    report_path: Path, claim_hash: str | None
) -> list[str]:
    """Checks that a coverage report names the registered claim.

    Args:
        report_path: Where the coverage report would be.
        claim_hash: The hash of the claim, or None if the claim is unreadable.

    Returns:
        A message if the report exists and names another claim; else empty.
    """
    if not report_path.exists():
        return []
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report = {}
    if claim_hash is not None and report.get("claim_sha256") != claim_hash:
        return [f"{coverage_run.REPORT_FILE}: names another claim"]
    return []


def check_coverage_claim(
    run_dir: Path, signing_key: str | bytes | None
) -> Finding | None:
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
            return Finding(name, FAIL, str(error))
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
        return Finding(name, SKIPPED, f"{CUSTODY_FILE} is not in the run")
    problems.extend(coverage_report_problems(report_path, found))
    if problems:
        return Finding(name, FAIL, "; ".join(problems))
    return Finding(name, PASS, "claim hash registered; the report names it")


def unregistered_coverage_finding(
    name: str, document: dict[str, Any], registered: dict[str, Any] | None
) -> Finding | None:
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
            return Finding(
                name, FAIL, "a registered claim has no evaluation document"
            )
        return Finding(name, SKIPPED, "the claim could not be tested on C")
    if registered is None:
        return Finding(name, FAIL, "the claim is not registered in the ledger")
    return None


def coverage_status_problems(
    document: dict[str, Any],
    redone: dict[str, Any],
    ledger: scmle.data.ledger.ledger.Ledger,
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


def check_coverage_confirmation(
    run_dir: Path, ledger: scmle.data.ledger.ledger.Ledger
) -> Finding | None:
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
    document = read_optional(run_dir / coverage_confirm.DOCUMENT)
    if document is None:
        return None
    registered = ledger.claims.get(coverage_confirm.CLAIM_ID)
    early = unregistered_coverage_finding(name, document, registered)
    if early is not None:
        return early
    problems = []
    claim = read_optional(run_dir / coverage_run.CLAIM_FILE)
    try:
        claim_hash = coverage_run.claim_hash(claim or {})
        choice = document["primary"]
        body = coverage_confirm.registration_body(claim_hash, choice)
        if document.get("coverage_claim_sha256") != claim_hash:
            problems.append("the document names another coverage claim")
        if scmle.data.ledger.ledger.digest(body) != registered["sha256"]:
            problems.append("the registration hash differs from the ledger's")
        redone = coverage_confirm.decide(
            int(document["covered"]), int(document["rows"]), choice
        )
    except (KeyError, TypeError, ValueError):
        return Finding(name, FAIL, "the document is malformed")
    if abs(redone["p_value"] - float(document["p_value"])) > TOLERANCE:
        problems.append("the p-value does not recompute from the counts")
    problems.extend(coverage_status_problems(document, redone, ledger))
    if problems:
        return Finding(name, FAIL, "; ".join(problems))
    return Finding(
        name,
        PASS,
        f"{document['covered']} of {document['rows']} covered; "
        f"status {document['status']}",
    )


def attachment_problems(
    run_dir: Path, attachments: dict[str, str]
) -> list[str]:
    """Checks that each attachment file hashes to its sealed digest.

    Args:
        run_dir: The run directory.
        attachments: The sealed digests by file name.

    Returns:
        A message for each attachment that is missing or differs; empty if
        all match.
    """
    problems = []
    for file_name, expected in sorted(attachments.items()):
        found = read_optional(run_dir / file_name)
        if found is None:
            problems.append(f"{file_name}: missing")
        elif scmle.data.ledger.ledger.digest(found) != expected:
            problems.append(f"{file_name}: differs from the sealed hash")
    return problems


def baseline_report_problems(
    run_dir: Path, ledger: scmle.data.ledger.ledger.Ledger
) -> list[str]:
    """Checks that the baseline report names the registered baseline claim.

    Args:
        run_dir: The run directory.
        ledger: The run's ledger.

    Returns:
        A message for each baseline claim whose registration the report does
        not name; empty if there is no report or all match.
    """
    baseline_doc = read_optional(run_dir / BASELINE_FILE)
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
    ledger: scmle.data.ledger.ledger.Ledger,
) -> Finding:
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
    problems.extend(attachment_problems(run_dir, attachments))
    if document.get("attachments") != attachments:
        problems.append("attachment hashes differ from the claims file")
    problems.extend(baseline_report_problems(run_dir, ledger))
    if problems:
        return Finding(name, FAIL, "; ".join(problems))
    return Finding(
        name,
        PASS,
        f"{len(ledger.claims)} claim(s), {len(attachments)} file(s) bound",
    )


def check_control(
    run_dir: Path, ledger: scmle.data.ledger.ledger.Ledger
) -> Finding:
    """Checks that the negative-control file is the one the ledger recorded.

    Args:
        run_dir: The run directory.
        ledger: The run's ledger.

    Returns:
        A finding: skipped without a control file or control row, passing if the
        file's hash equals the ledger's, otherwise failing.
    """
    name = "negative control matches the ledger"
    path = run_dir / CONTROL_FILE
    if not path.exists() or ledger.control is None:
        return Finding(name, SKIPPED, "no control file or no control row")
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return Finding(name, FAIL, f"{CONTROL_FILE} is not JSON")
    if scmle.data.ledger.ledger.digest(body) == ledger.control.get("sha256"):
        return Finding(name, PASS, "hash equal")
    return Finding(name, FAIL, "the file differs from the ledger's control row")


def evaluate_attestation(
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
        return None, [f"malformed attestation: {error}"], False
    problems = []
    if given.assumption not in allowed.get(given.attestor, set()):
        problems.append(f"{given.assumption}: {given.attestor} not permitted")
    if given.scope != charter.charter_hash:
        problems.append(f"{given.assumption}: scope is not this charter")
    day = str(report.get("evaluated_on", ""))
    try:
        valid = given.date <= day <= given.expires
    except TypeError:
        valid = False
    if not valid:
        problems.append(f"{given.assumption}: not valid on {day!r}")
    unchecked = False
    if signing_key is None and not signing.resolve_key(None):
        unchecked = True
    elif not given.verify(signing_key):
        problems.append(f"{given.assumption}: signature does not verify")
    return given, problems, unchecked


def evaluate_attestations(
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
        given, found, skipped = evaluate_attestation(
            item, report, charter, allowed, signing_key
        )
        problems.extend(found)
        unchecked = unchecked or skipped
        if given is not None:
            accepted[given.assumption] = given.attestor
    return accepted, problems, unchecked


def check_attestations(
    run_dir: Path, signing_key: str | bytes | None
) -> Finding:
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
    claims = read_optional(run_dir / CAUSAL_FILE)
    record = read_optional(run_dir / CHARTER_FILE)
    if claims is None or record is None:
        return Finding(name, SKIPPED, "no causal claims or charter file")
    try:
        charter = contracts.charter_from_record(record.get("charter", record))
    except errors.ConfigError as error:
        return Finding(name, FAIL, str(error))
    block = charter.estimand.causal
    if block is None:
        return Finding(name, SKIPPED, "the charter has no causal block")
    report = claims.get("attestations") or {}
    raised = claims.get("tier") in ("attested", "confirmed")
    problems = []
    if raised and not block.attestors:
        problems.append(
            "tier above unverified but the charter names no attestors"
        )
    allowed = {a.name: set(a.assumptions) for a in block.attestors}
    accepted, found, unchecked = evaluate_attestations(
        report, charter, allowed, signing_key
    )
    problems.extend(found)
    for assumption, named in (claims.get("assumptions") or {}).items():
        if named and accepted.get(assumption) != named:
            problems.append(
                f"{assumption}: {named} has no accepted attestation"
            )
    if raised and not accepted:
        problems.append("tier above unverified with no accepted attestation")
    if problems:
        return Finding(name, FAIL, "; ".join(problems))
    if unchecked:
        return Finding(name, SKIPPED, "no signing key to check the signatures")
    return Finding(
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
        ledger = scmle.data.ledger.ledger.Ledger.load(
            folder / scmle.data.ledger.ledger.FILE_NAME
        )
    except scmle.data.ledger.ledger.LedgerError as error:
        return Verification(str(folder), (Finding(chain, FAIL, str(error)),))
    findings.append(
        Finding(
            chain, PASS, f"{len(ledger.rows)} rows, head {ledger.head[:12]}"
        )
    )
    try:
        document = read_claims(folder)
    except errors.ArtifactError as error:
        findings.append(Finding("claims file reads", FAIL, str(error)))
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
