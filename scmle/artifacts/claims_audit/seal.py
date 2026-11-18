"""Checks of the claims file and the charter against the sealed ledger."""

from fractions import Fraction
import math
from pathlib import Path
from typing import Any

from scmle.artifacts.claims_audit import files
from scmle.artifacts.claims_audit import verdict
from scmle.assurance.charter import sealing
import scmle.assurance.charter.rules
from scmle.assurance.ledger import budget
from scmle.assurance.ledger import confirm
from scmle.assurance.ledger import multiplicity
import scmle.assurance.ledger.ledger
from scmle.contracts import errors
from scmle.contracts import qualification


def check_ledger_head(
    document: dict[str, Any], ledger: scmle.assurance.ledger.ledger.Ledger
) -> verdict.Finding:
    """Checks that the claims file names the ledger's current head.

    Args:
        document: The claims document.
        ledger: The run's ledger.

    Returns:
        A passing finding if the heads agree, otherwise a failing one.
    """
    name = "claims file names the ledger head"
    if document.get("heads", {}).get("ledger") == ledger.head:
        return verdict.Finding(name, verdict.PASS, f"head {ledger.head[:12]}")
    return verdict.Finding(
        name,
        verdict.FAIL,
        "confirmed_claims.json names a different ledger head",
    )


def authenticate(
    run_dir: Path,
    ledger: scmle.assurance.ledger.ledger.Ledger,
    signing_key: str | bytes | None,
) -> list[verdict.Finding]:
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
    path = run_dir / files.CHARTER_FILE
    name = "charter verifies and matches the ledger"
    if not path.exists():
        return [
            verdict.Finding(
                name, verdict.SKIPPED, f"{files.CHARTER_FILE} is not in the run"
            )
        ]
    try:
        loaded = sealing.read_verified(path, signing_key=signing_key)
    except errors.InputError:
        # ERR-20: an unusable signing key (empty) is the caller's input
        # error and propagates, as its ValueError did before the error
        # types were narrowed; it is not a finding about the run.
        raise
    except errors.ScmleError as error:
        # handled: returned as failed finding/problem/refusal with error text
        return [verdict.Finding(name, verdict.FAIL, str(error))]
    if loaded.charter_hash != ledger.header.get("charter_hash"):
        return [
            verdict.Finding(
                name, verdict.FAIL, "the ledger names another charter hash"
            )
        ]
    how = "signature verified" if loaded.signed else "hash only, not signed"
    return [
        verdict.Finding(
            name, verdict.PASS, f"{loaded.charter_hash[:12]} ({how})"
        ),
        ruled(loaded.charter.rules, ledger),
    ]


def ruled(
    rules: scmle.assurance.charter.rules.Procedure | None,
    ledger: scmle.assurance.ledger.ledger.Ledger,
) -> verdict.Finding:
    """Checks that the ledger and the confirmation used the chartered rules.

    Args:
        rules: The rules the charter fixes, or None.
        ledger: The run's ledger.

    Returns:
        A finding: skipped if the charter fixes no rules, failing if the
        ledger or confirmation used other rules, else passing.
    """
    rule_step = "chartered rules were used"
    if rules is None:
        return verdict.Finding(
            rule_step, verdict.SKIPPED, "the charter fixes no rules"
        )
    used = ledger.header["rule"]["rule"]
    confirmed_with = (ledger.confirmation or {}).get("rule")
    if used != rules.ledger or confirmed_with not in (
        None,
        rules.confirmation,
    ):
        return verdict.Finding(
            rule_step,
            verdict.FAIL,
            f"ledger rule {used!r} / confirmation rule "
            f"{confirmed_with!r} differ from the charter's "
            f"{rules.ledger!r} / {rules.confirmation!r}",
        )
    return verdict.Finding(
        rule_step, verdict.PASS, f"{used} and {rules.confirmation}"
    )


def compare(
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
    if document.get("digests", {}).get("survivors") != survivors.sha256:
        problems.append("survivor hash")
    if document.get("outcome") != sealed_row.get("outcome"):
        problems.append("outcome")
    if document.get("rule") != sealed_row.get("rule"):
        problems.append("rule")
    if document.get("alpha") != sealed_row.get("alpha_confirm"):
        problems.append("alpha_confirm")
    return problems


def crosscheck(
    document: dict[str, Any], ledger: scmle.assurance.ledger.ledger.Ledger
) -> verdict.Finding:
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
        return verdict.Finding(name, verdict.FAIL, "the ledger is not sealed")
    problems = compare(document, sealed_row, survivors)
    ids = [
        files.migrate(qualification.Claim, c)["identifier"]
        for c in document["claims"]
    ]
    if document.get("outcome") == confirm.NO_SURVIVORS:
        ok_ids = ids == []
    else:
        ok_ids = ids == list(survivors.hypothesis_ids)
    if not ok_ids:
        problems.append("claim ids differ from the frozen survivors")
    confirmed = [
        files.migrate(qualification.Claim, c)["identifier"]
        for c in document["claims"]
        if upheld(c)
    ]
    if confirmed != list(sealed_row.get("confirmed", [])):
        problems.append("confirmed set")
    if document.get("second") != sealed_row.get("stage_two"):
        problems.append("second-stage accounting")
    if problems:
        return verdict.Finding(
            name,
            verdict.FAIL,
            "differs from the ledger: " + ", ".join(problems),
        )
    return verdict.Finding(
        name, verdict.PASS, f"{len(confirmed)} confirmed of {len(ids)} claims"
    )


def upheld(claim: dict[str, Any]) -> bool:
    """Reports whether a claim record has tier `confirmed`.

    Args:
        claim: A claim record.

    Returns:
        True if its tier is `confirmed`.
    """
    return claim.get("tier") == "confirmed"


def rederive(document: dict[str, Any]) -> verdict.Finding:
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
        return verdict.Finding(
            name, verdict.SKIPPED, f"outcome is {document.get('outcome')!r}"
        )
    try:
        checks = {
            c["identifier"]: c
            for c in (
                files.migrate(qualification.Check, c)
                for c in document["checks"]
            )
        }
        tested = [
            c
            for c in (
                files.migrate(qualification.Claim, c)
                for c in document["claims"]
            )
            if f"{c['identifier']}:C" in checks
        ]
        values = [checks[f"{c['identifier']}:C"]["p_value"] for c in tested]
        adjustment = multiplicity.adjust(
            values,
            budget.Correction(document["rule"]),
            Fraction(document["alpha"]),
        )
        for index, claim in enumerate(tested):
            recorded = claim["adjusted_p_value"]
            if recorded is None or not math.isclose(
                recorded, adjustment.adjusted[index], abs_tol=verdict.TOLERANCE
            ):
                return verdict.Finding(
                    name,
                    verdict.FAIL,
                    f"{claim['identifier']}: adjusted p differs",
                )
            if upheld(claim) != adjustment.rejected[index]:
                return verdict.Finding(
                    name, verdict.FAIL, f"{claim['identifier']}: tier differs"
                )
    except (
        ValueError,
        KeyError,
        TypeError,
        ZeroDivisionError,
    ) as error:
        # handled: returned as failed finding/problem/refusal with error text
        return verdict.Finding(name, verdict.FAIL, f"cannot replay: {error!r}")
    return verdict.Finding(
        name, verdict.PASS, f"{len(tested)} hypotheses, rule {document['rule']}"
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
        record: The claims document's `second` record.
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


def tally(
    document: dict[str, Any], ledger: scmle.assurance.ledger.ledger.Ledger
) -> verdict.Finding:
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
    record = document.get("second")
    if record is None:
        return verdict.Finding(name, verdict.SKIPPED, "no withdrawal rule ran")
    survivors = ledger.survivor_set
    if survivors is None:
        return verdict.Finding(
            name, verdict.FAIL, "the ledger has no frozen survivors"
        )
    withdrawn = set(record["withdrawn"])
    family = [i for i in survivors.hypothesis_ids if i not in withdrawn]
    expected = scmle.assurance.ledger.ledger.digest(
        {"survivors_sha256": survivors.sha256, "family": family}
    )
    problems = stage_two_record_problems(
        record, survivors, withdrawn, family, expected
    )
    tested = {
        files.migrate(qualification.Check, c)["identifier"].split(":")[0]
        for c in document["checks"]
    }
    if tested & withdrawn:
        problems.append("a withdrawn survivor was tested on C")
    if tested - set(family):
        problems.append("a check on C is outside the family")
    problems.extend(
        "a withdrawn survivor is confirmed"
        for claim in document["claims"]
        if files.migrate(qualification.Claim, claim)["identifier"] in withdrawn
        and upheld(claim)
    )
    if problems:
        return verdict.Finding(name, verdict.FAIL, "; ".join(problems))
    return verdict.Finding(
        name,
        verdict.PASS,
        f"{len(family)} tested, {len(withdrawn)} withdrawn before C opened",
    )
