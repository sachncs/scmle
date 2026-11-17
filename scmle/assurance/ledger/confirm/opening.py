"""The single opening of `C` and the confirmation row it writes."""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import MutableMapping
from collections.abc import Sequence
from fractions import Fraction
from typing import Any

import pandas

from scmle.assurance.ledger import budget
from scmle.assurance.ledger import usage
from scmle.assurance.ledger.confirm import evaluation
from scmle.assurance.ledger.confirm import records
import scmle.assurance.ledger.ledger
from scmle.contracts import errors
from scmle.contracts import qualification


def finish(
    ledger: scmle.assurance.ledger.ledger.Ledger,
    *,
    outcome: str,
    reason: str,
    claims: tuple[qualification.Claim, ...],
    checks: tuple[qualification.Check, ...],
    rule: budget.Correction,
    alpha: Fraction,
    custody_head: str | None,
    stage_two: Mapping[str, Any] | None = None,
    runner: records.Docket | None = None,
    ran: records.Output | None = None,
) -> records.Confirmation:
    """Writes the confirmation row and builds the result.

    With a `runner`, `ran` holds what it produced; None means it was not
    evaluated (no `C`, or the opening failed).

    Args:
        ledger: The ledger with frozen survivors.
        outcome: Outcome code of the confirmation.
        reason: One sentence for the report.
        claims: One claim per survivor.
        checks: The checks on `C` (empty when `C` was not tested).
        rule: Multiplicity rule used.
        alpha: The confirmation level.
        custody_head: Custody log head after the open, or None.
        stage_two: Withdrawal accounting, or None.
        runner: Registered claim runner, or None.
        ran: What the runner produced, or None if it was not evaluated.

    Returns:
        The confirmation, after its row has been sealed into the ledger.

    Raises:
        LedgerError: If the survivors have not been frozen.
    """
    survivors = ledger.survivor_set
    if survivors is None:
        raise scmle.assurance.ledger.ledger.LedgerError(
            "freeze the survivors first"
        )
    body: dict[str, Any] = {
        "outcome": outcome,
        "reason": reason,
        "rule": rule.value,
        "alpha_confirm": str(alpha),
        "confirmed": [
            c.identifier
            for c in claims
            if c.tier is qualification.Tier.CONFIRMED
        ],
        "custody_head": custody_head,
    }
    if stage_two is not None:
        body["stage_two"] = dict(stage_two)
    registered, documents = account(body, runner, ran)
    ledger.seal(body)
    return records.Confirmation(
        outcome=outcome,
        reason=reason,
        claims=claims,
        checks=checks,
        rule=rule.value,
        alpha_confirm=alpha,
        survivors_sha256=survivors.sha256,
        ledger_head=ledger.head,
        custody_head=custody_head,
        control=ledger.control,
        stage_two=None if stage_two is None else dict(stage_two),
        registered=registered,
        documents=documents,
    )


def account(
    body: MutableMapping[str, Any],
    runner: records.Docket | None,
    ran: records.Output | None,
) -> tuple[dict[str, dict[str, str]] | None, dict[str, Mapping[str, Any]]]:
    """Adds the registered claims and the hashes of their files to the row.

    Args:
        body: The confirmation row; ``claims`` and ``attachments`` are added
            in place when there is a runner.
        runner: Registered claim runner, or None.
        ran: What the runner produced, or None if it was not evaluated.

    Returns:
        Claim id to its kind, hash and status (None without a runner), and
        the documents the runner produced.
    """
    if runner is None:
        return None, {}
    statuses = {} if ran is None else dict(ran.statuses)
    documents = {} if ran is None else dict(ran.documents)
    registered: dict[str, dict[str, str]] = {
        item.claim_id: {
            "kind": item.kind,
            "sha256": item.sha256,
            "status": statuses.get(item.claim_id, records.NOT_EVALUATED),
        }
        for item in runner.registered
    }
    body["claims"] = registered
    body["attachments"] = {
        name: scmle.assurance.ledger.ledger.digest(dict(content))
        for name, content in sorted(documents.items())
    }
    return registered, documents


def enforce(policy: str) -> None:
    """Refuses a failure policy that is not available.

    Args:
        policy: The failure policy name.

    Raises:
        ConfigError: If the policy is `thresholdout` (not implemented) or is
            not a known policy.
    """
    if policy == records.THRESHOLDOUT:
        raise errors.ConfigError(
            "the Thresholdout reusable holdout is not implemented; use the "
            "'unconfirmed' failure policy"
        )
    if policy not in records.POLICIES:
        raise errors.ConfigError(
            f"failure policy must be one of {records.POLICIES}"
        )


def confirm(
    ledger: scmle.assurance.ledger.ledger.Ledger,
    source: records.Source | None,
    *,
    alpha_confirm: Fraction,
    rule: budget.Correction = budget.Correction.BY,
    policy: str = records.UNCONFIRMED,
    require_control: bool = True,
    custody_log: records.Log | None = None,
    encode: Callable[[pandas.Series], Any] = evaluation.default_encoding,
    withdrawal: records.Withdrawal | None = None,
    claim_runner: records.Docket | None = None,
    register: usage.Register | None = None,
) -> records.Confirmation:
    """Freezes the survivors and tests them once on the confirmation set.

    An unavailable failure policy is refused first, by `enforce`, with a
    `ConfigError`.

    Args:
        ledger: The run's ledger, after exploration (and the negative
            control, recorded with `Ledger.negatives`).
        source: The sealed confirmation set, or None when no `C` exists.
        alpha_confirm: The confirmation level (the charter's).
        rule: `by` (default, arbitrary dependence), `bh`, `holm` or
            `bonferroni`.
        policy: Failure policy, `unconfirmed` (default).
        require_control: Whether a missing negative control blocks promotion.
        custody_log: Log that receives the `selection_frozen` event and, when
            they occur, `denied` and `claims_failed` (the sealed set's own
            log when omitted).
        encode: Turns the opened target into the floats the statistics read;
            it must code classes exactly as exploration did.
        withdrawal: Survivors an exploration-only rule removed from the
            family (module docstring, "Second stage"); the full survivor
            set is still frozen and hashed, and `C` tests the rest.
        claim_runner: Claims registered in the ledger and evaluated on the
            same opening (module docstring, "Registered claims").
        register: The durable record of this confirmation set across runs
            (`usage`); the use is written just before `C` is opened.

    Returns:
        The confirmation, also recorded in the ledger. Cases that stop
        before testing (no survivors, no control, floor not exceeded, no
        `C`, every survivor withdrawn) and any other failure while opening
        or testing are returned with a matching outcome and unverified
        claims, not raised.

    Raises:
        LeakageError: If the confirmation was already performed, in this run
            or, by `register`, in another run of the same charter on the same
            data. The attempt
            is written to the custody log. A `LeakageError` from opening the
            set is also propagated. A `LedgerError` from registering claims,
            freezing the survivors or building the withdrawal accounting
            (a withdrawn id that is not a survivor) is not caught either.
    """
    enforce(policy)
    if ledger.confirmation is not None:
        deny(custody_log or (source.log if source is not None else None))
        raise errors.LeakageError(
            "the confirmation set was already used for the confirmation"
        )
    record = enter(ledger, withdrawal, claim_runner)
    blocked = screen(ledger, source, record, require_control)
    terms = records.Terms(rule, alpha_confirm, encode, record, claim_runner)
    if source is None or (blocked is not None and claim_runner is None):
        return halt(ledger, blocked, terms)
    log = custody_log or source.log
    spent = None if register is None else register.record()
    if spent is not None:
        deny(log)
        raise errors.LeakageError(spent)
    return unseal(ledger, source, log, blocked, terms)


def enter(
    ledger: scmle.assurance.ledger.ledger.Ledger,
    withdrawal: records.Withdrawal | None,
    claim_runner: records.Docket | None,
) -> Mapping[str, Any] | None:
    """Registers the claims, freezes the survivors and accounts a withdrawal.

    Args:
        ledger: The run's ledger, after exploration.
        withdrawal: Survivors an exploration-only rule removed, or None.
        claim_runner: Claims to register in the ledger, or None.

    Returns:
        The withdrawal accounting from `stage_two_record`, or None when no
        withdrawal rule ran.
    """
    if claim_runner is not None:
        for item in claim_runner.registered:
            ledger.enroll(item.claim_id, item.kind, item.statement, item.sha256)
    ledger.freeze()
    return (
        None
        if withdrawal is None
        else evaluation.stage_two_record(ledger, withdrawal)
    )


def screen(
    ledger: scmle.assurance.ledger.ledger.Ledger,
    source: records.Source | None,
    record: Mapping[str, Any] | None,
    require_control: bool,
) -> tuple[str, str, tuple[qualification.Claim, ...]] | None:
    """Decides why the family cannot be promoted, including a full withdrawal.

    Args:
        ledger: The ledger with frozen survivors.
        source: The sealed confirmation set, or None when no `C` exists.
        record: Withdrawal accounting, or None.
        require_control: Whether a missing negative control blocks promotion.

    Returns:
        The outcome of `block`; when it is None and `C` exists but every
        survivor was withdrawn, the `all_withdrawn` outcome; otherwise None.
    """
    blocked = evaluation.block(ledger, require_control)
    if source is None or blocked is not None:
        return blocked
    if record is not None and not record["family"]:
        return (
            records.ALL_WITHDRAWN,
            "every survivor was withdrawn before C; C was not opened",
            tuple(evaluation.withdraw(ledger, record["withdrawn"]).values()),
        )
    return None


def halt(
    ledger: scmle.assurance.ledger.ledger.Ledger,
    blocked: tuple[str, str, tuple[qualification.Claim, ...]] | None,
    terms: records.Terms,
    log: Any = None,
    ran: records.Output | None = None,
) -> records.Confirmation:
    """Writes a confirmation that tested nothing on `C`.

    Args:
        ledger: The ledger with frozen survivors.
        blocked: The outcome code, sentence and unverified claims; None when
            no confirmation set exists.
        terms: The rule, level, withdrawal accounting and claim runner.
        log: The custody log whose head is recorded, or None before `C`
            was reached.
        ran: What the claim runner produced, or None if it was not
            evaluated.

    Returns:
        The confirmation, after its row has been sealed into the ledger.
    """
    if blocked is None:
        blocked = (
            records.NO_CONFIRMATION_SET,
            "no confirmation set exists for this run",
            evaluation.disclaim(ledger, "no confirmation set"),
        )
    outcome, reason, held = blocked
    return finish(
        ledger,
        outcome=outcome,
        reason=reason,
        claims=held,
        checks=(),
        rule=terms.rule,
        alpha=terms.alpha,
        custody_head=tip(log),
        stage_two=terms.record,
        runner=terms.runner,
        ran=ran,
    )


def deny(log: records.Log | None) -> None:
    """Writes a refused second confirmation to the custody log, if any.

    Args:
        log: The custody log, or None.
    """
    if log is not None:
        log.append(
            "C",
            "denied",
            records.ACTOR,
            {"reason": "confirmation was already performed"},
        )


def preregister(
    log: Any,
    ledger: scmle.assurance.ledger.ledger.Ledger,
    survivors: Any,
    rule: budget.Correction,
    record: Mapping[str, Any] | None,
    registered: Sequence[records.Commitment],
) -> str:
    """Writes the `selection_frozen` event and hashes the claim set.

    Args:
        log: The custody log.
        ledger: The ledger with frozen survivors.
        survivors: The frozen survivor set.
        rule: Multiplicity rule used.
        record: Withdrawal accounting, or None.
        registered: The registered claims.

    Returns:
        The hash of the family, combined with the registered claims when
        there are any.
    """
    log.append(
        "C",
        "selection_frozen",
        records.ACTOR,
        {
            "survivors_sha256": survivors.sha256,
            "survivors": len(survivors.hypothesis_ids),
            "ledger_head": ledger.head,
            "rule": rule.value,
            **(
                {}
                if record is None
                else {
                    "family_sha256": record["family_sha256"],
                    "family": len(record["family"]),
                    "withdrawn": len(record["withdrawn"]),
                }
            ),
            **(
                {}
                if not registered
                else {"registered_claims": [i.claim_id for i in registered]}
            ),
        },
    )
    claims_hash = (
        survivors.sha256 if record is None else record["family_sha256"]
    )
    if registered:
        claims_hash = scmle.assurance.ledger.ledger.digest(
            {
                "family": claims_hash,
                "registered": [[i.claim_id, i.sha256] for i in registered],
            }
        )
    return str(claims_hash)


def unseal(
    ledger: scmle.assurance.ledger.ledger.Ledger,
    source: records.Source,
    log: Any,
    blocked: tuple[str, str, tuple[qualification.Claim, ...]] | None,
    terms: records.Terms,
) -> records.Confirmation:
    """Opens `C` once, evaluates the registered claims and tests the family.

    The `selection_frozen` event is written and the claim set hashed
    (`preregister`) before the release is bound and opened.

    Args:
        ledger: The ledger with frozen survivors.
        source: The sealed confirmation set.
        log: The custody log.
        blocked: Why the family cannot be promoted, or None; `C` is then
            opened for the registered claims only.
        terms: The rule, level, encoder, withdrawal accounting and claim
            runner.

    Returns:
        The confirmation. A failure while opening or testing gives a
        `pipeline_failure` outcome with unverified claims, not an exception.

    Raises:
        LeakageError: If opening the set raises it.
    """
    registered = () if terms.runner is None else terms.runner.registered
    claims_hash = preregister(
        log, ledger, ledger.freeze(), terms.rule, terms.record, registered
    )
    ledger_hash = ledger.head
    try:
        if not source.committed:
            source.commit(claims_hash, ledger_hash)
        rows = source.open(
            source.mint(),
            actor=records.ACTOR,
            claims_hash=claims_hash,
            ledger_hash=ledger_hash,
        )
    except errors.LeakageError:
        raise
    except Exception as failure:  # noqa: BLE001 -- a failed opening is recorded as a pipeline-failure outcome, not raised
        # handled: opening failure returned as pipeline_failure outcome with
        # reason
        return halt(ledger, crash(ledger, "opening C", failure), terms, log)
    ran = evaluate(terms.runner, rows, log)
    if blocked is not None:
        outcome, reason, held = blocked
        reason += "; C was opened for the registered claims only"
        return halt(ledger, (outcome, reason, held), terms, log, ran)
    return adjudicate(ledger, rows, terms, log, ran)


def evaluate(
    runner: records.Docket | None,
    rows: tuple[pandas.DataFrame, pandas.Series],
    log: Any,
) -> records.Output | None:
    """Runs the registered claims on the opened rows.

    Args:
        runner: Registered claims, or None.
        rows: The opened rows of `C`.
        log: The custody log; a crash of the runner is written to it as
            `claims_failed`.

    Returns:
        What the runner produced; every claim `failed` when it crashed; None
        without a runner.
    """
    if runner is None:
        return None
    try:
        return runner.run(*rows)
    except Exception as failure:  # noqa: BLE001 -- a crashing claim runner marks its claims failed and is logged, not raised
        # handled: claim runner crash logged to custody log; claims marked
        # FAILED
        failed = phrase("evaluating the registered claims", failure)
        log.append("C", "claims_failed", records.ACTOR, {"reason": failed})
        return records.Output(
            {i.claim_id: records.FAILED for i in runner.registered},
            {},
        )


def adjudicate(
    ledger: scmle.assurance.ledger.ledger.Ledger,
    rows: tuple[pandas.DataFrame, pandas.Series],
    terms: records.Terms,
    log: Any,
    ran: records.Output | None,
) -> records.Confirmation:
    """Tests the family on the opened rows and writes the confirmation.

    Args:
        ledger: The ledger with frozen survivors.
        rows: The opened rows of `C`.
        terms: The rule, level, encoder, withdrawal accounting and claim
            runner.
        log: The custody log.
        ran: What the claim runner produced, or None.

    Returns:
        The `confirmed` outcome, or `pipeline_failure` with unverified claims
        when testing fails.
    """
    record = terms.record
    try:
        claims, checks = evaluation.verify(
            ledger, rows, terms.rule, terms.alpha, terms.encode, record
        )
    except Exception as failure:  # noqa: BLE001 -- a failed test on C is recorded as a pipeline-failure outcome, not raised
        # handled: testing failure returned as pipeline_failure outcome with
        # reason
        crashed = crash(ledger, "testing on C", failure)
        return halt(ledger, crashed, terms, log, ran)
    done = sum(1 for c in claims if c.tier is qualification.Tier.CONFIRMED)
    tested = len(claims) - (0 if record is None else len(record["withdrawn"]))
    return finish(
        ledger,
        outcome=records.CONFIRMED,
        reason=(
            f"{done} of {tested} survivors confirmed on C by "
            f"{terms.rule.value} at {float(terms.alpha):g}"
        ),
        claims=claims,
        checks=checks,
        rule=terms.rule,
        alpha=terms.alpha,
        custody_head=tip(log),
        stage_two=record,
        runner=terms.runner,
        ran=ran,
    )


def crash(
    ledger: scmle.assurance.ledger.ledger.Ledger, stage: str, failure: Exception
) -> tuple[str, str, tuple[qualification.Claim, ...]]:
    """Builds the `pipeline_failure` outcome of a crash after `C` was opened.

    Args:
        ledger: The ledger with frozen survivors.
        stage: What was being done when it failed.
        failure: The exception that was caught.

    Returns:
        The outcome code, the sentence from `phrase` and one unverified
        claim per survivor.
    """
    return (
        records.PIPELINE_FAILURE,
        phrase(stage, failure),
        evaluation.disclaim(ledger, "confirmation crashed"),
    )


def phrase(stage: str, failure: Exception) -> str:
    """Phrases why the confirmation crashed, under the 'unconfirmed' policy.

    Args:
        stage: What was being done when it failed.
        failure: The exception that was caught.

    Returns:
        A sentence naming the stage and the exception, and saying that all
        claims stay unverified.
    """
    return (
        f"the confirmation failed while {stage} "
        f"({type(failure).__name__}: {failure}); C is spent and all claims "
        "stay unverified (failure policy: unconfirmed)"
    )


def tip(log: Any) -> str | None:
    """Reads the hash of the custody log's last entry.

    Args:
        log: The custody log, read through its ``entries`` attribute.

    Returns:
        The last entry's hash, or None when the log has no entries.
    """
    entries: Sequence[dict[str, Any]] = getattr(log, "entries", ())
    return str(entries[-1]["hash"]) if entries else None
