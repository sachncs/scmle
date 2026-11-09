"""The single confirmation of the exploration survivors on `C`.

Order of events. The guarantee ``FDR <= alpha_confirm`` comes only
from this stage:

1. The survivors, selected on `X` only, are frozen in the ledger with a hash.
2. If the negative-control gate fails, there is nothing to promote and `C` is
   not opened. With no survivors, `C` is not opened either.
3. The selection is written to the custody log, the release is bound to the
   family's hash and the ledger head (`SealedSet.commit`), then `C` is opened
   exactly once through the custody API. A set already bound to another
   claim set or ledger state is refused by the vault, so a ledger edited
   after the binding cannot unlock `C`.
4. Each survivor's statistic runs on `C`; the family of survivors is adjusted
   by BH, BY (default) or Holm at ``alpha_confirm``; claims are written.

A second call raises `LeakageError` and is logged. This module takes the
sealed set as an argument; it imports nothing from custody, and no other
module of the ledger package can reach `C` (an import-graph test checks it).

Second stage (optional, pre-registered). Between the negative control and the
opening of `C` the caller may hand over a `Withdrawal`: survivors that a rule
computed from `X` alone (the stability filter) removes from the family that
goes to `C`. The ledger is untouched: it still freezes and hashes the full
survivor set, and the withdrawal is its own accounting, written to the custody
log (``selection_frozen``, with ``family_sha256``) before `C` opens and to the
confirmation row and ``confirmed_claims.json`` after (``stage_two``). Withdrawn
survivors are reported at tier `unverified` and are never tested on `C`.

What is controlled, and under what assumption (`STAGE_TWO_STATEMENT`). Let
``S`` be the frozen survivors and ``F`` the subset not withdrawn. If ``F`` is a
function of `X`, the charter and the seed only, and `C` is independent of `X`
(the existing assumption), then conditionally on ``X`` the p-values on `C` of
the true nulls in ``F`` keep their null distribution, so the rule's guarantee
holds for ``F``: FDR of the confirmed claims at most ``alpha_confirm`` for BY
(any dependence) and for BH under independence or positive dependence on `C`,
FWER at most ``alpha_confirm`` for Holm and Bonferroni. The bound is on the
claims actually made, which are a subset of ``F``; it averages over the
selection, so the same level holds unconditionally. The withdrawal spends no
alpha and adds none. What is not claimed: that the withdrawal raises power
(it can only remove hypotheses, a true effect withdrawn is lost), that a
withdrawn finding is false, or any guarantee if ``F`` was computed from rows
of `C` or `F`. The module cannot see how the caller computed ``F``; it checks
only that the withdrawn ids are survivors and that the family hash is
recorded before `C` opens. `tests/unit/test_ledger_withdrawal.py` simulates
the error rate with an `X`-only withdrawal rule that is correlated with the
signal, and checks the order of events.

Registered claims. A caller may hand over a `ClaimRunner`: claims that are not
single-feature statistics (the baseline comparison, the primary causal
hypothesis). Each is registered in the ledger by the hash of its
registration before the survivors are frozen, the sealed set is bound to the
survivor family and the registrations, and the *same* released frame is given
to the runner after the bivariate family was tested. The runner never sees `C`
twice: there is one opening. Their gates differ from the family's: a missing
or failed negative control and an empty family stop the bivariate promotion,
not the registered claims, which carry their own gates. The confirmation row
records the status of each claim and the hash of every report file the runner
wrote, so the files are bound to the ledger head. When `C` does not exist or
the opening fails, each claim is recorded as not evaluated and stays
unverified.

Failure policy. A statistical failure to reject is a result, not a failure.
If the confirmation run *crashes* after `C` was opened, `C` is spent and the
charter's policy applies: ``unconfirmed`` (default) keeps every claim at tier
`unverified` and says so. The Thresholdout reusable holdout of Dwork et al.
(2015) is not implemented (it needs a query interface in front of `C`, which
the one-time custody release does not offer) and is refused with a
`ConfigError` instead of being faked.

Assumptions (stated on every claim): the rows of `C` are independent of the
rows of `X` (grouped split by entity, embargo for temporal data); `C` is read
once; the statistics are valid under their null on `C` (the Spearman and
Kruskal tests assume independent rows within `C`, which the user must judge
for grouped or serial data); the family is exactly the frozen survivors.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from fractions import Fraction
import json
from pathlib import Path
from typing import Any
from typing import Protocol

import pandas

from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.core import randomness
from scmle.data.ledger import bivariate
from scmle.data.ledger import budget
from scmle.data.ledger import multiplicity
import scmle.data.ledger.ledger

UNCONFIRMED = "unconfirmed"
THRESHOLDOUT = "thresholdout"
POLICIES = (UNCONFIRMED, THRESHOLDOUT)
ACTOR = "confirmation"
CLAIMS_FILE = "confirmed_claims.json"

CONFIRMED = "confirmed"
NO_SURVIVORS = "no_survivors"
FLOOR_NOT_EXCEEDED = "floor_not_exceeded"
NO_CONFIRMATION_SET = "no_confirmation_set"
PIPELINE_FAILURE = "pipeline_failure"
NO_CONTROL = "no_negative_control"


class EventLog(Protocol):
    """The part of the custody log the confirmation writes to."""

    def append(
        self,
        set_id: str,
        event: str,
        actor: str,
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Adds an event and returns the stored entry.

        Args:
            set_id: Identifier of the sealed set the event concerns.
            event: Name of the event.
            actor: Who caused the event.
            detail: Extra JSON-ready detail, or None.

        Returns:
            The entry as stored in the log.
        """


class ConfirmationSource(Protocol):
    """A sealed set that opens once (the custody `SealedSet`)."""

    log: Any

    @property
    def committed(self) -> bool:
        """Whether the release is already bound to a claim set."""

    def issue_token(self) -> str:
        """Returns the release token."""

    def commit(self, claims_hash: str, ledger_hash: str) -> str:
        """Binds the single release to a claim set and a ledger state.

        Args:
            claims_hash: Hash of the family (and registered claims) to test.
            ledger_hash: Head hash of the ledger at the time of binding.

        Returns:
            The binding digest computed from the two hashes.
        """

    def open(
        self,
        token: str,
        actor: str = ...,
        claims_hash: str | None = ...,
        ledger_hash: str | None = ...,
    ) -> tuple[pandas.DataFrame, pandas.Series]:
        """Releases the rows a single time.

        Args:
            token: The release token from `issue_token`.
            actor: Who opens the set.
            claims_hash: The claim-set hash the release was bound to.
            ledger_hash: The ledger head the release was bound to.

        Returns:
            The features and target of the confirmation set.
        """


STAGE_TWO_STATEMENT = (
    "the confirmed claims keep the guarantee of the confirmation rule at "
    "alpha_confirm over the family that was not withdrawn, provided the "
    "withdrawal was computed from the exploration set only and C is "
    "independent of it; the withdrawal spends no alpha, and withdrawn "
    "findings are unverified, not shown false"
)
ALL_WITHDRAWN = "all_withdrawn"


@dataclasses.dataclass(frozen=True, slots=True)
class Withdrawal:
    """Survivors an exploration-only rule removed before `C` was opened.

    Attributes:
        reasons: Hypothesis id to the sentence that withdrew it. Every id
            must be a frozen survivor.
        rule: JSON-ready description of the rule (name, thresholds,
            certificate state), kept in the accounting.
        diagnostics: (source, check) pairs the rule produced on `X`; the
            caller logs them as wealth-free ledger diagnostics.
        accounted: False when the rule only produced diagnostics and no
            withdrawal was pre-registered; no second-stage accounting is then
            written and `reasons` must be empty.
    """

    reasons: Mapping[str, str]
    rule: Mapping[str, Any]
    diagnostics: tuple[tuple[str, qualification.Check], ...] = ()
    accounted: bool = True

    def __post_init__(self) -> None:
        """Refuses reasons without accounting.

        Raises:
            ValueError: If `reasons` is not empty while `accounted` is False.
        """
        if self.reasons and not self.accounted:
            raise ValueError("a withdrawal with reasons must be accounted")


NOT_EVALUATED = "not_evaluated"
FAILED = "failed"


@dataclasses.dataclass(frozen=True, slots=True)
class RegisteredClaim:
    """A claim fixed in the ledger before `C` is opened.

    Attributes:
        claim_id: Unique id (it must not collide with a hypothesis id).
        kind: What it is (`baseline`, `causal_primary` or `coverage`).
        statement: The claim in words.
        sha256: Hash of the registration (what was frozen before `C`).
    """

    claim_id: str
    kind: str
    statement: str
    sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class ClaimRun:
    """What a `ClaimRunner` produced from the opened rows.

    Attributes:
        statuses: Claim id to a one-word status.
        documents: File name to JSON-ready content; each is hashed into the
            confirmation row.
    """

    statuses: Mapping[str, str]
    documents: Mapping[str, Mapping[str, Any]]


@dataclasses.dataclass(frozen=True, slots=True)
class ClaimRunner:
    """Registered claims and the function that evaluates them on `C`.

    Attributes:
        registered: The claims, registered in the ledger by `confirm`.
        run: Evaluates them on the one released frame (features, target). It
            is called at most once, after the family was tested.
    """

    registered: tuple[RegisteredClaim, ...]
    run: Callable[[pandas.DataFrame, pandas.Series], ClaimRun]


def merge_runners(*runners: ClaimRunner | None) -> ClaimRunner | None:
    """Combines several runners into one, all evaluated on the same frame.

    A document name two runners produce is refused when it is evaluated,
    never overwritten silently. A runner that raises while evaluating has all
    its claims marked `failed`; the others still run.

    Args:
        *runners: The runners to combine; None entries are ignored.

    Returns:
        None when no runner is given, the single runner when only one is
        given, otherwise a runner registering every claim of all of them.

    Raises:
        ValueError: If two runners register the same claim id. The runner
            the function returns also raises it, when called, if two runners
            produce a document with the same name.
    """
    chosen = [r for r in runners if r is not None]
    if not chosen:
        return None
    if len(chosen) == 1:
        return chosen[0]
    registered = tuple(item for r in chosen for item in r.registered)
    names = [item.claim_id for item in registered]
    if len(set(names)) != len(names):
        raise ValueError("claim ids must be unique across runners")

    def run_all(features: pandas.DataFrame, target: pandas.Series) -> ClaimRun:
        statuses: dict[str, str] = {}
        documents: dict[str, Mapping[str, Any]] = {}
        for runner in chosen:
            try:
                found = runner.run(features, target)
            except Exception:
                statuses.update({i.claim_id: FAILED for i in runner.registered})
                continue
            clash = set(documents) & set(found.documents)
            if clash:
                raise ValueError(f"two claims wrote {sorted(clash)}")
            statuses.update(found.statuses)
            documents.update(found.documents)
        return ClaimRun(statuses, documents)

    return ClaimRunner(registered, run_all)


@dataclasses.dataclass(frozen=True, slots=True)
class Confirmation:
    """The outcome of the confirmation stage.

    Attributes:
        outcome: `confirmed` (C was tested), `no_survivors`,
            `floor_not_exceeded`, `no_confirmation_set`, `pipeline_failure`
            or `no_negative_control`.
        reason: One sentence for the report.
        claims: One claim per survivor.
        checks: The checks on `C` (empty when `C` was not tested).
        rule: Multiplicity rule used (or that would have been used).
        alpha_confirm: The confirmation level.
        survivors_sha256: Hash of the frozen survivor set.
        ledger_head: Ledger head hash when the confirmation was written.
        custody_head: Custody log head hash after the open, or None.
        control: The negative-control summary, or None.
        stage_two: Accounting of the withdrawal (family, hash, rule,
            statement), or None when no withdrawal rule ran.
        registered: Claim id to its registration hash and status, or None
            when no claim was registered.
        documents: Report files the registered claims produced.
    """

    outcome: str
    reason: str
    claims: tuple[qualification.Claim, ...]
    checks: tuple[qualification.Check, ...]
    rule: str
    alpha_confirm: Fraction
    survivors_sha256: str
    ledger_head: str
    custody_head: str | None
    control: dict[str, Any] | None
    stage_two: dict[str, Any] | None = None
    registered: dict[str, Any] | None = None
    documents: dict[str, Mapping[str, Any]] = dataclasses.field(
        default_factory=dict
    )

    @property
    def confirmed(self) -> tuple[qualification.Claim, ...]:
        """Claims at tier `confirmed`."""
        return tuple(
            c for c in self.claims if c.tier is qualification.Tier.CONFIRMED
        )

    def to_dict(self) -> dict[str, Any]:
        """Builds the JSON-ready record (``confirmed_claims.json``).

        Returns:
            A mapping with the outcome and its reason, the rule and its
            guarantee statement, the level, the hashes, the negative
            control, the claims, the checks and a note. ``stage_two``,
            ``registered_claims`` and ``attachments`` (a hash per document)
            are present only when they apply.
        """
        record = {
            "outcome": self.outcome,
            "reason": self.reason,
            "rule": self.rule,
            "guarantee": multiplicity.STATEMENTS[
                budget.MultiplicityRule(self.rule)
            ],
            "alpha_confirm": str(self.alpha_confirm),
            "survivors_sha256": self.survivors_sha256,
            "ledger_head": self.ledger_head,
            "custody_head": self.custody_head,
            "negative_control": self.control,
            "claims": [c.to_dict() for c in self.claims],
            "checks": [c.to_dict() for c in self.checks],
            "note": (
                "Only claims with tier 'confirmed' carry the FDR/FWER "
                "guarantee, and only under the assumptions listed on them. "
                "Exploration findings that are not confirmed do not change "
                "feature admission or preprocessing."
            ),
        }
        if self.stage_two is not None:
            record["stage_two"] = self.stage_two
        if self.registered is not None:
            record["registered_claims"] = self.registered
        if self.documents:
            record["attachments"] = {
                name: scmle.data.ledger.ledger.digest(dict(content))
                for name, content in sorted(self.documents.items())
            }
        return record

    def write(self, path: str | Path) -> None:
        """Writes the `to_dict` record as indented JSON.

        Args:
            path: File to write (``confirmed_claims.json`` by convention);
                missing parent directories are created and an existing
                file is overwritten.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )


def assumptions(
    rule: budget.MultiplicityRule, size: int, withdrawn: int = 0
) -> tuple[str, ...]:
    """Lists the assumptions a confirmed claim depends on.

    Args:
        rule: Multiplicity rule used for the family.
        size: Number of survivors in the tested family.
        withdrawn: Number of survivors withdrawn before `C` was opened.

    Returns:
        One sentence per assumption; a sentence on the withdrawal is added
        when `withdrawn` is positive.
    """
    lines = [
        f"{multiplicity.STATEMENTS[rule]} (rule {rule.value}, family of "
        f"{size} survivors fixed before C was opened)",
        "rows of C are independent of rows of X (grouped split by entity, "
        "embargo for temporal data)",
        "C was opened once and its rows were used only by this confirmation",
        "each statistic is valid under its null on C: Spearman and Kruskal "
        "assume independent rows",
        "exploration FDR is a triage budget, not part of this guarantee",
    ]
    if withdrawn:
        lines.insert(
            1,
            f"{withdrawn} survivor(s) were withdrawn before C was opened by "
            "a rule computed from the exploration set only; the family above "
            "is the survivors that remain",
        )
    return tuple(lines)


def statement(spec: scmle.data.ledger.ledger.HypothesisSpec) -> str:
    """Writes the sentence of a claim.

    Args:
        spec: The hypothesis specification.

    Returns:
        The hypothesis name followed by the null it is evidence against.
    """
    return f"{spec.name}: evidence against '{spec.null}'"


def unverified_claims(
    ledger: scmle.data.ledger.ledger.Ledger, why: str
) -> tuple[qualification.Claim, ...]:
    """Builds claims for survivors that were not (or could not be) confirmed.

    Args:
        ledger: The ledger whose survivors are frozen.
        why: Reason, recorded in each claim's assumptions.

    Returns:
        One unverified claim per survivor, in survivor order.

    Raises:
        LedgerError: If the survivors have not been frozen.
    """
    survivors = ledger.survivor_set
    if survivors is None:
        raise scmle.data.ledger.ledger.LedgerError("freeze the survivors first")
    claims = []
    for hypothesis_id in survivors.hypothesis_ids:
        result = ledger.results[hypothesis_id]
        spec = ledger.hypotheses[hypothesis_id].spec
        claims.append(
            qualification.Claim(
                claim_id=hypothesis_id,
                statement=statement(spec),
                check_ids=(f"{hypothesis_id}:X",),
                adjusted_p_value=None,
                effect=result.check.effect if result.check else 0.0,
                tier=qualification.Tier.UNVERIFIED,
                assumptions=(
                    f"not confirmed on C: {why}",
                    "exploration finding only; it does not change feature "
                    "admission or preprocessing",
                ),
            )
        )
    return tuple(claims)


def finish(
    ledger: scmle.data.ledger.ledger.Ledger,
    *,
    outcome: str,
    reason: str,
    claims: tuple[qualification.Claim, ...],
    checks: tuple[qualification.Check, ...],
    rule: budget.MultiplicityRule,
    alpha: Fraction,
    custody_head: str | None,
    stage_two: Mapping[str, Any] | None = None,
    runner: ClaimRunner | None = None,
    ran: ClaimRun | None = None,
) -> Confirmation:
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
        raise scmle.data.ledger.ledger.LedgerError("freeze the survivors first")
    body: dict[str, Any] = {
        "outcome": outcome,
        "reason": reason,
        "rule": rule.value,
        "alpha_confirm": str(alpha),
        "confirmed": [
            c.claim_id for c in claims if c.tier is qualification.Tier.CONFIRMED
        ],
        "custody_head": custody_head,
    }
    if stage_two is not None:
        body["stage_two"] = dict(stage_two)
    registered: dict[str, Any] | None = None
    documents: dict[str, Mapping[str, Any]] = {}
    if runner is not None:
        statuses = {} if ran is None else dict(ran.statuses)
        documents = {} if ran is None else dict(ran.documents)
        registered = {
            item.claim_id: {
                "kind": item.kind,
                "sha256": item.sha256,
                "status": statuses.get(item.claim_id, NOT_EVALUATED),
            }
            for item in runner.registered
        }
        body["claims"] = registered
        body["attachments"] = {
            name: scmle.data.ledger.ledger.digest(dict(content))
            for name, content in sorted(documents.items())
        }
    ledger.seal(body)
    return Confirmation(
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


def check_policy(policy: str) -> None:
    """Refuses a failure policy that is not available.

    Args:
        policy: The failure policy name.

    Raises:
        ConfigError: If the policy is `thresholdout` (not implemented) or is
            not a known policy.
    """
    if policy == THRESHOLDOUT:
        raise errors.ConfigError(
            "the Thresholdout reusable holdout is not implemented; use the "
            "'unconfirmed' failure policy"
        )
    if policy not in POLICIES:
        raise errors.ConfigError(f"failure policy must be one of {POLICIES}")


def withdrawn_claims(
    ledger: scmle.data.ledger.ledger.Ledger, reasons: Mapping[str, str]
) -> dict[str, qualification.Claim]:
    """Builds unverified claims for survivors withdrawn before `C` opened.

    Args:
        ledger: The ledger holding the survivors' results.
        reasons: Hypothesis id to the sentence that withdrew it.

    Returns:
        Hypothesis id to an unverified claim that records the reason.
    """
    claims = {}
    for hypothesis_id, why in reasons.items():
        result = ledger.results[hypothesis_id]
        spec = ledger.hypotheses[hypothesis_id].spec
        claims[hypothesis_id] = qualification.Claim(
            claim_id=hypothesis_id,
            statement=statement(spec),
            check_ids=(f"{hypothesis_id}:X",),
            adjusted_p_value=None,
            effect=result.check.effect if result.check else 0.0,
            tier=qualification.Tier.UNVERIFIED,
            assumptions=(
                f"withdrawn before C was opened, not tested on C: {why}",
                "exploration finding only; it does not change feature "
                "admission or preprocessing",
            ),
        )
    return claims


def stage_two_record(
    ledger: scmle.data.ledger.ledger.Ledger, withdrawal: Withdrawal
) -> dict[str, Any]:
    """Builds the accounting of a withdrawal, hashed over the surviving family.

    Args:
        ledger: The ledger; its survivors are frozen if they are not yet.
        withdrawal: The withdrawal to account for.

    Returns:
        A mapping with the survivors and their hash, the withdrawn ids with
        reasons, the remaining family and its hash, the rule and the
        guarantee statement.

    Raises:
        LedgerError: If a withdrawn id is not a frozen survivor.
    """
    survivors = ledger.freeze()
    unknown = sorted(set(withdrawal.reasons) - set(survivors.hypothesis_ids))
    if unknown:
        raise scmle.data.ledger.ledger.LedgerError(
            f"withdrawn ids {unknown} are not frozen survivors"
        )
    family = [
        i for i in survivors.hypothesis_ids if i not in withdrawal.reasons
    ]
    return {
        "survivors_sha256": survivors.sha256,
        "survivors": list(survivors.hypothesis_ids),
        "withdrawn": dict(sorted(withdrawal.reasons.items())),
        "family": family,
        "family_sha256": scmle.data.ledger.ledger.digest(
            {"survivors_sha256": survivors.sha256, "family": family}
        ),
        "rule": dict(withdrawal.rule),
        "statement": STAGE_TWO_STATEMENT,
    }


def run_confirmation(
    ledger: scmle.data.ledger.ledger.Ledger,
    rows: tuple[pandas.DataFrame, pandas.Series],
    rule: budget.MultiplicityRule,
    alpha: Fraction,
    encode: Callable[[pandas.Series], Any],
    stage_two: Mapping[str, Any] | None = None,
) -> tuple[tuple[qualification.Claim, ...], tuple[qualification.Check, ...]]:
    """Tests the survivors on the opened rows and builds checks and claims.

    Args:
        ledger: The ledger with frozen survivors.
        rows: The opened rows of `C`.
        rule: Multiplicity rule.
        alpha: Confirmation level.
        encode: Target encoder.
        stage_two: Accounting from `stage_two_record`; the survivors in its
            ``withdrawn`` mapping are left out of the tested family and get
            unverified claims instead.

    Returns:
        One claim per frozen survivor, in survivor order (withdrawn ones
        included), and the checks on `C`, which cover only the tested
        family.

    Raises:
        LedgerError: If the survivors have not been frozen.
    """
    survivors = ledger.survivor_set
    if survivors is None:
        raise scmle.data.ledger.ledger.LedgerError("freeze the survivors first")
    features, target = rows
    withdrawn = {} if stage_two is None else dict(stage_two["withdrawn"])
    ids = tuple(i for i in survivors.hypothesis_ids if i not in withdrawn)
    codes = encode(target)
    outcomes = []
    for hypothesis_id in ids:
        spec = ledger.hypotheses[hypothesis_id].spec
        column = features[spec.column] if spec.column else features.iloc[:, 0]
        outcomes.append(
            bivariate.run_statistic(
                spec.statistic,
                column.to_numpy(dtype=float),
                codes,
                spec.direction,
            )
        )
    adjustment = multiplicity.adjust([o.p for o in outcomes], rule, alpha)
    checks = []
    claims = []
    for index, hypothesis_id in enumerate(ids):
        spec = ledger.hypotheses[hypothesis_id].spec
        outcome = outcomes[index]
        passed = adjustment.rejected[index]
        tier = (
            qualification.Tier.CONFIRMED
            if passed
            else qualification.Tier.UNVERIFIED
        )
        checks.append(
            qualification.Check(
                check_id=f"{hypothesis_id}:C",
                null_hypothesis=spec.null,
                statistic=outcome.statistic,
                p_value=outcome.p,
                interval=None,
                effect=outcome.effect,
                n=outcome.n,
                decision=(
                    qualification.Decision.WARN
                    if passed
                    else qualification.Decision.PASS
                ),
                set_id=qualification.SetId.CONFIRMATION,
                seed_path=randomness.seed_path(
                    ledger.seed, f"confirm.{spec.statistic}", index
                ),
                tier=tier,
            )
        )
        claims.append(
            qualification.Claim(
                claim_id=hypothesis_id,
                statement=statement(spec),
                check_ids=(f"{hypothesis_id}:X", f"{hypothesis_id}:C"),
                adjusted_p_value=adjustment.adjusted[index],
                effect=outcome.effect,
                tier=tier,
                assumptions=assumptions(rule, len(ids), len(withdrawn)),
            )
        )
    if withdrawn:
        by_id = {c.claim_id: c for c in claims}
        by_id.update(withdrawn_claims(ledger, withdrawn))
        claims = [by_id[i] for i in survivors.hypothesis_ids]
    return tuple(claims), tuple(checks)


def default_encoding(target: pandas.Series) -> Any:
    """Encodes the target as floats; classes are coded in sorted order.

    Args:
        target: The opened confirmation target.

    Returns:
        A float array: the values of a numeric target, or the code of each
        label otherwise (labels sorted as text), NaN where missing.
    """
    if pandas.api.types.is_numeric_dtype(target):
        return target.to_numpy(dtype=float)
    classes = sorted(target.dropna().unique(), key=str)
    codes = {label: float(i) for i, label in enumerate(classes)}
    return target.map(codes).to_numpy(dtype=float)


def confirm(
    ledger: scmle.data.ledger.ledger.Ledger,
    source: ConfirmationSource | None,
    *,
    alpha_confirm: Fraction,
    rule: budget.MultiplicityRule = multiplicity.DEFAULT_RULE,
    policy: str = UNCONFIRMED,
    require_control: bool = True,
    custody_log: EventLog | None = None,
    encode: Callable[[pandas.Series], Any] = default_encoding,
    withdrawal: Withdrawal | None = None,
    claim_runner: ClaimRunner | None = None,
) -> Confirmation:
    """Freezes the survivors and tests them once on the confirmation set.

    An unavailable failure policy is refused first, by `check_policy`, with a
    `ConfigError`.

    Args:
        ledger: The run's ledger, after exploration (and the negative
            control, recorded with `Ledger.record_control`).
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

    Returns:
        The confirmation, also recorded in the ledger. Cases that stop
        before testing (no survivors, no control, floor not exceeded, no
        `C`, every survivor withdrawn) and any other failure while opening
        or testing are returned with a matching outcome and unverified
        claims, not raised.

    Raises:
        LeakageError: If the confirmation was already performed. The attempt
            is written to the custody log. A `LeakageError` from opening the
            set is also propagated. A `LedgerError` from registering claims,
            freezing the survivors or building the withdrawal accounting
            (a withdrawn id that is not a survivor) is not caught either.
    """
    check_policy(policy)
    if ledger.confirmation is not None:
        log = custody_log or (source.log if source is not None else None)
        if log is not None:
            log.append(
                "C",
                "denied",
                ACTOR,
                {"reason": "confirmation was already performed"},
            )
        raise errors.LeakageError(
            "the confirmation set was already used for the confirmation"
        )
    if claim_runner is not None:
        for item in claim_runner.registered:
            ledger.register_claim(
                item.claim_id, item.kind, item.statement, item.sha256
            )
    survivors = ledger.freeze()
    record = (
        None if withdrawal is None else stage_two_record(ledger, withdrawal)
    )
    blocked = family_block(ledger, record, alpha_confirm, require_control)
    if blocked is not None and (claim_runner is None or source is None):
        outcome, reason, held = blocked
        return finish(
            ledger,
            outcome=outcome,
            reason=reason,
            claims=held,
            checks=(),
            rule=rule,
            alpha=alpha_confirm,
            custody_head=None,
            stage_two=record,
            runner=claim_runner,
        )
    if source is None:
        return finish(
            ledger,
            outcome=NO_CONFIRMATION_SET,
            reason="no confirmation set exists for this run",
            claims=unverified_claims(ledger, "no confirmation set"),
            checks=(),
            rule=rule,
            alpha=alpha_confirm,
            custody_head=None,
            stage_two=record,
            runner=claim_runner,
        )
    if blocked is None and record is not None and not record["family"]:
        blocked = (
            ALL_WITHDRAWN,
            "every survivor was withdrawn before C; C was not opened",
            tuple(withdrawn_claims(ledger, record["withdrawn"]).values()),
        )
        if claim_runner is None:
            return finish(
                ledger,
                outcome=blocked[0],
                reason=blocked[1],
                claims=blocked[2],
                checks=(),
                rule=rule,
                alpha=alpha_confirm,
                custody_head=None,
                stage_two=record,
            )
    log = custody_log or source.log
    registered = () if claim_runner is None else claim_runner.registered
    log.append(
        "C",
        "selection_frozen",
        ACTOR,
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
        claims_hash = scmle.data.ledger.ledger.digest(
            {
                "family": claims_hash,
                "registered": [[i.claim_id, i.sha256] for i in registered],
            }
        )
    ledger_hash = ledger.head
    try:
        if not source.committed:
            source.commit(claims_hash, ledger_hash)
        rows = source.open(
            source.issue_token(),
            actor=ACTOR,
            claims_hash=claims_hash,
            ledger_hash=ledger_hash,
        )
    except errors.LeakageError:
        raise
    except Exception as failure:
        return finish(
            ledger,
            outcome=PIPELINE_FAILURE,
            reason=failure_reason("opening C", failure),
            claims=unverified_claims(ledger, "confirmation crashed"),
            checks=(),
            rule=rule,
            alpha=alpha_confirm,
            custody_head=head_of(log),
            stage_two=record,
            runner=claim_runner,
        )
    ran: ClaimRun | None = None
    if claim_runner is not None:
        try:
            ran = claim_runner.run(*rows)
        except Exception as failure:
            ran = ClaimRun(
                {i.claim_id: FAILED for i in claim_runner.registered}, {}
            )
            failed = failure_reason("evaluating the registered claims", failure)
            log.append("C", "claims_failed", ACTOR, {"reason": failed})
    if blocked is not None:
        outcome, reason, held = blocked
        return finish(
            ledger,
            outcome=outcome,
            reason=reason + "; C was opened for the registered claims only",
            claims=held,
            checks=(),
            rule=rule,
            alpha=alpha_confirm,
            custody_head=head_of(log),
            stage_two=record,
            runner=claim_runner,
            ran=ran,
        )
    try:
        claims, checks = run_confirmation(
            ledger, rows, rule, alpha_confirm, encode, record
        )
    except Exception as failure:
        return finish(
            ledger,
            outcome=PIPELINE_FAILURE,
            reason=failure_reason("testing on C", failure),
            claims=unverified_claims(ledger, "confirmation crashed"),
            checks=(),
            rule=rule,
            alpha=alpha_confirm,
            custody_head=head_of(log),
            stage_two=record,
            runner=claim_runner,
            ran=ran,
        )
    done = sum(1 for c in claims if c.tier is qualification.Tier.CONFIRMED)
    tested = len(claims) - (0 if record is None else len(record["withdrawn"]))
    return finish(
        ledger,
        outcome=CONFIRMED,
        reason=(
            f"{done} of {tested} survivors confirmed on C by "
            f"{rule.value} at {float(alpha_confirm):g}"
        ),
        claims=claims,
        checks=checks,
        rule=rule,
        alpha=alpha_confirm,
        custody_head=head_of(log),
        stage_two=record,
        runner=claim_runner,
        ran=ran,
    )


def family_block(
    ledger: scmle.data.ledger.ledger.Ledger,
    record: Mapping[str, Any] | None,
    alpha_confirm: Fraction,
    require_control: bool,
) -> tuple[str, str, tuple[qualification.Claim, ...]] | None:
    """Decides why the bivariate family cannot be promoted, if it cannot.

    The gates apply in this order: no survivors, no control, control floor
    not exceeded.

    Args:
        ledger: The ledger; its survivors are frozen if they are not yet.
        record: Withdrawal accounting, or None. Not read by the gates.
        alpha_confirm: The confirmation level. Not read by the gates.
        require_control: Whether a missing negative control blocks promotion.

    Returns:
        None when the family can be promoted, otherwise the outcome code, the
        sentence and the unverified claims.
    """
    survivors = ledger.freeze()
    if not survivors.hypothesis_ids:
        return (
            NO_SURVIVORS,
            "no hypothesis survived exploration; C was not opened",
            (),
        )
    control = ledger.control
    if control is None and require_control:
        return (
            NO_CONTROL,
            "no negative control was recorded; no claim is promoted",
            unverified_claims(ledger, "no negative control"),
        )
    if control is not None and not control["exceeds"]:
        return (
            FLOOR_NOT_EXCEEDED,
            str(control["message"]) + "; no claim is promoted",
            unverified_claims(
                ledger, "findings do not exceed the false-discovery floor"
            ),
        )
    return None


def failure_reason(stage: str, failure: Exception) -> str:
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


def head_of(log: Any) -> str | None:
    """Reads the hash of the custody log's last entry.

    Args:
        log: The custody log, read through its ``entries`` attribute.

    Returns:
        The last entry's hash, or None when the log has no entries.
    """
    entries: Sequence[dict[str, Any]] = getattr(log, "entries", ())
    return str(entries[-1]["hash"]) if entries else None
