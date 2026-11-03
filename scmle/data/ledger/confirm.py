"""The single confirmation of the exploration survivors on `C`.

Order of events (canon P2). The guarantee ``FDR <= alpha_confirm`` comes only
from this stage:

1. The survivors, selected on `X` only, are frozen in the ledger with a hash.
2. If the negative-control gate fails, there is nothing to promote and `C` is
   not opened. With no survivors, `C` is not opened either.
3. The selection is written to the custody log, then `C` is opened exactly
   once through the custody API.
4. Each survivor's statistic runs on `C`; the family of survivors is adjusted
   by BH, BY (default) or Holm at ``alpha_confirm``; claims are written.

A second call raises `LeakageError` and is logged. This module takes the
sealed set as an argument; it imports nothing from custody, and no other
module of the ledger package can reach `C` (an import-graph test checks it).

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
from collections.abc import Sequence
import dataclasses
from fractions import Fraction
import json
from pathlib import Path
from typing import Any
from typing import Protocol

import pandas as pd

from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.core import randomness
from scmle.data.ledger import bivariate
from scmle.data.ledger import budget as budget_module
from scmle.data.ledger import ledger as ledger_module
from scmle.data.ledger import multiplicity

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
        """Adds an event and returns the stored entry."""


class ConfirmationSource(Protocol):
    """A sealed set that opens once (the custody `SealedSet`)."""

    log: Any

    def issue_token(self) -> str:
        """Returns the release token."""

    def open(
        self, token: str, actor: str = ...
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Releases the rows a single time."""


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
        rule: Multiplicity rule used (or planned).
        alpha_confirm: The confirmation level.
        survivors_sha256: Hash of the frozen survivor set.
        ledger_head: Ledger head hash when the confirmation was written.
        custody_head: Custody log head hash after the open, or None.
        control: The negative-control summary, or None.
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

    @property
    def confirmed(self) -> tuple[qualification.Claim, ...]:
        """Claims at tier `confirmed`."""
        return tuple(
            c for c in self.claims if c.tier is qualification.Tier.CONFIRMED
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record (``confirmed_claims.json``)."""
        return {
            "outcome": self.outcome,
            "reason": self.reason,
            "rule": self.rule,
            "guarantee": multiplicity.STATEMENTS[
                budget_module.MultiplicityRule(self.rule)
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

    def write(self, path: str | Path) -> None:
        """Writes ``confirmed_claims.json``."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def assumptions(
    rule: budget_module.MultiplicityRule, size: int
) -> tuple[str, ...]:
    """The assumptions a confirmed claim depends on."""
    return (
        f"{multiplicity.STATEMENTS[rule]} (rule {rule.value}, family of "
        f"{size} survivors fixed before C was opened)",
        "rows of C are independent of rows of X (grouped split by entity, "
        "embargo for temporal data)",
        "C was opened once and its rows were used only by this confirmation",
        "each statistic is valid under its null on C: Spearman and Kruskal "
        "assume independent rows",
        "exploration FDR is a triage budget, not part of this guarantee",
    )


def statement(spec: ledger_module.HypothesisSpec) -> str:
    """A claim's sentence."""
    return f"{spec.name}: evidence against '{spec.null}'"


def unverified_claims(
    ledger: ledger_module.Ledger, why: str
) -> tuple[qualification.Claim, ...]:
    """Claims for survivors that were not (or could not be) confirmed."""
    survivors = ledger.survivor_set
    if survivors is None:
        raise ledger_module.LedgerError("freeze the survivors first")
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
    ledger: ledger_module.Ledger,
    *,
    outcome: str,
    reason: str,
    claims: tuple[qualification.Claim, ...],
    checks: tuple[qualification.Check, ...],
    rule: budget_module.MultiplicityRule,
    alpha: Fraction,
    custody_head: str | None,
) -> Confirmation:
    """Writes the confirmation row and builds the result."""
    survivors = ledger.survivor_set
    if survivors is None:
        raise ledger_module.LedgerError("freeze the survivors first")
    ledger.seal(
        {
            "outcome": outcome,
            "reason": reason,
            "rule": rule.value,
            "alpha_confirm": str(alpha),
            "confirmed": [
                c.claim_id
                for c in claims
                if c.tier is qualification.Tier.CONFIRMED
            ],
            "custody_head": custody_head,
        }
    )
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
    )


def check_policy(policy: str) -> None:
    """Refuses a failure policy that is not available."""
    if policy == THRESHOLDOUT:
        raise errors.ConfigError(
            "the Thresholdout reusable holdout is not implemented; use the "
            "'unconfirmed' failure policy"
        )
    if policy not in POLICIES:
        raise errors.ConfigError(f"failure policy must be one of {POLICIES}")


def run_confirmation(
    ledger: ledger_module.Ledger,
    rows: tuple[pd.DataFrame, pd.Series],
    rule: budget_module.MultiplicityRule,
    alpha: Fraction,
    encode: Callable[[pd.Series], Any],
) -> tuple[tuple[qualification.Claim, ...], tuple[qualification.Check, ...]]:
    """Tests the survivors on the opened rows and builds checks and claims."""
    survivors = ledger.survivor_set
    if survivors is None:
        raise ledger_module.LedgerError("freeze the survivors first")
    features, target = rows
    ids = survivors.hypothesis_ids
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
                assumptions=assumptions(rule, len(ids)),
            )
        )
    return tuple(claims), tuple(checks)


def default_encoding(target: pd.Series) -> Any:
    """Target values as floats; classes are coded in sorted order."""
    if pd.api.types.is_numeric_dtype(target):
        return target.to_numpy(dtype=float)
    classes = sorted(target.dropna().unique(), key=str)
    codes = {label: float(i) for i, label in enumerate(classes)}
    return target.map(codes).to_numpy(dtype=float)


def confirm(
    ledger: ledger_module.Ledger,
    source: ConfirmationSource | None,
    *,
    alpha_confirm: Fraction,
    rule: budget_module.MultiplicityRule = multiplicity.DEFAULT_RULE,
    policy: str = UNCONFIRMED,
    require_control: bool = True,
    custody_log: EventLog | None = None,
    encode: Callable[[pd.Series], Any] = default_encoding,
) -> Confirmation:
    """Freezes the survivors and tests them once on the confirmation set.

    Args:
        ledger: The run's ledger, after exploration (and the negative
            control, recorded with `Ledger.record_control`).
        source: The sealed confirmation set, or None when no `C` exists.
        alpha_confirm: The confirmation level (the charter's).
        rule: `by` (default, arbitrary dependence), `bh`, `holm` or
            `bonferroni`.
        policy: Failure policy, `unconfirmed` (default).
        require_control: Whether a missing negative control blocks promotion.
        custody_log: Log that receives the `selection_frozen` event (the
            sealed set's own log when omitted).
        encode: Turns the opened target into the floats the statistics read;
            it must code classes exactly as exploration did.

    Returns:
        The confirmation, also recorded in the ledger.

    Raises:
        LeakageError: If the confirmation was already performed. The attempt
            is written to the custody log.
        ConfigError: For an unavailable failure policy.
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
    survivors = ledger.freeze()
    if not survivors.hypothesis_ids:
        return finish(
            ledger,
            outcome=NO_SURVIVORS,
            reason="no hypothesis survived exploration; C was not opened",
            claims=(),
            checks=(),
            rule=rule,
            alpha=alpha_confirm,
            custody_head=None,
        )
    control = ledger.control
    if control is None and require_control:
        return finish(
            ledger,
            outcome=NO_CONTROL,
            reason="no negative control was recorded; no claim is promoted",
            claims=unverified_claims(ledger, "no negative control"),
            checks=(),
            rule=rule,
            alpha=alpha_confirm,
            custody_head=None,
        )
    if control is not None and not control["exceeds"]:
        return finish(
            ledger,
            outcome=FLOOR_NOT_EXCEEDED,
            reason=str(control["message"]) + "; no claim is promoted",
            claims=unverified_claims(
                ledger, "findings do not exceed the false-discovery floor"
            ),
            checks=(),
            rule=rule,
            alpha=alpha_confirm,
            custody_head=None,
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
        )
    log = custody_log or source.log
    log.append(
        "C",
        "selection_frozen",
        ACTOR,
        {
            "survivors_sha256": survivors.sha256,
            "survivors": len(survivors.hypothesis_ids),
            "ledger_head": ledger.head,
            "rule": rule.value,
        },
    )
    try:
        rows = source.open(source.issue_token(), actor=ACTOR)
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
        )
    try:
        claims, checks = run_confirmation(
            ledger, rows, rule, alpha_confirm, encode
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
        )
    done = sum(1 for c in claims if c.tier is qualification.Tier.CONFIRMED)
    return finish(
        ledger,
        outcome=CONFIRMED,
        reason=(
            f"{done} of {len(claims)} survivors confirmed on C by "
            f"{rule.value} at {float(alpha_confirm):g}"
        ),
        claims=claims,
        checks=checks,
        rule=rule,
        alpha=alpha_confirm,
        custody_head=head_of(log),
    )


def failure_reason(stage: str, failure: Exception) -> str:
    """Why the confirmation crashed, under the 'unconfirmed' policy."""
    return (
        f"the confirmation failed while {stage} "
        f"({type(failure).__name__}: {failure}); C is spent and all claims "
        "stay unverified (failure policy: unconfirmed)"
    )


def head_of(log: Any) -> str | None:
    """Hash of the custody log's last entry, when it has entries."""
    entries: Sequence[dict[str, Any]] = getattr(log, "entries", ())
    return str(entries[-1]["hash"]) if entries else None
