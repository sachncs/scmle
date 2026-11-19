"""Outcome codes, the custody protocols and the records of a confirmation."""

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
from fractions import Fraction
import json
import logging
from pathlib import Path
from typing import Any
from typing import ClassVar
from typing import Protocol

import pandas

from scmle.assurance.ledger import budget
from scmle.assurance.ledger import multiplicity
import scmle.assurance.ledger.ledger
from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.contracts import records

LOGGER = logging.getLogger(__name__)


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


class Log(Protocol):
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
        ...


class Source(Protocol):
    """A sealed set that opens once (the custody `Locker`)."""

    log: Any

    @property
    def committed(self) -> bool:
        """Whether the release is already bound to a claim set."""
        ...

    def mint(self) -> str:
        """Returns the release token."""
        ...

    def commit(self, claims_hash: str, ledger_hash: str) -> str:
        """Binds the single release to a claim set and a ledger state.

        Args:
            claims_hash: Hash of the family (and registered claims) to test.
            ledger_hash: Head hash of the ledger at the time of binding.

        Returns:
            The binding digest computed from the two hashes.
        """
        ...

    def open(
        self,
        token: str,
        actor: str = ...,
        claims_hash: str | None = ...,
        ledger_hash: str | None = ...,
    ) -> tuple[pandas.DataFrame, pandas.Series]:
        """Releases the rows a single time.

        Args:
            token: The release token from `mint`.
            actor: Who opens the set.
            claims_hash: The claim-set hash the release was bound to.
            ledger_hash: The ledger head the release was bound to.

        Returns:
            The features and target of the confirmation set.
        """
        ...


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
            raise errors.InputError(
                "a withdrawal with reasons must be accounted"
            )


NOT_EVALUATED = "not_evaluated"
FAILED = "failed"


@dataclasses.dataclass(frozen=True, slots=True)
class Commitment:
    """A claim fixed in the ledger before `C` is opened.

    Attributes:
        identifier: Unique id (it must not collide with a hypothesis id).
        kind: What it is (`baseline`, `causal_primary` or `coverage`).
        statement: The claim in words.
        sha256: Hash of the registration (what was frozen before `C`).
    """

    identifier: str
    kind: str
    statement: str
    sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class Output:
    """What a `Docket` produced from the opened rows.

    Attributes:
        statuses: Claim id to a one-word status.
        documents: File name to JSON-ready content; each is hashed into the
            confirmation row.
    """

    statuses: Mapping[str, str]
    documents: Mapping[str, Mapping[str, Any]]


@dataclasses.dataclass(frozen=True, slots=True)
class Docket:
    """Registered claims and the function that evaluates them on `C`.

    Attributes:
        registered: The claims, registered in the ledger by `confirm`.
        run: Evaluates them on the one released frame (features, target). It
            is called at most once, after the family was tested.
    """

    registered: tuple[Commitment, ...]
    run: Callable[[pandas.DataFrame, pandas.Series], Output]


def merge(*runners: Docket | None) -> Docket | None:
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
    names = [item.identifier for item in registered]
    if len(set(names)) != len(names):
        raise errors.InputError("claim ids must be unique across runners")

    def run_all(features: pandas.DataFrame, target: pandas.Series) -> Output:
        statuses: dict[str, str] = {}
        documents: dict[str, Mapping[str, Any]] = {}
        for runner in chosen:
            try:
                found = runner.run(features, target)
            except Exception:
                # ERR-8: the crash used to leave no type, message or log.
                LOGGER.exception(
                    "confirmation runner for %s crashed; its claims fail",
                    [i.identifier for i in runner.registered],
                )
                statuses.update(
                    {i.identifier: FAILED for i in runner.registered}
                )
                continue
            clash = set(documents) & set(found.documents)
            if clash:
                raise errors.InputError(f"two claims wrote {sorted(clash)}")
            statuses.update(found.statuses)
            documents.update(found.documents)
        return Output(statuses, documents)

    return Docket(registered, run_all)


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
        confirm: The confirmation level.
        survivors_sha256: Hash of the frozen survivor set.
        ledger_head: Ledger head hash when the confirmation was written.
        head: Custody log head hash after the open, or None.
        control: The negative-control summary, or None.
        second: Accounting of the withdrawal (family, hash, rule,
            statement), or None when no withdrawal rule ran.
        registered: Claim id to its registration hash and status, or None
            when no claim was registered.
        documents: Report files the registered claims produced.
        migrations: The written layout's version steps, oldest first;
            version 2 writes `schema_version`, `alpha`, `digests.survivors`,
            `heads.{ledger, custody}` and `second`. The ledger and custody
            rows keep their own keys.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            {"alpha_confirm": "alpha", "stage_two": "second"},
            groups={
                "digests": {"survivors_sha256": "survivors"},
                "heads": {"ledger_head": "ledger", "custody_head": "custody"},
            },
        ),
    )

    outcome: str
    reason: str
    claims: tuple[qualification.Claim, ...]
    checks: tuple[qualification.Check, ...]
    rule: str
    confirm: Fraction
    survivors_sha256: str
    ledger_head: str
    head: str | None
    control: dict[str, Any] | None
    second: dict[str, Any] | None = None
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
            A mapping with the `schema_version` (version 1 files carry
            none), the outcome and its reason, the rule and its
            guarantee statement, the level, the hashes, the negative
            control, the claims, the checks and a note. ``second``,
            ``registered_claims`` and ``attachments`` (a hash per document)
            are present only when they apply.
        """
        record = {
            records.KEY: records.ORIGIN + len(self.migrations),
            "outcome": self.outcome,
            "reason": self.reason,
            "rule": self.rule,
            "guarantee": multiplicity.STATEMENTS[budget.Correction(self.rule)],
            "alpha": str(self.confirm),
            "digests": {"survivors": self.survivors_sha256},
            "heads": {"ledger": self.ledger_head, "custody": self.head},
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
        if self.second is not None:
            record["second"] = self.second
        if self.registered is not None:
            record["registered_claims"] = self.registered
        if self.documents:
            record["attachments"] = {
                name: scmle.assurance.ledger.ledger.digest(dict(content))
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


@dataclasses.dataclass(frozen=True, slots=True)
class Terms:
    """What one confirmation applies once `C` is reached.

    Attributes:
        rule: Multiplicity rule for the family.
        alpha: The confirmation level.
        encode: Turns the opened target into the floats the statistics read.
        record: Withdrawal accounting, or None when no withdrawal rule ran.
        runner: Registered claims evaluated on the same opening, or None.
    """

    rule: budget.Correction
    alpha: Fraction
    encode: Callable[[pandas.Series], Any]
    record: Mapping[str, Any] | None
    runner: Docket | None
