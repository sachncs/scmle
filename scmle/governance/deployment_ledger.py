"""The deployment ledger: a hash-chained record of the closed loop.

The hypothesis ledger (`scmle.data.ledger.ledger`) closes at its
confirmation, so a production loop cannot append to it. This ledger has its
own chain, with the same row layout and hash (`ledger.digest`), and the same
limits: it shows that nothing was edited, removed from the middle or
reordered, not that the newest rows were never removed (compare `head` with a
copy kept elsewhere), and not that nobody rewrote file and copies together.

Row kinds (``open`` first and only once):

``open``
    Charter hash, the loop policy and its hash, the fingerprints of keys the
    system itself holds (an attestation signed with one of them is refused).
``observe_summary``, ``alarm``, ``retrain``
    What the loop saw and did, free-form but chained.
``challenger_result``
    A challenger's registration (``phase = "registered"``, before it is
    scored) and its decision (``phase = "decided"``) with the e-value, the
    e-LOND level and the wealth. Replay re-runs e-LOND over the decided rows
    and compares level, decision and wealth exactly.
``promotion_proposal``, ``rollback_proposal``
    Proposals only; the registry's signed approval still moves the alias.
``escalation``
    The full pending question, evidence references and the system's proposed
    default, written before anything waits (``phase = "open"``); a timeout
    closes it with the safe default (``phase = "timeout"``).
``attestation``
    A human reply, bound to one escalation and to the hash of its question,
    signed with a key the system does not hold, with an expiry.
``halt``
    Entering (``phase = "enter"``) or leaving (``phase = "release"``) the
    state in which every case is escalated.
``incident``
    An incident opened because of a halt: the reason, the batch and the
    status ``open``. It is a record that a person must look, not an incident
    workflow (`scmle.monitoring.incidents` wants an SLO evaluation, which a
    halt does not have).
``rollout_experiment``, ``rollout_assignment``, ``rollout_decision``
    The decision layer's rollout: an experiment is started with the
    level the `DecisionBudget` assigned and the budget after the assignment;
    each batch of units is logged with the arm and the probability that arm
    had (the propensity), and only for a started experiment; the decision
    (`hold`, `promote`, `hold_final`, `hold_invalid`, or a rollback) is
    chained with the e-process values. A promotion or rollback still needs the
    registry's signed approval.
``envelope``, ``envelope_used``, ``envelope_revoked``
    Authority a person signed in advance for an action that is not on the
    never list (`scmle.governance.envelope`): the actions with a use limit
    each, the scope, an expiry and the signing key's fingerprint. A use is
    chained before the action runs and is refused at ingest when the
    envelope is unknown, revoked, out of scope, does not cover the action or
    has spent its limit; revoking is allowed to anyone because it only
    removes authority.
``controller``
    The unattended controller's journal (`scmle.operations.controller`): the
    intent to apply an inbox file (by content digest) and that it was
    applied, a rejected file, and one row per tick. A digest is applied once.
``deployment``
    A release event confirmed or rejected by the runtime (`ReleaseRegistry`):
    the model, version, environment, revision and image digest, and whether
    the platform confirmed the promotion, the rollback or refused it.
``audit``
    A security event of the prediction service: a model activation, an
    access decision (allowed or denied) or an administrative action. Chained
    so an entry cannot be removed or edited afterwards; it carries who, what,
    the outcome and a request id, never a credential or a request body.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import datetime
import enum
from fractions import Fraction
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any
from typing import Self

from scmle.contracts import signing
from scmle.data.ledger import ledger as ledger_module
from scmle.monitoring import sequential_tests

FILE_NAME = "deployment_ledger.jsonl"
LEDGER_VERSION = 1
GENESIS = ledger_module.GENESIS
OPEN = "open"
OBSERVE_SUMMARY = "observe_summary"
ALARM = "alarm"
RETRAIN = "retrain"
CHALLENGER_RESULT = "challenger_result"
PROMOTION_PROPOSAL = "promotion_proposal"
ROLLBACK_PROPOSAL = "rollback_proposal"
ESCALATION = "escalation"
ATTESTATION = "attestation"
ATTESTATION_USED = "attestation_used"
HALT = "halt"
INCIDENT = "incident"
AUDIT = "audit"
DEPLOYMENT = "deployment"
ROLLOUT_EXPERIMENT = "rollout_experiment"
ROLLOUT_ASSIGNMENT = "rollout_assignment"
ROLLOUT_DECISION = "rollout_decision"
ENVELOPE = "envelope"
ENVELOPE_USED = "envelope_used"
ENVELOPE_REVOKED = "envelope_revoked"
CONTROLLER = "controller"
ENVELOPE_ACTIONS = ("train_challenger",)
INTENT = "intent"
APPLIED = "applied"
REJECTED = "rejected"
TICK = "tick"
CONTROLLER_PHASES = (INTENT, APPLIED, REJECTED, TICK)
CONTROLLER_INPUTS = ("batch", "labels")
KINDS = (
    OPEN,
    OBSERVE_SUMMARY,
    ALARM,
    RETRAIN,
    CHALLENGER_RESULT,
    PROMOTION_PROPOSAL,
    ROLLBACK_PROPOSAL,
    ESCALATION,
    ATTESTATION,
    ATTESTATION_USED,
    HALT,
    INCIDENT,
    AUDIT,
    DEPLOYMENT,
    ROLLOUT_EXPERIMENT,
    ROLLOUT_ASSIGNMENT,
    ROLLOUT_DECISION,
    ENVELOPE,
    ENVELOPE_USED,
    ENVELOPE_REVOKED,
    CONTROLLER,
)
REGISTERED = "registered"
DECIDED = "decided"
PHASE_OPEN = "open"
PHASE_TIMEOUT = "timeout"
ENTER = "enter"
RELEASE = "release"


class DeploymentLedgerError(ledger_module.LedgerError):
    """The deployment ledger was misused or its contents are inconsistent."""


class ReplyDecision(enum.StrEnum):
    """What a human reply decides."""

    APPROVE = "approve"
    REJECT = "reject"
    OVERRIDE = "override"
    CANNOT_DECIDE = "cannot_decide"


def key_fingerprint(key: str | bytes) -> str:
    """A short public fingerprint of a signing key (never the key).

    Args:
        key: The signing key.

    Returns:
        The first 16 hex characters of a domain-separated SHA-256 of the
        key; it cannot be used to recover the key.
    """
    raw = key.encode() if isinstance(key, str) else key
    return hashlib.sha256(b"scmle-key-fingerprint:" + raw).hexdigest()[:16]


def parse_date(value: str, name: str) -> datetime.date:
    """Parses an ISO 8601 calendar date, normally ``YYYY-MM-DD``.

    Args:
        value: The text to parse; whatever `datetime.date.fromisoformat`
            accepts, which is wider than ``YYYY-MM-DD``.
        name: The field name, for the error message.

    Returns:
        The date.

    Raises:
        DeploymentLedgerError: If `value` is not a string that
            `datetime.date.fromisoformat` accepts.
    """
    try:
        return datetime.date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise DeploymentLedgerError(f"{name} must be YYYY-MM-DD") from error


def question_hash(question: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of a question.

    Args:
        question: The question as a mapping.

    Returns:
        The hex digest.
    """
    return ledger_module.digest(question)


@dataclasses.dataclass(frozen=True, slots=True)
class Attestation:
    """A person's signed reply to one escalation.

    The fields are a domain owner's sign-off: who, in what role, on
    what date, for what scope. The signature covers every
    other field, so a reply cannot be moved to another question, scope or
    date. HMAC is symmetric: whoever can verify can also sign, so the verifier
    must keep the key where the system cannot read it; the ledger
    refuses the fingerprints of keys the system declared as its own.

    Attributes:
        escalation_id: The escalation answered.
        question_hash: Hash of that escalation's question.
        decision: What the person decides.
        statement: What the person asserts.
        evidence: Ticket, log table or document id.
        role: The person's role.
        name_or_id: The person.
        date: Date of the reply.
        scope: Model version hash and stream id the reply applies to.
        expires: Last day the reply is valid.
        key_id: Fingerprint of the signing key.
        signature: HMAC-SHA256 of the other fields.
    """

    escalation_id: str
    question_hash: str
    decision: ReplyDecision
    statement: str
    evidence: str
    role: str
    name_or_id: str
    date: str
    scope: str
    expires: str
    key_id: str
    signature: str = ""

    def __post_init__(self) -> None:
        """Validates the record's shape.

        Raises:
            DeploymentLedgerError: If a required text field is blank, the
                decision is not a `ReplyDecision`, a date is not
                ``YYYY-MM-DD``, or the expiry is before the date.
        """
        for name in (
            "escalation_id",
            "question_hash",
            "statement",
            "evidence",
            "role",
            "name_or_id",
            "scope",
            "key_id",
        ):
            if not str(getattr(self, name)).strip():
                raise DeploymentLedgerError(f"attestation {name} is required")
        if not isinstance(self.decision, ReplyDecision):
            raise DeploymentLedgerError("decision must be a ReplyDecision")
        issued = parse_date(self.date, "date")
        if parse_date(self.expires, "expires") < issued:
            raise DeploymentLedgerError("expires is before the date")

    def unsigned_dict(self) -> dict[str, Any]:
        """The signed content.

        Returns:
            The record's fields, with the decision as text and without the
            signature.
        """
        record = dataclasses.asdict(self)
        record["decision"] = self.decision.value
        del record["signature"]
        return record

    def to_dict(self) -> dict[str, Any]:
        """The full record.

        Returns:
            The signed content plus the signature.
        """
        return {**self.unsigned_dict(), "signature": self.signature}

    @classmethod
    def create(cls, key: str | bytes, **fields: Any) -> Self:
        """Builds and signs a reply with `key` (the person's key).

        Args:
            key: Signing key; its fingerprint becomes `key_id`.
            **fields: The other fields, all of them required except
                `signature`; `decision` may be given as its text value.

        Returns:
            The signed reply. A `decision` that is not a `ReplyDecision`
            value fails with `ValueError`, a missing one with `KeyError`; the
            fields are otherwise validated as in construction.
        """
        fields["decision"] = ReplyDecision(fields["decision"])
        unsigned = cls(key_id=key_fingerprint(key), **fields)
        signature = signing.sign_payload(unsigned.unsigned_dict(), key)
        return dataclasses.replace(unsigned, signature=signature)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a reply from `to_dict`.

        Args:
            record: A mapping as produced by `to_dict`.

        Returns:
            The reply.

        Raises:
            DeploymentLedgerError: If the fields differ from the reply's fields
                or fail the checks of construction. A `decision` that is not
                a `ReplyDecision` value fails with `ValueError`.
        """
        names = {f.name for f in dataclasses.fields(cls)}
        if set(record) != names:
            raise DeploymentLedgerError("unexpected or missing reply fields")
        values = dict(record)
        values["decision"] = ReplyDecision(values["decision"])
        return cls(**values)

    def verify(self, key: str | bytes) -> bool:
        """Whether the signature matches under `key`.

        Args:
            key: The verification key.

        Returns:
            True if the reply is signed, was made with a key of this
            fingerprint and the signature matches its content.
        """
        if not self.signature or self.key_id != key_fingerprint(key):
            return False
        expected = signing.sign_payload(self.unsigned_dict(), key)
        return hmac.compare_digest(self.signature, expected)


@dataclasses.dataclass(frozen=True, slots=True)
class Envelope:
    """Authority a person signs in advance for actions off the never list.

    The never list (`scmle.governance.never_list`) stays as it is: an
    envelope cannot name any of its actions, only the actions in
    `ENVELOPE_ACTIONS`. It bounds what the unattended controller may do by
    action, scope, count and time, and the signature covers every other
    field. HMAC is symmetric, so the verifier must keep the key where the
    system cannot read it; the ledger refuses the fingerprints of
    keys the system declared as its own.

    Attributes:
        envelope_id: Identifier, unique in the ledger.
        limits: For each permitted action, how many times it may be used.
        statement: What the person authorises, in words.
        role: The person's role.
        name_or_id: The person.
        date: First day the envelope is valid.
        scope: Model version hash and stream id it applies to.
        expires: Last day the envelope is valid.
        key_id: Fingerprint of the signing key.
        signature: HMAC-SHA256 of the other fields.
    """

    envelope_id: str
    limits: Mapping[str, int]
    statement: str
    role: str
    name_or_id: str
    date: str
    scope: str
    expires: str
    key_id: str
    signature: str = ""

    def __post_init__(self) -> None:
        """Validates the record's shape.

        Raises:
            DeploymentLedgerError: If a required text field is blank, no action
                is permitted, an action cannot be put in an envelope, a limit is
                not a positive integer, a date is not ``YYYY-MM-DD``, or the
                expiry is before the date.
        """
        for name in (
            "envelope_id",
            "statement",
            "role",
            "name_or_id",
            "scope",
            "key_id",
        ):
            if not str(getattr(self, name)).strip():
                raise DeploymentLedgerError(f"envelope {name} is required")
        if not self.limits:
            raise DeploymentLedgerError(
                "an envelope permits at least one action"
            )
        for action, limit in self.limits.items():
            if action not in ENVELOPE_ACTIONS:
                raise DeploymentLedgerError(
                    f"{action!r} cannot be put in an envelope; only "
                    f"{list(ENVELOPE_ACTIONS)} can"
                )
            if (
                isinstance(limit, bool)
                or not isinstance(limit, int)
                or limit < 1
            ):
                raise DeploymentLedgerError(
                    f"the limit for {action!r} must be a positive integer"
                )
        issued = parse_date(self.date, "date")
        if parse_date(self.expires, "expires") < issued:
            raise DeploymentLedgerError("expires is before the date")

    def unsigned_dict(self) -> dict[str, Any]:
        """The signed content.

        Returns:
            The record's fields without the signature.
        """
        record = dataclasses.asdict(self)
        record["limits"] = dict(self.limits)
        del record["signature"]
        return record

    def to_dict(self) -> dict[str, Any]:
        """The full record.

        Returns:
            The signed content plus the signature.
        """
        return {**self.unsigned_dict(), "signature": self.signature}

    @classmethod
    def create(cls, key: str | bytes, **fields: Any) -> Self:
        """Builds and signs an envelope with `key` (the person's key).

        Args:
            key: Signing key; its fingerprint becomes `key_id`.
            **fields: The other envelope fields, all of them required except
                `signature`; they are validated as in construction.

        Returns:
            The signed envelope.
        """
        unsigned = cls(key_id=key_fingerprint(key), **fields)
        signature = signing.sign_payload(unsigned.unsigned_dict(), key)
        return dataclasses.replace(unsigned, signature=signature)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds an envelope from `to_dict`.

        Args:
            record: A mapping as produced by `to_dict`.

        Returns:
            The envelope.

        Raises:
            DeploymentLedgerError: If the fields differ from the envelope's
                fields or fail the checks of construction.
        """
        names = {f.name for f in dataclasses.fields(cls)}
        if set(record) != names:
            raise DeploymentLedgerError("unexpected or missing envelope fields")
        return cls(**dict(record))

    def verify(self, key: str | bytes) -> bool:
        """Whether the signature matches under `key`.

        Args:
            key: The verification key.

        Returns:
            True if the envelope is signed, was made with a key of this
            fingerprint and the signature matches its content.
        """
        if not self.signature or self.key_id != key_fingerprint(key):
            return False
        expected = signing.sign_payload(self.unsigned_dict(), key)
        return hmac.compare_digest(self.signature, expected)


class DeploymentLedger:
    """The ledger of one deployment (file-backed or in memory)."""

    def __init__(self, path: Path | None) -> None:
        """Creates an empty, unopened ledger; use `create` or `load`.

        Args:
            path: File to persist rows to, or None for an in-memory ledger.
        """
        self.path = path
        self.rows: list[dict[str, Any]] = []
        self.header: dict[str, Any] = {}
        self.system_keys: frozenset[str] = frozenset()
        self.escalations: dict[str, dict[str, Any]] = {}
        self.closed: dict[str, str] = {}
        self.attestations: dict[str, dict[str, Any]] = {}
        self.used: dict[str, dict[str, Any]] = {}
        self.promotion: sequential_tests.PromotionSequence | None = None
        self.registered: list[str] = []
        self.decided: dict[str, dict[str, Any]] = {}
        self.halted = False
        self.experiments: dict[str, dict[str, Any]] = {}
        self.envelopes: dict[str, dict[str, Any]] = {}
        self.envelope_uses: dict[str, dict[str, int]] = {}
        self.envelope_revoked: dict[str, dict[str, Any]] = {}
        self.intents: dict[str, dict[str, Any]] = {}
        self.consumed: dict[str, dict[str, Any]] = {}
        self.rejected: dict[str, dict[str, Any]] = {}

    @classmethod
    def create(
        cls,
        path: str | Path | None,
        *,
        charter_hash: str,
        policy: Mapping[str, Any],
        alpha_promo: str,
        system_keys: Sequence[str] = (),
    ) -> Self:
        """Starts a ledger and writes its `open` row.

        Args:
            path: File to persist to, or None for an in-memory ledger.
            charter_hash: Hash of the governing charter.
            policy: The loop policy as a JSON-ready mapping.
            alpha_promo: Level of the e-LOND rule across challengers, as
                ``"p/q"``.
            system_keys: Fingerprints of keys the system holds; attestations
                signed with them are refused.

        Returns:
            The new ledger, holding its `open` row.

        Raises:
            DeploymentLedgerError: If the file exists.
        """
        target = Path(path) if path is not None else None
        if target is not None and target.exists():
            raise DeploymentLedgerError(f"{target} exists; use load")
        created = cls(target)
        created.commit(
            OPEN,
            {
                "ledger_version": LEDGER_VERSION,
                "charter_hash": charter_hash,
                "policy": dict(policy),
                "policy_hash": ledger_module.digest(policy),
                "alpha_promo": alpha_promo,
                "system_keys": sorted(system_keys),
            },
        )
        return created

    @classmethod
    def load(cls, path: str | Path, expected_head: str | None = None) -> Self:
        """Reads a ledger, verifying the chain and replaying e-LOND.

        Args:
            path: The file.
            expected_head: Head recorded elsewhere; detects deleted rows at
                the end.

        Returns:
            The ledger with every row applied.

        Raises:
            DeploymentLedgerError: On a missing file, a line that is not
                JSON, a broken chain, an invalid row, a replay mismatch, a
                file without rows, or a head that differs from
                `expected_head`.
        """
        source = Path(path)
        if not source.exists():
            raise DeploymentLedgerError(f"ledger {source} does not exist")
        loaded = cls(source)
        for index, line in enumerate(
            source.read_text(encoding="utf-8").splitlines()
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise DeploymentLedgerError(
                    f"ledger row {index} is not JSON"
                ) from error
            loaded.ingest(row)
        if not loaded.rows:
            raise DeploymentLedgerError("ledger is empty")
        if expected_head is not None and loaded.head != expected_head:
            raise DeploymentLedgerError(
                "ledger head does not match the recorded head: rows were "
                "removed from the end or the file was replaced"
            )
        return loaded

    @property
    def head(self) -> str:
        """Hash of the last row (genesis for an empty ledger)."""
        return str(self.rows[-1]["hash"]) if self.rows else GENESIS

    def commit(self, kind: str, body: Mapping[str, Any]) -> dict[str, Any]:
        """Chains a row, ingests it and persists it.

        A row that `ingest` refuses is neither kept nor written. A file
        ledger gets the row appended as one JSON line, creating the parent
        directory if needed; an in-memory ledger only keeps it.

        Args:
            kind: The row kind.
            body: The row content.

        Returns:
            The row: its sequence number (the count of earlier rows), kind,
            body, previous hash and own hash.
        """
        content = {
            "seq": len(self.rows),
            "kind": kind,
            "body": dict(body),
            "prev": self.head,
        }
        row = {**content, "hash": ledger_module.digest(content)}
        self.ingest(row)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        return row

    def ingest(self, row: Mapping[str, Any]) -> None:
        """Checks one row against the chain and the rules, then applies it.

        Args:
            row: A row with `seq`, `kind`, `body`, `prev` and `hash`.

        Raises:
            DeploymentLedgerError: If the row has the wrong fields, a wrong
                sequence number, previous hash or hash, an unknown kind, an
                `open` kind that is not first (or a second one), or a body
                that is malformed or fails the kind's rules, including a
                replay mismatch.
            TypeError: Raised internally for a body that is not an object; it
                is turned into `DeploymentLedgerError` and never reaches the
                caller.
        """
        if set(row) != {"seq", "kind", "body", "prev", "hash"}:
            raise DeploymentLedgerError(
                f"ledger row {len(self.rows)} has wrong fields"
            )
        content = {k: row[k] for k in ("seq", "kind", "body", "prev")}
        if (
            row["seq"] != len(self.rows)
            or row["prev"] != self.head
            or row["hash"] != ledger_module.digest(content)
        ):
            raise DeploymentLedgerError(
                f"ledger chain broken at row {len(self.rows)}"
            )
        kind, body = row["kind"], row["body"]
        if kind not in KINDS:
            raise DeploymentLedgerError(f"unknown ledger row kind {kind!r}")
        if not self.rows and kind != OPEN:
            raise DeploymentLedgerError("the first ledger row must be 'open'")
        if self.rows and kind == OPEN:
            raise DeploymentLedgerError("a ledger has exactly one 'open' row")
        handlers: dict[str, Callable[[Mapping[str, Any]], None]] = {
            OPEN: self.apply_open,
            CHALLENGER_RESULT: self.apply_challenger,
            ESCALATION: self.apply_escalation,
            ATTESTATION: self.apply_attestation,
            ATTESTATION_USED: self.apply_used,
            HALT: self.apply_halt,
            ROLLOUT_EXPERIMENT: self.apply_experiment,
            ROLLOUT_ASSIGNMENT: self.apply_assignment,
            ROLLOUT_DECISION: self.apply_rollout_decision,
            ENVELOPE: self.apply_envelope,
            ENVELOPE_USED: self.apply_envelope_used,
            ENVELOPE_REVOKED: self.apply_envelope_revoked,
            CONTROLLER: self.apply_controller,
        }
        try:
            if not isinstance(body, dict):
                raise TypeError("body must be an object")
            handlers.get(kind, self.apply_free)(body)
        except DeploymentLedgerError:
            raise
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            raise DeploymentLedgerError(
                f"malformed {kind} row {len(self.rows)}: {error!r}"
            ) from error
        self.rows.append(dict(row))

    def apply_free(self, body: Mapping[str, Any]) -> None:
        """Rows without a state machine are chained as they are.

        Args:
            body: The row content; not interpreted.
        """

    def apply_open(self, body: Mapping[str, Any]) -> None:
        """Reads the header and builds the e-LOND sequence.

        Args:
            body: The `open` row content.

        Raises:
            DeploymentLedgerError: If the ledger version is unsupported or the
                policy does not match its hash.
        """
        if body["ledger_version"] != LEDGER_VERSION:
            raise DeploymentLedgerError("unsupported ledger version")
        if body["policy_hash"] != ledger_module.digest(body["policy"]):
            raise DeploymentLedgerError("policy does not match its hash")
        self.header = dict(body)
        self.system_keys = frozenset(body["system_keys"])
        self.promotion = sequential_tests.PromotionSequence(body["alpha_promo"])

    def require_promotion(self) -> sequential_tests.PromotionSequence:
        """The e-LOND sequence, which exists once `open` is in.

        Returns:
            The e-LOND sequence.

        Raises:
            DeploymentLedgerError: If there is no `open` row yet.
        """
        if self.promotion is None:
            raise DeploymentLedgerError("ledger has no open row")
        return self.promotion

    def apply_challenger(self, body: Mapping[str, Any]) -> None:
        """Registration precedes the decision; the decision is replayed.

        Args:
            body: The challenger-result row content.

        Raises:
            DeploymentLedgerError: If the challenger is registered twice, the
                phase is neither registered nor decided, a decision names an
                unregistered challenger, one already decided or one that is
                not next in registration order, or the e-value is invalid, or
                the replay of the e-LOND step differs from what was
                recorded.
        """
        name = str(body["name"])
        sequence = self.require_promotion()
        if body["phase"] == REGISTERED:
            if name in self.registered:
                raise DeploymentLedgerError(f"{name!r} is already registered")
            sequence.register(name)
            self.registered.append(name)
            return
        if body["phase"] != DECIDED:
            raise DeploymentLedgerError("phase must be registered or decided")
        if name not in self.registered:
            raise DeploymentLedgerError(
                f"{name!r} was not registered before it was scored"
            )
        try:
            step = sequence.decide(name, float(body["e_value"]))
        except ValueError as error:
            raise DeploymentLedgerError(str(error)) from error
        recorded = (body["level"], body["rejected"], body["wealth_after"])
        replayed = (str(step.level), step.rejected, str(step.wealth_after))
        if recorded != replayed:
            raise DeploymentLedgerError(
                f"replay of e-LOND differs for {name!r}: recorded "
                f"{recorded}, replayed {replayed}"
            )
        self.decided[name] = dict(body)

    def apply_escalation(self, body: Mapping[str, Any]) -> None:
        """Opens an escalation or closes it with the safe default.

        Args:
            body: The escalation row content.

        Raises:
            DeploymentLedgerError: If an opening row's question does not match
                its hash, its id is not ``esc-`` followed by its row number
                or it lacks evidence or a proposed default, the phase is
                neither open nor timeout, or a timeout names an escalation
                that is not pending.
        """
        number = len(self.rows)
        if body["phase"] == PHASE_OPEN:
            question = body["question"]
            if body["question_hash"] != question_hash(question):
                raise DeploymentLedgerError(
                    "escalation question does not match its hash"
                )
            if body["id"] != f"esc-{number}":
                raise DeploymentLedgerError("escalation id is not its row")
            for name in ("evidence", "proposed_default"):
                if name not in body:
                    raise DeploymentLedgerError(f"escalation needs {name}")
            self.escalations[body["id"]] = dict(body)
            return
        if body["phase"] != PHASE_TIMEOUT:
            raise DeploymentLedgerError("phase must be open or timeout")
        identity = body["id"]
        if identity not in self.escalations or identity in self.closed:
            raise DeploymentLedgerError(f"{identity!r} is not pending")
        self.closed[identity] = PHASE_TIMEOUT

    def apply_attestation(self, body: Mapping[str, Any]) -> None:
        """Binds a reply to one pending escalation and its question hash.

        Args:
            body: The attestation row content.

        Raises:
            DeploymentLedgerError: If the row is not a valid reply, it answers
                no escalation, the escalation is already closed, the reply is
                bound to a different question, or it is signed with a key
                the system holds. The signature and expiry are not checked.
        """
        reply = Attestation.from_dict(body)
        escalation = self.escalations.get(reply.escalation_id)
        if escalation is None:
            raise DeploymentLedgerError("attestation answers no escalation")
        if reply.escalation_id in self.closed:
            raise DeploymentLedgerError("escalation is already closed")
        if reply.question_hash != escalation["question_hash"]:
            raise DeploymentLedgerError(
                "attestation is bound to a different question"
            )
        if reply.key_id in self.system_keys:
            raise DeploymentLedgerError(
                "attestation is signed with a key the system holds"
            )
        self.closed[reply.escalation_id] = ATTESTATION
        self.attestations[reply.escalation_id] = dict(body)

    def apply_used(self, body: Mapping[str, Any]) -> None:
        """Marks an answered escalation's attestation as spent, once.

        Args:
            body: The attestation-used row content.

        Raises:
            DeploymentLedgerError: If the attestation was not recorded or is
                already spent.
        """
        identity = str(body["escalation_id"])
        if identity not in self.attestations:
            raise DeploymentLedgerError(
                "an attestation can only be spent after it was recorded"
            )
        if identity in self.used:
            raise DeploymentLedgerError("the attestation is already spent")
        self.used[identity] = dict(body)

    def apply_envelope(self, body: Mapping[str, Any]) -> None:
        """Records a granted envelope once, never under a system key.

        Args:
            body: The envelope row content.

        Raises:
            DeploymentLedgerError: If the row is not a valid envelope, the
                envelope was already granted, or it is signed with a key the
                system holds. The signature and expiry are not checked.
        """
        grant = Envelope.from_dict(body)
        if grant.envelope_id in self.envelopes:
            raise DeploymentLedgerError(
                f"envelope {grant.envelope_id!r} was already granted"
            )
        if grant.key_id in self.system_keys:
            raise DeploymentLedgerError(
                "envelope is signed with a key the system holds"
            )
        self.envelopes[grant.envelope_id] = grant.to_dict()
        self.envelope_uses[grant.envelope_id] = {}

    def apply_envelope_used(self, body: Mapping[str, Any]) -> None:
        """Counts one use, refusing what the envelope does not allow.

        The envelope's expiry is not checked here.

        Args:
            body: The envelope-use row content.

        Raises:
            DeploymentLedgerError: If the envelope is unknown or revoked, does
                not cover the action, the scope differs or its uses are spent.
        """
        identity = str(body["envelope_id"])
        action = str(body["action"])
        grant = self.envelopes.get(identity)
        if grant is None:
            raise DeploymentLedgerError(f"envelope {identity!r} is unknown")
        if identity in self.envelope_revoked:
            raise DeploymentLedgerError(f"envelope {identity!r} was revoked")
        limit = grant["limits"].get(action)
        if limit is None:
            raise DeploymentLedgerError(
                f"envelope {identity!r} does not cover {action!r}"
            )
        if body["scope"] != grant["scope"]:
            raise DeploymentLedgerError("use is outside the envelope's scope")
        used = self.envelope_uses[identity].get(action, 0)
        if used >= limit:
            raise DeploymentLedgerError(
                f"envelope {identity!r} has spent its {limit} use(s) of "
                f"{action!r}"
            )
        self.envelope_uses[identity][action] = used + 1

    def apply_envelope_revoked(self, body: Mapping[str, Any]) -> None:
        """Removes an envelope's authority, once.

        Args:
            body: The revocation row content.

        Raises:
            DeploymentLedgerError: If the envelope is unknown or already
                revoked.
        """
        identity = str(body["envelope_id"])
        if identity not in self.envelopes:
            raise DeploymentLedgerError(f"envelope {identity!r} is unknown")
        if identity in self.envelope_revoked:
            raise DeploymentLedgerError(f"envelope {identity!r} is revoked")
        self.envelope_revoked[identity] = dict(body)

    def apply_controller(self, body: Mapping[str, Any]) -> None:
        """Keeps the controller's journal: each input digest is applied once.

        A `tick` row is accepted without any check and changes nothing.

        Args:
            body: The controller row content.

        Raises:
            DeploymentLedgerError: If the phase or input kind is not allowed,
                the input was already applied, an intent repeats, or an input is
                applied without an intent.
        """
        phase = body["phase"]
        if phase not in CONTROLLER_PHASES:
            raise DeploymentLedgerError(
                f"controller phase must be one of {list(CONTROLLER_PHASES)}"
            )
        if phase == TICK:
            return
        digest = str(body["digest"])
        if body["what"] not in CONTROLLER_INPUTS:
            raise DeploymentLedgerError(
                f"controller input must be one of {list(CONTROLLER_INPUTS)}"
            )
        if digest in self.consumed:
            raise DeploymentLedgerError(f"input {digest[:12]} was applied")
        if phase == INTENT:
            if digest in self.intents:
                raise DeploymentLedgerError(
                    f"input {digest[:12]} already has an intent"
                )
            self.intents[digest] = dict(body)
        elif phase == APPLIED:
            if digest not in self.intents:
                raise DeploymentLedgerError(
                    f"input {digest[:12]} was applied without an intent"
                )
            self.consumed[digest] = dict(body)
        else:
            self.rejected[digest] = dict(body)

    def apply_experiment(self, body: Mapping[str, Any]) -> None:
        """Starts an experiment once, at a level strictly between 0 and 1.

        Args:
            body: The experiment row content.

        Raises:
            DeploymentLedgerError: If the experiment already started or its
                level is not in (0, 1).
        """
        name = str(body["experiment_id"])
        if name in self.experiments:
            raise DeploymentLedgerError(f"experiment {name!r} already started")
        if not 0 < float(Fraction(body["level"])) < 1:
            raise DeploymentLedgerError("an experiment level lies in (0, 1)")
        self.experiments[name] = dict(body)

    def apply_assignment(self, body: Mapping[str, Any]) -> None:
        """Checks a batch of propensity-logged assignments.

        Args:
            body: The assignment row content.

        Raises:
            DeploymentLedgerError: If the experiment was not started, the lists
                differ in length, an arm is not 0 or 1, or a probability is not
                strictly in (0, 1).
        """
        name = str(body["experiment_id"])
        if name not in self.experiments:
            raise DeploymentLedgerError(
                f"assignment for {name!r}, which was not started"
            )
        units, arms, odds = body["units"], body["arm"], body["probability"]
        if not len(units) == len(arms) == len(odds):
            raise DeploymentLedgerError("assignment lists differ in length")
        if any(a not in (0, 1) for a in arms) or any(
            not 0 < float(p) < 1 for p in odds
        ):
            raise DeploymentLedgerError(
                "arms are 0 or 1 and probabilities lie strictly in (0, 1)"
            )

    def apply_rollout_decision(self, body: Mapping[str, Any]) -> None:
        """Requires the experiment a rollout decision names to have started.

        Nothing is recorded; a rollback decision also names its experiment.

        Args:
            body: The decision row content.

        Raises:
            DeploymentLedgerError: If the experiment was not started.
        """
        name = str(body["experiment_id"])
        if name not in self.experiments:
            raise DeploymentLedgerError(
                f"decision for {name!r}, which was not started"
            )

    def apply_halt(self, body: Mapping[str, Any]) -> None:
        """Enters or leaves the halted state.

        Args:
            body: The halt row content.

        Raises:
            DeploymentLedgerError: If a release occurs without a halt or
                without a recalibration entry in the row (its content is
                not examined), or the phase is neither enter nor release.
        """
        if body["phase"] == ENTER:
            self.halted = True
        elif body["phase"] == RELEASE:
            if not self.halted:
                raise DeploymentLedgerError("release without a halt")
            if not body.get("recalibration"):
                raise DeploymentLedgerError(
                    "a halt is released only on a passing recalibration"
                )
            self.halted = False
        else:
            raise DeploymentLedgerError("phase must be enter or release")

    def pending(self) -> list[dict[str, Any]]:
        """Lists the escalations with no reply and no timeout, oldest first.

        Returns:
            The bodies of the escalation rows that opened them (not the whole
            chained rows) and have no reply and no timeout, oldest first.
        """
        return [
            row
            for identity, row in self.escalations.items()
            if identity not in self.closed
        ]

    def verify(self) -> None:
        """Re-reads the file (or the rows) and checks it against memory.

        An in-memory ledger replays its rows into a fresh ledger instead.

        Raises:
            DeploymentLedgerError: If the file cannot be loaded or differs
                from this ledger, or the rows fail to replay.
        """
        if self.path is None:
            again = DeploymentLedger(None)
            for row in self.rows:
                again.ingest(row)
            return
        if DeploymentLedger.load(self.path).rows != self.rows:
            raise DeploymentLedgerError("ledger file differs from memory")
