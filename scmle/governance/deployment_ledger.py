"""The deployment ledger: a hash-chained record of the closed loop (F13).

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
    The decision layer's rollout (F11): an experiment is started with the
    level the `DecisionBudget` assigned and the budget after the assignment;
    each batch of units is logged with the arm and the probability that arm
    had (the propensity), and only for a started experiment; the decision
    (`hold`, `promote`, `hold_final`, `hold_invalid`, or a rollback) is
    chained with the e-process values. A promotion or rollback still needs the
    registry's signed approval.
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
HALT = "halt"
INCIDENT = "incident"
AUDIT = "audit"
DEPLOYMENT = "deployment"
ROLLOUT_EXPERIMENT = "rollout_experiment"
ROLLOUT_ASSIGNMENT = "rollout_assignment"
ROLLOUT_DECISION = "rollout_decision"
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
    HALT,
    INCIDENT,
    AUDIT,
    DEPLOYMENT,
    ROLLOUT_EXPERIMENT,
    ROLLOUT_ASSIGNMENT,
    ROLLOUT_DECISION,
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
    """A short public fingerprint of a signing key (never the key)."""
    raw = key.encode() if isinstance(key, str) else key
    return hashlib.sha256(b"scmle-key-fingerprint:" + raw).hexdigest()[:16]


def parse_date(value: str, name: str) -> datetime.date:
    """A ``YYYY-MM-DD`` date.

    Raises:
        DeploymentLedgerError: If `value` is not such a date.
    """
    try:
        return datetime.date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise DeploymentLedgerError(f"{name} must be YYYY-MM-DD") from error


def question_hash(question: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of a question."""
    return ledger_module.digest(question)


@dataclasses.dataclass(frozen=True, slots=True)
class Attestation:
    """A person's signed reply to one escalation.

    The fields follow the domain-owner sign-off of
    ``plan/validation-referents.md`` (section 5). The signature covers every
    other field, so a reply cannot be moved to another question, scope or
    date. HMAC is symmetric: whoever can verify can also sign, so the verifier
    must keep the key where the system cannot read it (canon P6); the ledger
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
        """Validates the record's shape."""
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
        """The signed content."""
        record = dataclasses.asdict(self)
        record["decision"] = self.decision.value
        del record["signature"]
        return record

    def to_dict(self) -> dict[str, Any]:
        """The full record."""
        return {**self.unsigned_dict(), "signature": self.signature}

    @classmethod
    def create(cls, key: str | bytes, **fields: Any) -> Self:
        """Builds and signs a reply with `key` (the person's key).

        Args:
            key: Signing key; its fingerprint becomes `key_id`.
            **fields: The other fields (`decision` may be a string).
        """
        fields["decision"] = ReplyDecision(fields["decision"])
        unsigned = cls(key_id=key_fingerprint(key), **fields)
        signature = signing.sign_payload(unsigned.unsigned_dict(), key)
        return dataclasses.replace(unsigned, signature=signature)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a reply from `to_dict`."""
        names = {f.name for f in dataclasses.fields(cls)}
        if set(record) != names:
            raise DeploymentLedgerError("unexpected or missing reply fields")
        values = dict(record)
        values["decision"] = ReplyDecision(values["decision"])
        return cls(**values)

    def verify(self, key: str | bytes) -> bool:
        """Whether the signature matches under `key`."""
        if not self.signature or self.key_id != key_fingerprint(key):
            return False
        expected = signing.sign_payload(self.unsigned_dict(), key)
        return hmac.compare_digest(self.signature, expected)


class DeploymentLedger:
    """The ledger of one deployment (file-backed or in memory)."""

    def __init__(self, path: Path | None) -> None:
        """Creates an empty, unopened ledger; use `create` or `load`."""
        self.path = path
        self.rows: list[dict[str, Any]] = []
        self.header: dict[str, Any] = {}
        self.system_keys: frozenset[str] = frozenset()
        self.escalations: dict[str, dict[str, Any]] = {}
        self.closed: dict[str, str] = {}
        self.attestations: dict[str, dict[str, Any]] = {}
        self.promotion: sequential_tests.PromotionSequence | None = None
        self.registered: list[str] = []
        self.decided: dict[str, dict[str, Any]] = {}
        self.halted = False
        self.experiments: dict[str, dict[str, Any]] = {}

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

        Raises:
            DeploymentLedgerError: On a missing file, a broken chain, an
                invalid row or a replay mismatch.
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
        """Chains a row, ingests it and persists it."""
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

        Raises:
            DeploymentLedgerError: On a broken link, wrong order, an invalid
                body or a replay mismatch.
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
            HALT: self.apply_halt,
            ROLLOUT_EXPERIMENT: self.apply_experiment,
            ROLLOUT_ASSIGNMENT: self.apply_assignment,
            ROLLOUT_DECISION: self.apply_rollout_decision,
        }
        try:
            if not isinstance(body, dict):
                raise TypeError("body must be an object")
            handlers.get(kind, self.apply_free)(body)
        except DeploymentLedgerError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise DeploymentLedgerError(
                f"malformed {kind} row {len(self.rows)}: {error!r}"
            ) from error
        self.rows.append(dict(row))

    def apply_free(self, body: Mapping[str, Any]) -> None:
        """Rows without a state machine are chained as they are."""

    def apply_open(self, body: Mapping[str, Any]) -> None:
        """Reads the header and builds the e-LOND sequence."""
        if body["ledger_version"] != LEDGER_VERSION:
            raise DeploymentLedgerError("unsupported ledger version")
        if body["policy_hash"] != ledger_module.digest(body["policy"]):
            raise DeploymentLedgerError("policy does not match its hash")
        self.header = dict(body)
        self.system_keys = frozenset(body["system_keys"])
        self.promotion = sequential_tests.PromotionSequence(body["alpha_promo"])

    def require_promotion(self) -> sequential_tests.PromotionSequence:
        """The e-LOND sequence, which exists once `open` is in."""
        if self.promotion is None:
            raise DeploymentLedgerError("ledger has no open row")
        return self.promotion

    def apply_challenger(self, body: Mapping[str, Any]) -> None:
        """Registration precedes the decision; the decision is replayed."""
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
        """Opens an escalation or closes it with the safe default."""
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
        """Binds a reply to one pending escalation and its question hash."""
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

    def apply_experiment(self, body: Mapping[str, Any]) -> None:
        """Starts an experiment once, at a positive level."""
        name = str(body["experiment_id"])
        if name in self.experiments:
            raise DeploymentLedgerError(f"experiment {name!r} already started")
        if not 0 < float(Fraction(body["level"])) < 1:
            raise DeploymentLedgerError("an experiment level lies in (0, 1)")
        self.experiments[name] = dict(body)

    def apply_assignment(self, body: Mapping[str, Any]) -> None:
        """Checks a batch of propensity-logged assignments."""
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
        """A decision names a started experiment or a monitored champion."""
        name = str(body["experiment_id"])
        if name not in self.experiments:
            raise DeploymentLedgerError(
                f"decision for {name!r}, which was not started"
            )

    def apply_halt(self, body: Mapping[str, Any]) -> None:
        """Enters or leaves the halted state."""
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
        """Escalations with no reply and no timeout, oldest first."""
        return [
            row
            for identity, row in self.escalations.items()
            if identity not in self.closed
        ]

    def verify(self) -> None:
        """Re-reads the file (or the rows) and checks it against memory.

        Raises:
            DeploymentLedgerError: If the file differs from this ledger.
        """
        if self.path is None:
            again = DeploymentLedger(None)
            for row in self.rows:
                again.ingest(row)
            return
        if DeploymentLedger.load(self.path).rows != self.rows:
            raise DeploymentLedgerError("ledger file differs from memory")
