"""The append-only, hash-chained hypothesis ledger.

Every comparison made on the exploration set `X` is a logged hypothesis. The
file ``hypothesis_ledger.jsonl`` holds one JSON object per line; each stores
the hash of the previous line, so editing, removing from the middle or
reordering any line breaks the chain from there on. Rows are:

``open``
    Charter hash, the online rule and its parameters, its guarantee, seeds.
``register``
    A hypothesis specification, written *before* its data is touched.
``result``
    A `Check`, the e-value used, the level assigned, the decision and the
    wealth before and after. Registration precedes evaluation: a result for an
    unregistered or already tested hypothesis is refused in code and, on
    load, rejected as a broken ledger.
``diagnostic``
    A `Check` from a detector run on `X` (a leakage suspicion).
    It spends no wealth and never enters the survivors; it is chained so a
    suspicion that was raised cannot be removed afterwards.
``control``
    The negative-control summary (optional, before the survivors are frozen).
``claim``
    A registered claim that is not a single-feature statistic (the baseline
    comparison, the primary causal hypothesis), written before
    the survivors are frozen with the hash of its registration. It spends no
    wealth, never enters the survivors and is evaluated once, on `C`, inside
    the single opening; the confirmation row records its outcome.
``survivors``
    The rejected hypotheses, frozen with a hash before `C` is unlocked.
``confirmation``
    The outcome of the single confirmation, with the claims file hash, the
    status of every registered claim and the hashes of the report files the
    registered claims produced (``attachments``), so they are bound to the
    ledger head.

Verification checks the chain and then replays every result through the
online rule from the stored evidence, comparing level, decision and wealth
exactly (the rule uses exact fractions, so replay is bit-for-bit identical on
every platform).

Limits. A hash chain shows that nothing was changed, removed from
the middle or reordered since it was written. It cannot show that the *last*
rows were never deleted: a prefix of a valid chain is valid. Pass the head
hash recorded elsewhere (the custody log and the claims file store it) to
`verify_file` to detect tail truncation. Someone able to rewrite the whole
file and every copy of the head can forge a ledger; this is tamper-evidence on
one machine, not protection from an adversary.
"""

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scmle.contracts import errors
from scmle.contracts import evidence as evidence_module
from scmle.contracts import qualification
from scmle.contracts import signing
from scmle.core import randomness
from scmle.data.ledger import bivariate
from scmle.data.ledger import online_fdr

GENESIS = "0" * 64
LEDGER_VERSION = 1
FILE_NAME = "hypothesis_ledger.jsonl"
REJECT = "reject"
RETAIN = "retain"
NOT_TESTED = "not_tested"
OPEN = "open"
REGISTER = "register"
RESULT = "result"
CONTROL = "control"
DIAGNOSTIC = "diagnostic"
CLAIM = "claim"
SURVIVORS = "survivors"
CONFIRMATION = "confirmation"


class LedgerError(errors.ArtifactError):
    """The ledger was misused or its contents are inconsistent."""


def digest(body: Mapping[str, Any]) -> str:
    """Hashes the canonical JSON of a record.

    Args:
        body: The JSON-ready record.

    Returns:
        The SHA-256 hex digest of the canonical JSON form of `body`.
    """
    return hashlib.sha256(signing.canonical_payload(body)).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class HypothesisSpec:
    """What is tested, fixed before the data is touched.

    Attributes:
        name: Human-readable name, unique within the ledger.
        family: Family the hypothesis belongs to (for example `bivariate`).
        statistic: Registered statistic (see `bivariate.STATISTICS`).
        null: The null hypothesis, in words.
        direction: `two-sided`, `greater` or `less`.
        set_id: Set the test reads; exploration tests read `X`.
        column: Feature column the statistic reads, if any.
        code_hash: Hash of the code path that computes the statistic.
    """

    name: str
    family: str
    statistic: str
    null: str
    direction: str
    set_id: qualification.SetId
    column: str | None
    code_hash: str

    def __post_init__(self) -> None:
        """Validates the specification.

        Raises:
            ValueError: If `direction` is not a known alternative or
                `set_id` is not a `SetId`. A blank text field is refused with
                a `ValueError` by `evidence.require_text`.
        """
        for field, value in (
            ("name", self.name),
            ("family", self.family),
            ("statistic", self.statistic),
            ("null", self.null),
            ("code_hash", self.code_hash),
        ):
            evidence_module.require_text(value, field)
        if self.direction not in bivariate.DIRECTIONS:
            raise ValueError(f"direction must be one of {bivariate.DIRECTIONS}")
        if not isinstance(self.set_id, qualification.SetId):
            raise ValueError("set_id must be a SetId")

    @classmethod
    def create(
        cls,
        *,
        name: str,
        family: str,
        statistic: str,
        null: str,
        column: str | None = None,
        direction: str = bivariate.TWO_SIDED,
        set_id: qualification.SetId = qualification.SetId.EXPLORATION,
    ) -> "HypothesisSpec":
        """Builds a specification, filling the code hash from the registry.

        Args:
            name: Human-readable name, unique within the ledger.
            family: Family the hypothesis belongs to.
            statistic: Registered statistic; an unknown name is refused by
                `bivariate.code_hash` with a `ValueError`.
            null: The null hypothesis, in words.
            column: Feature column the statistic reads, if any.
            direction: `two-sided`, `greater` or `less`.
            set_id: Set the test reads (exploration by default).

        Returns:
            The specification, with the hash of the statistic's code.
        """
        return cls(
            name=name,
            family=family,
            statistic=statistic,
            null=null,
            direction=direction,
            set_id=set_id,
            column=column,
            code_hash=bivariate.code_hash(statistic),
        )

    def to_dict(self) -> dict[str, Any]:
        """Builds the JSON-ready record.

        Returns:
            A mapping of every field; the set is stored under ``set`` as its
            value.
        """
        return {
            "name": self.name,
            "family": self.family,
            "statistic": self.statistic,
            "null": self.null,
            "direction": self.direction,
            "set": self.set_id.value,
            "column": self.column,
            "code_hash": self.code_hash,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "HypothesisSpec":
        """Rebuilds a specification, rejecting unknown or missing fields.

        Args:
            record: A mapping as written by `to_dict`.

        Returns:
            The specification described by the record.

        Raises:
            ValueError: If the record has unexpected or missing fields. An
                unknown set value or an invalid field is also refused with a
                `ValueError`.
        """
        names = {
            "name",
            "family",
            "statistic",
            "null",
            "direction",
            "set",
            "column",
            "code_hash",
        }
        if set(record) != names:
            raise ValueError("unexpected or missing hypothesis fields")
        return cls(
            name=record["name"],
            family=record["family"],
            statistic=record["statistic"],
            null=record["null"],
            direction=record["direction"],
            set_id=qualification.SetId(record["set"]),
            column=record["column"],
            code_hash=record["code_hash"],
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Hypothesis:
    """A registered hypothesis.

    Attributes:
        hypothesis_id: Identifier assigned at registration (`h000001`, ...).
        spec: The specification.
    """

    hypothesis_id: str
    spec: HypothesisSpec


@dataclasses.dataclass(frozen=True, slots=True)
class Result:
    """The recorded outcome of one hypothesis.

    Attributes:
        hypothesis_id: The hypothesis.
        check: The `Check` (None when the hypothesis was not tested).
        e_value: E-value used by e-LOND, or None.
        level: Test level the rule assigned.
        decision: `reject`, `retain` or `not_tested`.
        wealth_before: Wealth before the test.
        wealth_after: Wealth after the test.
        reason: Why a hypothesis was not tested, or an empty string.
    """

    hypothesis_id: str
    check: qualification.Check | None
    e_value: float | None
    level: Fraction
    decision: str
    wealth_before: Fraction
    wealth_after: Fraction
    reason: str = ""

    @property
    def rejected(self) -> bool:
        """Whether the hypothesis was rejected."""
        return self.decision == REJECT


@dataclasses.dataclass(frozen=True, slots=True)
class SurvivorSet:
    """The frozen selection that goes to the confirmation set.

    Attributes:
        hypothesis_ids: Rejected hypotheses, in registration order.
        sha256: Hash of their ids and specifications.
    """

    hypothesis_ids: tuple[str, ...]
    sha256: str


def survivors_hash(
    ids: tuple[str, ...], specs: Mapping[str, HypothesisSpec]
) -> str:
    """Hashes the survivor ids together with their specifications.

    Args:
        ids: Survivor hypothesis ids, in registration order.
        specs: Specification of every hypothesis, by id.

    Returns:
        The digest that binds the ids to their specifications.
    """
    return digest(
        {"ids": list(ids), "specs": [specs[i].to_dict() for i in ids]}
    )


def encode_fraction(value: Fraction) -> str:
    """Writes a fraction as text.

    Args:
        value: The fraction.

    Returns:
        The fraction as ``"p/q"`` (``"p"`` when the denominator is 1).
    """
    return str(value)


class Ledger:
    """The hypothesis ledger of one run (file-backed or in memory)."""

    def __init__(self, path: Path | None) -> None:
        """Creates an empty, unopened ledger; use `create`, `load`, `open`."""
        self.path = path
        self.rows: list[dict[str, Any]] = []
        self.hypotheses: dict[str, Hypothesis] = {}
        self.results: dict[str, Result] = {}
        self.header: dict[str, Any] = {}
        self.rule: online_fdr.OnlineRule | None = None
        self.config: online_fdr.RuleConfig | None = None
        self.state = online_fdr.State()
        self.survivor_set: SurvivorSet | None = None
        self.control: dict[str, Any] | None = None
        self.diagnostics: list[qualification.Check] = []
        self.confirmation: dict[str, Any] | None = None
        self.claims: dict[str, dict[str, Any]] = {}
        self.seed = 0

    @classmethod
    def create(
        cls,
        path: str | Path | None,
        *,
        charter_hash: str,
        config: online_fdr.RuleConfig,
        alpha_confirm: Fraction,
        seed: int,
    ) -> "Ledger":
        """Starts a new ledger and writes its `open` row.

        Args:
            path: File to persist to, or None for an in-memory ledger.
            charter_hash: Hash of the governing charter.
            config: The online rule and its parameters.
            alpha_confirm: Level of the later confirmation.
            seed: The run seed.

        Returns:
            The new ledger, holding only its `open` row.

        Raises:
            LedgerError: If the file already exists.
        """
        target = Path(path) if path is not None else None
        if target is not None and target.exists():
            raise LedgerError(f"{target} already exists; use Ledger.load")
        ledger = cls(target)
        rule = config.build()
        ledger.commit(
            OPEN,
            {
                "ledger_version": LEDGER_VERSION,
                "charter_hash": charter_hash,
                "rule": config.to_dict(),
                "guarantee": {
                    "statement": rule.guarantee.statement,
                    "assumptions": list(rule.guarantee.assumptions),
                    "citation": rule.guarantee.citation,
                    "verified": rule.guarantee.verified,
                    "dependence": rule.guarantee.dependence,
                },
                "alpha_confirm": encode_fraction(alpha_confirm),
                "seed": seed,
            },
        )
        return ledger

    @classmethod
    def load(
        cls, path: str | Path, expected_head: str | None = None
    ) -> "Ledger":
        """Reads a ledger file, verifying the chain and replaying the rule.

        Args:
            path: The ledger file.
            expected_head: Head hash recorded elsewhere. When given, the last
                row must carry exactly this hash, which detects deleted rows
                at the end of the file.

        Returns:
            The ledger rebuilt from the file.

        Raises:
            LedgerError: If the file is missing, a row fails its hash, the
                order of rows is invalid or a replayed decision differs.
        """
        source = Path(path)
        if not source.exists():
            raise LedgerError(f"ledger {source} does not exist")
        lines = [
            line
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        ledger = cls(source)
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except ValueError as error:
                raise LedgerError(f"ledger row {index} is not JSON") from error
            ledger.ingest(row)
        if not ledger.rows:
            raise LedgerError("ledger is empty")
        if expected_head is not None and ledger.head != expected_head:
            raise LedgerError(
                "ledger head does not match the recorded head: rows were "
                "removed from the end or the file was replaced"
            )
        return ledger

    @classmethod
    def verify_file(
        cls, path: str | Path, expected_head: str | None = None
    ) -> "Ledger":
        """Loads a ledger, which verifies it (see `load`).

        Args:
            path: The ledger file.
            expected_head: Head hash recorded elsewhere, or None.

        Returns:
            The verified ledger.
        """
        return cls.load(path, expected_head)

    @classmethod
    def open(
        cls,
        run_dir: str | Path,
        charter: Any,
        rule: str = online_fdr.DEFAULT_RULE,
        horizon: int | None = None,
    ) -> "Ledger":
        """Opens the run's ledger, creating it on first use.

        Args:
            run_dir: Run directory; the file is `hypothesis_ledger.jsonl`.
            charter: The `AnalysisCharter` governing the run. Its
                exploration wealth is the FDR level of the online rule and its
                confirmation level is carried into the ledger.
            rule: `e-lond` (default, arbitrary dependence), `lord++` or
                `saffron` (independence only).
            horizon: Discount horizon (None for ``1 / (j (j + 1))``).

        Returns:
            The existing ledger loaded from the run directory, or a new one
            if the file does not exist.

        Raises:
            LedgerError: If an existing ledger belongs to another charter or
                was created with a different rule.
        """
        directory = Path(run_dir)
        path = directory / FILE_NAME
        config = online_fdr.RuleConfig(
            rule=rule,
            alpha=charter.error_budget.exploration.initial,
            horizon=horizon,
        )
        if path.exists():
            ledger = cls.load(path)
            if ledger.header["charter_hash"] != charter.charter_hash:
                raise LedgerError("ledger belongs to a different charter")
            if ledger.config != config:
                raise LedgerError("ledger was created with a different rule")
            return ledger
        directory.mkdir(parents=True, exist_ok=True)
        return cls.create(
            path,
            charter_hash=charter.charter_hash,
            config=config,
            alpha_confirm=charter.error_budget.confirmation.alpha_confirm,
            seed=charter.seed,
        )

    @property
    def head(self) -> str:
        """Hash of the last row (the genesis hash for an empty ledger)."""
        return str(self.rows[-1]["hash"]) if self.rows else GENESIS

    @property
    def wealth(self) -> Fraction:
        """The wealth of the online rule at present."""
        return self.require_rule().wealth(self.state)

    @property
    def frozen(self) -> bool:
        """Whether the survivors were frozen (no more hypotheses)."""
        return self.survivor_set is not None

    def require_rule(self) -> online_fdr.OnlineRule:
        """Returns the online rule, which exists once the `open` row is in.

        Returns:
            The online rule built from the `open` row.

        Raises:
            LedgerError: If there is no `open` row yet.
        """
        if self.rule is None:
            raise LedgerError("ledger has no open row")
        return self.rule

    def commit(self, kind: str, body: dict[str, Any]) -> dict[str, Any]:
        """Chains a new row, ingests it and persists it.

        Args:
            kind: Row kind (`open`, `register`, `result`, ...).
            body: The row's JSON-ready content.

        Returns:
            The stored row, with its sequence number, previous hash and own
            hash.
        """
        content = {
            "seq": len(self.rows),
            "kind": kind,
            "body": body,
            "prev": self.head,
        }
        row = {**content, "hash": digest(content)}
        self.ingest(row)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        return row

    def ingest(self, row: Mapping[str, Any]) -> None:
        """Checks one row against the chain and the rules, then applies it.

        Args:
            row: The row.

        Raises:
            LedgerError: On a broken link, wrong order or replay mismatch, an
                unknown row kind, a row after the confirmation, or a row
                whose body is malformed.
        """
        if set(row) != {"seq", "kind", "body", "prev", "hash"}:
            raise LedgerError(f"ledger row {len(self.rows)} has wrong fields")
        content = {k: row[k] for k in ("seq", "kind", "body", "prev")}
        if (
            row["seq"] != len(self.rows)
            or row["prev"] != self.head
            or row["hash"] != digest(content)
        ):
            raise LedgerError(f"ledger chain broken at row {len(self.rows)}")
        kind, body = row["kind"], row["body"]
        if not self.rows and kind != OPEN:
            raise LedgerError("the first ledger row must be 'open'")
        if self.rows and kind == OPEN:
            raise LedgerError("a ledger has exactly one 'open' row")
        if self.confirmation is not None:
            raise LedgerError("no row may follow the confirmation")
        handlers: dict[str, Callable[[Mapping[str, Any]], None]] = {
            OPEN: self.apply_open,
            REGISTER: self.apply_register,
            RESULT: self.apply_result,
            CONTROL: self.apply_control,
            DIAGNOSTIC: self.apply_diagnostic,
            CLAIM: self.apply_claim,
            SURVIVORS: self.apply_survivors,
            CONFIRMATION: self.apply_confirmation,
        }
        if kind not in handlers:
            raise LedgerError(f"unknown ledger row kind {kind!r}")
        try:
            handlers[kind](body)
        except LedgerError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise LedgerError(
                f"malformed {kind} row {len(self.rows)}: {error!r}"
            ) from error
        self.rows.append(dict(row))

    def apply_open(self, body: Mapping[str, Any]) -> None:
        """Reads the header and builds the rule.

        Args:
            body: The `open` row's content.

        Raises:
            LedgerError: If the ledger version is unsupported or the stored
                rule is invalid.
        """
        if body.get("ledger_version") != LEDGER_VERSION:
            raise LedgerError("unsupported ledger version")
        try:
            self.config = online_fdr.RuleConfig.from_dict(body["rule"])
        except (KeyError, ValueError) as error:
            raise LedgerError(f"invalid rule in the ledger: {error}") from error
        self.rule = self.config.build()
        self.state = self.rule.initial()
        self.header = dict(body)
        self.seed = int(body["seed"])

    def require_unfrozen(self) -> None:
        """Requires that the survivors are not yet frozen.

        Hypotheses may only be added before the survivors are frozen.

        Raises:
            LedgerError: If the survivors are frozen.
        """
        if self.survivor_set is not None:
            raise LedgerError("survivors are frozen: no new hypotheses")

    def apply_register(self, body: Mapping[str, Any]) -> None:
        """Adds a hypothesis to the registry.

        Args:
            body: The `register` row's content.

        Raises:
            LedgerError: If the survivors are frozen, the registration is
                invalid, its id is not the next sequential one or its name
                is already registered.
        """
        self.require_unfrozen()
        try:
            spec = HypothesisSpec.from_dict(body["spec"])
            hypothesis_id = str(body["hypothesis_id"])
        except (KeyError, ValueError) as error:
            raise LedgerError(f"invalid registration: {error}") from error
        if hypothesis_id != f"h{len(self.hypotheses) + 1:06d}":
            raise LedgerError("hypothesis ids must be sequential")
        if any(h.spec.name == spec.name for h in self.hypotheses.values()):
            raise LedgerError(f"hypothesis name {spec.name!r} is registered")
        self.hypotheses[hypothesis_id] = Hypothesis(hypothesis_id, spec)

    def apply_result(self, body: Mapping[str, Any]) -> None:
        """Replays one result through the rule and stores it.

        Args:
            body: The `result` row's content.

        Raises:
            LedgerError: If the survivors are frozen, the hypothesis is
                unregistered or already tested, the evidence is invalid, the
                replayed level, decision or wealth differs from the stored
                one, the e-value is not the calibrated p-value, or the
                decision is invalid.
        """
        self.require_unfrozen()
        rule = self.require_rule()
        hypothesis_id = body.get("hypothesis_id")
        if hypothesis_id not in self.hypotheses:
            raise LedgerError(
                f"result for unregistered hypothesis {hypothesis_id!r}: "
                "registration must precede evaluation"
            )
        if hypothesis_id in self.results:
            raise LedgerError(f"hypothesis {hypothesis_id} was already tested")
        evidence = body["evidence"]
        try:
            step = rule.step(
                self.state,
                online_fdr.Evidence(p=evidence["p"], e=evidence["e"]),
            )
        except ValueError as error:
            raise LedgerError(f"invalid evidence: {error}") from error
        stored = (
            body["level"],
            body["rejected"],
            body["wealth_before"],
            body["wealth_after"],
        )
        replayed = (
            encode_fraction(step.level),
            step.rejected,
            encode_fraction(step.wealth_before),
            encode_fraction(step.wealth_after),
        )
        if stored != replayed:
            raise LedgerError(
                f"replay of {hypothesis_id} differs from the stored "
                f"decision: stored {stored}, replayed {replayed}"
            )
        self.check_calibration(rule, evidence)
        check = (
            None
            if body["check"] is None
            else qualification.Check.from_dict(body["check"])
        )
        decision = body["decision"]
        expected = REJECT if step.rejected else RETAIN
        if decision not in (expected, NOT_TESTED) or (
            decision == NOT_TESTED and (step.rejected or check is not None)
        ):
            raise LedgerError(f"invalid decision for {hypothesis_id}")
        self.results[hypothesis_id] = Result(
            hypothesis_id=hypothesis_id,
            check=check,
            e_value=body["e_value"],
            level=step.level,
            decision=decision,
            wealth_before=step.wealth_before,
            wealth_after=step.wealth_after,
            reason=body.get("reason", ""),
        )
        self.state = step.state

    def check_calibration(
        self, rule: online_fdr.OnlineRule, evidence: Mapping[str, Any]
    ) -> None:
        """Checks that a stored e-value is the calibrated p-value.

        Only e-LOND with both a p-value and an e-value is checked.

        Args:
            rule: The online rule.
            evidence: The stored evidence with its ``p`` and ``e`` entries.

        Raises:
            LedgerError: If the e-value differs from the calibrated p-value
                by more than a relative 1e-12.
        """
        if (
            isinstance(rule, online_fdr.ELond)
            and evidence["p"] is not None
            and evidence["e"] is not None
        ):
            expected = online_fdr.calibrate(evidence["p"], rule.kappa)
            if abs(evidence["e"] - expected) > 1e-12 * max(1.0, expected):
                raise LedgerError("stored e-value is not the calibrated p")

    def apply_control(self, body: Mapping[str, Any]) -> None:
        """Stores the negative-control summary (before the survivors).

        A control row after the survivors are frozen is refused with a
        `LedgerError`.

        Args:
            body: The `control` row's content.
        """
        self.require_unfrozen()
        self.control = dict(body)

    def apply_diagnostic(self, body: Mapping[str, Any]) -> None:
        """Stores a detector check; it never touches the online rule.

        Args:
            body: The `diagnostic` row's content.

        Raises:
            LedgerError: If the survivors are frozen or the check read a set
                other than the exploration set.
        """
        self.require_unfrozen()
        check = qualification.Check.from_dict(body["check"])
        if check.set_id is not qualification.SetId.EXPLORATION:
            raise LedgerError("diagnostics may only read the exploration set")
        self.diagnostics.append(check)

    def apply_claim(self, body: Mapping[str, Any]) -> None:
        """Registers a claim by the hash of its registration.

        Args:
            body: The `claim` row's content.

        Raises:
            LedgerError: If the survivors are frozen or the claim id is
                already used by a claim or a hypothesis.
        """
        self.require_unfrozen()
        claim_id = evidence_module.require_text(
            str(body["claim_id"]), "claim_id"
        )
        if claim_id in self.claims or claim_id in self.hypotheses:
            raise LedgerError(f"claim {claim_id!r} is already registered")
        self.claims[claim_id] = {
            "claim_id": claim_id,
            "kind": evidence_module.require_text(str(body["kind"]), "kind"),
            "statement": evidence_module.require_text(
                str(body["statement"]), "statement"
            ),
            "sha256": evidence_module.require_text(
                str(body["sha256"]), "sha256"
            ),
        }

    def apply_survivors(self, body: Mapping[str, Any]) -> None:
        """Freezes the survivor set, which must be the rejected hypotheses.

        Args:
            body: The `survivors` row's content.

        Raises:
            LedgerError: If the survivors are already frozen, the ids differ
                from the rejected hypotheses or the hash does not match.
        """
        self.require_unfrozen()
        ids = tuple(
            i
            for i in self.hypotheses
            if i in self.results and self.results[i].rejected
        )
        specs = {i: h.spec for i, h in self.hypotheses.items()}
        if tuple(body["ids"]) != ids:
            raise LedgerError("survivors differ from the rejected hypotheses")
        if body["sha256"] != survivors_hash(ids, specs):
            raise LedgerError("survivor hash does not match")
        self.survivor_set = SurvivorSet(ids, str(body["sha256"]))

    def apply_confirmation(self, body: Mapping[str, Any]) -> None:
        """Seals the ledger with the confirmation outcome.

        Args:
            body: The `confirmation` row's content.

        Raises:
            LedgerError: If the survivors are not frozen, the row names
                different survivors, or registered claims lack an outcome.
        """
        if self.survivor_set is None:
            raise LedgerError("confirmation needs frozen survivors")
        if body.get("survivors_sha256") != self.survivor_set.sha256:
            raise LedgerError("confirmation names different survivors")
        if self.claims:
            recorded = body.get("claims")
            if not isinstance(recorded, Mapping) or set(recorded) != set(
                self.claims
            ):
                raise LedgerError(
                    "the confirmation must state the outcome of every "
                    "registered claim"
                )
        self.confirmation = dict(body)

    def register(self, spec: HypothesisSpec) -> Hypothesis:
        """Registers a hypothesis before any data is touched.

        Args:
            spec: The specification of the hypothesis.

        Returns:
            The registered hypothesis, with its assigned id.

        Raises:
            LedgerError: If the survivors are frozen, the name repeats or the
                hypothesis reads a set other than `X`.
        """
        if spec.set_id is not qualification.SetId.EXPLORATION:
            raise LedgerError("only exploration-set hypotheses are registered")
        hypothesis_id = f"h{len(self.hypotheses) + 1:06d}"
        self.commit(
            REGISTER, {"hypothesis_id": hypothesis_id, "spec": spec.to_dict()}
        )
        return self.hypotheses[hypothesis_id]

    def register_claim(
        self, claim_id: str, kind: str, statement: str, sha256: str
    ) -> dict[str, Any]:
        """Registers a claim by hash before the survivors are frozen.

        The claim is evaluated once, on `C`, inside the single opening, and
        the confirmation row must record its outcome. Registering the same
        claim again with the same hash returns the stored record. The ledger
        refuses the registration with a `LedgerError` if the survivors are
        frozen, or the id is registered with another hash or is a hypothesis
        id.

        Args:
            claim_id: Unique id of the claim.
            kind: What the claim is.
            statement: The claim in words.
            sha256: Hash of the claim's registration.

        Returns:
            The stored record of the claim.
        """
        known = self.claims.get(claim_id)
        if known is not None and known["sha256"] == sha256:
            return known
        self.commit(
            CLAIM,
            {
                "claim_id": claim_id,
                "kind": kind,
                "statement": statement,
                "sha256": sha256,
            },
        )
        return self.claims[claim_id]

    def require_registered(self, hypothesis: Hypothesis) -> None:
        """Requires an untested hypothesis of this ledger.

        Args:
            hypothesis: The hypothesis to check.

        Raises:
            LedgerError: If it is not registered in this ledger or has
                already been tested.
        """
        known = self.hypotheses.get(hypothesis.hypothesis_id)
        if known is None or known != hypothesis:
            raise LedgerError(
                f"hypothesis {hypothesis.hypothesis_id} is not registered: "
                "registration must precede evaluation"
            )
        if hypothesis.hypothesis_id in self.results:
            raise LedgerError(
                f"hypothesis {hypothesis.hypothesis_id} was already tested"
            )

    def test(
        self,
        hypothesis: Hypothesis,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> Result:
        """Runs a hypothesis' statistic on exploration data and records it.

        If the rule has no level left (wealth exhausted), the statistic is
        not computed and the hypothesis is logged as not tested.

        Args:
            hypothesis: A registered hypothesis.
            features: Exploration features (never confirmation data).
            target: Exploration target, or class codes.

        Returns:
            The recorded result. A hypothesis that is unregistered or already
            tested is refused with a `LedgerError`.
        """
        self.require_registered(hypothesis)
        rule = self.require_rule()
        if rule.level(self.state) == 0:
            return self.record_not_tested(
                hypothesis, "wealth exhausted: descriptive only"
            )
        spec = hypothesis.spec
        column = features[spec.column] if spec.column else features.iloc[:, 0]
        outcome = bivariate.run_statistic(
            spec.statistic,
            column.to_numpy(dtype=float),
            target.to_numpy(dtype=float),
            spec.direction,
        )
        return self.record(hypothesis, outcome)

    def record(
        self, hypothesis: Hypothesis, outcome: bivariate.Outcome
    ) -> Result:
        """Records an outcome computed elsewhere for a registered hypothesis.

        Args:
            hypothesis: A registered, untested hypothesis.
            outcome: The p-value and statistics of its test on `X`.

        Returns:
            The recorded result.
        """
        self.require_registered(hypothesis)
        rule = self.require_rule()
        step = rule.step(self.state, online_fdr.Evidence(p=outcome.p))
        index = len(self.results)
        decision = REJECT if step.rejected else RETAIN
        check = qualification.Check(
            check_id=f"{hypothesis.hypothesis_id}:X",
            null_hypothesis=hypothesis.spec.null,
            statistic=outcome.statistic,
            p_value=outcome.p,
            interval=None,
            effect=outcome.effect,
            n=outcome.n,
            decision=(
                qualification.Decision.WARN
                if step.rejected
                else qualification.Decision.PASS
            ),
            set_id=qualification.SetId.EXPLORATION,
            seed_path=randomness.seed_path(
                self.seed, f"ledger.{hypothesis.spec.statistic}", index
            ),
            tier=qualification.Tier.UNVERIFIED,
        )
        return self.commit_result(
            hypothesis, check, outcome.p, step, decision, outcome.note
        )

    def record_not_tested(self, hypothesis: Hypothesis, reason: str) -> Result:
        """Logs a hypothesis that could not be tested (no level left).

        It still uses its place in the sequence, as an untestable hypothesis
        that is never rejected.

        Args:
            hypothesis: A registered, untested hypothesis.
            reason: Why it was not tested.

        Returns:
            The recorded result, with decision `not_tested`.
        """
        self.require_registered(hypothesis)
        rule = self.require_rule()
        step = rule.step(self.state, online_fdr.Evidence(p=1.0))
        return self.commit_result(
            hypothesis, None, 1.0, step, NOT_TESTED, reason
        )

    def commit_result(
        self,
        hypothesis: Hypothesis,
        check: qualification.Check | None,
        p: float,
        step: online_fdr.Step,
        decision: str,
        reason: str,
    ) -> Result:
        """Writes the result row for a computed step.

        Args:
            hypothesis: The hypothesis the step belongs to.
            check: The check on `X`, or None if it was not tested.
            p: The p-value given to the rule.
            step: The rule's step for this hypothesis.
            decision: `reject`, `retain` or `not_tested`.
            reason: Why it was not tested, or an empty string.

        Returns:
            The result as stored in the ledger.
        """
        self.commit(
            RESULT,
            {
                "hypothesis_id": hypothesis.hypothesis_id,
                "check": None if check is None else check.to_dict(),
                "evidence": {"p": p, "e": step.e_value},
                "e_value": step.e_value,
                "level": encode_fraction(step.level),
                "rejected": step.rejected,
                "decision": decision,
                "wealth_before": encode_fraction(step.wealth_before),
                "wealth_after": encode_fraction(step.wealth_after),
                "reason": reason,
            },
        )
        return self.results[hypothesis.hypothesis_id]

    def decision(self, hypothesis: Hypothesis) -> Result:
        """Looks up the recorded result of a hypothesis.

        Args:
            hypothesis: The hypothesis.

        Returns:
            Its recorded result.

        Raises:
            LedgerError: If it has not been tested.
        """
        result = self.results.get(hypothesis.hypothesis_id)
        if result is None:
            raise LedgerError(f"{hypothesis.hypothesis_id} has no result yet")
        return result

    def survivors(self) -> tuple[Hypothesis, ...]:
        """Lists the rejected hypotheses so far.

        Returns:
            The rejected hypotheses, in registration order.
        """
        return tuple(
            h
            for i, h in self.hypotheses.items()
            if i in self.results and self.results[i].rejected
        )

    def record_control(self, summary: Mapping[str, Any]) -> None:
        """Logs the negative-control summary; call before `freeze`.

        Args:
            summary: The JSON-ready summary; its hash is stored with it.
        """
        self.commit(CONTROL, {**summary, "sha256": digest(dict(summary))})

    def record_diagnostic(
        self, check: qualification.Check, source: str
    ) -> None:
        """Logs a detector check run on `X`; call before `freeze`.

        Args:
            check: The check; it must have read the exploration set.
            source: Which detector produced it, for the reader.
        """
        self.commit(DIAGNOSTIC, {"source": source, "check": check.to_dict()})

    def freeze(self) -> SurvivorSet:
        """Freezes the survivor set before the confirmation set is opened.

        Calling it again returns the same set.

        Returns:
            The frozen survivor set: the rejected hypotheses, in registration
            order, with their hash.

        Raises:
            LedgerError: If the freeze row was written but the set was not
                recorded (a defect, not a data error).
        """
        if self.survivor_set is not None:
            return self.survivor_set
        kept = self.survivors()
        ids = tuple(h.hypothesis_id for h in kept)
        specs = {h.hypothesis_id: h.spec for h in kept}
        self.commit(
            SURVIVORS, {"ids": list(ids), "sha256": survivors_hash(ids, specs)}
        )
        if self.survivor_set is None:  # pragma: no cover - ingest sets it
            raise LedgerError("survivors were not frozen")
        return self.survivor_set

    def seal(self, body: Mapping[str, Any]) -> None:
        """Appends the confirmation row; nothing may follow it.

        Args:
            body: The confirmation content; the survivors' hash is added.

        Raises:
            LedgerError: If the survivors have not been frozen.
        """
        if self.survivor_set is None:
            raise LedgerError("freeze the survivors before confirming")
        self.commit(
            CONFIRMATION,
            {**body, "survivors_sha256": self.survivor_set.sha256},
        )

    def verify(self) -> None:
        """Re-reads the file and checks it against memory.

        Raises:
            LedgerError: If the file differs from this ledger.
        """
        if self.path is None:
            reloaded = Ledger(None)
            for row in self.rows:
                reloaded.ingest(row)
            return
        again = Ledger.load(self.path)
        if again.rows != self.rows:
            raise LedgerError("ledger file differs from memory")
