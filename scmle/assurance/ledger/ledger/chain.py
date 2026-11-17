"""The hash chain of a ledger and the replay of its rows."""

from collections.abc import Callable
from collections.abc import Mapping
from fractions import Fraction
import json
from pathlib import Path
from typing import Any

from scmle.assurance.ledger import online_fdr
from scmle.assurance.ledger.ledger import records
from scmle.contracts import qualification
import scmle.contracts.evidence


class Chain:
    """The rows of a ledger and the replay of the exploration rows."""

    def __init__(self, path: Path | None) -> None:
        """Creates an empty, unopened ledger; use `create`, `load`, `open`."""
        self.path = path
        self.rows: list[dict[str, Any]] = []
        self.hypotheses: dict[str, records.Hypothesis] = {}
        self.results: dict[str, records.Outcome] = {}
        self.header: dict[str, Any] = {}
        self.rule: online_fdr.Tester | None = None
        self.config: online_fdr.Settings | None = None
        self.state = online_fdr.State()
        self.survivor_set: records.Selection | None = None
        self.control: dict[str, Any] | None = None
        self.diagnostics: list[qualification.Check] = []
        self.confirmation: dict[str, Any] | None = None
        self.claims: dict[str, dict[str, Any]] = {}
        self.seed = 0

    @property
    def head(self) -> str:
        """Hash of the last row (the genesis hash for an empty ledger)."""
        return str(self.rows[-1]["hash"]) if self.rows else records.GENESIS

    @property
    def wealth(self) -> Fraction:
        """The wealth of the online rule at present."""
        return self.tester().wealth(self.state)

    @property
    def frozen(self) -> bool:
        """Whether the survivors were frozen (no more hypotheses)."""
        return self.survivor_set is not None

    def tester(self) -> online_fdr.Tester:
        """Returns the online rule, which exists once the `open` row is in.

        Returns:
            The online rule built from the `open` row.

        Raises:
            LedgerError: If there is no `open` row yet.
        """
        if self.rule is None:
            raise records.LedgerError("ledger has no open row")
        return self.rule

    def guard(self) -> None:
        """Requires that the survivors are not yet frozen.

        Hypotheses may only be added before the survivors are frozen.

        Raises:
            LedgerError: If the survivors are frozen.
        """
        if self.survivor_set is not None:
            raise records.LedgerError("survivors are frozen: no new hypotheses")

    def apply_open(self, body: Mapping[str, Any]) -> None:
        """Reads the header and builds the rule.

        Args:
            body: The `open` row's content.

        Raises:
            LedgerError: If the ledger version is unsupported or the stored
                rule is invalid.
        """
        if body.get("ledger_version") != records.LEDGER_VERSION:
            raise records.LedgerError("unsupported ledger version")
        try:
            self.config = online_fdr.Settings.from_dict(body["rule"])
        except (KeyError, ValueError) as error:
            raise records.LedgerError(
                f"invalid rule in the ledger: {error}"
            ) from error
        self.rule = self.config.build()
        self.state = self.rule.initial()
        self.header = dict(body)
        self.seed = int(body["seed"])

    def enrol(self, body: Mapping[str, Any]) -> None:
        """Adds a hypothesis to the registry.

        Args:
            body: The `register` row's content.

        Raises:
            LedgerError: If the survivors are frozen, the registration is
                invalid, its id is not the next sequential one or its name
                is already registered.
        """
        self.guard()
        try:
            spec = records.Test.from_dict(body["spec"])
            hypothesis_id = str(body["hypothesis_id"])
        except (KeyError, ValueError) as error:
            raise records.LedgerError(
                f"invalid registration: {error}"
            ) from error
        if hypothesis_id != f"h{len(self.hypotheses) + 1:06d}":
            raise records.LedgerError("hypothesis ids must be sequential")
        if any(h.spec.name == spec.name for h in self.hypotheses.values()):
            raise records.LedgerError(
                f"hypothesis name {spec.name!r} is registered"
            )
        self.hypotheses[hypothesis_id] = records.Hypothesis(hypothesis_id, spec)

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
        self.guard()
        rule = self.tester()
        hypothesis_id = body.get("hypothesis_id")
        if hypothesis_id not in self.hypotheses:
            raise records.LedgerError(
                f"result for unregistered hypothesis {hypothesis_id!r}: "
                "registration must precede evaluation"
            )
        if hypothesis_id in self.results:
            raise records.LedgerError(
                f"hypothesis {hypothesis_id} was already tested"
            )
        evidence = body["evidence"]
        try:
            step = rule.step(
                self.state,
                online_fdr.Evidence(p=evidence["p"], e=evidence["e"]),
            )
        except ValueError as error:
            raise records.LedgerError(f"invalid evidence: {error}") from error
        stored = (
            body["level"],
            body["rejected"],
            body["wealth_before"],
            body["wealth_after"],
        )
        replayed = (
            str(step.level),
            step.rejected,
            str(step.wealth_before),
            str(step.wealth_after),
        )
        if stored != replayed:
            raise records.LedgerError(
                f"replay of {hypothesis_id} differs from the stored "
                f"decision: stored {stored}, replayed {replayed}"
            )
        self.audit(rule, evidence)
        check = (
            None
            if body["check"] is None
            else qualification.Check.from_dict(body["check"])
        )
        decision = body["decision"]
        expected = records.REJECT if step.rejected else records.RETAIN
        if decision not in (expected, records.NOT_TESTED) or (
            decision == records.NOT_TESTED
            and (step.rejected or check is not None)
        ):
            raise records.LedgerError(f"invalid decision for {hypothesis_id}")
        self.results[hypothesis_id] = records.Outcome(
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

    def audit(
        self, rule: online_fdr.Tester, evidence: Mapping[str, Any]
    ) -> None:
        """Checks that a stored e-value is the calibrated p-value.

        Only e-LOND with both a p-value and an e-value is checked.

        Args:
            rule: The online rule.
            evidence: The stored evidence with its ``p`` and ``e`` entries.

        Raises:
            LedgerError: If the e-value differs from the calibrated p-value
                by more than ``1e-12 * max(1, calibrated value)``.
        """
        if (
            isinstance(rule, online_fdr.Lond)
            and evidence["p"] is not None
            and evidence["e"] is not None
        ):
            expected = online_fdr.calibrate(evidence["p"], rule.kappa)
            if abs(evidence["e"] - expected) > 1e-12 * max(1.0, expected):
                raise records.LedgerError(
                    "stored e-value is not the calibrated p"
                )


class Replay(Chain):
    """Chains rows and replays the control, claim and closing rows."""

    def benchmark(self, body: Mapping[str, Any]) -> None:
        """Stores the negative-control summary (before the survivors).

        A control row after the survivors are frozen is refused with a
        `LedgerError`.

        Args:
            body: The `control` row's content.
        """
        self.guard()
        self.control = dict(body)

    def diagnose(self, body: Mapping[str, Any]) -> None:
        """Stores a detector check; it never touches the online rule.

        Args:
            body: The `diagnostic` row's content.

        Raises:
            LedgerError: If the survivors are frozen or the check read a set
                other than the exploration set.
        """
        self.guard()
        check = qualification.Check.from_dict(body["check"])
        if check.subset is not qualification.Subset.EXPLORATION:
            raise records.LedgerError(
                "diagnostics may only read the exploration set"
            )
        self.diagnostics.append(check)

    def lodge(self, body: Mapping[str, Any]) -> None:
        """Registers a claim by the hash of its registration.

        Args:
            body: The `claim` row's content.

        Raises:
            LedgerError: If the survivors are frozen or the claim id is
                already used by a claim or a hypothesis.
        """
        self.guard()
        claim_id = scmle.contracts.evidence.require_text(
            str(body["claim_id"]), "claim_id"
        )
        if claim_id in self.claims or claim_id in self.hypotheses:
            raise records.LedgerError(
                f"claim {claim_id!r} is already registered"
            )
        self.claims[claim_id] = {
            "claim_id": claim_id,
            "kind": scmle.contracts.evidence.require_text(
                str(body["kind"]), "kind"
            ),
            "statement": scmle.contracts.evidence.require_text(
                str(body["statement"]), "statement"
            ),
            "sha256": scmle.contracts.evidence.require_text(
                str(body["sha256"]), "sha256"
            ),
        }

    def survive(self, body: Mapping[str, Any]) -> None:
        """Freezes the survivor set, which must be the rejected hypotheses.

        Args:
            body: The `survivors` row's content.

        Raises:
            LedgerError: If the survivors are already frozen, the ids differ
                from the rejected hypotheses or the hash does not match.
        """
        self.guard()
        ids = tuple(
            i
            for i in self.hypotheses
            if i in self.results and self.results[i].rejected
        )
        specs = {i: h.spec for i, h in self.hypotheses.items()}
        if tuple(body["ids"]) != ids:
            raise records.LedgerError(
                "survivors differ from the rejected hypotheses"
            )
        if body["sha256"] != records.fingerprint(ids, specs):
            raise records.LedgerError("survivor hash does not match")
        self.survivor_set = records.Selection(ids, str(body["sha256"]))

    def conclude(self, body: Mapping[str, Any]) -> None:
        """Seals the ledger with the confirmation outcome.

        Args:
            body: The `confirmation` row's content.

        Raises:
            LedgerError: If the survivors are not frozen, the row names
                different survivors, or (when claims are registered) its
                ``claims`` mapping does not name exactly the registered
                claims.
        """
        if self.survivor_set is None:
            raise records.LedgerError("confirmation needs frozen survivors")
        if body.get("survivors_sha256") != self.survivor_set.sha256:
            raise records.LedgerError("confirmation names different survivors")
        if self.claims:
            recorded = body.get("claims")
            if not isinstance(recorded, Mapping) or set(recorded) != set(
                self.claims
            ):
                raise records.LedgerError(
                    "the confirmation must state the outcome of every "
                    "registered claim"
                )
        self.confirmation = dict(body)

    def commit(self, kind: str, body: dict[str, Any]) -> dict[str, Any]:
        """Chains a new row, ingests it and persists it.

        Args:
            kind: Row kind (`open`, `register`, `result`, ...).
            body: The row's JSON-ready content.

        Returns:
            The stored row, with its sequence number, previous hash and own
            hash.

        Raises:
            LedgerError: If `ingest` refuses the row; nothing is then written
                to the file.
        """
        content = {
            "seq": len(self.rows),
            "kind": kind,
            "body": body,
            "prev": self.head,
        }
        row = {**content, "hash": records.digest(content)}
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
            raise records.LedgerError(
                f"ledger row {len(self.rows)} has wrong fields"
            )
        content = {k: row[k] for k in ("seq", "kind", "body", "prev")}
        if (
            row["seq"] != len(self.rows)
            or row["prev"] != self.head
            or row["hash"] != records.digest(content)
        ):
            raise records.LedgerError(
                f"ledger chain broken at row {len(self.rows)}"
            )
        kind, body = row["kind"], row["body"]
        if not self.rows and kind != records.OPEN:
            raise records.LedgerError("the first ledger row must be 'open'")
        if self.rows and kind == records.OPEN:
            raise records.LedgerError("a ledger has exactly one 'open' row")
        if self.confirmation is not None:
            raise records.LedgerError("no row may follow the confirmation")
        handlers: dict[str, Callable[[Mapping[str, Any]], None]] = {
            records.OPEN: self.apply_open,
            records.REGISTER: self.enrol,
            records.RESULT: self.apply_result,
            records.CONTROL: self.benchmark,
            records.DIAGNOSTIC: self.diagnose,
            records.CLAIM: self.lodge,
            records.SURVIVORS: self.survive,
            records.CONFIRMATION: self.conclude,
        }
        if kind not in handlers:
            raise records.LedgerError(f"unknown ledger row kind {kind!r}")
        try:
            handlers[kind](body)
        except records.LedgerError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise records.LedgerError(
                f"malformed {kind} row {len(self.rows)}: {error!r}"
            ) from error
        self.rows.append(dict(row))
