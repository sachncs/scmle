"""Registration and testing of hypotheses, and the ledger itself."""

from collections.abc import Mapping
from fractions import Fraction
import json
from pathlib import Path
from typing import Any

import pandas

from scmle.assurance.ledger import bivariate
from scmle.assurance.ledger import online_fdr
from scmle.assurance.ledger.ledger import chain
from scmle.assurance.ledger.ledger import records
import scmle.assurance.ledger.online_fdr
from scmle.contracts import qualification
from scmle.runtime import randomness


class Journal(chain.Replay):
    """Registers hypotheses and claims and records their results."""

    def register(self, spec: records.Test) -> records.Hypothesis:
        """Registers a hypothesis before any data is touched.

        Args:
            spec: The specification of the hypothesis.

        Returns:
            The registered hypothesis, with its assigned id.

        Raises:
            LedgerError: If the survivors are frozen, the name repeats or the
                hypothesis reads a set other than `X`.
        """
        if spec.subset is not qualification.Subset.EXPLORATION:
            raise records.LedgerError(
                "only exploration-set hypotheses are registered"
            )
        hypothesis_id = f"h{len(self.hypotheses) + 1:06d}"
        self.commit(
            records.REGISTER,
            {"hypothesis_id": hypothesis_id, "spec": spec.to_dict()},
        )
        return self.hypotheses[hypothesis_id]

    def enroll(
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
            records.CLAIM,
            {
                "claim_id": claim_id,
                "kind": kind,
                "statement": statement,
                "sha256": sha256,
            },
        )
        return self.claims[claim_id]

    def pending(self, hypothesis: records.Hypothesis) -> None:
        """Requires an untested hypothesis of this ledger.

        Args:
            hypothesis: The hypothesis to check.

        Raises:
            LedgerError: If it is not registered in this ledger or has
                already been tested.
        """
        known = self.hypotheses.get(hypothesis.hypothesis_id)
        if known is None or known != hypothesis:
            raise records.LedgerError(
                f"hypothesis {hypothesis.hypothesis_id} is not registered: "
                "registration must precede evaluation"
            )
        if hypothesis.hypothesis_id in self.results:
            raise records.LedgerError(
                f"hypothesis {hypothesis.hypothesis_id} was already tested"
            )

    def test(
        self,
        hypothesis: records.Hypothesis,
        features: pandas.DataFrame,
        target: pandas.Series,
    ) -> records.Outcome:
        """Runs a hypothesis' statistic on exploration data and records it.

        If the rule has no level left (wealth exhausted), the statistic is
        not computed and the hypothesis is logged as not tested.

        Args:
            hypothesis: A registered hypothesis.
            features: Exploration features (never confirmation data). The
                column named by the specification is read, or the first
                column if it names none.
            target: Exploration target, or class codes.

        Returns:
            The recorded result. A hypothesis that is unregistered or already
            tested is refused with a `LedgerError`.
        """
        self.pending(hypothesis)
        rule = self.tester()
        if rule.level(self.state) == 0:
            return self.abstain(
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
        self, hypothesis: records.Hypothesis, outcome: bivariate.Outcome
    ) -> records.Outcome:
        """Records an outcome computed elsewhere for a registered hypothesis.

        Args:
            hypothesis: A registered, untested hypothesis.
            outcome: The p-value and statistics of its test on `X`.

        Returns:
            The recorded result.
        """
        self.pending(hypothesis)
        rule = self.tester()
        step = rule.step(self.state, online_fdr.Evidence(p=outcome.p))
        index = len(self.results)
        decision = records.REJECT if step.rejected else records.RETAIN
        check = qualification.Check(
            identifier=f"{hypothesis.hypothesis_id}:X",
            null=hypothesis.spec.null,
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
            subset=qualification.Subset.EXPLORATION,
            seeds=randomness.seed_path(
                self.seed, f"ledger.{hypothesis.spec.statistic}", index
            ),
            tier=qualification.Tier.UNVERIFIED,
        )
        return self.settle(
            hypothesis, check, outcome.p, step, decision, outcome.note
        )

    def abstain(
        self, hypothesis: records.Hypothesis, reason: str
    ) -> records.Outcome:
        """Logs a hypothesis that could not be tested (no level left).

        It still uses its place in the sequence, as an untestable hypothesis
        that is never rejected.

        Args:
            hypothesis: A registered, untested hypothesis.
            reason: Why it was not tested.

        Returns:
            The recorded result, with decision `not_tested`.
        """
        self.pending(hypothesis)
        rule = self.tester()
        step = rule.step(self.state, online_fdr.Evidence(p=1.0))
        return self.settle(
            hypothesis, None, 1.0, step, records.NOT_TESTED, reason
        )

    def settle(
        self,
        hypothesis: records.Hypothesis,
        check: qualification.Check | None,
        p: float,
        step: online_fdr.Step,
        decision: str,
        reason: str,
    ) -> records.Outcome:
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
            records.RESULT,
            {
                "hypothesis_id": hypothesis.hypothesis_id,
                "check": None if check is None else check.to_dict(),
                "evidence": {"p": p, "e": step.e_value},
                "e_value": step.e_value,
                "level": str(step.level),
                "rejected": step.rejected,
                "decision": decision,
                "wealth_before": str(step.wealth_before),
                "wealth_after": str(step.wealth_after),
                "reason": reason,
            },
        )
        return self.results[hypothesis.hypothesis_id]

    def decision(self, hypothesis: records.Hypothesis) -> records.Outcome:
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
            raise records.LedgerError(
                f"{hypothesis.hypothesis_id} has no result yet"
            )
        return result

    def survivors(self) -> tuple[records.Hypothesis, ...]:
        """Lists the rejected hypotheses so far.

        Returns:
            The rejected hypotheses, in registration order.
        """
        return tuple(
            h
            for i, h in self.hypotheses.items()
            if i in self.results and self.results[i].rejected
        )


class Ledger(Journal):
    """The hypothesis ledger of one run (file-backed or in memory)."""

    @classmethod
    def create(
        cls,
        path: str | Path | None,
        *,
        charter_hash: str,
        config: online_fdr.Settings,
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
            raise records.LedgerError(
                f"{target} already exists; use Ledger.load"
            )
        ledger = cls(target)
        rule = config.build()
        ledger.commit(
            records.OPEN,
            {
                "ledger_version": records.LEDGER_VERSION,
                "charter_hash": charter_hash,
                "rule": config.to_dict(),
                "guarantee": {
                    "statement": rule.guarantee.statement,
                    "assumptions": list(rule.guarantee.assumptions),
                    "citation": rule.guarantee.citation,
                    "verified": rule.guarantee.verified,
                    "dependence": rule.guarantee.dependence,
                },
                "alpha_confirm": str(alpha_confirm),
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
            LedgerError: If the file is missing or empty, a line is not JSON,
                a row fails its hash or `ingest` refuses it (broken chain,
                invalid order or content, a replayed level, decision or
                wealth that differs), or `expected_head` does not match.
        """
        source = Path(path)
        if not source.exists():
            raise records.LedgerError(f"ledger {source} does not exist")
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
                raise records.LedgerError(
                    f"ledger row {index} is not JSON"
                ) from error
            ledger.ingest(row)
        if not ledger.rows:
            raise records.LedgerError("ledger is empty")
        if expected_head is not None and ledger.head != expected_head:
            raise records.LedgerError(
                "ledger head does not match the recorded head: rows were "
                "removed from the end or the file was replaced"
            )
        return ledger

    @classmethod
    def open(
        cls,
        run_dir: str | Path,
        charter: Any,
        rule: str = scmle.assurance.ledger.online_fdr.ELOND,
        horizon: int | None = None,
    ) -> "Ledger":
        """Opens the run's ledger, creating it on first use.

        Args:
            run_dir: Run directory; the file is `hypothesis_ledger.jsonl`.
            charter: The `AnalysisCharter` governing the run. Its
                exploration wealth is the FDR level of the online rule and its
                confirmation level is recorded in the `open` row of a new
                ledger.
            rule: `e-lond` (default, arbitrary dependence), `lord++` or
                `saffron` (independence only).
            horizon: Discount horizon (None for ``1 / (j (j + 1))``).

        Returns:
            The existing ledger loaded from the run directory, or a new one
            if the file does not exist.

        Raises:
            LedgerError: If an existing ledger belongs to another charter or
                was created with a different rule configuration (rule, level,
                horizon or other parameters), or fails to load (see `load`).
                An unknown `rule` or invalid `horizon` is refused with a
                `ValueError` by `online_fdr.Settings`.
        """
        directory = Path(run_dir)
        path = directory / records.FILE_NAME
        config = online_fdr.Settings(
            rule=rule,
            alpha=charter.budget.exploration.initial,
            horizon=horizon,
        )
        if path.exists():
            ledger = cls.load(path)
            if ledger.header["charter_hash"] != charter.charter_hash:
                raise records.LedgerError(
                    "ledger belongs to a different charter"
                )
            if ledger.config != config:
                raise records.LedgerError(
                    "ledger was created with a different rule"
                )
            return ledger
        directory.mkdir(parents=True, exist_ok=True)
        return cls.create(
            path,
            charter_hash=charter.charter_hash,
            config=config,
            alpha_confirm=charter.budget.confirmation.alpha_confirm,
            seed=charter.seed,
        )

    def negatives(self, summary: Mapping[str, Any]) -> None:
        """Logs the negative-control summary; call before `freeze`.

        Args:
            summary: The JSON-ready summary; its hash is stored with it.
        """
        self.commit(
            records.CONTROL,
            {**summary, "sha256": records.digest(dict(summary))},
        )

    def record_diagnostic(
        self, check: qualification.Check, source: str
    ) -> None:
        """Logs a detector check run on `X`; call before `freeze`.

        Args:
            check: The check; it must have read the exploration set.
            source: Which detector produced it, for the reader.
        """
        self.commit(
            records.DIAGNOSTIC, {"source": source, "check": check.to_dict()}
        )

    def freeze(self) -> records.Selection:
        """Freezes the survivor set before the confirmation set is opened.

        Calling it again returns the same set.

        Returns:
            The frozen survivor set: the rejected hypotheses, in registration
            order, with their hash.

        Raises:
            LedgerError: If the freeze row was written but the set was not
                recorded (a defect, not a data error).
        """
        frozen = self.survivor_set
        if frozen is not None:
            return frozen
        kept = self.survivors()
        ids = tuple(h.hypothesis_id for h in kept)
        specs = {h.hypothesis_id: h.spec for h in kept}
        self.commit(
            records.SURVIVORS,
            {"ids": list(ids), "sha256": records.fingerprint(ids, specs)},
        )
        if self.survivor_set is None:  # pragma: no cover - ingest sets it
            raise records.LedgerError("survivors were not frozen")
        return self.survivor_set

    def seal(self, body: Mapping[str, Any]) -> None:
        """Appends the confirmation row; nothing may follow it.

        Args:
            body: The confirmation content; the survivors' hash is added.

        Raises:
            LedgerError: If the survivors have not been frozen or `ingest`
                refuses the row (for example, registered claims without an
                outcome in ``body["claims"]``).
        """
        if self.survivor_set is None:
            raise records.LedgerError("freeze the survivors before confirming")
        self.commit(
            records.CONFIRMATION,
            {**body, "survivors_sha256": self.survivor_set.sha256},
        )

    def verify(self) -> None:
        """Checks the ledger against its file, or replays it if in memory.

        A file-backed ledger is reloaded from its path and compared with the
        rows in memory. An in-memory ledger replays its rows into a fresh
        ledger.

        Raises:
            LedgerError: If the reloaded file fails to load or differs from
                this ledger, or a replayed row is refused.
        """
        if self.path is None:
            reloaded = Ledger(None)
            for row in self.rows:
                reloaded.ingest(row)
            return
        again = Ledger.load(self.path)
        if again.rows != self.rows:
            raise records.LedgerError("ledger file differs from memory")
