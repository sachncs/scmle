"""Chaining, loading and verifying the deployment ledger's rows."""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any
from typing import Self

from scmle.assurance.ledger import ledger
from scmle.governance.deployment_ledger import kinds
from scmle.governance.deployment_ledger import rules


class Ledger(rules.Automation):
    """The ledger of one deployment (file-backed or in memory)."""

    __module__ = "scmle.governance.deployment_ledger"  # pinned signatures

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
            MisuseError: If the file exists.
        """
        target = Path(path) if path is not None else None
        if target is not None and target.exists():
            raise kinds.MisuseError(f"{target} exists; use load")
        created = cls(target)
        created.commit(
            kinds.OPEN,
            {
                "ledger_version": kinds.LEDGER_VERSION,
                "charter_hash": charter_hash,
                "policy": dict(policy),
                "policy_hash": ledger.digest(policy),
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
            MisuseError: On a missing file, a line that is not
                JSON, a broken chain, an invalid row, a replay mismatch, a
                file without rows, or a head that differs from
                `expected_head`.
        """
        source = Path(path)
        if not source.exists():
            raise kinds.MisuseError(f"ledger {source} does not exist")
        loaded = cls(source)
        for index, line in enumerate(
            source.read_text(encoding="utf-8").splitlines()
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise kinds.MisuseError(
                    f"ledger row {index} is not JSON"
                ) from error
            loaded.ingest(row)
        if not loaded.rows:
            raise kinds.MisuseError("ledger is empty")
        if expected_head is not None and loaded.head != expected_head:
            raise kinds.MisuseError(
                "ledger head does not match the recorded head: rows were "
                "removed from the end or the file was replaced"
            )
        return loaded

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
        row = {**content, "hash": ledger.digest(content)}
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
            MisuseError: If the row has the wrong fields, a wrong
                sequence number, previous hash or hash, an unknown kind, an
                `open` kind that is not first (or a second one), or a body
                that is malformed or fails the kind's rules, including a
                replay mismatch.
            TypeError: Raised internally for a body that is not an object; it
                is turned into `MisuseError` and never reaches the
                caller.
        """
        if set(row) != {"seq", "kind", "body", "prev", "hash"}:
            raise kinds.MisuseError(
                f"ledger row {len(self.rows)} has wrong fields"
            )
        content = {k: row[k] for k in ("seq", "kind", "body", "prev")}
        if (
            row["seq"] != len(self.rows)
            or row["prev"] != self.head
            or row["hash"] != ledger.digest(content)
        ):
            raise kinds.MisuseError(
                f"ledger chain broken at row {len(self.rows)}"
            )
        kind, body = row["kind"], row["body"]
        if kind not in kinds.KINDS:
            raise kinds.MisuseError(f"unknown ledger row kind {kind!r}")
        if not self.rows and kind != kinds.OPEN:
            raise kinds.MisuseError("the first ledger row must be 'open'")
        if self.rows and kind == kinds.OPEN:
            raise kinds.MisuseError("a ledger has exactly one 'open' row")
        handlers: dict[str, Callable[[Mapping[str, Any]], None]] = {
            kinds.OPEN: self.apply,
            kinds.CHALLENGER_RESULT: self.replay,
            kinds.ESCALATION: self.escalate,
            kinds.ATTESTATION: self.attest,
            kinds.ATTESTATION_USED: self.spend,
            kinds.HALT: self.halt,
            kinds.ROLLOUT_EXPERIMENT: self.start,
            kinds.ROLLOUT_ASSIGNMENT: self.assign,
            kinds.ROLLOUT_DECISION: self.confirm,
            kinds.ENVELOPE: self.issue,
            kinds.ENVELOPE_USED: self.consume,
            kinds.ENVELOPE_REVOKED: self.revoke,
            kinds.CONTROLLER: self.journal,
        }
        try:
            kinds.mapped(body)
            handlers.get(kind, self.accept)(body)
        except kinds.MisuseError:
            raise
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            raise kinds.MisuseError(
                f"malformed {kind} row {len(self.rows)}: {error!r}"
            ) from error
        self.rows.append(dict(row))

    def verify(self) -> None:
        """Re-reads the file (or the rows) and checks it against memory.

        An in-memory ledger replays its rows into a fresh ledger instead.

        Raises:
            MisuseError: If the file cannot be loaded or differs
                from this ledger, or the rows fail to replay.
        """
        if self.path is None:
            again = Ledger(None)
            for row in self.rows:
                again.ingest(row)
            return
        if Ledger.load(self.path).rows != self.rows:
            raise kinds.MisuseError("ledger file differs from memory")
