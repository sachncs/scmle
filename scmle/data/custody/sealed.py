"""Sealed sets, the hash-chained open log and the exploration view.

Protection model (canon P6). A sealed set lives in a separate operating-system
process (`scmle.governance.vault`): after sealing, the orchestrator keeps only
row ids and a release token, and the rows travel to the token holder once, over
a channel keyed by a secret that only the parent and the vault hold. Code that
is given only an `ExplorationView` has no reference to a sealed set, a vault
or a token, so it cannot open one by accident or by attribute access. This is
tamper-evident custody on one machine, not adversary-proof: a process with the
same operating-system privileges could still attach to the vault or read the
original file. Every sealing, token issue, open and refused attempt is written
to a hash-chained log that holds ids and counts but never row values.
"""

from collections.abc import Mapping
import dataclasses
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scmle.contracts import errors
from scmle.contracts import signing
from scmle.data.splitting import splitters
from scmle.governance import vault as vault_module

GENESIS = "0" * 64
ORCHESTRATOR = "orchestrator"
VAULT_LEVEL = "vault process"
PROTOCOL_LEVEL = "in-process protocol"


def binding_digest(claims_hash: str, ledger_hash: str) -> str:
    """Digest that a release is bound to: the claim set and the ledger."""
    return hashlib.sha256(
        signing.canonical_payload(
            {"claims_hash": claims_hash, "ledger_hash": ledger_hash}
        )
    ).hexdigest()


def entry_hash(body: Mapping[str, Any]) -> str:
    """Hash of a log entry's content."""
    return hashlib.sha256(signing.canonical_payload(body)).hexdigest()


def entry_mac(entry_hash_value: str, key: bytes) -> str:
    """HMAC-SHA256 of an entry's hash under the log key."""
    return hmac.new(key, entry_hash_value.encode(), hashlib.sha256).hexdigest()


class OpenLog:
    """Append-only, hash-chained record of custody events.

    Each entry stores the hash of the previous one, so editing, removing or
    reordering any entry breaks the chain from that point on. With a key,
    each entry also carries an HMAC of its hash: someone who edits an entry
    and recomputes the chain still cannot produce the MAC without the key.
    The chain alone is a checksum, not an authentication. Neither detects
    the removal of the newest entries; compare `head` with a copy kept
    elsewhere (the confirmation record stores one) to detect that.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        signing_key: str | bytes | None = None,
    ) -> None:
        """Opens the log, loading and verifying an existing file.

        Args:
            path: File to persist to (one JSON object per line), or None for
                an in-memory log.
            signing_key: HMAC key; the environment key is used when None.
                Without any key the log is hash-chained but not MAC-ed.

        Raises:
            ArtifactError: If an existing file fails verification.
        """
        self.path = Path(path) if path is not None else None
        self.key = signing.resolve_key(signing_key)
        self.entries: list[dict[str, Any]] = []
        if self.path is not None and self.path.exists():
            self.entries = [
                json.loads(line)
                for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.verify()

    def append(
        self,
        set_id: str,
        event: str,
        actor: str,
        detail: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Adds an event and persists it.

        Args:
            set_id: Set concerned (`C`, `final_test`, ...).
            event: `sealed`, `token_issued`, `open` or `denied`.
            actor: Who acted.
            detail: Counts and reasons; never row values.

        Returns:
            The stored entry including its hash.
        """
        body = {
            "seq": len(self.entries),
            "set": set_id,
            "event": event,
            "actor": actor,
            "detail": dict(detail or {}),
            "prev": self.entries[-1]["hash"] if self.entries else GENESIS,
        }
        digest = entry_hash(body)
        entry = {**body, "hash": digest}
        if self.key is not None:
            entry["mac"] = entry_mac(digest, self.key)
        self.entries.append(entry)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def verify(self) -> None:
        """Checks every link of the chain.

        Raises:
            ArtifactError: At the first entry that does not verify.
        """
        previous = GENESIS
        for index, entry in enumerate(self.entries):
            body = {k: v for k, v in entry.items() if k not in ("hash", "mac")}
            if (
                entry.get("seq") != index
                or entry.get("prev") != previous
                or entry.get("hash") != entry_hash(body)
            ):
                raise errors.ArtifactError(
                    f"custody log chain broken at entry {index}"
                )
            if self.key is not None and not hmac.compare_digest(
                str(entry.get("mac", "")), entry_mac(entry["hash"], self.key)
            ):
                raise errors.ArtifactError(
                    f"custody log MAC invalid at entry {index}"
                )
            previous = entry["hash"]

    @property
    def head(self) -> str:
        """Hash of the newest entry, or the genesis value when empty."""
        return str(self.entries[-1]["hash"]) if self.entries else GENESIS

    @classmethod
    def verify_file(
        cls, path: str | Path, signing_key: str | bytes | None = None
    ) -> "OpenLog":
        """Loads a log file and verifies its chain and, with a key, its MACs."""
        return cls(path, signing_key)


class SealedSet:
    """A named set behind a one-time, token-gated release, with a log."""

    def __init__(
        self, set_id: str, locked: splitters.LockedTestSet, log: OpenLog
    ) -> None:
        """Wraps a locked set and records that it was sealed.

        Args:
            set_id: Name of the set.
            locked: The locked set (a vault-backed one for real custody).
            log: The custody log.
        """
        self.set_id = set_id
        self.locked = locked
        self.log = log
        self.binding: str | None = None
        log.append(set_id, "sealed", ORCHESTRATOR, {"rows": len(locked)})

    @property
    def ids(self) -> tuple[int, ...]:
        """Row ids of the set (the rows themselves are not held here)."""
        return tuple(self.locked.ids)

    @property
    def opened(self) -> bool:
        """Whether the rows were released."""
        return bool(self.locked.opened)

    def issue_token(self) -> str:
        """Returns the release token; call only once selection is frozen."""
        token = self.locked.issue_token()
        self.log.append(self.set_id, "token_issued", ORCHESTRATOR)
        return token

    @property
    def committed(self) -> bool:
        """Whether the release is bound to a claim set and a ledger."""
        return self.binding is not None

    def commit(self, claims_hash: str, ledger_hash: str) -> str:
        """Binds the single release to a claim set and a ledger state.

        After this, `open` releases the rows only when it is given the same
        two hashes; a release for any other selection is refused by the
        vault (or, for an in-process set, by the protocol check). The
        binding is logged with both hashes.

        Args:
            claims_hash: Digest of the claims (survivors) to be tested.
            ledger_hash: Head hash of the hypothesis ledger.

        Returns:
            The binding digest.

        Raises:
            LeakageError: If the set is already committed or was opened.
        """
        binding = binding_digest(claims_hash, ledger_hash)
        self.locked.bind(binding)
        self.binding = binding
        self.log.append(
            self.set_id,
            "bound",
            ORCHESTRATOR,
            {"claims_hash": claims_hash, "ledger_hash": ledger_hash},
        )
        return binding

    def open(
        self,
        token: str,
        actor: str = ORCHESTRATOR,
        claims_hash: str | None = None,
        ledger_hash: str | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Releases the rows a single time.

        Args:
            token: The token from `issue_token`.
            actor: Who is opening (recorded).
            claims_hash: Digest of the claims being tested; required, with
                `ledger_hash`, once the set is committed.
            ledger_hash: Head hash of the ledger.

        Raises:
            LeakageError: On a wrong token, a binding that does not match,
                or a repeated release.
        """
        offered = (
            binding_digest(claims_hash, ledger_hash)
            if claims_hash is not None and ledger_hash is not None
            else None
        )
        try:
            rows = self.locked.open(token, offered)
        except errors.LeakageError as error:
            self.log.append(
                self.set_id, "denied", actor, {"reason": str(error)}
            )
            raise
        self.log.append(self.set_id, "open", actor, {"rows": len(rows[0])})
        return rows


class CustodyVault:
    """Holds every sealed set of a run and the shared custody log."""

    def __init__(self, log: OpenLog, process: bool = True) -> None:
        """Creates an empty vault.

        Args:
            log: The custody log.
            process: Seal into a separate vault process (real custody). False
                keeps rows in this process behind the token protocol only,
                for tests that must not spawn processes.
        """
        self.log = log
        self.process = process
        self.sets: dict[str, SealedSet] = {}

    @property
    def level(self) -> str:
        """Isolation level this vault gives."""
        return VAULT_LEVEL if self.process else PROTOCOL_LEVEL

    def adopt(self, set_id: str, locked: splitters.LockedTestSet) -> SealedSet:
        """Registers an already-locked set under a name."""
        if set_id in self.sets:
            raise ValueError(f"set {set_id} is already sealed")
        held = SealedSet(set_id, locked, self.log)
        self.sets[set_id] = held
        return held

    def seal(
        self,
        set_id: str,
        features: pd.DataFrame,
        target: pd.Series,
        ids: tuple[int, ...],
    ) -> SealedSet:
        """Moves rows into custody and keeps only their ids here."""
        locked: splitters.LockedTestSet = (
            vault_module.SealedTestSet(features, target, ids)
            if self.process
            else splitters.ResidentTestSet(features, target, ids)
        )
        return self.adopt(set_id, locked)

    def close(self) -> None:
        """Ends every vault process."""
        for held in self.sets.values():
            held.locked.close()


@dataclasses.dataclass(frozen=True, slots=True)
class ExplorationView:
    """Everything exploration code may receive: `X` and nothing sealed.

    It holds no reference to a vault, a sealed set, a token or the log.

    Attributes:
        features: Exploration features.
        target: Exploration labels.
        ids: Row ids of the exploration rows.
        folds: Cross-validation folds over positions in these rows.
        manifest_sha256: Hash of the governing split manifest.
    """

    features: pd.DataFrame
    target: pd.Series
    ids: tuple[int, ...]
    folds: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    manifest_sha256: str
