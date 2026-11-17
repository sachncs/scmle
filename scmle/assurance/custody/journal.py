"""The custody journal: what a run did that its own files cannot unsay.

A run records its start, every checkpoint verdict and the opening of its
locked test set in `<runs>/custody/<run id>.jsonl`, outside the run
directory. Deleting `evaluation.json` or editing `config.json` (with its
hashes recomputed) therefore cannot make a resumed run open the same test
rows again, and deleting `events.jsonl` cannot forget a rejection. The lines
are hash-chained, so an edited journal is refused; a missing one is refused
too, because a run without a record cannot show its test set is still shut.
This is tamper-evident custody on one machine, not adversary-proof: a
process with the same privileges can delete the journal and the run together,
which leaves nothing to resume.
"""

import hashlib
import json
import pathlib
from typing import Self
from typing import TypedDict

import pandas

from scmle import durability
from scmle.contracts import errors
from scmle.contracts import events
from scmle.data.splitting import splitters

FOLDER = "custody"
GENESIS = hashlib.sha256(b"scmle.custody.journal").hexdigest()


class Entry(TypedDict, total=False):
    """One journal line.

    Attributes:
        kind: `started`, `decided` or `opened`.
        run: The run id.
        time: When, as ISO 8601 UTC.
        previous: The digest of the line before, or `GENESIS`.
        digest: This line's digest (absent while it is hashed).
        config: The configuration hash a run started with.
        gate: The state a checkpoint reviewed.
        approved: Whether the checkpoint approved.
        reviewer: Who decided.
        reason: Why.
    """

    kind: str
    run: str
    time: str
    previous: str
    digest: str
    config: str
    gate: str
    approved: bool
    reviewer: str
    reason: str


def digest(entry: Entry) -> str:
    """Hashes one entry as canonical JSON.

    Args:
        entry: The entry without its `digest`.

    Returns:
        The 64-character SHA-256 hex digest.
    """
    text = json.dumps(entry, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Journal:
    """The hash-chained custody record of one run.

    Attributes:
        path: The journal file.
        run: The run id the entries belong to.
    """

    def __init__(self, path: pathlib.Path, run: str) -> None:
        """Initializes the journal over a file that may not exist yet.

        Args:
            path: The journal file.
            run: The run id.
        """
        self.path = path
        self.run = run

    @classmethod
    def locate(cls, directory: str | pathlib.Path) -> Self:
        """The journal of a run directory, kept beside it, not inside.

        Args:
            directory: The run directory.

        Returns:
            Its journal.
        """
        run = pathlib.Path(directory)
        return cls(run.parent / FOLDER / f"{run.name}.jsonl", run.name)

    def read(self) -> list[Entry]:
        """Reads every entry and checks the chain.

        Returns:
            The entries, oldest first; empty when there is no journal.

        Raises:
            BlockedError: If an entry was edited, removed or reordered.
        """
        if not self.path.exists():
            return []
        entries: list[Entry] = []
        previous = GENESIS
        for line in self.path.read_text(encoding="utf-8").splitlines():
            entry: Entry = json.loads(line)
            stated = entry.pop("digest", "")
            if entry.get("previous") != previous or digest(entry) != stated:
                raise errors.BlockedError(
                    f"custody journal {self.path} was edited; run "
                    f"{self.run} cannot be resumed",
                    {"journal": str(self.path)},
                )
            entries.append(entry)
            previous = stated
        return entries

    def append(self, entry: Entry) -> None:
        """Adds one chained entry and flushes it to disk.

        Args:
            entry: Its kind and data; the run, time and chain are added.
        """
        with durability.latch(self.path):
            entries = self.read()
            previous = digest(entries[-1]) if entries else GENESIS
            entry = entry | Entry(
                run=self.run, time=events.stamp(), previous=previous
            )
            line = json.dumps(entry | Entry(digest=digest(entry)))
            durability.append_text(self.path, line + "\n")
            durability.sync(self.path)

    def current(self) -> list[Entry]:
        """The entries since the run's latest start.

        A run id reused after its directory was deleted starts afresh.

        Returns:
            The latest `started` entry and every entry after it, or an empty
            list when the run never started.
        """
        entries = self.read()
        starts = [i for i, e in enumerate(entries) if e["kind"] == "started"]
        return entries[starts[-1] :] if starts else []

    def start(self, config: str) -> None:
        """Records that the run was created.

        Args:
            config: The configuration hash it was created with.
        """
        self.append(Entry(kind="started", config=config))

    def decide(
        self, gate: str, approved: bool, reviewer: str, reason: str
    ) -> None:
        """Records a checkpoint verdict.

        Args:
            gate: The state the checkpoint reviewed.
            approved: Whether it approved.
            reviewer: Who decided.
            reason: Why.
        """
        self.append(
            Entry(
                kind="decided",
                gate=gate,
                approved=approved,
                reviewer=reviewer,
                reason=reason,
            )
        )

    def witness(self) -> None:
        """Records that the locked test set is being opened."""
        self.append(Entry(kind="opened"))

    def rejected(self) -> dict[str, str]:
        """The gates whose latest verdict is a rejection.

        Returns:
            Each rejected gate mapped to its reviewer.
        """
        verdicts: dict[str, str] = {}
        for entry in self.current():
            if entry["kind"] != "decided":
                continue
            if entry["approved"]:
                verdicts.pop(entry["gate"], None)
            else:
                verdicts[entry["gate"]] = str(entry["reviewer"])
        return verdicts

    def admit(self) -> None:
        """Refuses to resume a run whose test set may have been opened.

        Raises:
            BlockedError: If the run has no custody record (it was copied,
                predates the journal, or the journal was deleted), the
                journal was edited, or the test set was already opened.
        """
        entries = self.current()
        if not entries:
            raise errors.BlockedError(
                f"run {self.run} has no custody record at {self.path}; it "
                "cannot show its test set is still locked, so start a new run",
                {"journal": str(self.path)},
            )
        if any(entry["kind"] == "opened" for entry in entries):
            raise errors.BlockedError(
                f"the test set of run {self.run} was already opened; a "
                "resumed or edited run cannot score it again, so start a "
                "new run",
                {"journal": str(self.path)},
            )


class Custodian(splitters.Lockbox):
    """A locked test set that records its opening before it opens.

    Attributes:
        inner: The test set it guards.
        journal: Where the opening is recorded.
    """

    def __init__(self, inner: splitters.Lockbox, journal: Journal) -> None:
        """Wraps a test set.

        Args:
            inner: The test set.
            journal: The run's journal.
        """
        self.inner = inner
        self.journal = journal

    @property
    def ids(self) -> tuple[int, ...]:
        """The guarded set's row ids."""
        return self.inner.ids

    @ids.setter
    def ids(self, value: tuple[int, ...]) -> None:
        """Sets the guarded set's row ids."""
        self.inner.ids = value

    @property
    def opened(self) -> bool:
        """Whether the guarded set was released."""
        return self.inner.opened

    @opened.setter
    def opened(self, value: bool) -> None:
        """Sets whether the guarded set was released."""
        self.inner.opened = value

    def mint(self) -> str:
        """Returns the guarded set's release token.

        Returns:
            The token.
        """
        return self.inner.mint()

    def bind(self, binding: str) -> None:
        """Binds the guarded set.

        Args:
            binding: Digest of the selection the release is meant to test.
        """
        self.inner.bind(binding)

    def open(
        self, token: str, binding: str | None = None
    ) -> tuple[pandas.DataFrame, pandas.Series]:
        """Records the opening, then releases the guarded set.

        The record comes first, so a crash after the release cannot lose it.

        Args:
            token: The token from `mint`.
            binding: The binding fixed by `bind`.

        Returns:
            Test features and labels.
        """
        self.journal.witness()
        return self.inner.open(token, binding)

    def close(self) -> None:
        """Releases the guarded set's resources."""
        self.inner.close()
