"""The steps an erasure is made of, each one idempotent.

A plan is a list of steps written to disk before the first one runs. Each
step names its target and carries the evidence measured when it was planned
(file count, bytes and a SHA-256 over the files), so the receipt can say what
was removed after it is gone. `perform` may be called again after an
interruption: removing what is already gone, marking what is already marked
and purging rows that are already purged change nothing.
"""

import abc
from collections.abc import Iterable
import dataclasses
import hashlib
import io
import os
import pathlib
import shutil
from typing import ClassVar
from typing import Self

import pandas

from scmle import durability
from scmle.artifacts.bundle import layout
from scmle.artifacts.bundle import seal
from scmle.artifacts.bundle import tombstone
from scmle.contracts import errors
from scmle.contracts import events
from scmle.contracts import records
from scmle.contracts import signing
from scmle.data.readers import tabular
from scmle.deploy import release_state
from scmle.governance import registry
import scmle.pipeline.orchestrator
from scmle.storage import stores


@dataclasses.dataclass(frozen=True)
class Order:
    """Who asked for an erasure and with what authority.

    Attributes:
        identifier: The erasure's id, written into every mark and audit row.
        reason: Why it was asked for.
        key: The signing key of the receipt and of retirement marks.
    """

    identifier: str
    reason: str
    key: bytes


def fingerprint(value: object) -> str:
    """Hashes a value as text, so a subject's identifiers are never stored.

    Args:
        value: An identifier as given or as read from a table.

    Returns:
        The hex SHA-256 of `str(value)`.
    """
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def inventory(path: pathlib.Path) -> records.Layout:
    """Measures a file or a directory tree before it is erased.

    Args:
        path: The file or directory.

    Returns:
        `files`, `bytes` and `sha256`: for a file its own digest; for a
        directory a digest over each file's relative path and digest, in
        sorted order. A missing path measures as zero files.
    """
    if path.is_file():
        return {
            "files": 1,
            "bytes": path.stat().st_size,
            "sha256": tabular.raw_source_hash(path),
        }
    digest = hashlib.sha256()
    count = size = 0
    if path.is_dir():
        for found in sorted(p for p in path.rglob("*") if p.is_file()):
            relative = found.relative_to(path).as_posix()
            line = f"{relative}\0{tabular.raw_source_hash(found)}\n"
            digest.update(line.encode("utf-8"))
            count += 1
            size += found.stat().st_size
    return {"files": count, "bytes": size, "sha256": digest.hexdigest()}


@dataclasses.dataclass(frozen=True)
class Step(abc.ABC):
    """One idempotent action of an erasure plan.

    Attributes:
        action: The name a stored plan gives the step's class.
        target: The absolute path the step acts on.
        evidence: What the target held when the step was planned.
    """

    action: ClassVar[str]
    target: str
    evidence: records.Layout = dataclasses.field(default_factory=dict)

    @abc.abstractmethod
    def perform(self, order: Order) -> records.Layout:
        """Carries the step out; calling it again changes nothing.

        Args:
            order: The erasure's id, reason and key.

        Returns:
            What the step did, for the receipt.
        """

    def to_dict(self) -> records.Layout:
        """Returns the stored form of the step.

        Returns:
            The step's fields with its `action`.
        """
        return {"action": self.action, **dataclasses.asdict(self)}

    @staticmethod
    def load(stored: records.Layout) -> "Step":
        """Rebuilds a step from its stored form.

        Args:
            stored: A record from `to_dict`.

        Returns:
            The step, of the class its `action` names.

        Raises:
            ArtifactError: If no step class has that action.
        """
        fields = dict(stored)
        action = fields.pop("action", None)
        for kind in Step.__subclasses__():
            if kind.action == action:
                return kind(**fields)
        raise errors.ArtifactError(f"unknown erasure step '{action}'")


@dataclasses.dataclass(frozen=True)
class Removal(Step):
    """Deletes a file or a directory tree."""

    action: ClassVar[str] = "remove"

    @classmethod
    def of(cls, path: pathlib.Path) -> Self:
        """Plans the removal of a path, measuring it first.

        Args:
            path: The file or directory.

        Returns:
            The step.
        """
        return cls(str(path.resolve()), inventory(path))

    def perform(self, order: Order) -> records.Layout:  # noqa: ARG002 -- Step.perform interface
        """Deletes the target; a target already gone is not an error.

        Args:
            order: Unused; part of the step interface.

        Returns:
            `removed`: whether anything was there to delete.
        """
        path = pathlib.Path(self.target)
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
            durability.sync_directory(path.parent)
            return {"removed": True}
        if path.exists() or path.is_symlink():
            path.unlink()
            durability.sync_directory(path.parent)
            return {"removed": True}
        return {"removed": False}


@dataclasses.dataclass(frozen=True)
class Custody(Step):
    """Keeps a record the law requires; the receipt lists its digest."""

    action: ClassVar[str] = "keep"

    @classmethod
    def of(cls, path: pathlib.Path) -> Self:
        """Plans keeping a path, measuring it.

        Args:
            path: The file or directory kept.

        Returns:
            The step.
        """
        return cls(str(path.resolve()), inventory(path))

    def perform(self, order: Order) -> records.Layout:  # noqa: ARG002 -- Step.perform interface
        """Checks the record is still there.

        Args:
            order: Unused; part of the step interface.

        Returns:
            `kept`: whether the record exists.
        """
        return {"kept": pathlib.Path(self.target).exists()}


@dataclasses.dataclass(frozen=True)
class Mark(Step):
    """Retires a model: its bundle is refused and its registry state moves.

    Attributes:
        registry: The registry root holding the bundle as a version, or
            empty when the bundle is not registered.
        model: The registered model name, or empty.
        version: The registered version, or 0.
    """

    action: ClassVar[str] = "retire"
    registry: str = ""
    model: str = ""
    version: int = 0

    def perform(self, order: Order) -> records.Layout:
        """Writes the retirement mark and retires the registry version.

        An existing mark is kept, so the first retirement's reason stands.

        Args:
            order: The erasure's id, reason and key, written into the mark.

        Returns:
            `marked` and the registry `state` (empty when not registered).
        """
        root = pathlib.Path(self.target)
        if tombstone.read(root) is None:
            record = {
                "reason": order.reason,
                "retired": events.stamp(),
                "operation": order.identifier,
                "content": self.evidence.get("content", ""),
            }
            record["signature"] = signing.mac(record, order.key)
            tombstone.mark(root, record)
        state = ""
        if self.registry:
            store = registry.ModelRegistryStore(self.registry)
            machine = release_state.ReleaseRegistry(store)
            retired = release_state.ReleaseState.RETIRED
            if machine.lookup(self.model, self.version) is not retired:
                machine.transition(
                    self.model,
                    self.version,
                    retired,
                    f"{order.reason} ({order.identifier})",
                )
            state = retired.value
        return {"marked": True, "state": state}


class Table:
    """A source file read and written as text, so it round-trips.

    Attributes:
        suffixes: The file suffixes this format reads.
    """

    suffixes: ClassVar[tuple[str, ...]] = (".csv",)
    separator: ClassVar[str] = ","

    def read(self, path: pathlib.Path) -> pandas.DataFrame:
        """Reads every cell as text, keeping empty cells empty.

        Args:
            path: The file.

        Returns:
            The table.
        """
        return pandas.read_csv(
            path, dtype=str, keep_default_na=False, sep=self.separator
        )

    def write(self, frame: pandas.DataFrame) -> bytes:
        """Renders a table in this format.

        Args:
            frame: The table.

        Returns:
            The file's bytes.
        """
        text = frame.to_csv(index=False, sep=self.separator)
        return text.encode("utf-8")

    @staticmethod
    def find(path: pathlib.Path) -> "Table":
        """Returns the format of a file by its suffix.

        Args:
            path: The file.

        Returns:
            The format.

        Raises:
            InputError: If no format reads the suffix.
        """
        formats: Iterable[Table] = (Table(), Tabs(), Columns(), Lines())
        for found in formats:
            if path.name.lower().endswith(found.suffixes):
                return found
        raise errors.InputError(
            f"cannot rewrite {path.name}: a subject is purged from CSV, TSV, "
            "Parquet or JSON Lines files"
        )


class Tabs(Table):
    """A tab-separated file."""

    suffixes = (".tsv", ".tab")
    separator = "\t"


class Columns(Table):
    """A Parquet file."""

    suffixes = (".parquet", ".pq")

    def read(self, path: pathlib.Path) -> pandas.DataFrame:
        """Reads the file with its stored types.

        Args:
            path: The file.

        Returns:
            The table.
        """
        return pandas.read_parquet(path)

    def write(self, frame: pandas.DataFrame) -> bytes:
        """Renders a table as Parquet.

        Args:
            frame: The table.

        Returns:
            The file's bytes.
        """
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        return buffer.getvalue()


class Lines(Table):
    """A JSON Lines file."""

    suffixes = (".jsonl", ".ndjson")

    def read(self, path: pathlib.Path) -> pandas.DataFrame:
        """Reads one record per line without converting types.

        Args:
            path: The file.

        Returns:
            The table.
        """
        return pandas.read_json(path, lines=True, dtype=False)

    def write(self, frame: pandas.DataFrame) -> bytes:
        """Renders a table as JSON Lines.

        Args:
            frame: The table.

        Returns:
            The file's bytes.
        """
        text = frame.to_json(orient="records", lines=True, force_ascii=False)
        return (text or "").encode("utf-8")


def matches(
    frame: pandas.DataFrame, column: str, hashes: Iterable[str]
) -> "pandas.Series[bool]":
    """Flags the rows whose `column` value hashes to one of `hashes`.

    Args:
        frame: The table.
        column: The identifying column.
        hashes: `fingerprint`s of the subject's identifiers.

    Returns:
        A boolean Series aligned with `frame`.

    Raises:
        InputError: If the column is missing.
    """
    if column not in frame.columns:
        raise errors.InputError(
            f"the source has no column '{column}'; columns: "
            f"{list(frame.columns)[:20]}"
        )
    wanted = frozenset(hashes)
    return frame[column].astype(str).map(fingerprint).isin(wanted)


@dataclasses.dataclass(frozen=True)
class Purge(Step):
    """Removes a subject's rows from a source file, rewriting it atomically.

    Attributes:
        column: The identifying column.
        hashes: `fingerprint`s of the identifiers; the values themselves are
            never written anywhere.
    """

    action: ClassVar[str] = "purge"
    column: str = ""
    hashes: tuple[str, ...] = ()

    def perform(self, order: Order) -> records.Layout:  # noqa: ARG002 -- Step.perform interface
        """Rewrites the source without the subject's rows.

        Args:
            order: Unused; part of the step interface.

        Returns:
            `rows` removed and the source's `sha256` afterwards.
        """
        path = pathlib.Path(self.target)
        table = Table.find(path)
        frame = table.read(path)
        flagged = matches(frame, self.column, self.hashes)
        removed = int(flagged.sum())
        if removed:
            durability.write_atomically(path, table.write(frame[~flagged]))
        return {"rows": removed, "sha256": tabular.raw_source_hash(path)}


@dataclasses.dataclass(frozen=True)
class Retraining(Step):
    """Trains a run's configuration again on its purged source.

    Attributes:
        source: The purged source file the new run reads.
    """

    action: ClassVar[str] = "retrain"
    source: str = ""

    def perform(self, order: Order) -> records.Layout:  # noqa: ARG002 -- Step.perform interface
        """Runs the stored configuration in a new run beside the old one.

        Args:
            order: Unused; part of the step interface.

        Returns:
            `run`: the new run directory.
        """
        frame = tabular.read(self.source)
        run = pathlib.Path(self.target)
        orchestrator = scmle.pipeline.orchestrator.Orchestrator.reload(
            run, frame=frame
        )
        result = orchestrator.run(run.parent)
        return {"run": os.fspath(result.run_dir)}


@dataclasses.dataclass(frozen=True)
class Unlisting(Step):
    """Deletes a run's rows from the tracking database beside it.

    Attributes:
        run: The run id.
    """

    action: ClassVar[str] = "unlist"
    run: str = ""

    def perform(self, order: Order) -> records.Layout:  # noqa: ARG002 -- Step.perform interface
        """Deletes the run's trial and run rows in one transaction.

        Args:
            order: Unused; part of the step interface.

        Returns:
            `rows`: how many rows were deleted.
        """
        path = pathlib.Path(self.target)
        if not path.is_file():
            return {"rows": 0}
        database = stores.SqliteExperimentStore(path)
        with database.session() as connection:
            trials = connection.execute(
                "DELETE FROM trials WHERE run_id = ?", (self.run,)
            ).rowcount
            runs = connection.execute(
                "DELETE FROM runs WHERE run_id = ?", (self.run,)
            ).rowcount
        return {"rows": trials + runs}


# The bundle files kept when a model is retired by an erasure: the manifest
# (checksums, versions and the training data's digest, no row values), its
# signature and the retirement mark. Every other member is fitted on the
# data and is removed.
KEPT = frozenset({layout.MANIFEST_FILE, seal.SIGNATURE_FILE, tombstone.FILE})
