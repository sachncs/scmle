"""Forgetting data, retiring models and deleting runs, with signed receipts.

Each erasure is planned once, journaled step by step and closed by a signed
receipt (`receipts`). The three kinds differ only in what they plan:

* `Forgetting` finds every run, model, trail and prediction log of a
  dataset (`Dataset`) or of one data subject's rows (`Subject`), removes
  what was derived from the data, keeps the custody records, retires the
  models and, for a subject, purges the rows from the source and can train
  each run again without them;
* `Retirement` retires one model: its bundle is refused from then on and,
  if registered, its release state becomes `retired`;
* `Deletion` deletes runs or a whole project folder, keeping custody.
"""

import abc
from collections.abc import Iterable
import hashlib
import os
import pathlib
from typing import ClassVar

from scmle.api.erasure import holdings
from scmle.api.erasure import receipts
from scmle.api.erasure import steps
from scmle.artifacts.bundle import layout
from scmle.assurance.custody import journal
from scmle.contracts import errors
from scmle.contracts import events
from scmle.contracts import records
from scmle.contracts import signing
from scmle.data.readers import tabular
from scmle.governance import registry

# The tracking database's companion files, removed with it.
COMPANIONS = ("-wal", "-shm", "-journal")


def text(value: object, name: str) -> str:
    """Checks a required text argument.

    Args:
        value: The argument.
        name: Its name, for the error.

    Returns:
        The stripped text.

    Raises:
        InputError: If it is not text or is blank.
    """
    if not isinstance(value, str) or not value.strip():
        raise errors.InputError(f"{name} must be non-empty text")
    return value.strip()


class Dataset:
    """A dataset to forget, named by its digests.

    Attributes:
        digests: Any of the row-content hash, snapshot, identifier or raw
            file SHA-256 that a run's `dataset_manifest.json` records.
    """

    def __init__(self, digests: Iterable[str]) -> None:
        """Names the dataset.

        Args:
            digests: One or more digests.

        Raises:
            InputError: If none is given or one is blank.
        """
        self.digests = tuple(
            sorted({text(d, "a dataset digest") for d in digests})
        )
        if not self.digests:
            raise errors.InputError("name the dataset by at least one digest")

    @classmethod
    def of(cls, source: str | pathlib.Path) -> "Dataset":
        """Names a dataset by the file it was read from.

        Args:
            source: The data file, as a run read it.

        Returns:
            The dataset, by its row-content hash and its file's SHA-256.

        Raises:
            InputError: If the file does not exist.
        """
        path = pathlib.Path(source)
        if not path.is_file():
            raise errors.InputError(f"no such data file: {path}")
        frame = tabular.read(path)
        return cls((tabular.data_hash(frame), tabular.raw_source_hash(path)))

    def describe(self) -> records.Layout:
        """Returns what the receipt says was forgotten.

        Returns:
            The digests.
        """
        return {"digests": list(self.digests)}

    def identities(self) -> frozenset[str]:
        """Returns the digests a run or a bundle is matched by.

        Returns:
            The digests.
        """
        return frozenset(self.digests)

    def prepare(self) -> list[steps.Step]:
        """Returns the steps run before the stored artefacts are erased.

        Returns:
            None for a dataset: the source is not rewritten.
        """
        return []


class Subject(Dataset):
    """One data subject's rows of a source file.

    The identifiers are kept only as SHA-256 hashes of their text, so
    neither the plan nor the receipt holds them.

    Attributes:
        source: The source file, purged of the rows.
        column: The identifying column.
        hashes: The hashed identifiers.
    """

    def __init__(
        self,
        source: str | pathlib.Path,
        column: str,
        values: Iterable[object],
        earlier: Iterable[str] = (),
    ) -> None:
        """Names the subject and the source holding their rows.

        Args:
            source: CSV, TSV, Parquet or JSON Lines file.
            column: The identifying column.
            values: The subject's identifiers, compared as text with the
                column's stored values.
            earlier: Digests of earlier versions of the source that also
                held the rows; runs trained on them are found too.

        Raises:
            InputError: If the file is missing or of another format, the
                column is blank or no identifier is given.
        """
        self.source = pathlib.Path(source).resolve()
        if not self.source.is_file():
            raise errors.InputError(f"no such data file: {self.source}")
        steps.Table.find(self.source)
        self.column = text(column, "the identifying column")
        self.hashes = tuple(sorted({steps.fingerprint(v) for v in values}))
        if not self.hashes:
            raise errors.InputError("name at least one identifier to forget")
        self.digests = tuple(sorted({text(d, "a digest") for d in earlier}))

    def describe(self) -> records.Layout:
        """Returns what the receipt says was forgotten.

        Returns:
            The source, column, hashed identifiers and earlier digests.
        """
        return {
            "source": os.fspath(self.source),
            "column": self.column,
            "hashes": list(self.hashes),
            "earlier": list(self.digests),
        }

    def identities(self) -> frozenset[str]:
        """Returns the digests of the source as it is now, with the rows.

        Returns:
            Its row-content hash and file SHA-256, and the earlier digests.

        Raises:
            InputError: If no row of the source has one of the identifiers.
        """
        table = steps.Table.find(self.source)
        flagged = steps.matches(
            table.read(self.source), self.column, self.hashes
        )
        if not flagged.any():
            raise errors.InputError(
                f"no row of {self.source.name} has one of the identifiers in "
                f"'{self.column}'"
            )
        frame = tabular.read(self.source)
        current = {
            tabular.data_hash(frame),
            tabular.raw_source_hash(self.source),
        }
        return frozenset(current | set(self.digests))

    def prepare(self) -> list[steps.Step]:
        """Plans purging the rows from the source first.

        Returns:
            One `Purge`.
        """
        return [
            steps.Purge(
                os.fspath(self.source),
                steps.inventory(self.source),
                self.column,
                self.hashes,
            )
        ]


class Erasure(abc.ABC):
    """An erasure: planned once, journaled, closed by a signed receipt.

    Attributes:
        kind: The receipt's `kind`.
        receipts: The folder holding plans, journals and receipts.
        reason: Why the erasure was asked for.
    """

    kind: ClassVar[str]

    def __init__(self, receipts: str | pathlib.Path, reason: str) -> None:
        """Sets where the receipt goes and why.

        Args:
            receipts: The receipts folder (created when needed).
            reason: Why the erasure was asked for.
        """
        self.receipts = pathlib.Path(receipts).resolve()
        self.reason = text(reason, "the reason")

    @abc.abstractmethod
    def request(self) -> records.Layout:
        """Returns what was asked, which also names the erasure."""

    @abc.abstractmethod
    def survey(self) -> list[steps.Step]:
        """Finds what to erase and plans it."""

    @property
    def identifier(self) -> str:
        """The erasure's id: its kind and a digest of its request."""
        payload = signing.canonicalise(self.request())
        return f"{self.kind}-{hashlib.sha256(payload).hexdigest()[:16]}"

    def preview(self) -> list[steps.Step]:
        """Returns the plan without carrying anything out or writing it.

        Returns:
            The stored plan if the erasure was started, else a fresh one.
        """
        stored = receipts.Journal(self.receipts / self.identifier).plan()
        if stored is None:
            return self.check(self.survey())
        return [steps.Step.load(step) for step in stored["steps"]]

    def check(self, planned: list[steps.Step]) -> list[steps.Step]:
        """Refuses a plan that would remove its own receipts.

        Args:
            planned: The plan.

        Returns:
            The plan.

        Raises:
            InputError: If the receipts folder lies inside a removed path.
        """
        for step in planned:
            if isinstance(step, steps.Removal) and self.receipts.is_relative_to(
                step.target
            ):
                raise errors.InputError(
                    f"the receipts folder {self.receipts} is inside "
                    f"{step.target}, which this erasure removes"
                )
        return planned

    def plan(self, ledger: receipts.Journal) -> records.Layout:
        """Returns the stored plan, writing it first if there is none.

        Args:
            ledger: The erasure's journal.

        Returns:
            The plan: request, reason, start time and steps.
        """
        stored = ledger.plan()
        if stored is None:
            stored = {
                "identifier": self.identifier,
                "kind": self.kind,
                "reason": self.reason,
                "request": self.request(),
                "started": events.stamp(),
                "steps": [s.to_dict() for s in self.check(self.survey())],
            }
            ledger.record(stored)
        return stored

    def execute(self, key: str | bytes | None = None) -> receipts.Receipt:
        """Carries out the plan, resuming where an earlier attempt stopped.

        Args:
            key: The signing key, or None for `SCMLE_SIGNING_KEY`.

        Returns:
            The signed receipt; the stored one if the erasure had finished.

        Raises:
            ConfigError: If no signing key is given or configured.
        """
        resolved = signing.keying(key)
        if resolved is None:
            raise errors.ConfigError(
                "an erasure receipt is signed: pass a key or set "
                f"{signing.SIGNING_KEY_ENV}"
            )
        path = self.receipts / f"{self.identifier}.json"
        if path.is_file():
            return receipts.Receipt.read(path)
        ledger = receipts.Journal(self.receipts / self.identifier)
        stored = self.plan(ledger)
        order = steps.Order(self.identifier, stored["reason"], resolved)
        done = ledger.done()
        planned = [steps.Step.load(step) for step in stored["steps"]]
        for index, step in enumerate(planned):
            if index not in done:
                done[index] = step.perform(order)
                ledger.finish(index, done[index])
        receipt = receipts.Receipt(
            self.identifier,
            self.kind,
            stored["reason"],
            stored["request"],
            tuple(
                {**step.to_dict(), "result": done[index]}
                for index, step in enumerate(planned)
            ),
            stored["started"],
            events.stamp(),
        ).sign(resolved)
        receipt.write(path)
        return receipt


class Forgetting(Erasure):
    """Forgets a dataset or a subject everywhere under some folders.

    Attributes:
        roots: The folders searched.
        subject: What is forgotten.
        retrain: Whether each run trained on a subject's rows is trained
            again on the purged source; its old model is retired either way.
    """

    kind = "forget"

    def __init__(
        self,
        roots: Iterable[str | pathlib.Path],
        subject: Dataset,
        receipts: str | pathlib.Path,
        reason: str,
        retrain: bool = False,
    ) -> None:
        """Describes the forgetting.

        Args:
            roots: The folders to search (runs, registries, trails, logs).
            subject: A `Dataset` or a `Subject`.
            receipts: The receipts folder.
            reason: Why, for example the request's reference.
            retrain: Train each affected run again without the subject's
                rows; only a `Subject` leaves rows to train on.

        Raises:
            InputError: If no folder is given, or `retrain` is asked for a
                whole dataset.
        """
        super().__init__(receipts, reason)
        self.roots = tuple(sorted({pathlib.Path(r).resolve() for r in roots}))
        if not self.roots:
            raise errors.InputError("name at least one folder to search")
        if retrain and not isinstance(subject, Subject):
            raise errors.InputError(
                "only a subject's rows can be retrained without: a forgotten "
                "dataset leaves no data"
            )
        self.subject = subject
        self.retrain = retrain

    def request(self) -> records.Layout:
        """Returns the folders, the subject and whether to retrain.

        Returns:
            The request.
        """
        return {
            "roots": [os.fspath(r) for r in self.roots],
            "subject": self.subject.describe(),
            "retrain": self.retrain,
        }

    def survey(self) -> list[steps.Step]:
        """Plans the purge, the retraining and the erasure of what was found.

        Returns:
            The plan.
        """
        found = holdings.scan(
            self.roots, self.subject.identities(), self.receipts
        )
        planned = self.subject.prepare()
        if self.retrain and isinstance(self.subject, Subject):
            planned += [
                steps.Retraining(
                    os.fspath(run.resolve()),
                    {},
                    os.fspath(self.subject.source),
                )
                for run in found.runs
            ]
        return planned + found.plan


class Retirement(Erasure):
    """Retires one model, by its bundle or its registry version.

    Attributes:
        target: The bundle directory or the registry root.
        model: The registered model name, or empty for a bundle.
        version: The registered version, or 0 for a bundle.
    """

    kind = "retire"

    def __init__(
        self,
        target: str | pathlib.Path,
        receipts: str | pathlib.Path,
        reason: str,
        model: str = "",
        version: int = 0,
    ) -> None:
        """Names the model.

        Args:
            target: A bundle directory, or a registry root with `model` and
                `version`.
            receipts: The receipts folder.
            reason: Why the model is retired.
            model: Registered model name.
            version: Registered version.

        Raises:
            InputError: If only one of `model` and `version` is given.
        """
        super().__init__(receipts, reason)
        self.target = pathlib.Path(target).resolve()
        if bool(model) != bool(version):
            raise errors.InputError(
                "a registered model is named by both its name and version"
            )
        self.model = model
        self.version = int(version)

    def request(self) -> records.Layout:
        """Returns the model retired.

        Returns:
            The target, model and version.
        """
        return {
            "target": os.fspath(self.target),
            "model": self.model,
            "version": self.version,
        }

    def survey(self) -> list[steps.Step]:
        """Plans one `Mark`; nothing is removed.

        Returns:
            The plan.

        Raises:
            ArtifactError: If the target is not a bundle, or not a registry
                holding that version.
        """
        if self.model:
            if not (self.target / registry.INDEX_FILE).is_file():
                raise errors.ArtifactError(f"no registry at {self.target}")
            store = registry.ModelRegistryStore(self.target)
            record = store.resolve(self.model, self.version)
            bundle = self.target / record.path
            root, model, version = (
                os.fspath(self.target),
                self.model,
                self.version,
            )
        else:
            bundle = self.target
            root, model, version = holdings.registration(bundle)
        if not (bundle / layout.MANIFEST_FILE).is_file():
            raise errors.ArtifactError(f"no model bundle at {bundle}")
        content = registry.identify(bundle)
        return [
            steps.Mark(
                os.fspath(bundle), {"content": content}, root, model, version
            )
        ]


class Deletion(Erasure):
    """Deletes run directories or whole project folders, keeping custody.

    A run is a folder holding `config.json`; a project is an output folder
    holding `experiments.sqlite`, whose runs, tracking database and
    caches are deleted while its custody journals stay.

    Attributes:
        targets: The runs and projects.
    """

    kind = "delete"

    def __init__(
        self,
        targets: Iterable[str | pathlib.Path],
        receipts: str | pathlib.Path,
        reason: str,
    ) -> None:
        """Names what to delete.

        Args:
            targets: Run directories and project folders.
            receipts: The receipts folder.
            reason: Why.

        Raises:
            InputError: If nothing is named.
        """
        super().__init__(receipts, reason)
        self.targets = tuple(
            sorted({pathlib.Path(t).resolve() for t in targets})
        )
        if not self.targets:
            raise errors.InputError("name at least one run or project")

    def request(self) -> records.Layout:
        """Returns the targets.

        Returns:
            The request.
        """
        return {"targets": [os.fspath(t) for t in self.targets]}

    def survey(self) -> list[steps.Step]:
        """Plans each run's erasure, and each project's database and runs.

        Returns:
            The plan.

        Raises:
            InputError: If a target is neither a run nor a project.
        """
        planned: list[steps.Step] = []
        for target in self.targets:
            if (target / holdings.CONFIG).is_file():
                planned += holdings.dossier(target)
            elif (target / holdings.TRACKING).is_file():
                planned += self.project(target)
            else:
                raise errors.InputError(
                    f"{target} is neither a run (config.json) nor a project "
                    f"({holdings.TRACKING})"
                )
        return planned

    @staticmethod
    def project(folder: pathlib.Path) -> list[steps.Step]:
        """Plans deleting a project folder's runs and tracking database.

        Files scmle did not write are left where they are.

        Args:
            folder: The project folder.

        Returns:
            The plan.
        """
        planned: list[steps.Step] = []
        for run in sorted(folder.iterdir()):
            if (run / holdings.CONFIG).is_file():
                planned += holdings.dossier(run)
        custody = folder / journal.FOLDER
        if custody.is_dir():
            planned.append(steps.Custody.of(custody))
        tracking = folder / holdings.TRACKING
        for suffix in ("", *COMPANIONS):
            companion = tracking.with_name(tracking.name + suffix)
            if companion.exists():
                planned.append(steps.Removal.of(companion))
        return planned
