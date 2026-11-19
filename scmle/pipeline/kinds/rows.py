"""What every kind whose items become table rows shares.

A document, a graph node, a media file or a vector feature becomes one row
of a table, which the shared stages profile, split, featurise and decode as
they do a table read from a file. `Rows` is the base of those kinds: the
portrait, the split, the featuriser, the decoder, the cost bound and the
capability records are written once here, and a kind adds its reader, its
findings (`notice`), its limits and, where it has them, its spatial blocks
(`places`).

A kind ties rows together through the cohort column (`KEY`) its reader
writes: rows that are copies or near-copies of one another, or chunks of
one document, share a cohort. `Cohorts` is the one splitter of every kind:
it joins those cohorts with the declared group and the kind's spatial
blocks, and keeps each on one side of the locked split and of every fold
(register rows MM-8, MM-10, MM-11, MM-22). A kind whose `role` is `time`
writes a time instead, and its latest rows are held out.
"""

import abc
from collections.abc import Iterable
from collections.abc import Sequence
import dataclasses
import pathlib
from typing import Any
from typing import ClassVar

import numpy
import pandas
from sklearn import model_selection

from scmle.contracts import capabilities
from scmle.contracts import errors
from scmle.contracts import tasks
import scmle.contracts.data
from scmle.data.modalities import cohorts
from scmle.data.modalities import curation
from scmle.data.readers import streaming
from scmle.data.readers import tabular
from scmle.data.splitting import splitters
from scmle.data.splitting.splitters import lockbox
from scmle.data.splitting.splitters import splitter
from scmle.pipeline import profiler
from scmle.preparation import preprocessing

KEY = "cohort"
"""The column a kind's reader writes each row's cohort (or, for a kind whose
`role` is `time`, its time) to. `Cohorts` reads it and drops it from the
split's rows, so it is never a feature; a kind refuses a source that already
has a column of this name."""

TASKS = (tasks.Task.BINARY, tasks.Task.MULTICLASS, tasks.Task.REGRESSION)
"""The tasks a row's label can pose."""

FAMILIES = (
    "baseline",
    "linear",
    "tree",
    "boosting",
    "kernel",
    "neighbors",
    "neural",
    "bagging",
)
"""The model families the shared stages search on the rows: every tabular
family that trains on numeric rows without an optional extra."""

LIFECYCLE = (
    "experimental, not supported: supported means the whole lifecycle is "
    "proven, and deployment, monitoring and recovery are not proven for "
    "this kind; a run needs the opt-in `models.experimental_models`"
)
"""Why a kind other than tabular is declared experimental: the breadth
guard (`tests/unit/test_breadth.py`) keeps `supported` to tabular library
inference."""

TIMED = splitter.TIMED
"""Strategies that hold out the latest rows."""


def unite(
    size: int, pairs: Iterable[tuple[int, int]], *keys: Sequence[Any]
) -> list[str]:
    """Joins rows into cohorts: linked pairs and rows sharing a key value.

    Joining is transitive (union-find), so a cohort is a connected
    component. A missing key value joins nothing.

    Args:
        size: Rows.
        pairs: Row positions linked directly (near-copies).
        *keys: Per-row values that tie rows together, such as a content
            digest, a document id, the declared group or a spatial block.

    Returns:
        Each row's cohort, named after its first row (`cohort-<row>`).
    """
    parent = list(range(size))

    def root(row: int) -> int:
        while parent[row] != row:
            parent[row] = parent[parent[row]]
            row = parent[row]
        return row

    def join(first: int, second: int) -> None:
        low, high = sorted((root(first), root(second)))
        parent[high] = low

    for first, second in pairs:
        join(int(first), int(second))
    for values in keys:
        seen: dict[Any, int] = {}
        for row, value in enumerate(values):
            if not pandas.isna(value):
                join(seen.setdefault(value, row), row)
    return [f"cohort-{root(row)}" for row in range(size)]


class Strata(splitters.splits.Group):
    """The group split whose holdout is drawn stratified by label (MM-8).

    The test groups are allocated with `curation.allocate` under each
    group's most frequent label (`cohorts.consensus`), so every class with
    enough groups is held out in proportion; the folds over the development
    groups are the group split's (each holding every class for
    classification).
    """

    def partition(self, frame: pandas.DataFrame) -> splitters.Partition:
        """Stratified group holdout and group K-fold.

        Args:
            frame: The full dataset with the group column.

        Returns:
            The development and test positions, the folds and any notes.

        Raises:
            BlockedError: From the group split's own gate
                (`splits.Group.partition`), which runs first: fewer than
                `folds + 2` groups, or (for classification) no group folds
                that keep every class.
        """
        super().partition(frame)
        column = str(self.data.group_column)
        groups = frame[column].astype(str).to_numpy()
        folds = self.validation.folds
        strata = (
            cohorts.consensus(groups.tolist(), frame[self.data.target].tolist())
            if self.task.is_classification
            else None
        )
        test = curation.allocate(
            set(groups),
            0.0,
            self.validation.test_fraction,
            self.validation.seed,
            strata,
        )[1]
        held = numpy.isin(groups, sorted(test))
        dev_ids = numpy.flatnonzero(~held)
        test_ids = numpy.flatnonzero(held)
        dev_groups = groups[dev_ids]
        count = min(folds, len(set(dev_groups)))
        notes: list[str] = ["test groups drawn stratified by label"]
        if not self.task.is_classification:
            return splitters.Partition(
                dev_ids,
                test_ids,
                list(
                    model_selection.GroupKFold(n_splits=count).split(
                        numpy.arange(len(dev_ids)), groups=dev_groups
                    )
                ),
                notes,
            )
        labels = frame[self.data.target].to_numpy()[dev_ids]
        chosen = self.fold(labels, dev_groups, count, notes)
        return splitters.Partition(dev_ids, test_ids, chosen, notes)


def purge(split: splitters.Split, labels: numpy.ndarray) -> splitters.Split:
    """Drops the training rows that share a cohort with held-out rows.

    Args:
        split: A time-ordered split.
        labels: The cohort of each row of the whole data.

    Returns:
        The split without the development rows whose cohort has a test row,
        its folds without the training rows whose cohort has a row in the
        fold's validation part, re-indexed over the kept development rows.

    Raises:
        BlockedError: If purging empties a fold's training or validation
            part, or the development rows.
    """
    manifest = split.manifest
    train = numpy.asarray(manifest.training, dtype=int)
    test = numpy.asarray(manifest.test, dtype=int)
    keep = numpy.flatnonzero(~numpy.isin(labels[train], labels[test]))
    position = numpy.full(len(train), -1)
    position[keep] = numpy.arange(len(keep))
    folds = []
    for fit, check in manifest.folds:
        fitted = numpy.asarray(fit, dtype=int)
        checked = numpy.asarray(check, dtype=int)
        fitted = fitted[
            ~numpy.isin(labels[train[fitted]], labels[train[checked]])
        ]
        fitted = position[fitted][position[fitted] >= 0]
        checked = position[checked][position[checked] >= 0]
        if not len(fitted) or not len(checked):
            raise errors.BlockedError(
                "purging cohorts empties a validation fold; provide more "
                "distinct rows or fewer folds"
            )
        folds.append((splitters.tuplify(fitted), splitters.tuplify(checked)))
    if not len(keep):
        raise errors.BlockedError(
            "purging the cohorts of the test rows leaves no development row"
        )
    return splitters.Split(
        dataclasses.replace(
            manifest,
            training=splitters.tuplify(train[keep]),
            folds=tuple(folds),
            notes=(
                *manifest.notes,
                (
                    f"{len(train) - len(keep)} development rows purged: their "
                    "cohort is tested"
                ),
            ),
        ),
        split.features.iloc[keep].reset_index(drop=True),
        split.target.iloc[keep].reset_index(drop=True),
        split.test,
    )


def strip(split: splitters.Split, notes: list[str]) -> splitters.Split:
    """Drops the cohort column from a split's rows and adds notes.

    Args:
        split: The split.
        notes: Notes for the end of its manifest's notes.

    Returns:
        The split without `KEY` in its development or locked rows.
    """
    locked = split.test
    if isinstance(locked, lockbox.Resident):
        locked = dataclasses.replace(
            locked,
            features=locked.features.drop(columns=[KEY], errors="ignore"),
        )
    return splitters.Split(
        dataclasses.replace(
            split.manifest, notes=(*split.manifest.notes, *notes)
        ),
        split.features.drop(columns=[KEY], errors="ignore"),
        split.target,
        locked,
    )


class Cohorts(splitters.Splitter):
    """The splitter of every kind: no cohort straddles a split or a fold.

    A row's cohort joins its `KEY` (copies, near-copies, chunks of one
    document), the declared group (a row without one is tied by nothing
    else) and the kind's spatial blocks (`places`).
    A time-ordered protocol (`temporal`, `rolling`, or `auto` with a time
    column) keeps its windows and purges the development rows that share a
    cohort with a held-out row; a declared group then joins only when its
    entities are declared `unseen`. Any other protocol becomes the group
    split over the cohorts, its test cohorts drawn stratified by label
    (`Strata`), with a note when the declared protocol was another. When
    every cohort is one row and the kind has no blocks, the declared
    protocol applies unchanged. A kind whose `role` is `time` orders its
    rows by `KEY` when no time column is declared.

    Attributes:
        kind: The kind whose rows are split.
    """

    def __init__(
        self, spec: tasks.Config, task: tasks.Task, kind: "Rows"
    ) -> None:
        """Binds the declared protocol and the kind.

        Args:
            spec: Project specification.
            task: Resolved task type.
            kind: The kind whose rows are split.
        """
        super().__init__(spec, task)
        self.kind = kind

    def order(self, frame: pandas.DataFrame) -> tuple[tasks.Config, list[str]]:
        """The specification a time-role kind's `KEY` orders.

        Args:
            frame: The rows.

        Returns:
            The specification, with `KEY` as the time column and a
            time-ordered protocol when the kind's role is `time`, the rows
            carry `KEY` and no time column is declared; and notes on every
            change.
        """
        spec = self.spec
        if (
            self.kind.role != "time"
            or KEY not in frame.columns
            or spec.data.time_column
        ):
            return spec, []
        notes = []
        validation = spec.validation
        if validation.strategy not in TIMED:
            notes.append(
                f"{validation.strategy} split replaced by the temporal "
                "split: the latest rows are held out"
            )
            validation = validation.model_copy(update={"strategy": "temporal"})
        data = spec.data.model_copy(update={"time_column": KEY})
        return spec.model_copy(
            update={"data": data, "validation": validation}
        ), notes

    def split(self, frame: pandas.DataFrame) -> splitters.Split:
        """Splits so no cohort is on both sides of the split or any fold.

        Args:
            frame: The whole data, with the kind's `KEY` column when it has
                one, and the target.

        Returns:
            The split without `KEY`, its manifest noting how many cohorts
            hold more than one row and every change to the declared
            protocol.

        Raises:
            BlockedError: If too few cohorts exist for the folds, or purging
                empties a fold.
        """
        spec, notes = self.order(frame)
        data = spec.data
        strategy = spec.validation.strategy
        timed = strategy in TIMED or (strategy == "auto" and data.time_column)
        keys: list[Sequence[Any]] = []
        if KEY in frame.columns and data.time_column != KEY:
            keys.append(frame[KEY].tolist())
        if data.group_column and (
            not timed or spec.validation.entities == "unseen"
        ):
            keys.append(frame[data.group_column].tolist())
        blocks, note = self.kind.places(spec, self.task, frame, bool(timed))
        if note:
            notes.append(note)
        if blocks is not None:
            keys.append(blocks)
        labels = numpy.asarray(unite(len(frame), (), *keys))
        sizes = pandas.Series(labels).value_counts()
        joined = sizes[sizes > 1]
        notes.append(
            f"{len(joined)} cohorts (copies, near-copies, declared groups or "
            f"blocks) hold {int(joined.sum())} of {len(frame)} rows; each "
            "is kept on one side of the split and of every fold"
        )
        rows = (
            frame
            if data.time_column == KEY
            else frame.drop(columns=[KEY], errors="ignore")
        )
        plain = splitters.Splitter(spec, self.task)
        if timed:
            found = plain.split(rows)
            if len(joined):
                found = purge(found, labels)
        elif not len(joined) and blocks is None:
            found = plain.split(rows)
        else:
            if strategy not in {"auto", "group"}:
                notes.append(
                    f"{strategy} split replaced by the group split: rows of "
                    "one cohort stay together"
                )
            found = self.group(spec, rows, labels)
        return strip(found, notes)

    def group(
        self, spec: tasks.Config, frame: pandas.DataFrame, labels: numpy.ndarray
    ) -> splitters.Split:
        """The stratified group split with the cohorts as the groups.

        Args:
            spec: The specification of the split.
            frame: The rows, without `KEY`.
            labels: The cohort of each row.

        Returns:
            The split, without the cohort column.
        """
        data = spec.data.model_copy(update={"group_column": KEY})
        validation = spec.validation.model_copy(update={"strategy": "group"})
        grouped = spec.model_copy(
            update={"data": data, "validation": validation}
        )
        marked = frame.assign(**{KEY: labels})
        part = Strata(grouped, self.task).partition(marked)
        splitters.Splitter(grouped, self.task).support(marked, part)
        manifest = splitters.Manifest(
            strategy=Strata.name,
            seed=validation.seed,
            training=splitters.tuplify(part.development),
            test=splitters.tuplify(part.test),
            folds=tuple(
                (splitters.tuplify(fit), splitters.tuplify(held))
                for fit, held in part.folds
            ),
            notes=tuple(part.notes),
        )
        target = marked[data.target]
        features = marked.drop(columns=[data.target, KEY])
        return splitters.Split(
            manifest,
            features.iloc[part.development].reset_index(drop=True),
            target.iloc[part.development].reset_index(drop=True),
            lockbox.Resident(
                features.iloc[part.test].reset_index(drop=True),
                target.iloc[part.test].reset_index(drop=True),
                manifest.test,
            ),
        )


def catalogue(path: pathlib.Path, column: str) -> pandas.DataFrame:
    """Reads a table whose identity column keeps the text it was written as.

    A delimited file's identity column is read as text, so `007` stays
    `007` and `NA` or `null` stay identities (MM-7, MM-16); other formats
    are read by the tabular reader. A present identity is then converted
    to text; a missing one stays missing.

    Args:
        path: The table file.
        column: The identity column.

    Returns:
        The table.

    Raises:
        DataError: If the file cannot be read (see `tabular.read`).
    """
    if path.suffix.lower() in {".csv", ".tsv"}:
        try:
            frame = pandas.read_csv(
                path,
                sep="\t" if path.suffix.lower() == ".tsv" else ",",
                converters={column: str},
            )
        except (
            OSError,
            UnicodeDecodeError,
            pandas.errors.ParserError,
            pandas.errors.EmptyDataError,
        ) as error:
            raise errors.DataError(
                f"failed to read {path.name}: {error}"
            ) from error
    else:
        frame = tabular.read(path)
    if column in frame.columns:
        frame[column] = [
            value if value is None or pandas.isna(value) else str(value)
            for value in frame[column].tolist()
        ]
    return frame


class Rows(abc.ABC):
    """A kind whose items become one table row each.

    Attributes:
        name: The modality a data block declares.
        role: What `KEY` holds: `group` (cohorts) or `time` (when a row is
            predicted).
        status: The capability status the kind is declared with.
        reasons: Why the kind is experimental.
        extras: The optional extras reading needs.
        seconds: Upper bound on the seconds reading takes per row.
        texts: Columns the portrait profiles as free text, however few rows
            there are.
        failures: The kind's known failure modes.
        evidence: The test file that evidences the kind's records.
    """

    name: str
    role: ClassVar[str] = "group"
    status: ClassVar[capabilities.Status] = capabilities.Status.EXPERIMENTAL
    reasons: ClassVar[tuple[str, ...]] = (LIFECYCLE,)
    extras: ClassVar[tuple[str, ...]] = ("core",)
    seconds: ClassVar[float]
    texts: ClassVar[tuple[str, ...]] = ()
    failures: ClassVar[tuple[str, ...]] = (
        "unreadable or undecodable source file",
        "too few cohorts for the folds",
        "model-family dependency unavailable",
    )
    evidence: ClassVar[str]

    @abc.abstractmethod
    def read(
        self, spec: tasks.Config
    ) -> tuple[
        pandas.DataFrame, tuple[streaming.Feed, streaming.Totals] | None
    ]:
        """Reads the items into rows, with `KEY` when it applies.

        Args:
            spec: The run's specification.

        Returns:
            The rows and no stream.
        """

    @abc.abstractmethod
    def limits(self) -> tuple[str, ...]:
        """The boundaries the kind's capabilities state.

        Returns:
            Plain statements of what the kind does not cover.
        """

    def notice(self, frame: pandas.DataFrame) -> list[str]:
        """What the items show beyond the table's profile.

        Args:
            frame: The rows, with `KEY` when they have one.

        Returns:
            A finding counting the rows joined into cohorts, when the rows
            carry them and any cohort holds two or more.
        """
        if self.role != "group" or KEY not in frame.columns:
            return []
        sizes = frame[KEY].value_counts()
        joined = sizes[sizes > 1]
        if not len(joined):
            return []
        return [
            (
                f"{self.name}: {int(joined.sum())} of {len(frame)} rows in "
                f"{len(joined)} cohorts (copies, near-copies and chunks of one "
                "item) are kept on one side of every split"
            )
        ]

    def places(
        self,
        spec: tasks.Config,  # noqa: ARG002 -- a kind with places reads it
        task: tasks.Task,  # noqa: ARG002 -- a kind with places reads it
        frame: pandas.DataFrame,  # noqa: ARG002 -- a kind with places reads it
        timed: bool,  # noqa: ARG002 -- a kind with places reads it
    ) -> tuple[list[object] | None, str | None]:
        """The spatial block of each row, which `Cohorts` keeps whole.

        Args:
            spec: The specification of the split.
            task: The resolved task.
            frame: The rows.
            timed: Whether the split is time-ordered.

        Returns:
            No blocks and no note; a kind with places gives its blocks and
            a manifest note.
        """
        return None, None

    def survey(
        self,
        spec: tasks.Config,
        frame: pandas.DataFrame,
        manifest: scmle.contracts.data.DatasetManifest,
        task: tasks.Task,
        seconds: float = 0.0,
        total: int | None = None,
    ) -> scmle.contracts.data.Portrait:
        """The profiler's portrait of the rows, with the kind's findings.

        Args:
            spec: The run's specification.
            frame: The rows to profile.
            manifest: The identity of the whole dataset.
            task: The resolved task.
            seconds: Seconds the read took.
            total: Rows in the whole dataset when `frame` is a subset.

        Returns:
            The portrait of the rows without `KEY`, the kind's `texts`
            profiled as text, its quality findings extended by `notice`.
        """
        portrait = profiler.Profiler(spec, self.texts).profile(
            frame.drop(columns=[KEY], errors="ignore"),
            manifest,
            task,
            seconds,
            total,
        )
        return dataclasses.replace(
            portrait,
            quality_findings=(*portrait.quality_findings, *self.notice(frame)),
        )

    def constrain(
        self, spec: tasks.Config, task: tasks.Task
    ) -> splitters.Splitter:
        """The splitter that keeps every cohort on one side (`Cohorts`).

        Args:
            spec: The run's specification.
            task: The resolved task.

        Returns:
            The cohort splitter of the kind.
        """
        return Cohorts(spec, task, self)

    def featurise(
        self,
        plan: scmle.contracts.data.Blueprint,
        options: scmle.contracts.data.Preprocessing,
    ) -> Any:
        """The column transformer of the feature plan, fitted in each fold.

        Args:
            plan: The feature plan.
            options: The preprocessing switches.

        Returns:
            The unfitted column transformer
            (`preprocessing.build_preprocessor`).
        """
        return preprocessing.build_preprocessor(plan, options)

    def decode(self, labels: tuple[Any, ...] | None) -> tuple[Any, ...] | None:
        """Ordinal ranks decode to the recorded labels; else the classes.

        Args:
            labels: The ordinal labels, lowest first, or None.

        Returns:
            `labels`.
        """
        return labels

    def cost(
        self, rows: int, columns: int, classes: int, budget: float
    ) -> float:
        """The table's bound plus reading the items.

        Args:
            rows: Development rows.
            columns: Feature columns.
            classes: Classes of the target.
            budget: The run's time budget in seconds.

        Returns:
            `profiler.estimate` plus `seconds` per row.
        """
        estimate = profiler.estimate(rows, columns, classes, budget)
        return estimate + rows * self.seconds

    def declare(self) -> tuple[capabilities.Capability, ...]:
        """The capabilities the rows are run with.

        Returns:
            One library-inference record per task of `TASKS`, family of
            `FAMILIES` and fairness declaration (none, and mitigation when
            sensitive columns pass through), with the kind's status, extras,
            failure modes and limits (its `reasons` last).
        """
        return tuple(
            capabilities.Capability(
                key=capabilities.Coordinate(
                    modality=capabilities.Modality(self.name),
                    task=task,
                    model_family=family,
                    evaluation=capabilities.Evaluation.STANDARD,
                    uncertainty=capabilities.Uncertainty.NONE,
                    fairness=fairness,
                    explainability=capabilities.Explainability.NONE,
                    inference=capabilities.Inference.LIBRARY,
                ),
                status=self.status,
                required_extras=self.extras,
                resources=capabilities.Resources(memory_gb=1.0),
                failure_modes=self.failures,
                evidence_links=(self.evidence,),
                limitations=(*self.limits(), *self.reasons),
            )
            for task in TASKS
            for family in FAMILIES
            for fairness in (
                capabilities.Fairness.NONE,
                capabilities.Fairness.MITIGATION,
            )
        )
