"""The media kinds' base: files described into one numeric row each.

A data block of a media kind (audio, raster, image, video) names a
manifest table with one row per file next to the target and any other
columns, or a directory whose subdirectories name the labels of the files
inside them. The file is named by the `path` column, or else by the one
column whose every value names an existing file of the kind; a path is read
relative to the table. `Media.read` describes every file (`trace`) into
numeric feature columns, so the shared stages search, cross-validate, test,
explain and package media as they do a table, with every model family the
task allows.

Files that are copies (one content digest) or near-copies (`pairs`) of each
other, or that the kind ties by other keys (`keys`, such as a raster's
spatial blocks), share a cohort (`rows.KEY`), which `rows.Cohorts` keeps on
one side of every split (MM-10, MM-11). Rows are ordered by their file, so
the order of the listing changes nothing (MM-12).
"""

import abc
from collections.abc import Iterable
import dataclasses
import hashlib
import pathlib
from typing import Any
from typing import ClassVar
from typing import Protocol

import numpy
import pandas

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.readers import streaming
from scmle.data.readers import tabular
from scmle.pipeline import profiler
from scmle.pipeline.kinds import rows

SOURCE = "path"
"""The manifest column naming each file, relative to the manifest."""

SHOWN = 20
"""Unreadable files named in a read error; the rest are counted."""

BLOCK = 128
"""Rows compared at once when pairing envelopes: a block of 128 rows
against 100,000 others is 100 MB of float64 distances."""


class Trace(Protocol):
    """What a media kind learns from one file.

    Attributes:
        features: Feature name to finite value.
        digest: SHA-256 of the content (exact copies).
        envelope: Cells whose distance measures likeness (near-copies).
    """

    @property
    def features(self) -> dict[str, float]:
        """Feature name to finite value."""
        ...

    @property
    def digest(self) -> str:
        """SHA-256 of the content."""
        ...

    @property
    def envelope(self) -> tuple[float, ...]:
        """Cells whose distance measures likeness."""
        ...


@dataclasses.dataclass(frozen=True)
class Description:
    """A file's features and content digest, for a kind that pairs by them.

    Attributes:
        features: Feature name to finite value.
        digest: SHA-256 of the file.
        envelope: Unused; the kind's `pairs` reads the features.
    """

    features: dict[str, float]
    digest: str
    envelope: tuple[float, ...] = ()


def digest(path: pathlib.Path) -> str:
    """The SHA-256 of a file, read in chunks.

    Args:
        path: The file.

    Returns:
        The hex digest.
    """
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def near(
    envelopes: numpy.ndarray, masks: numpy.ndarray, radius: float
) -> list[tuple[int, int]]:
    """The pairs of rows whose envelopes are within a radius.

    The distance is the root-mean-square difference over the cells either
    row marks informative; cells neither marks are left out, so a quiet
    band two clips share does not make them alike. Rows are compared a
    `BLOCK` at a time.

    Args:
        envelopes: Rows by cells. Cells a row does not mark hold the same
            value in every row.
        masks: Rows by cells; True where a row's cell is informative.
        radius: Largest root-mean-square difference of two near-copies.

    Returns:
        Each linked pair, the first row before the second.
    """
    norms = numpy.square(envelopes).sum(axis=1)
    marked = masks.astype(numpy.float64)
    counts = marked.sum(axis=1)
    found: list[tuple[int, int]] = []
    for start in range(0, len(envelopes), BLOCK):
        stop = start + BLOCK
        union = (
            counts[start:stop, None]
            + counts[None, :]
            - marked[start:stop] @ marked.T
        )
        distance = (
            norms[start:stop, None]
            + norms[None, :]
            - 2.0 * envelopes[start:stop] @ envelopes.T
        ) / numpy.maximum(union, 1.0)
        found.extend(
            (int(start + first), int(second))
            for first, second in zip(
                *numpy.nonzero(distance <= radius * radius), strict=True
            )
            if start + first < second
        )
    return found


class Media(rows.Rows):
    """A kind whose rows are files described into numeric features.

    Attributes:
        suffixes: Lower-case file suffixes read as this kind.
        radius: Largest root-mean-square envelope difference of two
            near-copies.
        pair: Seconds to compare two files for likeness, set by
            observation.
        consumed: Manifest columns that declare how to read a file; they
            are no feature, so the rows leave them out.
    """

    suffixes: ClassVar[frozenset[str]]
    radius: ClassVar[float] = 0.0
    pair: ClassVar[float]
    consumed: ClassVar[frozenset[str]] = frozenset()
    failures = (
        "unreadable or undecodable file",
        "decoder or model-family dependency unavailable",
        "too few cohorts for the folds",
    )

    @abc.abstractmethod
    def trace(self, path: pathlib.Path, row: pandas.Series) -> Trace:
        """Describes one file.

        Args:
            path: The file.
            row: Its manifest row, for per-file declarations.

        Returns:
            Its features, digest and envelope.
        """

    def ready(self) -> None:
        """Raises the kind's typed error when a decoder is missing."""

    def pairs(
        self,
        traces: list[Any],
        features: pandas.DataFrame,  # noqa: ARG002 -- a kind pairing features reads it
    ) -> Iterable[tuple[int, int]]:
        """The near-copy pairs: envelopes within `radius`.

        Args:
            traces: Each file's trace.
            features: The feature rows.

        Returns:
            The linked pairs (`near`), over the cells `mask` marks.
        """
        envelopes = numpy.asarray([trace.envelope for trace in traces])
        return near(envelopes, self.mask(envelopes), self.radius)

    def mask(self, envelopes: numpy.ndarray) -> numpy.ndarray:
        """Marks the informative envelope cells.

        Args:
            envelopes: Rows by cells.

        Returns:
            True for every cell; a kind with a floor marks fewer.
        """
        return numpy.ones(envelopes.shape, dtype=bool)

    def keys(
        self,
        spec: tasks.Config,  # noqa: ARG002 -- a kind with places reads it
        listed: pandas.DataFrame,  # noqa: ARG002 -- a kind with places reads it
        traces: list[Any],  # noqa: ARG002 -- a kind with places reads it
    ) -> list[list[object]]:
        """Further values that tie rows into one cohort.

        Args:
            spec: The run's specification.
            listed: The manifest.
            traces: Each row's trace.

        Returns:
            None here; a kind with places adds its spatial blocks.
        """
        return []

    def manifest(self, data: tasks.Dataset) -> pandas.DataFrame:
        """Lists the declared files with their table columns.

        Args:
            data: The data block.

        Returns:
            One row per file, ordered by path, with the absolute path in the
            `SOURCE` column.

        Raises:
            DataError: If the source does not exist, names no file of the
                kind or a missing file, or no single column names the files.
        """
        root = pathlib.Path(data.uri).expanduser()
        if root.is_dir():
            found = [
                (str(path.resolve()), folder.name)
                for folder in sorted(root.iterdir())
                if folder.is_dir() and not folder.name.startswith(".")
                for path in sorted(folder.rglob("*"))
                if path.is_file() and path.suffix.lower() in self.suffixes
            ]
            table = pandas.DataFrame(found, columns=[SOURCE, data.target])
        elif root.is_file():
            table = tabular.read_table(root)
            table = table.rename(
                columns={self.column(table, root.parent): SOURCE}
            )
            table[SOURCE] = [
                str((root.parent / str(value)).expanduser().resolve())
                for value in table[SOURCE]
            ]
        else:
            raise errors.DataError(
                f"{self.name} source does not exist",
                {"uri": data.uri, "modality": self.name},
            )
        if table.empty:
            raise errors.DataError(
                f"{self.name} source names no file in a label subdirectory "
                "or a manifest row",
                {"uri": data.uri, "suffixes": sorted(self.suffixes)},
            )
        missing = [
            value
            for value in table[SOURCE]
            if not pathlib.Path(value).is_file()
        ]
        if missing:
            raise errors.DataError(
                f"{len(missing)} listed file(s) do not exist",
                {"missing": missing[:SHOWN]},
            )
        return table.sort_values(SOURCE, kind="stable").reset_index(drop=True)

    def column(self, table: pandas.DataFrame, base: pathlib.Path) -> str:
        """Finds the column of a manifest that names the files.

        Args:
            table: The manifest.
            base: The directory relative paths are read from.

        Returns:
            `SOURCE` when present; else the one column whose every value
            names an existing file of the kind.

        Raises:
            DataError: If no column, or more than one, names the files.
        """
        if SOURCE in table.columns:
            return SOURCE
        candidates = [
            str(name)
            for name in table.columns
            if len(table)
            and all(
                isinstance(value, str)
                and pathlib.Path(value).suffix.lower() in self.suffixes
                and (base / value).expanduser().is_file()
                for value in table[name]
            )
        ]
        if len(candidates) != 1:
            raise errors.DataError(
                f"the manifest has no '{SOURCE}' column naming each "
                f"{self.name} file",
                {"candidates": candidates, "columns": list(table.columns)},
            )
        return candidates[0]

    def gather(self, listed: pandas.DataFrame) -> list[Any]:
        """Describes every listed file, naming the files that fail.

        Args:
            listed: The manifest, with the `SOURCE` column.

        Returns:
            The traces, in manifest order.

        Raises:
            DataError: The first failing file's error, its details naming
                that file (`source`), the first `SHOWN` failures and their
                count.
        """
        made = []
        failures: list[tuple[str, errors.DataError]] = []
        for index in range(len(listed)):
            row = listed.iloc[index]
            try:
                made.append(self.trace(pathlib.Path(row[SOURCE]), row))
            except errors.DataError as error:
                # handled: collected, then the first is raised naming them all
                failures.append((str(row[SOURCE]), error))
        if failures:
            source, first = failures[0]
            first.details = {
                **first.details,
                "source": source,
                "failures": [
                    {"source": path, "error": str(error)}
                    for path, error in failures[:SHOWN]
                ],
                "count": len(failures),
            }
            raise first
        return made

    def describe(self, paths: Iterable[str | pathlib.Path]) -> pandas.DataFrame:
        """Describes files into the feature columns a run trained on.

        A packaged model of this kind predicts from these rows.

        Args:
            paths: The files.

        Returns:
            One row of features per file, in the order given.
        """
        self.ready()
        listed = pandas.DataFrame({SOURCE: [str(path) for path in paths]})
        return pandas.DataFrame(
            [trace.features for trace in self.gather(listed)]
        )

    def read(
        self, spec: tasks.Config
    ) -> tuple[
        pandas.DataFrame, tuple[streaming.Feed, streaming.Totals] | None
    ]:
        """Describes every listed file into a row of features.

        Args:
            spec: The run's specification.

        Returns:
            One row per file, ordered by source: the manifest's columns
            other than the `consumed` ones and the file (kept when the data
            block gives it a role), the features and `rows.KEY`; no stream.

        Raises:
            DataError: If the source cannot be listed (`manifest`), a file
                cannot be read (`gather`), or a manifest column clashes with
                a feature or `rows.KEY`. The kind's dependency error when
                its decoder is missing.
        """
        self.ready()
        data = spec.data
        listed = self.manifest(data)
        traces = self.gather(listed)
        features = pandas.DataFrame([trace.features for trace in traces])
        clashes = sorted(
            {rows.KEY, *map(str, features.columns)}
            & set(map(str, listed.columns))
        )
        if clashes:
            raise errors.DataError(
                f"manifest columns clash with the {self.name} rows",
                {"columns": clashes},
            )
        tied = rows.unite(
            len(traces),
            self.pairs(traces, features),
            [trace.digest for trace in traces],
            *self.keys(spec, listed, traces),
        )
        roles = {data.group_column, data.time_column, *data.id_columns}
        dropped = {SOURCE} - roles | self.consumed
        kept = listed.drop(columns=[c for c in listed.columns if c in dropped])
        frame = pandas.concat([kept, features], axis=1)
        return frame.assign(**{rows.KEY: tied}), None

    def cost(
        self, rows: int, columns: int, classes: int, budget: float
    ) -> float:
        """Bounds the seconds of describing, pairing and profiling.

        Args:
            rows: Development rows.
            columns: Feature columns.
            classes: Classes of the target.
            budget: The run's time budget in seconds.

        Returns:
            `profiler.estimate` plus `seconds` per file and `pair` per pair
            of files.
        """
        return (
            profiler.estimate(rows, columns, classes, budget)
            + rows * self.seconds
            + rows * rows * self.pair
        )
