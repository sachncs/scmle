"""Data kinds: what the shared stages need to know about one kind of data.

A `Kind` is the one seam between the shared stages and a kind of data. It
reads the declared source (`read`), reports what profiling finds in the
development rows (`survey`), states the split constraints as the splitter
that keeps related rows on one side (`constrain`), builds the featuriser
that turns rows into model inputs (`featurise`), names what a served model's
outputs decode to (`decode`) and bounds the seconds featurising and
profiling take (`cost`). Ingestion reads through it, profiling surveys and
splits through it, and packaging decodes through it.

`Tabular` is the first kind. A new kind is a class in `kinds/<name>.py`
with these six methods, added to `KINDS` under its modality name; `pick`
resolves the kind a data block declares and refuses one that has none.
"""

from collections.abc import Mapping
from typing import Any
from typing import Protocol
from typing import runtime_checkable

import pandas

from scmle.contracts import capabilities
from scmle.contracts import tasks
import scmle.contracts.data
from scmle.data.readers import streaming
from scmle.data.readers import tabular
from scmle.data.splitting import splitters
from scmle.pipeline import profiler
from scmle.preparation import preprocessing

GIGABYTE = 1e9
"""Bytes in the gigabyte `memory_budget_gb` is stated in."""


@runtime_checkable
class Kind(Protocol):
    """What the shared stages need of one kind of data.

    Attributes:
        name: The modality the kind serves, as a data block declares it.
    """

    name: str

    def read(
        self, spec: tasks.Config
    ) -> tuple[
        pandas.DataFrame, tuple[streaming.Feed, streaming.Totals] | None
    ]:
        """Reads the declared source.

        Args:
            spec: The run's specification; its data block names the source.

        Returns:
            The rows, and the chunked source with its totals when the rows
            are a sample kept from one streaming pass, else None.
        """
        ...

    def survey(
        self,
        spec: tasks.Config,
        frame: pandas.DataFrame,
        manifest: scmle.contracts.data.DatasetManifest,
        task: tasks.Task,
        seconds: float = 0.0,
        total: int | None = None,
    ) -> scmle.contracts.data.Portrait:
        """Profiles rows into findings and a feature plan.

        Args:
            spec: The run's specification.
            frame: The rows to profile (the development rows).
            manifest: The identity of the whole dataset.
            task: The resolved task.
            seconds: Seconds the read took.
            total: Rows in the whole dataset when `frame` is a subset.

        Returns:
            The portrait: findings, statistics and the feature plan.
        """
        ...

    def constrain(
        self, spec: tasks.Config, task: tasks.Task
    ) -> splitters.Splitter:
        """States the split constraints of the data.

        Args:
            spec: The run's specification.
            task: The resolved task.

        Returns:
            The splitter that keeps the rows the constraints tie together
            on one side of every split.
        """
        ...

    def featurise(
        self,
        plan: scmle.contracts.data.Blueprint,
        options: scmle.contracts.data.Preprocessing,
    ) -> Any:
        """Builds the unfitted featuriser of a feature plan.

        Args:
            plan: The feature plan.
            options: The preprocessing switches.

        Returns:
            A scikit-learn transformer from rows to model inputs.
        """
        ...

    def decode(self, labels: tuple[Any, ...] | None) -> tuple[Any, ...] | None:
        """Names the labels a served model's outputs decode to.

        Args:
            labels: The labels the run recorded, lowest first (ordinal
                tasks), or None.

        Returns:
            The labels per output column, or None when the model's own
            classes name them.
        """
        ...

    def cost(
        self, rows: int, columns: int, classes: int, budget: float
    ) -> float:
        """Bounds the seconds featurising and profiling take.

        Args:
            rows: Development rows.
            columns: Feature columns.
            classes: Classes of the target; below two means regression.
            budget: The run's time budget in seconds.

        Returns:
            An upper bound on the seconds.
        """
        ...


class Tabular:
    """Rows and columns read from a table file or given in memory.

    Attributes:
        name: "tabular".
    """

    name = "tabular"

    @staticmethod
    def overflows(data: tasks.Dataset) -> bool:
        """Whether the file is bigger than the memory budget.

        Args:
            data: The data block, giving the source and the optional memory
                budget in gigabytes.

        Returns:
            True if a budget is set and the source's estimated size exceeds
            it; False if no budget is set.
        """
        if data.memory_budget_gb is None:
            return False
        size = streaming.locate(data.uri).footprint()
        return size > data.memory_budget_gb * GIGABYTE

    def read(
        self, spec: tasks.Config
    ) -> tuple[
        pandas.DataFrame, tuple[streaming.Feed, streaming.Totals] | None
    ]:
        """Reads the table whole, or in one pass when it overflows.

        A streamed pass assigns each row to the locked or the development
        side by a hash of its position and keeps a capped sample of each.

        Args:
            spec: The run's specification.

        Returns:
            The rows (with a `split_side` column when streamed) and the
            stream, or None when the file was read whole.
        """
        data = spec.data
        if not self.overflows(data):
            return tabular.read_table(data.uri), None
        source = streaming.locate(data.uri)
        development, locked = streaming.budgets(
            source.shape()[1],
            data.memory_budget_gb or 1.0,
            spec.validation.test_fraction,
        )
        frame, totals = streaming.sweep(
            source,
            development,
            locked,
            spec.validation.test_fraction,
            spec.validation.seed,
        )
        return frame, (source, totals)

    def survey(
        self,
        spec: tasks.Config,
        frame: pandas.DataFrame,
        manifest: scmle.contracts.data.DatasetManifest,
        task: tasks.Task,
        seconds: float = 0.0,
        total: int | None = None,
    ) -> scmle.contracts.data.Portrait:
        """Profiles the rows with the column profiler.

        Args:
            spec: The run's specification.
            frame: The rows to profile.
            manifest: The identity of the whole dataset.
            task: The resolved task.
            seconds: Seconds the read took.
            total: Rows in the whole dataset when `frame` is a subset.

        Returns:
            The profiler's portrait (`profiler.Profiler.profile`).
        """
        return profiler.Profiler(spec).profile(
            frame, manifest, task, seconds, total
        )

    def constrain(
        self, spec: tasks.Config, task: tasks.Task
    ) -> splitters.Splitter:
        """The splitter of the declared protocol (groups, time, strata).

        Args:
            spec: The run's specification.
            task: The resolved task.

        Returns:
            The splitter.
        """
        return splitters.Splitter(spec, task)

    def featurise(
        self,
        plan: scmle.contracts.data.Blueprint,
        options: scmle.contracts.data.Preprocessing,
    ) -> Any:
        """The column transformer of the feature plan.

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
        """The profiler's bound on the seconds of data preparation.

        Args:
            rows: Development rows.
            columns: Feature columns.
            classes: Classes of the target.
            budget: The run's time budget in seconds.

        Returns:
            `profiler.estimate`.
        """
        return profiler.estimate(rows, columns, classes, budget)


TABULAR = Tabular()
"""The tabular kind; rows given in memory are read as a table."""

KINDS: Mapping[str, Kind] = {TABULAR.name: TABULAR}
"""Every registered kind, by the modality name a data block declares."""


def pick(data: tasks.Dataset) -> Kind:
    """The kind of a declared data source.

    Args:
        data: The data block.

    Returns:
        The kind registered for its modality.

    Raises:
        UnsupportedError: If no kind is registered for the modality.
    """
    found = KINDS.get(data.modality)
    if found is None:
        raise capabilities.UnsupportedError(
            "file-backed modality has no registered ingestion adapter",
            {
                "status": "unsupported",
                "reason": "file-backed modality has no ingestion adapter",
                "modality": data.modality,
                "uri": data.uri,
                "supported_modalities": sorted(KINDS),
            },
        )
    return found
