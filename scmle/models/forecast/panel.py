"""Panel data: one series per entity, each ordered in time.

A panel is a long table with an entity column (a store, a sensor, a
patient), a time column and the value forecast. A single series is a panel
without the entity column. Steps are counted in rows of an entity, as the
forecast lags and horizon of the project configuration are.
"""

from collections.abc import Hashable
import dataclasses
from typing import Any
from typing import Self

import numpy
import pandas

from scmle.contracts import errors


@dataclasses.dataclass(frozen=True)
class Panel:
    """A validated multi-series table ordered by entity, then time.

    Attributes:
        frame: The rows, sorted by entity then time, index reset.
        time: Column ordering each series (numbers or timestamps).
        value: Column holding the series' value.
        entity: Column naming the series, or None for a single series.
    """

    frame: pandas.DataFrame
    time: str
    value: str
    entity: str | None = None

    @classmethod
    def build(
        cls,
        frame: pandas.DataFrame,
        time: str,
        value: str,
        entity: str | None = None,
    ) -> Self:
        """Checks and orders a long table.

        Args:
            frame: The raw rows.
            time: Time column; numeric values stay numbers, others are parsed
                as timestamps.
            value: Value column; must be numeric.
            entity: Entity column, or None for a single series.

        Returns:
            The panel, with the time column parsed and the value as float.

        Raises:
            InputError: If a named column is not in the frame.
            DataError: If an entity or time is missing, a time is
                unparseable, a value is missing or not numeric, or an entity
                repeats a time.
        """
        named = [c for c in (entity, time, value) if c is not None]
        absent = [c for c in named if c not in frame.columns]
        if absent:
            raise errors.InputError(
                f"panel column(s) {', '.join(repr(c) for c in absent)} are "
                "not in the frame"
            )
        table = frame[named].copy()
        if entity is not None and table[entity].isna().any():
            raise errors.DataError(
                f"entity column '{entity}' has missing values"
            )
        if not pandas.api.types.is_numeric_dtype(table[time]):
            table[time] = pandas.to_datetime(table[time], errors="coerce")
        if table[time].isna().any():
            raise errors.DataError(
                f"time column '{time}' has missing or unparseable values"
            )
        table[value] = pandas.to_numeric(table[value], errors="coerce")
        bad = int(table[value].isna().sum())
        if bad:
            raise errors.DataError(
                f"value column '{value}' has {bad} missing or non-numeric "
                "value(s); fill or drop them before forecasting"
            )
        keys = [c for c in (entity, time) if c is not None]
        repeated = int(table.duplicated(subset=keys).sum())
        if repeated:
            raise errors.DataError(
                f"{repeated} repeated {' and '.join(keys)} row(s): each series "
                "needs one value per time"
            )
        table[value] = table[value].astype(float)
        ordered = table.sort_values(keys, kind="stable").reset_index(drop=True)
        return cls(ordered, time, value, entity)

    def series(self) -> dict[Hashable, numpy.ndarray]:
        """Each entity's values in time order.

        Returns:
            Entity to float array, entities sorted; a single series is keyed
            by the value column's name.
        """
        if self.entity is None:
            return {self.value: self.frame[self.value].to_numpy(float)}
        grouped = self.frame.groupby(self.entity, sort=True)[self.value]
        return {key: part.to_numpy(float) for key, part in grouped}

    def moments(self) -> numpy.ndarray:
        """The distinct times of the panel, ascending."""
        return numpy.sort(self.frame[self.time].unique())

    def history(self, moment: Any) -> Self:
        """The history known at `moment`.

        Args:
            moment: A time of the panel's time column.

        Returns:
            The panel of the rows at or before `moment`.
        """
        kept = self.frame[self.frame[self.time] <= moment]
        return dataclasses.replace(self, frame=kept.reset_index(drop=True))

    def ahead(self, moment: Any, steps: int) -> dict[Hashable, numpy.ndarray]:
        """What each entity did in the `steps` rows after `moment`.

        Args:
            moment: The forecast origin.
            steps: Rows ahead to read.

        Returns:
            Entity to its first `steps` values after `moment` (fewer, or
            none, where its series ends sooner), for every entity.
        """
        later = dataclasses.replace(
            self, frame=self.frame[self.frame[self.time] > moment]
        )
        found = later.series()
        return {
            key: found.get(key, numpy.empty(0))[:steps] for key in self.series()
        }
