"""Declared tables, their times, and the links that relate them.

A `Table` names its key and how it is timed: the event time (when the fact
happened) and the recorded time (when the system knew it). A row is known at
its recorded time; without one, or where the recorded cell is empty, at its
event time plus the declared recording `delay`. A table with no event time
is declared static: every row is visible at every cutoff, which is a
statement about the table that nothing here can check.

A `Link` relates the target rows to a table. `many_to_one` joins each
target row to the one row (the latest version visible at its cutoff) its
key names; `one_to_many` gathers every visible row its key names, for
aggregates. `Schema.check` holds the data to the declared cardinality: a
`many_to_one` table may hold several versions of a key, but never two rows
with the same key, event time and recorded time, which no as-of rule could
order.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import math
from typing import Any

import pandas

from scmle.contracts import errors

ONE = "many_to_one"
MANY = "one_to_many"
CARDINALITIES = (ONE, MANY)


@dataclasses.dataclass(frozen=True, slots=True)
class Table:
    """A table and how its rows are timed.

    Attributes:
        name: The table's name in its store.
        event: Event time column, or None for a static table.
        known: Recorded time column, or None.
        delay: Declared largest recording delay in days, added to the event
            time where no recorded time is given.
    """

    name: str
    event: str | None = None
    known: str | None = None
    delay: float = 0.0

    def __post_init__(self) -> None:
        """Validates the timing.

        Raises:
            ConfigError: If the name is empty, a recorded time is given
                without an event time, or the delay is negative or not
                finite.
        """
        if not self.name:
            raise errors.ConfigError("a table needs a name")
        if self.known is not None and self.event is None:
            raise errors.ConfigError(
                f"table {self.name!r} names a recorded time without an "
                "event time"
            )
        if not math.isfinite(self.delay) or self.delay < 0:
            raise errors.ConfigError(
                f"table {self.name!r} has delay {self.delay}; it must be "
                "finite and non-negative"
            )

    def columns(self) -> tuple[str, ...]:
        """The time columns the table must hold.

        Returns:
            The event and recorded time columns that are declared.
        """
        return tuple(c for c in (self.event, self.known) if c is not None)


@dataclasses.dataclass(frozen=True, slots=True)
class Link:
    """How the target rows reach a table.

    Attributes:
        table: The related table's name.
        key: Column of the related table.
        target: Column of the target rows (or of an earlier
            `many_to_one` join) whose value `key` matches.
        cardinality: `many_to_one` or `one_to_many`.
        name: The link's name, which prefixes its columns; the table's name
            when empty.
    """

    table: str
    key: str
    target: str
    cardinality: str = MANY
    name: str = ""

    def __post_init__(self) -> None:
        """Defaults the name and validates the cardinality.

        Raises:
            ConfigError: If the cardinality is not one of `CARDINALITIES`.
        """
        if self.cardinality not in CARDINALITIES:
            raise errors.ConfigError(
                f"link to {self.table!r} has cardinality "
                f"{self.cardinality!r}; use one of {CARDINALITIES}"
            )
        if not self.name:
            object.__setattr__(self, "name", self.table)


def duplicates(frame: pandas.DataFrame, table: Table, key: str) -> int:
    """Rows a `many_to_one` join could not tell apart.

    Args:
        frame: The related table.
        table: Its declaration.
        key: The link's key column.

    Returns:
        Rows beyond the first sharing a key, event time and recorded time
        (a key alone, for a static table); rows with no key are ignored.
    """
    columns = [key, *table.columns()]
    keyed = frame.loc[frame[key].notna(), columns]
    return int(keyed.duplicated(keep="first").sum())


@dataclasses.dataclass(frozen=True, slots=True)
class Schema:
    """The declared tables and links.

    Attributes:
        tables: Every table a link names.
        links: The links, applied in order; a link's target may be a column
            an earlier `many_to_one` link added.
    """

    tables: tuple[Table, ...]
    links: tuple[Link, ...]

    def __post_init__(self) -> None:
        """Validates the names.

        Raises:
            ConfigError: If two tables or two links share a name, or a link
                names an undeclared table.
        """
        for kind, names in (
            ("table", [t.name for t in self.tables]),
            ("link", [link.name for link in self.links]),
        ):
            twice = sorted({n for n in names if names.count(n) > 1})
            if twice:
                raise errors.ConfigError(f"{kind} names repeat: {twice}")
        declared = {t.name for t in self.tables}
        absent = sorted({link.table for link in self.links} - declared)
        if absent:
            raise errors.ConfigError(f"links name undeclared tables {absent}")

    def table(self, name: str) -> Table:
        """The declaration of a table.

        Args:
            name: The table's name.

        Returns:
            Its declaration.

        Raises:
            ConfigError: If no table has that name.
        """
        for table in self.tables:
            if table.name == name:
                return table
        raise errors.ConfigError(f"no table {name!r} is declared")

    def link(self, name: str) -> Link:
        """A link by name.

        Args:
            name: The link's name.

        Returns:
            The link.

        Raises:
            ConfigError: If no link has that name.
        """
        for link in self.links:
            if link.name == name:
                return link
        raise errors.ConfigError(f"no link {name!r} is declared")

    def check(self, frames: Mapping[str, pandas.DataFrame]) -> None:
        """Holds the tables to their declarations.

        Args:
            frames: Each declared table by name.

        Raises:
            ConfigError: If a declared table is missing.
            SchemaError: If a table lacks a declared column, or a
                `many_to_one` table holds rows its key, event time and
                recorded time do not tell apart.
        """
        for link in self.links:
            table = self.table(link.table)
            if link.table not in frames:
                raise errors.ConfigError(f"table {link.table!r} was not read")
            frame = frames[link.table]
            needed = [link.key, *table.columns()]
            absent = [c for c in needed if c not in frame.columns]
            if absent:
                raise errors.SchemaError(
                    f"table {link.table!r} lacks declared columns {absent}"
                )
            if link.cardinality != ONE:
                continue
            twice = duplicates(frame, table, link.key)
            if twice:
                raise errors.SchemaError(
                    f"link {link.name!r} is declared many_to_one but "
                    f"{link.table!r} holds {twice} rows that repeat a "
                    f"({', '.join(needed)}) value"
                )


def text(value: Any, what: str) -> str:
    """A required non-empty string of a schema document.

    Args:
        value: The entry.
        what: What it is, for the message.

    Returns:
        `value`.

    Raises:
        ConfigError: If `value` is not a non-empty string.
    """
    if not isinstance(value, str) or not value:
        raise errors.ConfigError(f"schema {what} must be a non-empty string")
    return value


def optional(value: Any, what: str) -> str | None:
    """An optional non-empty string of a schema document.

    Args:
        value: The entry, or None when absent.
        what: What it is, for the message.

    Returns:
        `value`, or None.

    Raises:
        ConfigError: If `value` is neither None nor a non-empty string.
    """
    return None if value is None else text(value, what)


def parse(document: Mapping[str, Any]) -> Schema:
    """Builds a schema from its JSON form.

        {"tables": [{"name": "orders", "event": "created",
                     "known": "recorded", "delay": 0.0}],
         "links": [{"table": "orders", "key": "customer_id",
                    "target": "customer_id",
                    "cardinality": "one_to_many", "name": "orders"}]}

    Args:
        document: The mapping.

    Returns:
        The schema.

    Raises:
        ConfigError: If an entry is missing or of the wrong type.
    """
    tables: Sequence[Any] = document.get("tables", [])
    links: Sequence[Any] = document.get("links", [])
    if not isinstance(tables, list) or not isinstance(links, list):
        raise errors.ConfigError("schema tables and links must be lists")
    if not all(isinstance(e, dict) for e in [*tables, *links]):
        raise errors.ConfigError("schema entries must be objects")
    delays = [e.get("delay", 0.0) for e in tables]
    if not all(isinstance(d, int | float) for d in delays):
        raise errors.ConfigError("schema table delay must be a number")
    return Schema(
        tuple(
            Table(
                text(e.get("name"), "table name"),
                optional(e.get("event"), "table event"),
                optional(e.get("known"), "table known"),
                float(delay),
            )
            for e, delay in zip(tables, delays, strict=True)
        ),
        tuple(
            Link(
                text(e.get("table"), "link table"),
                text(e.get("key"), "link key"),
                text(e.get("target"), "link target"),
                text(e.get("cardinality", MANY), "link cardinality"),
                optional(e.get("name"), "link name") or "",
            )
            for e in links
        ),
    )
