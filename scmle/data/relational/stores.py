"""Where related tables live: a SQLite file, a Parquet folder or a CSV set.

Every store answers the same three questions, so the engine and the ingest
stage never ask which kind they hold:

* `names`: the tables it holds;
* `read`: one table, whole, in its stored row order (the order breaks ties
  between rows with the same event and recorded time);
* `factory`: read-only SQLite connections over the tables, for the code that
  profiles and queries tables in SQL. A database answers with itself; a
  folder is mirrored once into a SQLite file, and the mirror's row count is
  checked against each file before it is used.

A folder holds a table either as one file (`orders.parquet`) or as a
directory of part files (`orders/part-0.parquet`, ..., or the Spark and
pyarrow layout `orders.parquet/part-0.parquet`), read in name order and
stacked; a table held by two entries is refused. A new kind of store is one
subclass added to `KINDS`.
"""

import abc
from collections.abc import Callable
import pathlib
import sqlite3
from typing import ClassVar

import pandas

from scmle.contracts import errors
from scmle.data.readers import tabular

# The first sixteen bytes of every SQLite 3 database file.
HEADER = b"SQLite format 3\x00"
MIRROR = "mirror.sqlite"


def quote(name: str) -> str:
    """Quotes a table name for SQLite.

    Args:
        name: The table name.

    Returns:
        The name in double quotes, inner double quotes doubled.
    """
    return '"' + name.replace('"', '""') + '"'


def opener(path: pathlib.Path) -> Callable[[], sqlite3.Connection]:
    """A factory of read-only connections to a SQLite file.

    Args:
        path: The database file.

    Returns:
        A callable that opens a new read-only connection on each call.
    """
    address = f"file:{path.resolve()}?mode=ro"

    def connect() -> sqlite3.Connection:
        return sqlite3.connect(address, uri=True)

    return connect


class Store(abc.ABC):
    """Named tables, read whole.

    Attributes:
        path: Where the tables live.
    """

    def __init__(self, path: str | pathlib.Path) -> None:
        """Opens a store.

        Args:
            path: The database file or the folder.

        Raises:
            ConfigError: If nothing exists at `path`.
        """
        self.path = pathlib.Path(path)
        if not self.path.exists():
            raise errors.ConfigError(f"no table store at {self.path}")

    @abc.abstractmethod
    def names(self) -> tuple[str, ...]:
        """Lists the tables.

        Returns:
            The table names, sorted.
        """

    @abc.abstractmethod
    def load(self, name: str) -> pandas.DataFrame:
        """Reads a table known to exist.

        Args:
            name: One of `names()`.

        Returns:
            The rows in stored order.
        """

    def read(self, name: str) -> pandas.DataFrame:
        """Reads one table whole.

        Args:
            name: The table.

        Returns:
            The rows in stored order, indexed from zero.

        Raises:
            ConfigError: If the store holds no such table.
        """
        if name not in self.names():
            raise errors.ConfigError(
                f"{self.path} holds no table {name!r}; it holds "
                f"{list(self.names())}"
            )
        return self.load(name).reset_index(drop=True)

    def factory(
        self, folder: str | pathlib.Path
    ) -> Callable[[], sqlite3.Connection]:
        """Read-only SQLite connections over every table.

        This default writes the tables into `folder/mirror.sqlite`, replacing
        an earlier mirror, and checks each table's row count there.

        Args:
            folder: Where the mirror is written.

        Returns:
            A callable that opens a new read-only connection on each call.

        Raises:
            DataError: If a mirrored table does not hold the rows read.
        """
        target = pathlib.Path(folder) / MIRROR
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        with sqlite3.connect(target) as connection:
            for name in self.names():
                frame = self.read(name)
                frame.to_sql(name, connection, index=False)
                query = f"SELECT COUNT(*) FROM {quote(name)}"  # noqa: S608 -- quoted name of a table this store lists # nosec B608
                stored = connection.execute(query).fetchone()[0]
                if stored != len(frame):
                    raise errors.DataError(
                        f"the mirror of {name!r} holds {stored} rows, "
                        f"not the {len(frame)} read"
                    )
        connection.close()
        return opener(target)


class Database(Store):
    """A SQLite database file.

    A table is read by one statement, which SQLite answers from one
    consistent snapshot, in rowid order (key order for a table without
    rowids).
    """

    def names(self) -> tuple[str, ...]:
        """Lists the tables.

        Returns:
            The table names, sorted, without SQLite's own tables.
        """
        query = (
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        connection = opener(self.path)()
        try:
            return tuple(row[0] for row in connection.execute(query))
        finally:
            connection.close()

    def load(self, name: str) -> pandas.DataFrame:
        """Reads a table known to exist.

        Args:
            name: One of `names()`.

        Returns:
            The rows in rowid order.
        """
        connection = opener(self.path)()
        try:
            flags = connection.execute(
                "SELECT sql FROM sqlite_master WHERE name = ?", (name,)
            ).fetchone()[0]
            plain = "WITHOUT ROWID" not in str(flags).upper()
            order = " ORDER BY rowid" if plain else ""
            query = f"SELECT * FROM {quote(name)}{order}"  # noqa: S608 -- quoted name of a table this store lists # nosec B608
            return pandas.read_sql_query(query, connection)
        finally:
            connection.close()

    def factory(
        self, folder: str | pathlib.Path
    ) -> Callable[[], sqlite3.Connection]:
        """Read-only connections to the database itself.

        Args:
            folder: Unused; a database needs no mirror.

        Returns:
            A callable that opens a new read-only connection on each call.
        """
        del folder
        return opener(self.path)


class Folder(Store):
    """A folder of table files of one format.

    Attributes:
        suffixes: File suffixes of the format, lower case.
    """

    suffixes: ClassVar[tuple[str, ...]] = ()

    def __init__(self, path: str | pathlib.Path) -> None:
        """Opens a folder.

        Args:
            path: The folder.

        Raises:
            ConfigError: If `path` is not a folder.
        """
        super().__init__(path)
        if not self.path.is_dir():
            raise errors.ConfigError(f"{self.path} is not a folder")

    def stem(self, path: pathlib.Path) -> str | None:
        """The table name a file holds, if it has this format.

        Args:
            path: A file name.

        Returns:
            The name without the suffix, or None for another format.
        """
        lowered = path.name.lower()
        for suffix in self.suffixes:
            if lowered.endswith(suffix):
                return path.name[: -len(suffix)]
        return None

    def title(self, entry: pathlib.Path) -> str | None:
        """The table an entry of the folder holds.

        Args:
            entry: A file or directory directly in the folder.

        Returns:
            The table's name: a file of the format less its suffix, or a
            directory holding a part file of the format, less the suffix
            its name may carry (`orders.parquet/part-0.parquet` holds
            `orders`); None for anything else.
        """
        if entry.is_file():
            return self.stem(entry)
        if not entry.is_dir() or not self.pieces(entry):
            return None
        return self.stem(entry) or entry.name

    def pieces(self, directory: pathlib.Path) -> list[pathlib.Path]:
        """The part files of a directory.

        Args:
            directory: A directory of the folder.

        Returns:
            Its files of the format, in name order.
        """
        return [
            entry
            for entry in sorted(directory.iterdir())
            if entry.is_file() and self.stem(entry) is not None
        ]

    def parts(self, name: str) -> list[pathlib.Path]:
        """The files that hold a table.

        Args:
            name: The table.

        Returns:
            The single file, or the part files of its directory in name
            order; empty when the folder does not hold the table.

        Raises:
            DataError: If more than one entry holds the table (a file and
                a directory, or `orders/` beside `orders.parquet/`).
        """
        holders = [
            entry
            for entry in sorted(self.path.iterdir())
            if self.title(entry) == name
        ]
        if len(holders) > 1:
            raise errors.DataError(
                f"table {name!r} is held {len(holders)} times in "
                f"{self.path}: {[h.name for h in holders]}"
            )
        if not holders:
            return []
        if holders[0].is_file():
            return holders
        return self.pieces(holders[0])

    def names(self) -> tuple[str, ...]:
        """Lists the tables.

        Returns:
            The names of the format's files and of the directories holding
            at least one of them, sorted.
        """
        found = {self.title(entry) for entry in self.path.iterdir()}
        return tuple(sorted(n for n in found if n is not None))

    def load(self, name: str) -> pandas.DataFrame:
        """Reads a table known to exist.

        Args:
            name: One of `names()`.

        Returns:
            The rows of its file, or of its part files stacked in name
            order.

        Raises:
            DataError: If the part files do not share their columns.
        """
        frames = [tabular.read_table(path) for path in self.parts(name)]
        columns = list(frames[0].columns)
        if any(list(frame.columns) != columns for frame in frames):
            raise errors.DataError(
                f"the parts of table {name!r} do not share their columns"
            )
        return pandas.concat(frames, ignore_index=True)


class Parquet(Folder):
    """A folder of Parquet tables."""

    suffixes = tabular.Parquet.suffixes


class Csv(Folder):
    """A folder of CSV tables (a CSV set)."""

    suffixes = tabular.Csv.suffixes


KINDS: dict[str, type[Store]] = {
    "sqlite": Database,
    "parquet": Parquet,
    "csv": Csv,
}


def locate(path: str | pathlib.Path, kind: str | None = None) -> Store:
    """Opens the store at `path`, of a stated or recognised kind.

    Without `kind`, a file that starts with the SQLite header is a database,
    and a folder is a Parquet folder or a CSV set by the tables it holds.

    Args:
        path: The database file or the folder.
        kind: One of `KINDS`, or None to recognise it.

    Returns:
        The store.

    Raises:
        ConfigError: If `kind` is unknown, nothing is at `path`, or the kind
            cannot be recognised (a file that is not SQLite, a folder with
            tables of both or neither format).
    """
    if kind is not None:
        if kind not in KINDS:
            raise errors.ConfigError(
                f"unknown store kind {kind!r}; use one of {sorted(KINDS)}"
            )
        return KINDS[kind](path)
    where = pathlib.Path(path)
    if where.is_file():
        with where.open("rb") as handle:
            if handle.read(len(HEADER)) == HEADER:
                return Database(where)
        raise errors.ConfigError(f"{where} is not a SQLite database")
    if not where.is_dir():
        raise errors.ConfigError(f"no table store at {where}")
    found = [store for store in (Parquet(where), Csv(where)) if store.names()]
    if len(found) != 1:
        raise errors.ConfigError(
            f"{where} holds {'both' if found else 'neither'} Parquet "
            "and CSV tables; state the store kind"
        )
    return found[0]
