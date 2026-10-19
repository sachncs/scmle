"""Dataset ingestion and manifest construction.

Readers are polymorphic: `DatasetLoader` dispatches on file suffix to a
`TableReader` subclass, so supporting a new format means adding one class.
"""

import abc
import hashlib
import pathlib
from typing import ClassVar

import numpy as np
import pandas as pd

from scml.contracts import data as data_contracts
from scml.contracts import errors


class TableReader(abc.ABC):
    """Reads one tabular file format into a DataFrame.

    Attributes:
        suffixes: Lower-case file suffixes the reader handles.
    """

    suffixes: ClassVar[tuple[str, ...]]

    @abc.abstractmethod
    def load(self, path: pathlib.Path) -> pd.DataFrame:
        """Reads `path`.

        Args:
            path: Existing file with a supported suffix.

        Returns:
            The parsed table.
        """


class CsvReader(TableReader):
    """Comma-separated values, including compressed CSV files."""

    suffixes = (".csv", ".csv.gz", ".csv.bz2", ".csv.xz", ".csv.zip")

    def load(self, path: pathlib.Path) -> pd.DataFrame:
        """Reads a CSV file."""
        return pd.read_csv(path)


class TsvReader(TableReader):
    """Tab-separated values."""

    suffixes = (".tsv", ".tab", ".tsv.gz", ".tab.gz")

    def load(self, path: pathlib.Path) -> pd.DataFrame:
        """Reads a tab-separated table."""
        return pd.read_csv(path, sep="\t")


class JsonReader(TableReader):
    """JSON records or a JSON array of records."""

    suffixes = (".json",)

    def load(self, path: pathlib.Path) -> pd.DataFrame:
        """Reads JSON records into a table."""
        return pd.read_json(path)


class JsonLinesReader(TableReader):
    """Newline-delimited JSON records."""

    suffixes = (".jsonl", ".ndjson", ".jsonl.gz", ".ndjson.gz")

    def load(self, path: pathlib.Path) -> pd.DataFrame:
        """Reads one JSON record per line."""
        return pd.read_json(path, lines=True)


class ExcelReader(TableReader):
    """Excel workbooks, using the first sheet by default."""

    suffixes = (".xlsx", ".xlsm")

    def load(self, path: pathlib.Path) -> pd.DataFrame:
        """Reads the first worksheet from an Excel workbook.

        Raises:
            DataError: If the optional spreadsheet dependency is absent.
        """
        try:
            return pd.read_excel(path)
        except ImportError as error:
            raise errors.DataError(
                "Excel input needs the optional 'spreadsheets' dependency; "
                "install super-charge-ml[spreadsheets]"
            ) from error


class ParquetReader(TableReader):
    """Apache Parquet."""

    suffixes = (".parquet", ".pq")

    def load(self, path: pathlib.Path) -> pd.DataFrame:
        """Reads a Parquet file."""
        return pd.read_parquet(path)


class FeatherReader(TableReader):
    """Arrow IPC / Feather files."""

    suffixes = (".feather", ".arrow")

    def load(self, path: pathlib.Path) -> pd.DataFrame:
        """Reads a Feather file."""
        return pd.read_feather(path)


class DatasetLoader:
    """Dispatches file reads to the matching `TableReader`."""

    def __init__(self, readers: tuple[TableReader, ...] | None = None) -> None:
        """Initializes the loader.

        Args:
            readers: Readers to dispatch to; defaults to common open formats.
        """
        self.readers = readers or (
            CsvReader(),
            TsvReader(),
            JsonReader(),
            JsonLinesReader(),
            ExcelReader(),
            ParquetReader(),
            FeatherReader(),
        )

    def supported_suffixes(self) -> list[str]:
        """Returns every suffix some reader handles, sorted."""
        return sorted(s for reader in self.readers for s in reader.suffixes)

    def reader_for(self, path: pathlib.Path) -> TableReader:
        """Selects the reader for `path`.

        Args:
            path: File whose suffix decides the reader.

        Returns:
            The matching reader.

        Raises:
            DataError: If no reader supports the suffix.
        """
        name = path.name.lower()
        for reader in self.readers:
            if any(name.endswith(suffix) for suffix in reader.suffixes):
                return reader
        raise errors.DataError(
            f"unsupported file type '{path.name}'; "
            f"use one of {self.supported_suffixes()}"
        )

    def load(self, uri: str | pathlib.Path) -> pd.DataFrame:
        """Reads and sanity-checks a dataset.

        Args:
            uri: Path to a supported file.

        Returns:
            A non-empty DataFrame with unique column names.

        Raises:
            DataError: If the file is missing, unreadable, empty or has
                duplicate column names.
        """
        path = pathlib.Path(uri)
        if not path.exists():
            raise errors.DataError(f"dataset not found: {path}")
        reader = self.reader_for(path)
        try:
            frame = reader.load(path)
        except Exception as error:  # Parser errors vary by engine.
            raise errors.DataError(
                f"failed to read {path.name}: {error}"
            ) from error
        if frame.empty:
            raise errors.DataError(f"{path.name} contains no rows")
        duplicated = frame.columns[frame.columns.duplicated()]
        if len(duplicated):
            names = sorted(set(duplicated.astype(str)))
            raise errors.DataError(f"duplicate column names: {names}")
        return frame


def read_table(uri: str | pathlib.Path) -> pd.DataFrame:
    """Reads a supported file dataset with the default loader."""
    return DatasetLoader().load(uri)


def schema_hash(frame: pd.DataFrame) -> str:
    """Returns a short hash of column names and dtypes."""
    pairs = zip(frame.columns, frame.dtypes.astype(str), strict=True)
    payload = "|".join(f"{name}:{dtype}" for name, dtype in pairs)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def data_hash(frame: pd.DataFrame) -> str:
    """Returns a short hash of the row contents (index ignored)."""
    rows = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    return hashlib.sha256(rows.tobytes()).hexdigest()[:16]


def build_manifest(
    frame: pd.DataFrame, target: str, uri: str = ""
) -> data_contracts.DatasetManifest:
    """Builds the immutable identity of a dataset.

    Args:
        frame: The ingested data (not modified).
        target: Name of the target column.
        uri: Where the data came from.

    Returns:
        The dataset manifest.

    Raises:
        DataError: If `target` is not a column.
    """
    if target not in frame.columns:
        raise errors.DataError(
            f"target column '{target}' not found; "
            f"columns: {list(frame.columns)[:20]}"
        )
    content_hash = data_hash(frame)
    return data_contracts.DatasetManifest(
        dataset_id=f"ds_{content_hash[:8]}",
        schema_hash=schema_hash(frame),
        data_hash=content_hash,
        row_count=len(frame),
        column_count=frame.shape[1],
        target=target,
        uri=str(uri),
    )


def replace_infinities(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Maps +/-inf in numeric columns to NaN (they carry no usable value).

    Args:
        frame: The ingested data (not modified).

    Returns:
        A cleaned copy (or the same frame when clean) and the number of
        replaced values per affected column.
    """
    numeric = frame.select_dtypes("number")
    infinite = np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan))
    counts = {
        str(column): int(count)
        for column, count in zip(
            numeric.columns, infinite.sum(axis=0), strict=True
        )
        if count
    }
    if not counts:
        return frame, {}
    return frame.replace([np.inf, -np.inf], np.nan), counts
