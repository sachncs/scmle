"""Dataset ingestion and manifest construction.

Readers are polymorphic: `Loader` dispatches on file suffix to a
`Reader` subclass, so supporting a new format means adding one class.
`Loader.stream` reads a file a chunk at a time: delimited text, JSON lines
and Parquet are streamed with bounded memory; the other formats are read
whole and then split.
"""

import abc
from collections.abc import Iterator
import hashlib
import pathlib
from typing import ClassVar
from urllib.parse import urlsplit
from urllib.parse import urlunsplit
from urllib.request import url2pathname

import numpy
import pandas

from scmle.contracts import data
from scmle.contracts import errors
from scmle.data.readers import delimited
from scmle.data.readers import stream


class Reader(abc.ABC):
    """Reads one tabular file format into a DataFrame.

    Attributes:
        suffixes: Lower-case file suffixes the reader handles.
    """

    suffixes: ClassVar[tuple[str, ...]]

    @abc.abstractmethod
    def load(self, path: pathlib.Path) -> pandas.DataFrame:
        """Reads `path`.

        Args:
            path: Existing file with a supported suffix.

        Returns:
            The parsed table.
        """

    def chunks(
        self, path: pathlib.Path, rows: int
    ) -> Iterator[pandas.DataFrame]:
        """Reads `path` at most `rows` rows at a time.

        This default reads the whole file and splits it; a format that can
        be read incrementally overrides it.

        Args:
            path: Existing file with a supported suffix.
            rows: Rows per chunk.

        Yields:
            The table's rows, in order, as consecutive chunks.
        """
        frame = self.load(path)
        for start in range(0, len(frame), rows):
            yield frame.iloc[start : start + rows].reset_index(drop=True)


class Csv(Reader):
    """Delimited text, including compressed files.

    The bytes are decoded by byte-order mark, as UTF-8 or as plausible
    cp1252, and the delimiter is found from the header unless stated. A
    file with no single right reading is refused (see `delimited`).

    Attributes:
        delimiter: Field separator, or None to detect it.
        encoding: Codec name, or None to detect it.
    """

    suffixes: ClassVar[tuple[str, ...]] = (
        ".csv",
        ".csv.gz",
        ".csv.bz2",
        ".csv.xz",
        ".csv.zip",
    )

    def __init__(
        self, delimiter: str | None = None, encoding: str | None = None
    ) -> None:
        """Initializes the reader.

        Args:
            delimiter: Field separator; detected from the header when None.
            encoding: Codec name; detected from the bytes when None.
        """
        self.delimiter = delimiter
        self.encoding = encoding

    def load(self, path: pathlib.Path) -> pandas.DataFrame:
        """Reads a delimited file.

        Args:
            path: Existing delimited file, optionally compressed.

        Returns:
            The parsed table.

        Raises:
            InputError: If the file cannot be decoded, its delimiter is
                ambiguous, it has duplicate headers or ragged records, or
                a column has no exact reading.
        """
        try:
            text = delimited.decode(delimited.inflate(path), self.encoding)
            delimiter = self.delimiter or delimited.sniff(text)
            delimited.check(text, delimiter)
            return delimited.parse(text, delimiter)
        except errors.InputError as error:
            raise errors.InputError(f"{path.name}: {error}") from error

    def chunks(
        self, path: pathlib.Path, rows: int
    ) -> Iterator[pandas.DataFrame]:
        """Streams a delimited file `rows` records at a time.

        Args:
            path: Existing delimited file, optionally compressed.
            rows: Records per chunk.

        Yields:
            Each chunk parsed as `load` parses a file (see `stream`).

        Raises:
            InputError: As `load`, naming the file.
        """
        try:
            yield from stream.frames(path, rows, self.delimiter, self.encoding)
        except errors.InputError as error:
            raise errors.InputError(f"{path.name}: {error}") from error


class Tsv(Csv):
    """Tab-separated values."""

    suffixes = (".tsv", ".tab", ".tsv.gz", ".tab.gz")

    def __init__(self, encoding: str | None = None) -> None:
        """Initializes the reader.

        Args:
            encoding: Codec name; detected from the bytes when None.
        """
        super().__init__("\t", encoding)


class Json(Reader):
    """JSON records or a JSON array of records."""

    suffixes = (".json",)

    def load(self, path: pathlib.Path) -> pandas.DataFrame:
        """Reads JSON records into a table.

        Args:
            path: Existing JSON file.

        Returns:
            The parsed table.
        """
        return pandas.read_json(path)


class Jsonl(Reader):
    """Newline-delimited JSON records."""

    suffixes = (".jsonl", ".ndjson", ".jsonl.gz", ".ndjson.gz")

    def load(self, path: pathlib.Path) -> pandas.DataFrame:
        """Reads one JSON record per line.

        Args:
            path: Existing newline-delimited JSON file.

        Returns:
            The parsed table.
        """
        return pandas.read_json(path, lines=True)

    def chunks(
        self, path: pathlib.Path, rows: int
    ) -> Iterator[pandas.DataFrame]:
        """Streams the records `rows` lines at a time.

        Args:
            path: Existing newline-delimited JSON file.
            rows: Lines per chunk.

        Yields:
            Each chunk of records as a table.
        """
        with pandas.read_json(path, lines=True, chunksize=rows) as reader:
            for chunk in reader:
                yield chunk.reset_index(drop=True)


class Excel(Reader):
    """Excel workbooks, using the first sheet by default."""

    suffixes = (".xlsx", ".xlsm")

    def load(self, path: pathlib.Path) -> pandas.DataFrame:
        """Reads the first worksheet from an Excel workbook.

        Args:
            path: Existing Excel workbook.

        Returns:
            The first worksheet as a table.

        Raises:
            DataError: If the optional spreadsheet dependency is absent.
        """
        try:
            return pandas.read_excel(path)
        except ImportError as error:
            raise errors.DataError(
                "Excel input needs the optional 'spreadsheets' dependency; "
                "install scmle[spreadsheets]"
            ) from error


class Parquet(Reader):
    """Apache Parquet."""

    suffixes = (".parquet", ".pq")

    def load(self, path: pathlib.Path) -> pandas.DataFrame:
        """Reads a Parquet file.

        Args:
            path: Existing Parquet file.

        Returns:
            The parsed table.
        """
        return pandas.read_parquet(path)

    def chunks(
        self, path: pathlib.Path, rows: int
    ) -> Iterator[pandas.DataFrame]:
        """Streams the file `rows` rows at a time, batch by batch.

        Args:
            path: Existing Parquet file.
            rows: Rows per chunk.

        Yields:
            Each record batch as a table.
        """
        import pyarrow.parquet  # noqa: PLC0415 -- only streamed reads need it

        for batch in pyarrow.parquet.ParquetFile(path).iter_batches(
            batch_size=rows
        ):
            yield batch.to_pandas()


class Feather(Reader):
    """Arrow IPC / Feather files."""

    suffixes = (".feather", ".arrow")

    def load(self, path: pathlib.Path) -> pandas.DataFrame:
        """Reads a Feather file.

        Args:
            path: Existing Feather or Arrow IPC file.

        Returns:
            The parsed table.
        """
        return pandas.read_feather(path)


class Loader:
    """Dispatches file reads to the matching `Reader`.

    Attributes:
        readers: The readers consulted, in order, when choosing by suffix.
    """

    def __init__(self, readers: tuple[Reader, ...] | None = None) -> None:
        """Initializes the loader.

        Args:
            readers: Readers to dispatch to; defaults to common open formats.
        """
        self.readers = readers or (
            Csv(),
            Tsv(),
            Json(),
            Jsonl(),
            Excel(),
            Parquet(),
            Feather(),
        )

    def extensions(self) -> list[str]:
        """Lists every suffix some reader handles.

        Returns:
            All reader suffixes, sorted alphabetically.
        """
        return sorted(s for reader in self.readers for s in reader.suffixes)

    def pick(self, path: pathlib.Path) -> Reader:
        """Selects the reader for `path`.

        Args:
            path: File whose suffix decides the reader.

        Returns:
            The first reader, in registration order, whose suffix the
            lower-cased file name ends with.

        Raises:
            DataError: If no reader supports the suffix.
        """
        name = path.name.lower()
        for reader in self.readers:
            if any(name.endswith(suffix) for suffix in reader.suffixes):
                return reader
        raise errors.DataError(
            f"unsupported file type '{path.name}'; "
            f"use one of {self.extensions()}"
        )

    def load(self, uri: str | pathlib.Path) -> pandas.DataFrame:
        """Reads and sanity-checks a dataset.

        Args:
            uri: Path to a supported file.

        Returns:
            A non-empty DataFrame with unique column names.

        Raises:
            DataError: If the file is missing, has an unsupported suffix,
                cannot be parsed, has no rows or has duplicate column names.
        """
        path = pathlib.Path(uri)
        if not path.exists():
            raise errors.DataError(f"dataset not found: {path}")
        reader = self.pick(path)
        try:
            frame = reader.load(path)
        except errors.DataError:
            raise
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

    def stream(
        self, uri: str | pathlib.Path, rows: int
    ) -> Iterator[pandas.DataFrame]:
        """Reads a dataset `rows` rows at a time, checked as `load` checks.

        Args:
            uri: Path to a supported file.
            rows: Rows per chunk.

        Yields:
            Non-empty tables of at most `rows` rows with the same unique
            column names, in file order.

        Raises:
            DataError: If `rows` is not positive, the file is missing, has
                an unsupported suffix, cannot be parsed, has no rows or has
                duplicate column names.
        """
        if rows < 1:
            raise errors.DataError("rows per chunk must be positive")
        path = pathlib.Path(uri)
        if not path.exists():
            raise errors.DataError(f"dataset not found: {path}")
        reader = self.pick(path)
        found = 0
        chunks = reader.chunks(path, rows)
        while True:
            try:
                frame = next(chunks, None)
            except errors.DataError:
                raise
            except Exception as error:  # Parser errors vary by engine.
                raise errors.DataError(
                    f"failed to read {path.name}: {error}"
                ) from error
            if frame is None:
                break
            if frame.empty:
                continue
            duplicated = frame.columns[frame.columns.duplicated()]
            if len(duplicated):
                names = sorted(set(duplicated.astype(str)))
                raise errors.DataError(f"duplicate column names: {names}")
            found += len(frame)
            yield frame
        if not found:
            raise errors.DataError(f"{path.name} contains no rows")


def read(uri: str | pathlib.Path) -> pandas.DataFrame:
    """Reads a supported file dataset with the default loader.

    Args:
        uri: Path to a supported file.

    Returns:
        The table, as validated by `Loader.load`.
    """
    return Loader().load(uri)


def schema_hash(frame: pandas.DataFrame) -> str:
    """Hashes the column names and dtypes of a frame.

    Args:
        frame: The table whose schema is hashed.

    Returns:
        The first 16 hexadecimal characters of a SHA-256 digest over the
        ordered ``name:dtype`` pairs.
    """
    pairs = zip(frame.columns, frame.dtypes.astype(str), strict=True)
    payload = "|".join(f"{name}:{dtype}" for name, dtype in pairs)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def data_hash(frame: pandas.DataFrame) -> str:
    """Hashes the row contents of a frame, ignoring the index.

    Args:
        frame: The table whose rows are hashed.

    Returns:
        The first 16 hexadecimal characters of a SHA-256 digest over the
        per-row pandas hashes, in row order.
    """
    rows = pandas.util.hash_pandas_object(frame, index=False).to_numpy()
    return hashlib.sha256(rows.tobytes()).hexdigest()[:16]


def canonicalize(uri: str | pathlib.Path) -> str:
    """Returns the canonical URI used for snapshot identity and lineage.

    Empty strings and the two supported in-memory markers are represented as
    ``memory://``. Local paths become absolute ``file://`` URIs. Other URI
    schemes retain their resource semantics while normalizing the scheme
    name. The caller-provided URI remains available on ``DatasetManifest``.

    Args:
        uri: A local path, remote URI, or in-memory marker.

    Returns:
        A stable URI representation for identity and lineage records.
    """
    value = str(uri).strip()
    if not value or value in {"<memory>", "<in-memory>"}:
        return "memory://"
    parsed = urlsplit(value)
    if parsed.scheme and not (len(parsed.scheme) == 1 and value[1:2] == ":"):
        return urlunsplit(
            (
                parsed.scheme.lower(),
                parsed.netloc,
                parsed.path,
                parsed.query,
                parsed.fragment,
            )
        )
    return pathlib.Path(value).expanduser().resolve(strict=False).as_uri()


def semantic(uri: str | pathlib.Path) -> str:
    """Returns the acquisition semantics directly implied by ``uri``.

    Args:
        uri: A local path, remote URI, or in-memory marker.

    Returns:
        ``memory``, ``local-file``, or ``<scheme>-uri``.
    """
    scheme = urlsplit(canonicalize(uri)).scheme
    if scheme == "memory":
        return "memory"
    if scheme == "file":
        return "local-file"
    return f"{scheme}-uri"


def raw_source_hash(uri: str | pathlib.Path) -> str:
    """Hashes the raw bytes of an available local source file.

    Remote and in-memory sources intentionally return an empty value because
    this reader has no transport or immutable-byte snapshot authority for
    them. The parsed-frame hash remains available in ``data_hash``.

    Args:
        uri: A local path, remote URI, or in-memory marker.

    Returns:
        The SHA-256 hex digest of the file's bytes, or an empty string if
        the source is not a local file or not an existing regular file.

    Raises:
        DataError: If the file exists but cannot be read; an empty digest
            would be indistinguishable from "not a file" (ERR-14).
    """
    parsed = urlsplit(canonicalize(uri))
    if parsed.scheme != "file":
        return ""
    path = pathlib.Path(url2pathname(parsed.path))
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as error:
        raise errors.DataError(
            f"cannot read {path} to hash it: {error}"
        ) from error
    return digest.hexdigest()


def identify(frame: pandas.DataFrame, uri: str | pathlib.Path) -> str:
    """Returns a full deterministic identity for a tabular snapshot.

    The identity includes the ordered schema, ordered row content, shape,
    canonical URI semantics, and raw source bytes when an available local
    file is the source. The DataFrame index is intentionally excluded.

    Args:
        frame: The ingested table.
        uri: The source path, URI, or in-memory marker.

    Returns:
        A ``snap_``-prefixed SHA-256 identity.
    """
    schema = "|".join(
        f"{name}:{dtype}"
        for name, dtype in zip(
            frame.columns, frame.dtypes.astype(str), strict=True
        )
    )
    schema_digest = hashlib.sha256(schema.encode("utf-8")).hexdigest()
    rows = pandas.util.hash_pandas_object(frame, index=False).to_numpy(
        dtype="uint64"
    )
    content_digest = hashlib.sha256(
        rows.astype("<u8", copy=False).tobytes()
    ).hexdigest()
    payload = "|".join(
        (
            schema_digest,
            content_digest,
            str(frame.shape[0]),
            str(frame.shape[1]),
            canonicalize(uri),
            raw_source_hash(uri),
        )
    )
    return f"snap_{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def manifested(
    frame: pandas.DataFrame,
    target: str,
    uri: str | pathlib.Path = "",
    modality: str = "tabular",
) -> data.Manifest:
    """Builds the immutable identity of a dataset.

    Args:
        frame: The ingested data (not modified).
        target: Name of the target column.
        uri: Where the data came from.
        modality: Dataset family declared by the caller.

    Returns:
        The manifest holding the content and schema hashes, row and column
        counts, snapshot identity and lineage of `frame`.

    Raises:
        DataError: If `target` is not a column.
    """
    if target not in frame.columns:
        raise errors.DataError(
            f"target column '{target}' not found; "
            f"columns: {list(frame.columns)[:20]}"
        )
    content_hash = data_hash(frame)
    identity = identify(frame, uri)
    raw_hash = raw_source_hash(uri)
    object_hashes = {"tabular-frame": content_hash}
    if raw_hash:
        object_hashes["raw-source"] = raw_hash
    return data.Manifest(
        identifier=f"ds_{content_hash[:8]}",
        digests={"schema": schema_hash(frame), "data": content_hash},
        rows=len(frame),
        columns=frame.shape[1],
        target=target,
        uri=str(uri),
        snapshot=identity,
        modality=modality,
        source=data.Origin(
            acquisition=semantic(uri),
        ),
        lineage=data.Ancestry(
            snapshot=identity,
            digests=object_hashes,
        ),
    )


def sanitize(
    frame: pandas.DataFrame,
) -> tuple[pandas.DataFrame, dict[str, int]]:
    """Maps +/-inf in numeric columns to NaN (they carry no usable value).

    Args:
        frame: The ingested data (not modified).

    Returns:
        A cleaned copy (or the same frame when clean) and a mapping from
        column name to the number of replaced values, containing only the
        affected columns (empty when the frame is clean).
    """
    numeric = frame.select_dtypes("number")
    infinite = numpy.isinf(numeric.to_numpy(dtype=float, na_value=numpy.nan))
    counts = {
        str(column): int(count)
        for column, count in zip(
            numeric.columns, infinite.sum(axis=0), strict=True
        )
        if count
    }
    if not counts:
        return frame, {}
    return frame.replace([numpy.inf, -numpy.inf], numpy.nan), counts
