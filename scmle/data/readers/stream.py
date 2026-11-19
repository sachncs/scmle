"""Delimited text read a bounded number of records at a time.

`Csv.load` decodes and parses a whole file; `frames` reads the same file
record by record, so memory holds one chunk however large the file is. The
encoding and the delimiter come from the first `HEAD` bytes by the rules
`delimited` applies to a whole file, and each chunk is parsed by
`delimited.parse`, so a chunk reads as the same rows would as a file of
their own: a column's type is chosen per chunk.
"""

import bz2
import codecs
from collections.abc import Callable
from collections.abc import Iterator
import contextlib
import csv
import gzip
import io
import lzma
import pathlib
from typing import Any
from typing import IO
import zipfile

import pandas

from scmle.contracts import errors
from scmle.data.readers import delimited

# Bytes read to detect the encoding and the delimiter: enough for the header
# and a few thousand characters of records, the same evidence a person
# opening the file sees, and small beside any chunk.
HEAD = 64 * 1024
# Decompressing readers by suffix; each returns a binary file object.
OPENERS: dict[str, Callable[[pathlib.Path], Any]] = {
    ".gz": gzip.GzipFile,
    ".bz2": bz2.BZ2File,
    ".xz": lzma.LZMAFile,
}


@contextlib.contextmanager
def opened(path: pathlib.Path) -> Iterator[IO[bytes]]:
    """Opens a file's bytes, decompressing them by its suffix as they are read.

    Args:
        path: Plain, gzip, bzip2, xz or single-member zip file.

    Yields:
        A binary stream of the uncompressed bytes.

    Raises:
        InputError: If a zip archive does not hold exactly one file.
    """
    suffix = path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if not n.endswith("/")]
            if len(names) != 1:
                raise errors.InputError(
                    f"a zip archive must hold one file, found {len(names)}"
                )
            with archive.open(names[0]) as handle:
                yield handle
        return
    opener = OPENERS.get(suffix)
    if opener is None:
        with path.open("rb") as handle:
            yield handle
        return
    with opener(path) as handle:
        yield handle


def detect(head: bytes, encoding: str | None = None) -> str:
    """Chooses the codec of a file from its first bytes.

    Args:
        head: The first bytes of the file, possibly cut inside a character.
        encoding: Encoding to use instead of detecting one.

    Returns:
        The codec name: the stated one, the byte-order mark's, UTF-8, or
        cp1252 when the bytes are not UTF-8 and read as plausible text.

    Raises:
        InputError: If no detected encoding reads the bytes plausibly.
    """
    if encoding is not None:
        return encoding
    for mark, name in delimited.MARKS:
        if head.startswith(mark):
            return name
    try:
        codecs.getincrementaldecoder("utf-8")().decode(head, final=False)
    except UnicodeDecodeError:
        # handled: not UTF-8; `delimited.decode` decides cp1252 or refuses
        pass
    else:
        if b"\x00" not in head:
            return "utf-8"
    delimited.decode(head)
    return delimited.LEGACY


def frames(
    path: pathlib.Path,
    rows: int,
    delimiter: str | None = None,
    encoding: str | None = None,
) -> Iterator[pandas.DataFrame]:
    """Reads a delimited file `rows` records at a time.

    Args:
        path: Existing delimited file, optionally compressed.
        rows: Records per chunk.
        delimiter: Field separator; detected from the header when None.
        encoding: Codec name; detected from the first bytes when None.

    Yields:
        One table per chunk of at most `rows` records, blank lines skipped.

    Raises:
        InputError: If `rows` is not positive, the file cannot be decoded,
            its delimiter is ambiguous, it has duplicate headers or ragged
            records, or a column of a chunk has no exact reading.
    """
    if rows < 1:
        raise errors.InputError("rows per chunk must be positive")
    with opened(path) as handle:
        head = handle.read(HEAD)
    codec = detect(head, encoding)
    sample = codecs.getincrementaldecoder(codec)(errors="replace").decode(head)
    separator = delimiter or delimited.sniff(sample)
    try:
        with opened(path) as handle:
            text = io.TextIOWrapper(handle, encoding=codec, newline="")
            yield from split(csv.reader(text, delimiter=separator), rows)
    except UnicodeDecodeError as error:
        raise errors.InputError(
            f"cannot decode as {codec}: {error}; state the encoding"
        ) from error
    except csv.Error as error:
        raise errors.InputError(f"malformed delimited text: {error}") from error


def split(reader: Iterator[list[str]], rows: int) -> Iterator[pandas.DataFrame]:
    """Groups the records of a csv reader into parsed chunks.

    Args:
        reader: A csv reader positioned before the header.
        rows: Records per chunk.

    Yields:
        One parsed table per chunk.

    Raises:
        InputError: If a header name repeats or a record's width differs
            from the header's.
    """
    header = next(reader, None)
    if header is None:
        return
    separator = getattr(getattr(reader, "dialect", None), "delimiter", ",")
    delimited.check(render(header, [], separator), separator)
    batch: list[list[str]] = []
    for record in reader:
        if not record:
            continue
        if len(record) != len(header):
            line = getattr(reader, "line_num", 0)
            raise errors.InputError(
                f"line {line} has {len(record)} fields, the header "
                f"{len(header)}"
            )
        batch.append(record)
        if len(batch) == rows:
            yield delimited.parse(render(header, batch, separator), separator)
            batch = []
    if batch:
        yield delimited.parse(render(header, batch, separator), separator)


def render(header: list[str], batch: list[list[str]], separator: str) -> str:
    """Writes a header and records back as delimited text.

    Args:
        header: The header fields.
        batch: The records.
        separator: Field separator.

    Returns:
        The text, quoted where a field needs it.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=separator, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(batch)
    return buffer.getvalue()
