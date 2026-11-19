"""Find input rows, undeclared categories and secrets in what a run wrote.

C9 asks that no raw row, raw category string or secret reach an artifact by
default. `probes` turns an input table into the values that must not
appear (`Probe`), and `scan` looks for each of them in every file under a
folder: a run directory, its store, a bundle, logs.

What must not appear:

* `row`: every string cell of a column the bundle schema does not declare
  categorical (names, identifiers, free text), and every string cell of a
  declared column that is not a declared level;
* `category`: every level of a declared column seen in fewer than `FLOOR`
  rows, declared or not. A level shared by at least `FLOOR` rows says no
  more about any one person than a k-anonymous release with k = `FLOOR`;
  one seen fewer times singles people out, so it may not be stored;
* `secret`: the values given, and the value of every environment variable
  whose name says it holds a key, token, secret or password;
* `cell`: every numeric cell with at least `PRECISION` significant digits,
  float or integer (account numbers, phone numbers, long identifiers).
  One such cell found alone is an aggregate that equals a value (a minimum,
  a quantile) and is reported, not counted; `FRAGMENT` cells of the same row
  in the same file are a raw row fragment and count as a `row` leak.

Forms searched. Each value is looked for as UTF-8, UTF-16 and UTF-32 (NumPy
string arrays), JSON- and HTML-escaped, and as its unsalted SHA-256, SHA-1
and MD5 hex digest; a float as its `repr` and its float64 bytes; an
integer as its decimal text and its int64 and float64 bytes.
Containers are opened, never executed: zlib, gzip, bz2 and xz streams are
decompressed, scmle envelopes unwrapped, zip members read, Parquet and SQLite
tables read as text. Pickles are searched as bytes (their strings are stored
as UTF-8) and never unpickled. The report names values by a keyed digest
prefix under a random salt drawn for each scan (`Probe.label`), never by
the value and never by an unsalted digest a dictionary could reverse.

Limits. A value shorter than `SHORTEST` characters cannot be told from
chance and is counted as `unsearchable`, never dropped silently. A value
encrypted, salted with an unknown salt or split across files is not found.

    python -m scmle.security.leaks RUN_FOLDER --data train.csv \
        --schema RUN_FOLDER/model/schema.json --secret-env SCMLE_SIGNING_KEY
"""

import abc
import argparse
import bz2
import collections
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
import dataclasses
import gzip
import hashlib
import hmac
import html
import io
import json
import lzma
import math
import os
import pathlib
import re
import sqlite3
import struct
import sys
import tempfile
from typing import Any
import zipfile
import zlib

import pandas
import pyarrow
import pyarrow.parquet

from scmle.contracts import errors

ROW = "row"
CATEGORY = "category"
SECRET = "secret"  # noqa: S105  # nosec B105 -- a kind name, not a password
CELL = "cell"
KINDS = (ROW, CATEGORY, SECRET, CELL)
LEAKS = (ROW, CATEGORY, SECRET)
# k-anonymity: a level shared by at least k rows identifies any one of them
# with probability at most 1/k; k = 5 (1/k = 20%) is the usual floor for
# released microdata (El Emam and Dankar 2008).
FLOOR = 5
# A random string of n characters over 36 symbols matches by chance in an
# M-byte file with probability about M / 36^n; n = 6 keeps that under 1% for
# a 16 MiB artifact (16 * 2^20 / 36^6 = 0.0076).
SHORTEST = 6
# Two independent random doubles agree to 12 significant digits with
# probability about 1e-12, so a match is the value, not chance; a value with
# fewer digits (an age, a count) matches by chance and is not searched.
PRECISION = 12
# Two cells of one row in one file are a row fragment.
FRAGMENT = 2
# Opened containers nest at most this deep, and decompress to at most this
# many bytes each, so a decompression bomb cannot exhaust the scan.
DEPTH = 4
LIMIT = 1 << 30
# Bytes of the per-scan label salt: 128 bits, so a label cannot be matched
# against a precomputed dictionary of candidate values.
SALT = 16
# A label keeps 12 hex characters (48 bits): two probes of one scan share a
# label with probability about n^2 / 2^49, under 1e-6 for n = 10^4 probes.
LABEL = 12
# The range of an int64, the widest integer form searched.
INT64 = (-(1 << 63), (1 << 63) - 1)
SECRET_NAMES = re.compile(
    r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL", re.IGNORECASE
)
ENVELOPE = b"SCMLE-ENVELOPE 1\n"


@dataclasses.dataclass(frozen=True)
class Probe:
    """A value that must not appear in any artifact.

    Attributes:
        text: The value as text.
        kind: One of `KINDS`.
        column: The input column it came from; "" for a secret.
        row: The input row of a `cell`; -1 otherwise.
        number: The value of a `cell`, an `int` for an integer column;
            None otherwise.
    """

    text: str
    kind: str
    column: str = ""
    row: int = -1
    number: int | float | None = None

    def label(self, salt: bytes) -> str:
        """A name for the value that does not reveal it.

        Args:
            salt: The scan's random salt.

        Returns:
            The first `LABEL` hex characters of the value's HMAC-SHA256
            under `salt`: stable within one scan, unrelated across scans.
        """
        keyed = hmac.new(salt, self.text.encode("utf-8"), hashlib.sha256)
        return keyed.hexdigest()[:LABEL]

    def forms(self) -> dict[str, bytes]:
        """Every byte form the value is searched in.

        Returns:
            Form name to bytes; a form equal to an earlier one is left out.
        """
        if isinstance(self.number, int):
            found = {"decimal": str(self.number).encode()}
            if INT64[0] <= self.number <= INT64[1]:
                found["int64"] = struct.pack("<q", self.number)
            found["float64"] = struct.pack("<d", float(self.number))
        elif self.number is not None:
            found = {
                "repr": repr(self.number).encode(),
                "float64": struct.pack("<d", self.number),
            }
        else:
            raw = self.text.encode("utf-8")
            found = {
                "utf-8": raw,
                "json": json.dumps(self.text)[1:-1].encode("utf-8"),
                "html": html.escape(self.text).encode("utf-8"),
                "utf-16": self.text.encode("utf-16-le"),
                "utf-32": self.text.encode("utf-32-le"),
                "sha256": hashlib.sha256(raw).hexdigest().encode(),
                "sha1": hashlib.sha1(raw, usedforsecurity=False)
                .hexdigest()
                .encode(),
                "md5": hashlib.md5(raw, usedforsecurity=False)
                .hexdigest()
                .encode(),
            }
        unique: dict[str, bytes] = {}
        for name, form in found.items():
            if form not in unique.values():
                unique[name] = form
        return unique


@dataclasses.dataclass(frozen=True)
class Finding:
    """One value found in one file.

    Attributes:
        path: The file, relative to the folder scanned.
        kind: The probe's kind.
        column: The probe's column.
        form: The form it was found in.
        label: The probe's label, never the value itself.
        row: The probe's row, for a `cell` or a row fragment.
    """

    path: str
    kind: str
    column: str
    form: str
    label: str
    row: int = -1


@dataclasses.dataclass(frozen=True)
class Report:
    """What one scan found.

    Attributes:
        files: Files scanned.
        size: Bytes read, before decompression.
        probes: Probes searched, by kind.
        unsearchable: Probes too short to search, by kind.
        findings: Every value found, lone cells included.
    """

    files: int
    size: int
    probes: dict[str, int]
    unsearchable: dict[str, int]
    findings: list[Finding]

    def leaks(self) -> list[Finding]:
        """The findings that are leaks: all but lone numeric cells.

        Returns:
            Row, category and secret findings, and one `row` finding per
            row with at least `FRAGMENT` cells found in the same file.
        """
        found = [f for f in self.findings if f.kind != CELL]
        cells: dict[tuple[str, int], set[str]] = collections.defaultdict(set)
        for each in self.findings:
            if each.kind == CELL:
                cells[each.path, each.row].add(each.column)
        found.extend(
            Finding(path, ROW, ",".join(sorted(columns)), "cells", "", row)
            for (path, row), columns in sorted(cells.items())
            if len(columns) >= FRAGMENT
        )
        return found

    def counts(self) -> dict[str, int]:
        """Leaks by kind, every kind listed.

        Returns:
            Kind to the number of leaked findings.
        """
        counted = collections.Counter(f.kind for f in self.leaks())
        return {kind: counted[kind] for kind in LEAKS}

    def to_dict(self) -> dict[str, Any]:
        """The report as JSON-ready data; no value is included.

        Returns:
            Counts, and every leak by file, kind, column, form and label.
        """
        return {
            "files": self.files,
            "size": self.size,
            "probes": self.probes,
            "unsearchable": self.unsearchable,
            "leaks": self.counts(),
            "cells": sum(f.kind == CELL for f in self.findings),
            "findings": [dataclasses.asdict(f) for f in self.leaks()],
        }


def precise(number: float) -> bool:
    """Whether a number has `PRECISION` significant digits.

    Args:
        number: The value.

    Returns:
        True for a finite value whose shortest repr has enough digits.
    """
    if not math.isfinite(number):
        return False
    mantissa = repr(abs(number)).split("e")[0].replace(".", "").lstrip("0")
    return len(mantissa) >= PRECISION


def long(number: int) -> bool:
    """Whether an integer has `PRECISION` digits.

    Args:
        number: The value.

    Returns:
        True when its decimal form, sign left out, has enough digits.
    """
    return len(str(abs(number))) >= PRECISION


def strings(frame: pandas.DataFrame, column: Any) -> list[str]:
    """The string cells of a column.

    Args:
        frame: The input table.
        column: The column label.

    Returns:
        Every non-missing string cell, in row order.
    """
    return [v for v in frame[column].tolist() if isinstance(v, str)]


def cells(frame: pandas.DataFrame, column: Any) -> list[Probe]:
    """The numeric cells of a column precise enough to identify a row.

    Args:
        frame: The input table.
        column: The column label.

    Returns:
        A `cell` probe per float cell with `PRECISION` significant digits
        and per integer cell with `PRECISION` digits, in row order.
    """
    name = str(column)
    values = frame[column].tolist()
    if pandas.api.types.is_float_dtype(frame[column]):
        return [
            Probe(repr(v), CELL, name, row, float(v))
            for row, v in enumerate(values)
            if precise(v)
        ]
    if pandas.api.types.is_integer_dtype(frame[column]):
        return [
            Probe(str(v), CELL, name, row, int(v))
            for row, v in enumerate(values)
            if isinstance(v, int) and long(v)
        ]
    return []


def probes(
    frame: pandas.DataFrame,
    declared: Mapping[str, Iterable[str]],
    secrets: Iterable[str] = (),
    floor: int = FLOOR,
) -> list[Probe]:
    """Every value of a table that no artifact may hold.

    Args:
        frame: The input table, as the run read it.
        declared: The categorical levels the bundle schema declares, by
            column.
        secrets: Secret values to look for.
        floor: The fewest rows a level must cover to be stored.

    Returns:
        The probes, one per kind, column and value.
    """
    found: dict[tuple[str, str, str], Probe] = {}
    for column in frame.columns:
        name = str(column)
        levels = {str(v) for v in declared.get(name, ())}
        for value, count in collections.Counter(strings(frame, column)).items():
            kind = ""
            if count < floor:
                kind = CATEGORY if name in declared else ROW
            elif value not in levels:
                kind = ROW
            if kind:
                found.setdefault((kind, name, value), Probe(value, kind, name))
        for probe in cells(frame, column):
            found[CELL, f"{name}:{probe.row}", probe.text] = probe
    for secret in secrets:
        found.setdefault((SECRET, "", secret), Probe(secret, SECRET))
    return list(found.values())


def environment(names: Iterable[str] = ()) -> list[str]:
    """Secret values held in the environment.

    Args:
        names: Variables to read whatever their name.

    Returns:
        The values of `names` and of every variable whose name matches
        `SECRET_NAMES`, non-empty, deduplicated.
    """
    chosen = set(names) | {n for n in os.environ if SECRET_NAMES.search(n)}
    return sorted({os.environ[n] for n in chosen if os.environ.get(n)})


def declared(schema: pathlib.Path) -> dict[str, list[str]]:
    """The categorical levels a bundle schema declares.

    Args:
        schema: A bundle's `schema.json`.

    Returns:
        Column to its declared levels; a categorical column with no levels
        maps to an empty list.

    Raises:
        InputError: If the schema cannot be read.
    """
    try:
        document = json.loads(schema.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise errors.InputError(f"cannot read schema {schema}") from error
    levels = document.get("categories") or {}
    return {
        str(c): [str(v) for v in levels.get(c, [])]
        for c in document.get("categorical", [])
    }


class Reader(abc.ABC):
    """Opens one kind of container and yields the blobs inside it.

    Each reader names the leading bytes it recognises (`magic`) and how to
    `open` the container.

    Attributes:
        magic: The leading bytes of the containers this reader opens.
    """

    magic: tuple[bytes, ...] = ()

    def accepts(self, data: bytes) -> bool:
        """Whether `data` starts with one of this reader's magic numbers.

        Args:
            data: The bytes.

        Returns:
            True when this reader can open them.
        """
        return bool(self.magic) and data.startswith(self.magic)

    @abc.abstractmethod
    def open(self, data: bytes) -> Iterator[bytes]:
        """The blobs inside the container.

        Args:
            data: The container's bytes.

        Yields:
            Each blob, to be searched and opened in turn.
        """


class Stream(Reader):
    """A compressed stream: zlib (joblib's default), gzip, bz2 or xz."""

    magic = (
        b"\x78\x01",
        b"\x78\x5e",
        b"\x78\x9c",
        b"\x78\xda",
        b"\x1f\x8b",
        b"BZh",
        b"\xfd7zXZ\x00",
    )

    def open(self, data: bytes) -> Iterator[bytes]:
        """Decompresses the stream, at most `LIMIT` bytes.

        Args:
            data: The compressed bytes.

        Yields:
            The decompressed bytes; nothing when they do not decompress.
        """
        try:
            if data.startswith(b"\x1f\x8b"):
                with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
                    opened = stream.read(LIMIT)
            elif data.startswith(b"BZh"):
                opened = bz2.BZ2Decompressor().decompress(data, LIMIT)
            elif data.startswith(b"\xfd7zXZ\x00"):
                opened = lzma.LZMADecompressor().decompress(data, LIMIT)
            else:
                opened = zlib.decompressobj().decompress(data, LIMIT)
        except (OSError, EOFError, zlib.error, lzma.LZMAError):
            # handled: not a stream after all; the raw bytes were searched
            return
        yield opened


class Envelope(Reader):
    """An scmle envelope: a header line and a (pickled) payload."""

    magic = (ENVELOPE,)

    def open(self, data: bytes) -> Iterator[bytes]:
        """The payload after the header line.

        Args:
            data: The envelope.

        Yields:
            The payload bytes.
        """
        end = data.find(b"\n", len(ENVELOPE))
        if end >= 0:
            yield data[end + 1 :]


class Archive(Reader):
    """A zip archive; each member is a blob."""

    magic = (b"PK\x03\x04",)

    def open(self, data: bytes) -> Iterator[bytes]:
        """Each member of at most `LIMIT` bytes.

        Args:
            data: The archive.

        Yields:
            The bytes of each member.
        """
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                members = [
                    m for m in archive.infolist() if m.file_size <= LIMIT
                ]
                opened = [archive.read(m) for m in members]
        except (zipfile.BadZipFile, OSError):
            # handled: not a zip after all; the raw bytes were searched
            return
        yield from opened


class Parquet(Reader):
    """A Parquet table: column names and every value, as text."""

    magic = (b"PAR1",)

    def open(self, data: bytes) -> Iterator[bytes]:
        """The table as JSON, the column names first.

        Args:
            data: The Parquet file.

        Yields:
            The column names, then each row, as UTF-8 JSON.
        """
        try:
            table = pyarrow.parquet.read_table(io.BytesIO(data))
        except (OSError, pyarrow.ArrowException):
            # handled: unreadable; the raw bytes were searched
            return
        yield json.dumps(table.column_names).encode("utf-8")
        for row in table.to_pylist():
            yield json.dumps(row, default=str).encode("utf-8")


class Database(Reader):
    """A SQLite database: every row of every table, as text."""

    magic = (b"SQLite format 3\x00",)

    def open(self, data: bytes) -> Iterator[bytes]:
        """Every statement of the database's SQL dump.

        SQLite opens files only, so the bytes are copied to a temporary
        file, opened read-only and deleted.

        Args:
            data: The database.

        Yields:
            Each statement of the dump as UTF-8.
        """
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "copy.sqlite"
            path.write_bytes(data)
            try:
                connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
                try:
                    lines = list(connection.iterdump())
                finally:
                    connection.close()
            except sqlite3.Error:
                # handled: unreadable; the raw bytes were searched
                return
        for line in lines:
            yield line.encode("utf-8")


READERS: tuple[Reader, ...] = (
    Stream(),
    Envelope(),
    Archive(),
    Parquet(),
    Database(),
)


def blobs(data: bytes, depth: int = 0) -> Iterator[bytes]:
    """The bytes themselves and everything opened out of them.

    An envelope's payload, and a joblib file, may be compressed, so every
    opened blob is offered to every reader again, down to `DEPTH`.

    Args:
        data: A file's bytes.
        depth: How deep in nested containers `data` is.

    Yields:
        `data`, then each blob opened from it.
    """
    yield data
    if depth >= DEPTH:
        return
    for reader in READERS:
        if reader.accepts(data):
            for inner in reader.open(data):
                yield from blobs(inner, depth + 1)


def search(
    data: bytes,
    path: str,
    chosen: list[tuple[Probe, dict[str, bytes]]],
    salt: bytes,
) -> list[Finding]:
    """The probes found in one file.

    Args:
        data: The file's bytes.
        path: The file's name in the report.
        chosen: The probes with their forms.
        salt: The scan's label salt.

    Returns:
        One finding per probe found, in the first form found.
    """
    found: dict[Probe, str] = {}
    for blob in blobs(data):
        for probe, forms in chosen:
            if probe in found:
                continue
            for name, form in forms.items():
                if form in blob:
                    found[probe] = name
                    break
    return [
        Finding(path, p.kind, p.column, form, p.label(salt), p.row)
        for p, form in found.items()
    ]


def scan(root: pathlib.Path, chosen: Iterable[Probe]) -> Report:
    """Searches every file under a folder for every probe.

    Args:
        root: A folder (a run directory, a store) or one file.
        chosen: The values that must not appear.

    Returns:
        What was searched and found.

    Raises:
        InputError: If `root` does not exist.
    """
    if not root.exists():
        raise errors.InputError(f"nothing to scan at {root}")
    probing = list(chosen)
    searchable = [p for p in probing if len(p.text) >= SHORTEST]
    formed = [(p, p.forms()) for p in searchable]
    if root.is_file():
        files = {root.name: root}
    else:
        files = {
            str(p.relative_to(root)): p
            for p in sorted(root.rglob("*"))
            if p.is_file()
        }
    salt = os.urandom(SALT)
    findings = []
    size = 0
    for name, path in files.items():
        data = path.read_bytes()
        size += len(data)
        findings += search(data, name, formed, salt)
    counted = collections.Counter(p.kind for p in probing)
    short = collections.Counter(
        p.kind for p in probing if len(p.text) < SHORTEST
    )
    return Report(
        len(files),
        size,
        {k: counted[k] for k in KINDS},
        {k: short[k] for k in KINDS},
        findings,
    )


def main(arguments: list[str] | None = None) -> int:
    """Scans a folder for the values of a table; exits 1 on any leak.

    Args:
        arguments: Command-line arguments; None reads `sys.argv`.

    Returns:
        0 when nothing leaked, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=pathlib.Path)
    parser.add_argument("--data", type=pathlib.Path, required=True)
    parser.add_argument("--schema", type=pathlib.Path)
    parser.add_argument("--secret", action="append", default=[])
    parser.add_argument("--secret-env", action="append", default=[])
    parser.add_argument("--floor", type=int, default=FLOOR)
    options = parser.parse_args(arguments)
    if options.data.suffix == ".parquet":
        frame = pandas.read_parquet(options.data)
    else:
        frame = pandas.read_csv(options.data)
    levels = declared(options.schema) if options.schema else {}
    secrets = [*options.secret, *environment(options.secret_env)]
    report = scan(options.folder, probes(frame, levels, secrets, options.floor))
    print(json.dumps(report.to_dict(), indent=2))  # noqa: T201 - the report is the output
    return int(any(report.counts().values()))


if __name__ == "__main__":
    sys.exit(main())
