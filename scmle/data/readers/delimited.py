"""Delimited text read so that every file ends correct or refused.

`pandas.read_csv` alone reads a ";" file as one column, renames a
duplicate header to "a.1", rounds 2**53 + 1 to a float when the column
has a gap, drops the zero of "02134", pads a short row with NaN and
rejects latin-1. The functions here decode the bytes, find the delimiter,
check the records and settle each column's type, and raise
`errors.InputError` wherever the file has no single right reading.
"""

import bz2
import codecs
from collections.abc import Iterator
import csv
import gzip
import io
import lzma
import pathlib
import re
import zipfile

import pandas

from scmle.contracts import errors

# Delimiters tried when none is stated; the header must split on exactly one.
CANDIDATES = (",", ";", "|", "\t")
# Byte-order marks, longest first: the UTF-32 LE mark begins with UTF-16's.
MARKS = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)
# The single-byte fallback: windows-1252 reads every printable latin-1 byte
# the same and leaves its five unassigned bytes undecodable.
LEGACY = "cp1252"
# Non-letter characters a Western European text plausibly holds.
SYMBOLS = frozenset(
    "€£¥°§©®\u00b1\u00b5¶·\u2013—\u2018\u2019\u201a“”„"
    "†‡•…‰\u2039\u203a«»¿¡\u00a0"
)
# Letters of Latin-1 Supplement and Latin Extended-A (U+00C0 to U+017F).
LETTERS = range(0xC0, 0x180)
# Longest run of non-ASCII characters accepted from the fallback. Western
# European words run at most three accented letters together ("ção",
# "ÆØÅ"), while each CJK character takes two high bytes, so a two-character
# word decoded with the wrong code page already gives a run of four.
RUN = 3
# float64 has a 53-bit significand: 2**53 + 1 is the first integer it
# cannot hold.
EXACT = 2**53
# Decimal comma read with ";": "1,500" may mean 1.5 or 1500.
GROUPED = re.compile(r"[+-]?\d{1,3},\d{3}")
INTEGER = re.compile(r"[+-]?\d+")
# A zero-padded integer token: an identifier ("02134"), not a number.
PADDED = r"[+-]?0\d+"
INFLATERS = {
    ".gz": gzip.decompress,
    ".bz2": bz2.decompress,
    ".xz": lzma.decompress,
}


def inflate(path: pathlib.Path) -> bytes:
    """Reads a file's bytes, decompressed by its suffix.

    Args:
        path: Plain, gzip, bzip2, xz or single-member zip file.

    Returns:
        The uncompressed bytes.

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
            return archive.read(names[0])
    inflater = INFLATERS.get(suffix)
    payload = path.read_bytes()
    return payload if inflater is None else inflater(payload)


def decode(payload: bytes, encoding: str | None = None) -> str:
    """Decodes bytes as stated, by byte-order mark, as UTF-8 or as cp1252.

    Args:
        payload: The file's bytes.
        encoding: Encoding to use instead of detecting one.

    Returns:
        The text.

    Raises:
        InputError: If the stated encoding fails, or none is stated and no
            detected one reads the bytes plausibly.
    """
    if encoding is not None:
        return strict(payload, encoding)
    for mark, name in MARKS:
        if payload.startswith(mark):
            return strict(payload, name)
    if b"\x00" in payload:
        raise errors.InputError(
            "the file holds NUL bytes, likely UTF-16 or UTF-32 without a "
            "byte-order mark; state the encoding"
        )
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        # handled: not UTF-8; the legacy decoding below decides or refuses
        pass
    text = strict(payload, LEGACY)
    if not plausible(text):
        raise errors.InputError(
            "the file is neither UTF-8 nor plausible latin-1 / cp1252 text; "
            "state the encoding"
        )
    return text


def strict(payload: bytes, encoding: str) -> str:
    """Decodes bytes with one encoding, refusing any undecodable byte.

    Args:
        payload: The file's bytes.
        encoding: A Python codec name.

    Returns:
        The text.

    Raises:
        InputError: If the codec is unknown or a byte does not decode.
    """
    try:
        return payload.decode(encoding)
    except (UnicodeDecodeError, LookupError) as error:
        raise errors.InputError(
            f"cannot decode as {encoding}: {error}; state the encoding"
        ) from error


def plausible(text: str) -> bool:
    """Whether cp1252-decoded text reads as Western European text.

    Args:
        text: Text decoded with the single-byte fallback.

    Returns:
        True when every non-ASCII character is a Latin letter or common
        symbol and no run of them is longer than `RUN`.
    """
    run = 0
    for char in text:
        if char.isascii():
            run = 0
            continue
        run += 1
        latin = ord(char) in LETTERS and char.isalpha()
        if run > RUN or not (latin or char in SYMBOLS):
            return False
    return True


def records(text: str, delimiter: str) -> Iterator[list[str]]:
    """Iterates the records of delimited text.

    Args:
        text: Decoded text.
        delimiter: Field separator.

    Returns:
        A csv reader over `text`.
    """
    return csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)


def sniff(text: str) -> str:
    """Finds the delimiter from the header record.

    Args:
        text: Decoded text.

    Returns:
        The one candidate the header splits on, or "," when the header is
        a single field under every candidate.

    Raises:
        InputError: If the header splits on more than one candidate.
    """
    found = [
        candidate
        for candidate in CANDIDATES
        if len(next(records(text, candidate), [])) > 1
    ]
    if len(found) > 1:
        raise errors.InputError(
            f"the header splits on each of {found}; state the delimiter"
        )
    return found[0] if found else ","


def check(text: str, delimiter: str) -> None:
    """Refuses duplicate headers and records of the wrong width.

    Args:
        text: Decoded text.
        delimiter: Field separator.

    Raises:
        InputError: If a non-empty header name repeats, or a non-blank
            record has more or fewer fields than the header.
    """
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    header = next(reader, None)
    if header is None:
        return
    named = [name for name in header if name]
    repeated = sorted({name for name in named if named.count(name) > 1})
    if repeated:
        raise errors.InputError(f"duplicate column names: {repeated}")
    for record in reader:
        if record and len(record) != len(header):
            raise errors.InputError(
                f"line {reader.line_num} has {len(record)} fields, the "
                f"header {len(header)}"
            )


def parse(text: str, delimiter: str) -> pandas.DataFrame:
    """Parses checked text, keeping every value exact.

    A ";" file reads "1,5" as 1.5. A numeric column keeps its text when a
    value is zero-padded, becomes nullable Int64 when an integer beyond
    2**53 shares it with a gap, and holds Python ints and floats when such
    an integer shares it with a float.

    Args:
        text: Decoded text that passed `check`.
        delimiter: Field separator.

    Returns:
        The table.

    Raises:
        InputError: If a column has no exact reading.
    """
    decimal = "," if delimiter == ";" else "."
    frame = pandas.read_csv(io.StringIO(text), sep=delimiter, decimal=decimal)
    numeric = [
        position
        for position, dtype in enumerate(frame.dtypes)
        if pandas.api.types.is_numeric_dtype(dtype)
        and not pandas.api.types.is_bool_dtype(dtype)
    ]
    if not numeric:
        return frame
    raw = pandas.read_csv(
        io.StringIO(text),
        sep=delimiter,
        dtype=str,
        usecols=numeric,
    )
    for position, name in zip(numeric, raw.columns, strict=True):
        frame.isetitem(
            position,
            settle(frame.iloc[:, position], raw[name], decimal).array,
        )
    return frame


def settle(
    column: pandas.Series, raw: pandas.Series, decimal: str = "."
) -> pandas.Series:
    """Chooses the exact form of one numeric column.

    Args:
        column: The column as pandas typed it.
        raw: The same column as text.
        decimal: Decimal mark of the file.

    Returns:
        `column`, its text, its nullable Int64 form, or, when an integer
        beyond 2**53 shares it with a float, an object column of Python
        ints and floats (float64 cannot keep such an integer exact).

    Raises:
        InputError: If its decimal commas could be thousands separators, or
            integers beyond int64 share it with a gap.
    """
    tokens = raw.dropna()
    if tokens.str.fullmatch(PADDED).any():
        return raw
    commas = tokens[tokens.str.contains(",", regex=False)]
    if len(commas) and commas.str.fullmatch(GROUPED.pattern).all():
        raise errors.InputError(
            f"column '{column.name}' reads {commas.iloc[0]!r} with a decimal "
            "comma, but it may be a thousands separator; state the delimiter "
            "and fix the file"
        )
    if not pandas.api.types.is_float_dtype(column.dtype):
        return column
    integral = tokens[tokens.str.fullmatch(INTEGER.pattern)]
    if not any(abs(int(token)) > EXACT for token in integral):
        return column
    if len(integral) < len(tokens):
        mixed = [
            None
            if pandas.isna(token)
            else int(token)
            if INTEGER.fullmatch(token)
            else float(token.replace(decimal, "."))
            for token in raw
        ]
        return pandas.Series(mixed, index=column.index, dtype=object)
    values = [None if pandas.isna(token) else int(token) for token in raw]
    try:
        return pandas.Series(values, index=column.index, dtype="Int64")
    except (OverflowError, TypeError) as error:
        raise errors.InputError(
            f"column '{column.name}' holds integers beyond int64 and a gap"
        ) from error
