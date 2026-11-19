"""Where events come from: a file, a stream of lines, or a growing file.

Every source yields raw records; `Source.events` turns them into events by
the source's `Columns`. A strict source raises on the first malformed
record; a lenient one skips it and keeps the reason in `skipped`, so a bad
line in a live stream does not stop the stream.
"""

import abc
from collections.abc import Iterator
from collections.abc import Mapping
import json
import pathlib
from typing import TextIO

import pandas

from scmle.data.events import event
from scmle.data.readers import tabular


class Source(abc.ABC):
    """A supply of event records.

    Attributes:
        columns: Where each record keeps each part of an event.
        strict: Whether a malformed record raises instead of being skipped.
        skipped: The position and reason of each record skipped.
    """

    def __init__(
        self, columns: event.Columns | None = None, *, strict: bool = True
    ) -> None:
        """Builds the source.

        Args:
            columns: Where each record keeps each part of an event.
            strict: Whether a malformed record raises.
        """
        self.columns = columns or event.Columns()
        self.strict = strict
        self.skipped: list[tuple[int, str]] = []

    @abc.abstractmethod
    def records(self) -> Iterator[tuple[int, Mapping[str, object] | str]]:
        """Yields the records available now with their positions.

        Yields:
            The record's position (row or line number) and the record, or
            the reason the record could not be read.
        """

    def events(self) -> Iterator[event.Event]:
        """Yields the events available now, in arrival order.

        Yields:
            Each well-formed record's event.

        Raises:
            EventError: If a record is malformed and the source is strict.
        """
        for position, record in self.records():
            try:
                found = event.parse(readable(record), self.columns)
            except event.EventError as error:
                if self.strict:
                    raise event.EventError(
                        f"record {position}: {error}",
                        {"position": position, **error.details},
                    ) from error
                self.skipped.append((position, str(error)))
                continue
            yield found


def readable(record: Mapping[str, object] | str) -> Mapping[str, object]:
    """Passes a record through, or raises the reason it could not be read.

    Args:
        record: A record, or the reason it could not be read.

    Returns:
        The record.

    Raises:
        EventError: If `record` is a reason.
    """
    if isinstance(record, str):
        raise event.EventError(record)
    return record


def decode(line: str) -> Mapping[str, object] | str:
    """Reads one JSON object line.

    Args:
        line: The line, without its newline.

    Returns:
        The object, or the reason it is not one.
    """
    try:
        found = json.loads(line)
    except json.JSONDecodeError as error:
        return f"not JSON ({error.msg})"
    if not isinstance(found, dict):
        return "not a JSON object"
    return found


class File(Source):
    """A whole event file of any format the tabular reader supports.

    Attributes:
        path: The file.
    """

    def __init__(
        self,
        path: str | pathlib.Path,
        columns: event.Columns | None = None,
        *,
        strict: bool = True,
    ) -> None:
        """Builds the source.

        Args:
            path: The file (CSV, JSON lines, Parquet and the rest).
            columns: Where each record keeps each part of an event.
            strict: Whether a malformed record raises.
        """
        super().__init__(columns, strict=strict)
        self.path = pathlib.Path(path)

    def records(self) -> Iterator[tuple[int, Mapping[str, object] | str]]:
        """Yields the file's rows, numbered from 1.

        Yields:
            Each row's number and its record.
        """
        frame = tabular.read_table(self.path)
        column = self.columns.time
        if column in frame.columns and not pandas.api.types.is_numeric_dtype(
            frame[column]
        ):
            stamps = pandas.to_datetime(
                frame[column], errors="coerce", utc=True, format="mixed"
            )
            frame[column] = stamps.astype(object).where(
                stamps.notna(), frame[column]
            )
        for number, row in enumerate(frame.to_dict("records"), start=1):
            yield number, {str(k): v for k, v in row.items()}


class Lines(Source):
    """JSON object lines read from an open text stream, such as stdin.

    Attributes:
        stream: The stream; reading stops at its end.
    """

    def __init__(
        self,
        stream: TextIO,
        columns: event.Columns | None = None,
        *,
        strict: bool = True,
    ) -> None:
        """Builds the source.

        Args:
            stream: The text stream.
            columns: Where each record keeps each part of an event.
            strict: Whether a malformed record raises.
        """
        super().__init__(columns, strict=strict)
        self.stream = stream

    def records(self) -> Iterator[tuple[int, Mapping[str, object] | str]]:
        """Yields each non-blank line's object as it is read.

        Yields:
            The line number and the object, or why it is not one.
        """
        for number, line in enumerate(self.stream, start=1):
            if line.strip():
                yield number, decode(line)


class Tail(Source):
    """A JSON-lines file another process appends to, read as it grows.

    Each call to `records` reads the complete lines written since the last
    call; a line still being written (no newline yet) waits for the next
    call. `offset` is the byte position read up to, so a reader that saves
    it resumes without reading an event twice.

    Attributes:
        path: The file.
        offset: Bytes consumed so far.
        line: Number of the last line consumed.
    """

    def __init__(
        self,
        path: str | pathlib.Path,
        columns: event.Columns | None = None,
        *,
        offset: int = 0,
        strict: bool = True,
    ) -> None:
        """Builds the source.

        Args:
            path: The file; it need not exist yet.
            columns: Where each record keeps each part of an event.
            offset: Byte position to resume from.
            strict: Whether a malformed record raises.

        Raises:
            EventError: If the offset is negative.
        """
        super().__init__(columns, strict=strict)
        if offset < 0:
            raise event.EventError("a tail offset must not be negative")
        self.path = pathlib.Path(path)
        self.offset = offset
        self.line = 0

    def records(self) -> Iterator[tuple[int, Mapping[str, object] | str]]:
        """Yields the complete lines appended since the last call.

        Yields:
            The line number and the object, or why it is not one.

        Raises:
            EventError: If the file shrank below the offset (it was
                truncated or replaced, so the offset no longer means
                anything).
        """
        if not self.path.exists():
            return
        if self.path.stat().st_size < self.offset:
            raise event.EventError(
                f"{self.path} is shorter than the offset {self.offset}: "
                "it was truncated or replaced",
                {"path": str(self.path), "offset": self.offset},
            )
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            for raw in handle:
                if not raw.endswith(b"\n"):
                    break
                self.offset += len(raw)
                self.line += 1
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    yield self.line, "not UTF-8 text"
                    continue
                if text.strip():
                    yield self.line, decode(text)
