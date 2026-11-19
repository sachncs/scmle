"""One event of an append-only log, and how a raw record becomes one.

An event says that something of a `kind` happened to an `entity` at a
`time`, with a `payload` of named scalar values. Times are held as integer
nanoseconds since the epoch in UTC, so ordering and window arithmetic are
exact; a timestamp without a zone is read as UTC. Nested payload objects are
flattened into dotted names (`{"card": {"type": "visa"}}` gives `card.type`).
"""

from collections.abc import Mapping
import dataclasses
import datetime
import json
import math
import numbers
import re

import pandas

from scmle.contracts import errors

# A day count written with a lower-case `d` unit, as in `7d` or `1d12h`.
DAYS = re.compile(r"(\d)\s*d(?![a-z])")


class EventError(errors.InputError):
    """A record is not an event, or an event cannot enter the log."""


class LatenessError(EventError):
    """An event arrived later than the declared lateness bound allows."""


@dataclasses.dataclass(frozen=True, slots=True)
class Event:
    """One immutable event.

    Attributes:
        entity: Identity of the thing the event happened to.
        time: When it happened, in nanoseconds since the epoch (UTC).
        kind: What happened, such as `purchase` or `login`.
        payload: Named values the event carries: numbers (int, float,
            bool) or text.
    """

    entity: str
    time: int
    kind: str
    payload: Mapping[str, str | float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, slots=True)
class Columns:
    """Where a record keeps each part of an event.

    Attributes:
        entity: Field holding the entity.
        time: Field holding the event time.
        kind: Field holding the event type.
        payload: Field holding the payload object or JSON text; when a
            record has no such field, every other field is the payload.
    """

    entity: str = "entity"
    time: str = "time"
    kind: str = "type"
    payload: str = "payload"


def instant(value: object) -> int:
    """Converts a timestamp to nanoseconds since the epoch in UTC.

    Args:
        value: A timestamp, datetime or ISO-8601 text; without a zone it is
            read as UTC.

    Returns:
        The instant in nanoseconds since the epoch.

    Raises:
        EventError: If `value` is missing, a number (a Python or numpy
            scalar), or is not a timestamp.
    """
    if isinstance(value, numbers.Number):
        shown = value.item() if hasattr(value, "item") else value
        raise EventError(f"time must be a timestamp, got the number {shown!r}")
    if isinstance(value, pandas.Timestamp | datetime.datetime):
        stamp = pandas.Timestamp(value)
        return int(
            (stamp.tz_localize("UTC") if stamp.tz is None else stamp).value
        )
    stamp = pandas.to_datetime(
        pandas.Series([value], dtype=object),
        errors="coerce",
        utc=True,
        format="mixed",
    ).iloc[0]
    if pandas.isna(stamp):
        raise EventError(f"time must be a timestamp, got {value!r}")
    return int(stamp.value)


def span(value: object) -> int:
    """Converts a duration to nanoseconds.

    Args:
        value: A timedelta or text such as `7d`, `12h` or `30min`.

    Returns:
        The duration in nanoseconds; never negative.

    Raises:
        EventError: If `value` is not a duration or is negative.
    """
    if isinstance(value, str):
        # pandas deprecates the lower-case day unit people write ("7d").
        value = DAYS.sub(r"\1D", value)
    duration: object = pandas.to_timedelta(
        pandas.Series([value], dtype=object), errors="coerce"
    ).iloc[0]
    if not isinstance(duration, pandas.Timedelta) or duration.value < 0:
        raise EventError(f"a duration must be non-negative, got {value!r}")
    return int(duration.value)


def missing(value: object) -> bool:
    """Whether a cell holds no value (None, NaN or NaT).

    Args:
        value: Any cell.

    Returns:
        True for None and for a float or pandas missing marker.
    """
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return value is pandas.NaT or value is pandas.NA


def flatten(
    content: Mapping[str, object], prefix: str = ""
) -> dict[str, str | float]:
    """Flattens a nested payload into dotted scalar fields.

    Missing values are left out: a field an event does not carry is absent,
    not null.

    Args:
        content: A payload object, possibly nested.
        prefix: The dotted name of `content` within the payload.

    Returns:
        The scalar fields by dotted name.

    Raises:
        EventError: If a value is a list or another non-scalar.
    """
    found: dict[str, str | float] = {}
    for key, value in content.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            found.update(flatten(value, f"{name}."))
        elif missing(value):
            continue
        elif isinstance(value, pandas.Timestamp):
            found[name] = value.isoformat()
        elif isinstance(value, bool | int | float | str):
            found[name] = value
        elif hasattr(value, "item"):
            found[name] = value.item()
        else:
            raise EventError(
                f"payload field {name!r} must be a scalar or an object, "
                f"got {type(value).__name__}"
            )
    return found


def contents(value: object) -> Mapping[str, object]:
    """Reads a payload cell: an object, JSON text, or nothing.

    Args:
        value: The payload cell of a record.

    Returns:
        The payload object.

    Raises:
        EventError: If the cell is not an object or JSON object text.
    """
    if missing(value):
        return {}
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value) if value.strip() else {}
        except json.JSONDecodeError as error:
            raise EventError(f"payload is not JSON: {value[:80]!r}") from error
        if isinstance(decoded, Mapping):
            return decoded
    raise EventError(f"payload must be an object, got {value!r}")


def identity(value: object) -> str:
    """Spells an entity the same way whichever reader produced it.

    A CSV column of ids may read as int, or as float when a cell is empty;
    `7`, `7.0` and `"7"` all name the entity `"7"`.

    Args:
        value: The entity cell.

    Returns:
        The entity's text.
    """
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def parse(record: Mapping[str, object], columns: Columns) -> Event:
    """Builds an event from one raw record.

    Args:
        record: A record of a file or a stream, by field name.
        columns: Where the record keeps each part of the event.

    Returns:
        The event.

    Raises:
        EventError: If the entity, time or type is missing, or the payload
            is malformed.
    """
    for field in (columns.entity, columns.time, columns.kind):
        if field not in record or missing(record[field]):
            raise EventError(
                f"an event needs the field {field!r}",
                {
                    "field": field,
                    "record": {k: str(v) for k, v in record.items()},
                },
            )
    if columns.payload in record:
        payload = flatten(contents(record[columns.payload]))
    else:
        named = {columns.entity, columns.time, columns.kind}
        payload = flatten(
            {str(k): v for k, v in record.items() if k not in named}
        )
    return Event(
        entity=identity(record[columns.entity]),
        time=instant(record[columns.time]),
        kind=str(record[columns.kind]),
        payload=payload,
    )
