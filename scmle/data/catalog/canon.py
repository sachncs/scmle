"""Deterministic JSON content and the digests taken over it."""

from collections.abc import Mapping
import datetime
import hashlib
import json
import math
from typing import Any

import numpy
import pandas

from scmle.contracts import errors


def canonize(value: Any) -> str:
    """Returns deterministic JSON for an already JSON-compatible value.

    Args:
        value: A JSON-compatible value (mappings, lists, strings, finite
            numbers, booleans, or None).

    Returns:
        Compact JSON text with sorted keys, non-ASCII characters kept, and no
        whitespace between tokens.
    """
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def fingerprint(value: Any) -> str:
    """Returns a SHA-256 identity for deterministic JSON content.

    Args:
        value: A JSON-compatible value, serialized with ``canonize``.

    Returns:
        The string ``"sha256:"`` followed by 64 lowercase hex digits.
    """
    return f"sha256:{hashlib.sha256(canonize(value).encode()).hexdigest()}"


def canonicalize(value: Any) -> Any:
    """Converts one supported value into deterministic JSON-native content.

    Args:
        value: A scalar, bytes, date or time, mapping, list, tuple, set, or
            NumPy value.

    Returns:
        None for missing values; a length-and-SHA-256 mapping for bytes;
        an ISO string for dates and times; a mapping with string keys
        sorted by key for mappings; lists in order for sequences; sorted
        lists for sets; a ``{"non_finite": ...}`` mapping for NaN and
        infinities; otherwise the plain string, number, or boolean.

    Raises:
        TypeError: If the value is of a type that cannot be converted
            deterministically.
    """
    if value is None or value is pandas.NA:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return {
            "bytes_length": len(raw),
            "bytes_sha256": hashlib.sha256(raw).hexdigest(),
        }
    if isinstance(value, numpy.generic):
        return canonicalize(value.item())
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        return convert(value)
    return normalise(value)


def convert(
    value: Mapping[Any, Any]
    | list[Any]
    | tuple[Any, ...]
    | set[Any]
    | frozenset[Any],
) -> Any:
    """Converts a mapping, sequence or set into JSON-native content.

    Args:
        value: A mapping, list, tuple, set or frozenset.

    Returns:
        A mapping with string keys sorted by key for a mapping, a list in
        order for a list or tuple, and a list sorted by canonical JSON for a
        set.
    """
    if isinstance(value, Mapping):
        return {
            str(key): canonicalize(item)
            for key, item in sorted(
                value.items(), key=lambda pair: str(pair[0])
            )
        }
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    values = [canonicalize(item) for item in value]
    return sorted(values, key=canonize)


def normalise(value: Any) -> Any:
    """Converts a value that is not a container, bytes, date or NumPy scalar.

    Args:
        value: A float, string, integer, boolean, NumPy array or another
            value pandas treats as missing.

    Returns:
        A ``{"non_finite": ...}`` mapping for NaN and infinities, None for
        a missing value, the value itself for a string, number or boolean,
        and a list for a NumPy array.

    Raises:
        TypeError: If the value is of a type that cannot be converted
            deterministically.
    """
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return {"non_finite": "nan"}
        return {"non_finite": "-inf" if value < 0 else "inf"}
    missing = pandas.isna(value)
    if isinstance(missing, (bool, numpy.bool_)) and bool(missing):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, numpy.ndarray):
        return [canonicalize(item) for item in value.tolist()]
    raise errors.KindError(
        f"cannot deterministically catalog value of type {type(value).__name__}"
    )
