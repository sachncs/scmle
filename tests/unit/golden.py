"""Characterization snapshots used to pin survivors of mutation testing.

`snapshot` turns a result into rounded JSON-ready data; `check` compares it
with a stored value in a fixture file and writes the fixture when the
environment variable SCMLE_WRITE_GOLDEN is set. A snapshot pins the behaviour
of the committed code on a fixed input. It does not show that the behaviour is
correct, only that a change to it is noticed.
"""

import dataclasses
import enum
from fractions import Fraction
import json
import os
from pathlib import Path
from typing import Any

import numpy

DIGITS = 9


def snapshot(value: Any) -> Any:
    """JSON-ready, rounded copy of `value`."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: snapshot(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, enum.Enum):
        return str(value.value)
    if isinstance(value, numpy.ndarray):
        return [snapshot(item) for item in value.tolist()]
    if isinstance(value, numpy.generic):
        return snapshot(value.item())
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value:
            return "nan"
        if value in (float("inf"), float("-inf")):
            return "inf" if value > 0 else "-inf"
        return round(value, DIGITS)
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {str(key): snapshot(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        items = (
            sorted(value, key=repr)
            if isinstance(value, set | frozenset)
            else value
        )
        return [snapshot(item) for item in items]
    return repr(value)


def check(path: Path, name: str, run: Any) -> tuple[Any, Any]:
    """Returns `(expected, actual)` for one case, writing when asked."""
    actual = snapshot(run())
    stored: dict[str, Any] = {}
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
    if os.environ.get("SCMLE_WRITE_GOLDEN"):
        stored[name] = actual
        path.write_text(
            json.dumps(stored, indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return stored.get(name), actual


def same(expected: Any, actual: Any, tol: float = 1e-4) -> bool:
    """Structural equality with a relative tolerance on floats.

    Learners that use threaded or platform-specific numerics (boosted trees)
    differ in the last digits between operating systems, so their snapshots
    are compared to `tol` instead of to the stored digits.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected.keys() == actual.keys() and all(
            same(expected[key], actual[key], tol) for key in expected
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            same(left, right, tol)
            for left, right in zip(expected, actual, strict=True)
        )
    if (
        isinstance(expected, float)
        and isinstance(actual, float)
        and not isinstance(expected, bool)
    ):
        return abs(expected - actual) <= tol * max(1.0, abs(expected))
    return bool(expected == actual)
