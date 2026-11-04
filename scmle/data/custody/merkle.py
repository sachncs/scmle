"""Merkle roots over per-row content hashes.

The split manifest lists row ids, so a change to a value inside a row leaves
it unchanged. A root over the content of the rows of each set closes that:
the manifest then commits to the values as well as to which rows belong to
which set. The root does not reveal a value, and the manifest stores only the
root.

Each row is encoded as a canonical string (the row id, then every column in
frame order, `name=type:value`), so the same table gives the same bytes on
any platform: floats use the shortest round-trip `repr`, integers and
booleans their decimal text, missing values `null`, timestamps ISO 8601.
Leaves and inner nodes are hashed with different prefixes, and an odd node
is carried up unchanged rather than duplicated, so two different lists of
leaves cannot share a root.
"""

from collections.abc import Sequence
import hashlib
import math
from typing import Any

import pandas as pd

LEAF = b"\x00"
NODE = b"\x01"
SEPARATOR = "\x1f"
EMPTY_ROOT = hashlib.sha256(b"scmle.custody.merkle.empty").hexdigest()


def canonical(value: Any) -> str:
    """Platform-independent text of one cell, tagged with its type."""
    if value is None or value is pd.NaT:
        return "null"
    if hasattr(value, "item") and not isinstance(value, pd.Timestamp):
        value = value.item()
    if isinstance(value, bool):
        return f"b:{int(value)}"
    if isinstance(value, int):
        return f"i:{value}"
    if isinstance(value, float):
        if math.isnan(value):
            return "null"
        return f"f:{value!r}"
    if isinstance(value, pd.Timestamp):
        return f"t:{value.isoformat()}"
    return f"s:{value}"


def row_hash(row_id: int, cells: Sequence[tuple[str, Any]]) -> str:
    """Hash of one row: its id and every `(column, value)` pair."""
    text = SEPARATOR.join(
        [f"id={int(row_id)}"]
        + [f"{name}={canonical(value)}" for name, value in cells]
    )
    return hashlib.sha256(LEAF + text.encode("utf-8")).hexdigest()


def row_hashes(frame: pd.DataFrame, ids: Sequence[int]) -> list[str]:
    """Leaf hashes of `frame` rows at positions `ids`, in id order."""
    ordered = sorted(int(i) for i in ids)
    names = [str(c) for c in frame.columns]
    columns = [frame.iloc[ordered, k].tolist() for k in range(len(names))]
    return [
        row_hash(row_id, list(zip(names, cells, strict=True)))
        for row_id, cells in zip(
            ordered, zip(*columns, strict=True), strict=True
        )
    ]


def combine(left: str, right: str) -> str:
    """Parent of two nodes."""
    return hashlib.sha256(
        NODE + bytes.fromhex(left) + bytes.fromhex(right)
    ).hexdigest()


def merkle_root(leaves: Sequence[str]) -> str:
    """Root over leaf hashes, in the order given."""
    level = list(leaves)
    if not level:
        return EMPTY_ROOT
    while len(level) > 1:
        paired = [
            combine(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)
        ]
        if len(level) % 2:
            paired.append(level[-1])
        level = paired
    return level[0]


def frame_root(frame: pd.DataFrame, ids: Sequence[int]) -> str:
    """Merkle root over the content of the rows at positions `ids`."""
    return merkle_root(row_hashes(frame, ids))
