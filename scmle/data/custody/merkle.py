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
    """Renders one cell as platform-independent text tagged with its type.

    Args:
        value: A cell value: None, NaT, NaN, a bool, int, float, Timestamp,
            numpy scalar or anything else (rendered with `str`).

    Returns:
        `null` for None, NaT or NaN; otherwise a type tag and the value
        (`b:`, `i:`, `f:` with the round-trip `repr`, `t:` ISO 8601, or `s:`).
    """
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
    """Hashes one row from its id and every `(column, value)` pair.

    Args:
        row_id: Position of the row in the dataset.
        cells: The `(column name, value)` pairs in frame order.

    Returns:
        The SHA-256 hex digest of the leaf tag followed by the row text.
    """
    text = SEPARATOR.join(
        [f"id={int(row_id)}"]
        + [f"{name}={canonical(value)}" for name, value in cells]
    )
    return hashlib.sha256(LEAF + text.encode("utf-8")).hexdigest()


def row_hashes(frame: pd.DataFrame, ids: Sequence[int]) -> list[str]:
    """Hashes the rows of a frame at the given positions.

    Args:
        frame: The dataset; columns are read in frame order.
        ids: Row positions to hash; sorted before use.

    Returns:
        One leaf hash per id, in ascending id order.
    """
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
    """Hashes two child nodes into their parent.

    Args:
        left: Hex digest of the left child.
        right: Hex digest of the right child.

    Returns:
        The SHA-256 hex digest of the node tag and both children.
    """
    return hashlib.sha256(
        NODE + bytes.fromhex(left) + bytes.fromhex(right)
    ).hexdigest()


def merkle_root(leaves: Sequence[str]) -> str:
    """Folds leaf hashes into a Merkle root.

    Args:
        leaves: Leaf hex digests, in the order they are combined.

    Returns:
        The root hex digest, or `EMPTY_ROOT` when there are no leaves.
    """
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
    """Computes the Merkle root over the content of selected rows.

    Args:
        frame: The dataset.
        ids: Row positions to include.

    Returns:
        The root hex digest of the rows in ascending id order.
    """
    return merkle_root(row_hashes(frame, ids))
