"""Keeps copies of one record on the same side of every split.

If a record appears twice and the copies land in training and in the locked
test set, the test score rewards memorising rather than generalising. Rows
whose features match (numbers compared to six decimals, text exactly) form a
group, and a post-processing pass moves each group wholly to one side.
"""

from collections import Counter

import numpy as np
import pandas as pd

DECIMALS = 6
NOT_DUPLICATED = -1


def duplicate_groups(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Group id per row; rows with no copy get `NOT_DUPLICATED`.

    Args:
        frame: The full dataset.
        columns: Feature columns that define a record.

    Returns:
        An integer array aligned with the frame's rows.
    """
    if not columns:
        return np.full(len(frame), NOT_DUPLICATED)
    keyed = frame[columns].copy()
    for column in keyed.columns:
        if pd.api.types.is_float_dtype(keyed[column]):
            keyed[column] = keyed[column].round(DECIMALS)
        elif not pd.api.types.is_numeric_dtype(keyed[column]):
            keyed[column] = keyed[column].astype(str)
    hashes = pd.util.hash_pandas_object(keyed, index=False).to_numpy()
    codes, uniques = pd.factorize(hashes)
    sizes = np.bincount(codes, minlength=len(uniques))
    return np.where(sizes[codes] > 1, codes, NOT_DUPLICATED)


def majority(values: list[int]) -> int:
    """Most common value; the smallest wins ties (the conservative side)."""
    counts = Counter(values)
    best = max(counts.values())
    return min(value for value, count in counts.items() if count == best)


def keep_together(
    dev_ids: np.ndarray,
    test_ids: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]], int]:
    """Moves each duplicate group wholly into development or the test.

    A group goes where most of its rows already sit (development on a tie).
    Inside development, each group is held out in the single fold where most
    of its rows were held out.

    Args:
        dev_ids: Development row ids.
        test_ids: Test row ids.
        folds: Folds as positions within `dev_ids`.
        groups: Group id per row of the frame.

    Returns:
        The new development ids, test ids and folds, and the number of
        duplicate groups kept together.
    """
    side = {int(i): 0 for i in dev_ids} | {int(i): 1 for i in test_ids}
    members: dict[int, list[int]] = {}
    for row, group in enumerate(groups):
        if group != NOT_DUPLICATED:
            members.setdefault(int(group), []).append(row)
    for rows in members.values():
        target = majority([side[row] for row in rows])
        for row in rows:
            side[row] = target
    new_dev = np.array(sorted(r for r, s in side.items() if s == 0))
    new_test = np.array(sorted(r for r, s in side.items() if s == 1))
    held_in: dict[int, int] = {}
    for index, fold in enumerate(folds):
        for position in fold[1]:
            held_in[int(dev_ids[position])] = index
    for rows in members.values():
        inside = [held_in[row] for row in rows if row in held_in]
        if not inside:
            continue
        chosen = majority(inside)
        for row in rows:
            if side[row] == 0:
                held_in[row] = chosen
    rebuilt = []
    for index in range(len(folds)):
        held_rows = {row for row, fold in held_in.items() if fold == index}
        held = np.array(
            [p for p, row in enumerate(new_dev) if int(row) in held_rows]
        )
        fit = np.array(
            [p for p, row in enumerate(new_dev) if int(row) not in held_rows]
        )
        rebuilt.append((fit, held))
    return new_dev, new_test, rebuilt, len(members)
