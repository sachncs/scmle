"""Carving the confirmation set `C` out of the development rows.

The final-evaluation rows are already held out by `SplitManager`. This module
takes what is left (the development rows) and divides it into the
exploration set `X`, the confirmation set `C` and, for temporal data, an
embargo of rows dropped between them. The result is a pure function of the
data hash, the charter hash and the seed.

Units. Rows that share a group value, or are copies of one record, form one
unit and always land on the same side. For temporal data `C` is drawn as
contiguous time blocks inside the exploration period (it tests replication;
the forward set tests stability), with `embargo_rows` rows dropped on each
side of every block so serial dependence cannot cross the boundary.
"""

import dataclasses
import hashlib
import math

import numpy as np
import pandas as pd

from scmle.contracts import errors
from scmle.core import randomness

MAX_FRACTION = 0.5
BLOCKS = 20
NO_DUPLICATE = -1
COMPONENT = "custody.carve"


@dataclasses.dataclass(frozen=True, slots=True)
class Carve:
    """The three-way division of the development rows.

    Attributes:
        exploration_ids: Row ids of `X`, in development order.
        confirmation_ids: Row ids of `C`, in development order.
        embargo_ids: Row ids dropped from both sets.
        folds: Cross-validation folds as positions within `exploration_ids`.
        notes: What was done and why.
    """

    exploration_ids: np.ndarray
    confirmation_ids: np.ndarray
    embargo_ids: np.ndarray
    folds: list[tuple[np.ndarray, np.ndarray]]
    notes: tuple[str, ...]


def derived_seed(charter_hash: str, data_hash: str, seed: int) -> int:
    """32-bit run seed that depends on all three inputs."""
    digest = hashlib.sha256(
        f"{data_hash}|{charter_hash}|{seed}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big")


def unit_labels(
    size: int, groups: np.ndarray | None, duplicates: np.ndarray | None
) -> np.ndarray:
    """Label each row with its unit: the smallest row index in its unit.

    Rows sharing a group value or a duplicate-group id are merged
    transitively, so a group that contains a copy of a record in another
    group pulls both groups into one unit.
    """
    parent = np.arange(size)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    for keys, none in ((groups, None), (duplicates, NO_DUPLICATE)):
        if keys is None:
            continue
        codes = pd.factorize(np.asarray(keys))[0]
        if none is not None:
            codes = np.where(np.asarray(keys) == none, -1, codes)
        first: dict[int, int] = {}
        for index, raw in enumerate(codes):
            code = int(raw)
            if code < 0:
                continue
            if code in first:
                union(first[code], index)
            else:
                first[code] = index
    return np.array([find(i) for i in range(size)])


def pick_units(
    units: np.ndarray,
    strata: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Marks rows of randomly chosen whole units, stratum by stratum."""
    chosen = np.zeros(len(units), dtype=bool)
    for stratum in sorted(set(strata.tolist())):
        in_stratum = strata == stratum
        stratum_units = np.unique(units[in_stratum])
        target = round(fraction * int(in_stratum.sum()))
        rows_taken = 0
        for units_taken, unit in enumerate(rng.permutation(stratum_units)):
            if rows_taken >= target or units_taken >= len(stratum_units) - 1:
                break
            members = units == unit
            chosen |= members
            rows_taken += int(members.sum())
    return chosen


def pick_blocks(
    times: np.ndarray, fraction: float, embargo: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Marks contiguous time blocks as `C`; returns the mask and the order."""
    size = len(times)
    order = np.argsort(times, kind="stable")
    block = max(4 * embargo, math.ceil(size / BLOCKS), 1)
    starts = list(range(0, size, block))
    target = round(fraction * size)
    in_c = np.zeros(size, dtype=bool)
    taken = 0
    for index in rng.permutation(len(starts)):
        if taken >= target or taken >= size - block:
            break
        rows = order[starts[index] : starts[index] + block]
        in_c[rows] = True
        taken += len(rows)
    return in_c, order


def close_over_units(in_c: np.ndarray, units: np.ndarray) -> np.ndarray:
    """Moves any unit split between C and the rest wholly out of C."""
    for unit in np.unique(units[in_c]):
        members = units == unit
        if not in_c[members].all():
            in_c[members] = False
    return in_c


def embargo_mask(in_c: np.ndarray, order: np.ndarray, rows: int) -> np.ndarray:
    """Rows within `rows` places (in time order) of a `C` row, not in `C`."""
    if rows == 0 or not in_c.any():
        return np.zeros(len(in_c), dtype=bool)
    flags = in_c[order].astype(int)
    near = np.convolve(flags, np.ones(2 * rows + 1, dtype=int), "same") > 0
    mask = np.zeros(len(in_c), dtype=bool)
    mask[order] = near & ~in_c[order]
    return mask


def restrict_folds(
    folds: list[tuple[np.ndarray, np.ndarray]],
    dev_ids: np.ndarray,
    exploration_ids: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Re-expresses folds over the exploration rows only.

    Every exploration row stays in the same validation fold it had before;
    carved-out rows disappear from both sides of every fold.

    Raises:
        ConfigError: If a validation fold would be left empty.
    """
    position = {int(row): index for index, row in enumerate(exploration_ids)}
    mapping = np.array([position.get(int(row), -1) for row in dev_ids])
    restricted = []
    for fit, held in folds:
        kept_fit, kept_held = mapping[fit], mapping[held]
        kept_fit, kept_held = kept_fit[kept_fit >= 0], kept_held[kept_held >= 0]
        if not len(kept_held):
            raise errors.ConfigError(
                "carving the confirmation set left a validation fold empty"
            )
        restricted.append((kept_fit, kept_held))
    return restricted


def check_inputs(
    size: int,
    fraction: float,
    arrays: dict[str, np.ndarray | None],
) -> None:
    """Validates the fraction and that every array is aligned."""
    if not 0 < fraction <= MAX_FRACTION:
        raise ValueError(f"fraction must be in (0, {MAX_FRACTION}]")
    for name, values in arrays.items():
        if values is not None and len(values) != size:
            raise ValueError(f"{name} must have one entry per development row")


def carve_confirmation(
    dev_ids: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    fraction: float,
    seed: int,
    charter_hash: str,
    data_hash: str,
    labels: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    times: np.ndarray | None = None,
    duplicates: np.ndarray | None = None,
    embargo_rows: int = 0,
) -> Carve:
    """Divides the development rows into `X`, `C` and the embargo.

    Args:
        dev_ids: Row ids of the development rows.
        folds: Existing folds as positions within `dev_ids`.
        fraction: Share of development rows to place in `C`, in (0, 0.5].
        seed: Seed from the charter.
        charter_hash: Hash of the charter (part of the pure function).
        data_hash: Hash of the data (part of the pure function).
        labels: Class label per development row (stratifies the draw).
        groups: Group value per development row.
        times: Ordering value per development row; given for temporal
            strategies, it switches the draw to contiguous blocks.
        duplicates: Duplicate-group id per development row (-1 = none).
        embargo_rows: Rows dropped around temporal blocks.

    Returns:
        The division, with folds restricted to `X`.

    Raises:
        ConfigError: If the carve would leave a validation fold empty. This is
            an explicit refusal; `apply_custody` turns it into a run with no
            confirmation set.
    """
    dev_ids = np.asarray(dev_ids)
    size = len(dev_ids)
    check_inputs(
        size,
        fraction,
        {
            "labels": labels,
            "groups": groups,
            "times": times,
            "duplicates": duplicates,
        },
    )
    rng = randomness.component_rng(
        derived_seed(charter_hash, data_hash, seed), COMPONENT
    )
    units = unit_labels(size, groups, duplicates)
    notes: list[str] = []
    order = np.arange(size)
    if times is not None:
        in_c, order = pick_blocks(
            np.asarray(times), fraction, embargo_rows, rng
        )
        in_c = close_over_units(in_c, units)
        embargo = embargo_mask(in_c, order, embargo_rows)
        notes.append(
            f"confirmation drawn as time blocks with an embargo of "
            f"{embargo_rows} rows"
        )
    else:
        strata = (
            np.asarray(labels).astype(str)
            if labels is not None
            else np.full(size, "all")
        )
        first = pd.Series(strata).groupby(units).transform("first").to_numpy()
        in_c = pick_units(units, first, fraction, rng)
        embargo = np.zeros(size, dtype=bool)
    if len(np.unique(units)) < 2 or not in_c.any():
        notes.append("no confirmation set carved: too few units to separate")
        in_c[:] = False
        embargo[:] = False
    exploration = ~in_c & ~embargo
    explored = dev_ids[exploration]
    return Carve(
        exploration_ids=explored,
        confirmation_ids=dev_ids[in_c],
        embargo_ids=dev_ids[embargo],
        folds=restrict_folds(folds, dev_ids, explored),
        notes=tuple(notes),
    )
