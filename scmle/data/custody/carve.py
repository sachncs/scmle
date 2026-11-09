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

import numpy
import pandas

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

    exploration_ids: numpy.ndarray
    confirmation_ids: numpy.ndarray
    embargo_ids: numpy.ndarray
    folds: list[tuple[numpy.ndarray, numpy.ndarray]]
    notes: tuple[str, ...]


def derived_seed(charter_hash: str, data_hash: str, seed: int) -> int:
    """Derives the run seed from the data, the charter and the seed.

    Args:
        charter_hash: Hash of the charter.
        data_hash: Hash of the data.
        seed: Seed from the charter.

    Returns:
        A 32-bit integer taken from the SHA-256 of all three inputs.
    """
    digest = hashlib.sha256(
        f"{data_hash}|{charter_hash}|{seed}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big")


def unit_labels(
    size: int, groups: numpy.ndarray | None, duplicates: numpy.ndarray | None
) -> numpy.ndarray:
    """Labels each row with its unit: the smallest row index in its unit.

    Rows sharing a group value or a duplicate-group id are merged
    transitively, so a group that contains a copy of a record in another
    group pulls both groups into one unit.

    Args:
        size: Number of rows.
        groups: Group value per row, or None for no grouping.
        duplicates: Duplicate-group id per row (`NO_DUPLICATE` = none), or
            None for no duplicate information.

    Returns:
        An integer array of length `size`: the unit label of each row.
    """
    parent = numpy.arange(size)

    def find(index: int) -> int:
        """Finds the root of a row's unit, compressing the path.

        Args:
            index: Row position.

        Returns:
            The position of the unit's root row.
        """
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(a: int, b: int) -> None:
        """Merges the units of two rows, keeping the smaller root.

        Args:
            a: Position of the first row.
            b: Position of the second row.
        """
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    for keys, none in ((groups, None), (duplicates, NO_DUPLICATE)):
        if keys is None:
            continue
        codes = pandas.factorize(numpy.asarray(keys))[0]
        if none is not None:
            codes = numpy.where(numpy.asarray(keys) == none, -1, codes)
        first: dict[int, int] = {}
        for index, raw in enumerate(codes):
            code = int(raw)
            if code < 0:
                continue
            if code in first:
                union(first[code], index)
            else:
                first[code] = index
    return numpy.array([find(i) for i in range(size)])


def pick_units(
    units: numpy.ndarray,
    strata: numpy.ndarray,
    fraction: float,
    rng: numpy.random.Generator,
) -> numpy.ndarray:
    """Marks rows of randomly chosen whole units, stratum by stratum.

    In each stratum units are taken in random order until about `fraction` of
    the stratum's rows are marked; at least one unit is always left unmarked.

    Args:
        units: Unit label per row.
        strata: Stratum label per row.
        fraction: Share of each stratum's rows to mark.
        rng: Random generator that orders the units.

    Returns:
        A boolean mask over the rows, True for the marked rows.
    """
    chosen = numpy.zeros(len(units), dtype=bool)
    for stratum in sorted(set(strata.tolist())):
        in_stratum = strata == stratum
        stratum_units = numpy.unique(units[in_stratum])
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
    times: numpy.ndarray,
    fraction: float,
    embargo: int,
    rng: numpy.random.Generator,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Marks contiguous time blocks as `C`.

    Args:
        times: Ordering value per row.
        fraction: Share of rows to mark.
        embargo: Embargo in rows; blocks are at least `4 * embargo` rows
            long (and at least a twentieth of the rows).
        rng: Random generator that orders the blocks.

    Returns:
        A pair of the boolean mask of marked rows and the row positions in
        stable time order.
    """
    size = len(times)
    order = numpy.argsort(times, kind="stable")
    block = max(4 * embargo, math.ceil(size / BLOCKS), 1)
    starts = list(range(0, size, block))
    target = round(fraction * size)
    in_c = numpy.zeros(size, dtype=bool)
    taken = 0
    for index in rng.permutation(len(starts)):
        if taken >= target or taken >= size - block:
            break
        rows = order[starts[index] : starts[index] + block]
        in_c[rows] = True
        taken += len(rows)
    return in_c, order


def close_over_units(
    in_c: numpy.ndarray, units: numpy.ndarray
) -> numpy.ndarray:
    """Moves any unit split between C and the rest wholly out of C.

    Args:
        in_c: Boolean mask of rows in `C`; modified in place.
        units: Unit label per row.

    Returns:
        The same `in_c` array after the update.
    """
    for unit in numpy.unique(units[in_c]):
        members = units == unit
        if not in_c[members].all():
            in_c[members] = False
    return in_c


def embargo_mask(
    in_c: numpy.ndarray, order: numpy.ndarray, rows: int
) -> numpy.ndarray:
    """Finds rows within `rows` places (in time order) of a `C` row.

    Args:
        in_c: Boolean mask of rows in `C`.
        order: Row positions in time order.
        rows: Embargo length in rows on each side of a `C` row.

    Returns:
        A boolean mask of the rows to drop, which excludes the `C` rows
        themselves. All False when `rows` is 0 or `C` is empty.
    """
    if rows == 0 or not in_c.any():
        return numpy.zeros(len(in_c), dtype=bool)
    flags = in_c[order].astype(int)
    spread = numpy.convolve(flags, numpy.ones(2 * rows + 1, dtype=int), "full")
    near = spread[rows : rows + len(flags)] > 0
    mask = numpy.zeros(len(in_c), dtype=bool)
    mask[order] = near & ~in_c[order]
    return mask


def autocorrelation(values: numpy.ndarray, max_lag: int) -> numpy.ndarray:
    """Computes the sample autocorrelation (biased, mean-removed).

    Args:
        values: Numeric series.
        max_lag: Largest lag.

    Returns:
        An array of length `max_lag` with the autocorrelation at lags
        1 to `max_lag`; zeros for a constant series.
    """
    series = numpy.asarray(values, dtype=float)
    series = series - series.mean()
    variance = float(series @ series)
    if variance == 0.0:
        return numpy.zeros(max_lag)
    return numpy.array(
        [
            float(series[:-lag] @ series[lag:]) / variance
            for lag in range(1, max_lag + 1)
        ]
    )


def acf_embargo(values: numpy.ndarray, max_lag: int | None = None) -> int:
    """Embargo length from the measured serial dependence of a series.

    `values` must be in time order. The embargo is the number of leading
    lags whose autocorrelation lies outside the large-sample 95 percent band
    for white noise, +-1.96/sqrt(n): the first lag inside the band ends it.
    Dependence that returns at a later lag (a seasonal peak) is not counted,
    and a series with no significant lag-1 autocorrelation measures 0. The
    band is a rule of thumb for a stationary series, not a guarantee that
    rows beyond the embargo are independent.

    Args:
        values: Numeric series in time order.
        max_lag: Largest lag examined; defaults to a quarter of the length.
            Either way it is capped at half the length and at 1000.

    Returns:
        The embargo in rows (0 for fewer than 8 values or a constant series).
        When every examined lag is outside the band, the largest lag
        examined.
    """
    size = len(values)
    if size < 8:
        return 0
    limit = min(size // 4 if max_lag is None else max_lag, size // 2, 1000)
    band = 1.96 / math.sqrt(size)
    inside = numpy.abs(autocorrelation(values, limit)) <= band
    return int(numpy.argmax(inside)) if inside.any() else limit


def restrict_folds(
    folds: list[tuple[numpy.ndarray, numpy.ndarray]],
    dev_ids: numpy.ndarray,
    exploration_ids: numpy.ndarray,
) -> list[tuple[numpy.ndarray, numpy.ndarray]]:
    """Re-expresses folds over the exploration rows only.

    Every exploration row stays in the same validation fold it had before;
    carved-out rows disappear from both sides of every fold.

    Args:
        folds: Folds as `(fit, validation)` positions within `dev_ids`.
        dev_ids: Row ids of the development rows.
        exploration_ids: Row ids that remain, in their new order.

    Returns:
        The folds as positions within `exploration_ids`.

    Raises:
        ConfigError: If a validation fold would be left empty.
    """
    position = {int(row): index for index, row in enumerate(exploration_ids)}
    mapping = numpy.array([position.get(int(row), -1) for row in dev_ids])
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
    arrays: dict[str, numpy.ndarray | None],
) -> None:
    """Validates the fraction and that every array is aligned.

    Args:
        size: Number of development rows.
        fraction: Requested share for `C`.
        arrays: Named optional arrays, each expected to have `size` entries.

    Raises:
        ValueError: If `fraction` is outside (0, `MAX_FRACTION`] or a given
            array does not have `size` entries.
    """
    if not 0 < fraction <= MAX_FRACTION:
        raise ValueError(f"fraction must be in (0, {MAX_FRACTION}]")
    for name, values in arrays.items():
        if values is not None and len(values) != size:
            raise ValueError(f"{name} must have one entry per development row")


def carve_confirmation(
    dev_ids: numpy.ndarray,
    folds: list[tuple[numpy.ndarray, numpy.ndarray]],
    *,
    fraction: float,
    seed: int,
    charter_hash: str,
    data_hash: str,
    labels: numpy.ndarray | None = None,
    groups: numpy.ndarray | None = None,
    times: numpy.ndarray | None = None,
    duplicates: numpy.ndarray | None = None,
    embargo_rows: int = 0,
) -> Carve:
    """Divides the development rows into `X`, `C` and the embargo.

    Errors raised by the helpers propagate. `check_inputs` raises ValueError
    when `fraction` is outside (0, 0.5] or an array does not have one entry
    per development row. `restrict_folds` raises ConfigError when the carve
    would leave a validation fold empty; `apply_custody` turns that refusal
    into a run with no confirmation set.

    Args:
        dev_ids: Row ids of the development rows.
        folds: Existing folds as positions within `dev_ids`.
        fraction: Share of development rows to place in `C`, in (0, 0.5].
        seed: Seed from the charter.
        charter_hash: Hash of the charter (part of the pure function).
        data_hash: Hash of the data (part of the pure function).
        labels: Class label per development row (stratifies the draw of
            whole units; not used when `times` is given).
        groups: Group value per development row.
        times: Ordering value per development row; given for temporal
            strategies, it switches the draw to contiguous blocks.
        duplicates: Duplicate-group id per development row (-1 = none).
        embargo_rows: Rows dropped around temporal blocks (used only when
            `times` is given).

    Returns:
        The division, with folds restricted to `X`. When fewer than two
        units exist or nothing was drawn, `C` and the embargo are empty,
        every development row is in `X` and a note says so.
    """
    dev_ids = numpy.asarray(dev_ids)
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
    order = numpy.arange(size)
    if times is not None:
        in_c, order = pick_blocks(
            numpy.asarray(times), fraction, embargo_rows, rng
        )
        in_c = close_over_units(in_c, units)
        embargo = embargo_mask(in_c, order, embargo_rows)
        notes.append(
            f"confirmation drawn as time blocks with an embargo of "
            f"{embargo_rows} rows"
        )
    else:
        strata = (
            numpy.asarray(labels).astype(str)
            if labels is not None
            else numpy.full(size, "all")
        )
        first = (
            pandas.Series(strata).groupby(units).transform("first").to_numpy()
        )
        in_c = pick_units(units, first, fraction, rng)
        embargo = numpy.zeros(size, dtype=bool)
    if len(numpy.unique(units)) < 2 or not in_c.any():
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
