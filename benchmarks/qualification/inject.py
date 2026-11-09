"""Seeded defect injection with exact ground truth.

Each injector copies a real table, plants one kind of defect and returns the
mutated copy with a boolean mask of the same shape that marks every cell it
changed (a whole appended row, or a whole added column, counts as changed).
The mask is built from the cells the injector chose, and each chosen cell is
guaranteed to change, so `changed_cells(original, planted.frame)` reproduces
it exactly; the tests compare the two cell by cell.

Randomness comes only from keyed streams (`scmle.core.randomness`), so equal
inputs and seeds give equal output in any process. Plants are synthetic, so
detection rates measured on them are optimistic.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import math
from typing import Any

import numpy
import pandas

from scmle.contracts import qualification
from scmle.core import randomness

MECHANISMS = ("mcar", "mar", "mnar")
LABEL_MODES = ("symmetric", "class_conditional")
LEAK_KINDS = ("copy", "noisy_copy", "missing_for_class")
MAD_SCALE = 1.4826
MAX_LABEL_CLASSES = 10
BISECTION_STEPS = 80
LOGIT_BRACKET = 60.0
TEXT_SENTINEL = "N/A"
MINIMUM_CLASSES = 2


@dataclasses.dataclass(frozen=True, eq=False)
class Injection:
    """A mutated copy of a table and where it was mutated.

    Attributes:
        frame: The table with the defect planted.
        mask: Boolean frame shaped like `frame`; True where a cell was
            changed, appended or added.
        defect_type: Name of the defect, used as the certificate key.
        seed_path: (run seed, component, index) of the random stream.
        info: Parameters and ground truth beyond the mask.
    """

    frame: pandas.DataFrame
    mask: pandas.DataFrame
    defect_type: str
    seed_path: qualification.SeedPath
    info: Mapping[str, Any]

    @property
    def row_mask(self) -> pandas.Series:
        """Rows that hold at least one planted cell."""
        return self.mask.any(axis=1)


def changed_cells(
    original: pandas.DataFrame, mutated: pandas.DataFrame
) -> pandas.DataFrame:
    """Independent before/after diff used to check a mask.

    Args:
        original: The table before injection.
        mutated: The table after injection.

    Returns:
        A boolean frame shaped like `mutated`; True where the value differs
        from `original` (missing on both sides counts as equal), and True
        for every cell of a row or column that `original` lacks.
    """
    index = mutated.index
    known = index.isin(original.index)
    result = pandas.DataFrame(data=False, index=index, columns=mutated.columns)
    for name in mutated.columns:
        if name not in original.columns:
            result[name] = True
            continue
        before = original[name].reindex(index)
        after = mutated[name]
        left = before.to_numpy(dtype=object)
        right = after.to_numpy(dtype=object)
        both_missing = before.isna().to_numpy() & after.isna().to_numpy()
        equal = numpy.array(
            [a == b for a, b in zip(left, right, strict=True)], dtype=bool
        )
        result[name] = ~((equal | both_missing) & known)
    return result


def check_frame(frame: pandas.DataFrame) -> None:
    """Requires a non-empty table with unique rows and column names.

    Args:
        frame: The table to check.

    Raises:
        ValueError: If it is not a non-empty DataFrame, or its column names
            or index labels repeat.
    """
    candidate: object = frame  # callers may pass anything
    if not isinstance(candidate, pandas.DataFrame) or candidate.empty:
        raise ValueError("frame must be a non-empty DataFrame")
    if not frame.columns.is_unique or not frame.index.is_unique:
        raise ValueError("frame needs unique column names and index labels")


def check_columns(frame: pandas.DataFrame, columns: Sequence[str]) -> list[str]:
    """Returns `columns` as a list, requiring that they exist.

    Args:
        frame: The table.
        columns: Column names to require.

    Returns:
        The names as a list, in the given order.

    Raises:
        ValueError: If the list is empty or a name is not a column of `frame`.
    """
    names = list(columns)
    if not names:
        raise ValueError("columns must not be empty")
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"columns not in frame: {missing}")
    return names


def check_rate(rate: float, name: str = "rate") -> float:
    """Requires a rate in (0, 1].

    Args:
        rate: The rate.
        name: Name used in the error message.

    Returns:
        The rate as a float.

    Raises:
        ValueError: If it is outside (0, 1].
    """
    value = qualification.finite(rate, name)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be within (0, 1]")
    return value


def cell_count(rate: float, available: int, what: str) -> int:
    """Number of cells to plant: `round(rate * available)`, at least one.

    Args:
        rate: Share of the available cells to plant.
        available: Number of cells that could be chosen.
        what: Noun for the error message, such as `cell` or `label`.

    Returns:
        The rounded count, capped at `available`.

    Raises:
        ValueError: If the count would be less than one.
    """
    chosen = min(round(rate * available), available)
    if chosen < 1:
        raise ValueError(f"rate is too small: no {what} would be planted")
    return chosen


def is_numeric(series: pandas.Series) -> bool:
    """Whether a column is numeric and not boolean.

    Args:
        series: The column.

    Returns:
        True for a numeric, non-boolean dtype.
    """
    return bool(
        pandas.api.types.is_numeric_dtype(series)
        and not pandas.api.types.is_bool_dtype(series)
    )


def require_numeric(frame: pandas.DataFrame, columns: Sequence[str]) -> None:
    """Requires every named column to be numeric.

    Args:
        frame: The table.
        columns: Names of the columns to check.

    Raises:
        ValueError: If any named column is not numeric (see `is_numeric`).
    """
    bad = [name for name in columns if not is_numeric(frame[name])]
    if bad:
        raise ValueError(f"columns must be numeric: {bad}")


def stream(seed: int, family: str, index: int) -> numpy.random.Generator:
    """Random stream for one injector call.

    Args:
        seed: Run seed.
        family: Injector family, part of the stream key.
        index: Repeat counter for the stream.

    Returns:
        The generator for `inject.<family>` at that index.
    """
    return randomness.component_rng(seed, f"inject.{family}", index)


def blank_mask(frame: pandas.DataFrame) -> pandas.DataFrame:
    """An all-False mask shaped like `frame`.

    Args:
        frame: The table whose shape and labels the mask copies.

    Returns:
        A boolean frame of False values.
    """
    return pandas.DataFrame(
        data=False, index=frame.index, columns=frame.columns
    )


def mark(mask: pandas.DataFrame, column: str, positions: numpy.ndarray) -> None:
    """Sets mask cells at row `positions` of `column`.

    Args:
        mask: The mask, changed in place.
        column: Name of the column to mark.
        positions: Row positions (not labels) to set to True.
    """
    flags = mask[column].to_numpy(dtype=bool, copy=True)
    flags[positions] = True
    mask[column] = flags


def finish(
    frame: pandas.DataFrame,
    mask: pandas.DataFrame,
    defect_type: str,
    family: str,
    seed: int,
    index: int,
    info: Mapping[str, Any],
) -> Injection:
    """Packages an injection with its random-stream key.

    Args:
        frame: The mutated table.
        mask: The mask of changed cells.
        defect_type: Name of the defect, the certificate key.
        family: Injector family, part of the stream key.
        seed: Run seed.
        index: Repeat counter for the stream.
        info: Parameters and ground truth beyond the mask; copied.

    Returns:
        The injection.
    """
    return Injection(
        frame=frame,
        mask=mask,
        defect_type=defect_type,
        seed_path=randomness.seed_path(seed, f"inject.{family}", index),
        info=dict(info),
    )


def other_classes(
    rng: numpy.random.Generator, current: numpy.ndarray, n_classes: int
) -> numpy.ndarray:
    """Uniformly random class indices that differ from `current`.

    Args:
        rng: Random generator.
        current: Current class index of each row.
        n_classes: Number of classes.

    Returns:
        New class indices, never equal to the current ones.
    """
    offset = rng.integers(1, n_classes, size=len(current))
    return (current + offset) % n_classes


def flip_labels(
    frame: pandas.DataFrame,
    target: str,
    rate: float,
    seed: int,
    index: int = 0,
    mode: str = "symmetric",
    class_rates: Mapping[Any, float] | None = None,
) -> Injection:
    """Flips class labels to a different, uniformly chosen class.

    Args:
        frame: The table to copy.
        target: Name of the label column.
        rate: Share of labelled rows to flip (symmetric mode only).
        seed: Run seed.
        index: Repeat counter for the random stream.
        mode: `symmetric` (one rate for all rows) or `class_conditional`.
        class_rates: Per-class flip rate in [0, 1]; used and required by
            `class_conditional`, where `rate` is ignored.

    Returns:
        The injection; `mask[target]` marks the flipped rows.

    Raises:
        ValueError: On a bad mode, rate, column or fewer than two classes.
    """
    check_frame(frame)
    check_columns(frame, [target])
    if mode not in LABEL_MODES:
        raise ValueError(f"mode must be one of {LABEL_MODES}")
    series = frame[target]
    classes = list(pandas.unique(series.dropna()))
    if len(classes) < MINIMUM_CLASSES:
        raise ValueError("flipping needs at least two classes")
    rng = stream(seed, "label_flip", index)
    labelled = numpy.flatnonzero(series.notna().to_numpy())
    if mode == "symmetric":
        check_rate(rate)
        chosen = rng.choice(
            labelled, cell_count(rate, len(labelled), "label"), replace=False
        )
    else:
        if not class_rates:
            raise ValueError("class_conditional mode needs class_rates")
        picks = []
        for label, share in class_rates.items():
            if label not in classes:
                raise ValueError(f"class {label!r} is not in the target")
            if not 0.0 <= qualification.finite(share, "class rate") <= 1.0:
                raise ValueError("class rates must be within [0, 1]")
            members = labelled[(series.iloc[labelled] == label).to_numpy()]
            count = min(round(share * len(members)), len(members))
            picks.append(rng.choice(members, count, replace=False))
        chosen = numpy.concatenate(picks)
        if len(chosen) < 1:
            raise ValueError("class rates are too small: no label flipped")
    chosen = numpy.sort(chosen)
    codes = pandas.Index(classes).get_indexer(series.iloc[chosen])
    new_codes = other_classes(rng, codes, len(classes))
    flipped = series.copy()
    flipped.iloc[chosen] = [classes[code] for code in new_codes]
    out = frame.copy()
    out[target] = flipped
    mask = blank_mask(out)
    mark(mask, target, chosen)
    return finish(
        out,
        mask,
        "label_flip",
        "label_flip",
        seed,
        index,
        {"mode": mode, "target": target, "flipped": len(chosen)},
    )


def standardized(values: pandas.Series) -> numpy.ndarray:
    """Z-scores with missing values set to zero.

    Args:
        values: The column.

    Returns:
        The standardized values as an array.

    Raises:
        ValueError: If the column has no spread (zero or undefined standard
            deviation).
    """
    numbers = values.astype(float)
    spread = float(numbers.std())
    if not math.isfinite(spread) or spread == 0.0:
        raise ValueError("driver column has no spread")
    z = (numbers - float(numbers.mean())) / spread
    return z.fillna(0.0).to_numpy()


def calibrated_probabilities(
    z: numpy.ndarray, strength: float, rate: float
) -> numpy.ndarray:
    """Logistic selection probabilities whose mean equals `rate`.

    Args:
        z: Standardized driver values.
        strength: Logit slope per standard deviation.
        rate: Target mean probability.

    Returns:
        The probabilities `1 / (1 + exp(-(strength * z + b)))`, with the
        intercept `b` found by bisection so that the mean is `rate`.
    """
    low, high = -LOGIT_BRACKET, LOGIT_BRACKET
    for _ in range(BISECTION_STEPS):
        middle = (low + high) / 2.0
        mean = float(
            numpy.mean(1.0 / (1.0 + numpy.exp(-(strength * z + middle))))
        )
        if mean < rate:
            low = middle
        else:
            high = middle
    return 1.0 / (1.0 + numpy.exp(-(strength * z + (low + high) / 2.0)))


def missing_strength(mechanism: str, strength: float) -> float:
    """Validates the mechanism and the logit slope of a missingness model.

    Args:
        mechanism: `mcar`, `mar` or `mnar`.
        strength: Logit slope per standard deviation.

    Returns:
        The slope as a finite float.

    Raises:
        ValueError: On an invalid mechanism or a zero slope for MAR or MNAR.
    """
    if mechanism not in MECHANISMS:
        raise ValueError(f"mechanism must be one of {MECHANISMS}")
    scale = qualification.finite(strength, "strength")
    if mechanism != "mcar" and scale == 0.0:
        raise ValueError("strength must be nonzero for MAR and MNAR")
    return scale


def check_missing_driver(
    frame: pandas.DataFrame,
    names: Sequence[str],
    mechanism: str,
    driver: str | None,
) -> None:
    """Validates the driver column of a missingness model.

    Args:
        frame: The table to copy.
        names: The columns to delete from.
        mechanism: `mcar`, `mar` or `mnar`.
        driver: Numeric column MAR depends on.

    Raises:
        ValueError: If MAR lacks a usable driver, another mechanism has one,
            or a column the mechanism reads is not numeric.
    """
    if mechanism == "mar":
        if driver is None or driver not in frame.columns:
            raise ValueError("MAR needs a driver column in the frame")
        if driver in names:
            raise ValueError("the driver must not be one of the columns")
        require_numeric(frame, [driver])
    elif driver is not None:
        raise ValueError("only MAR takes a driver")
    if mechanism == "mnar":
        require_numeric(frame, names)


def inject_missing(
    frame: pandas.DataFrame,
    columns: Sequence[str],
    rate: float,
    seed: int,
    index: int = 0,
    mechanism: str = "mcar",
    driver: str | None = None,
    strength: float = 1.5,
) -> Injection:
    """Deletes observed cells under a known missingness mechanism.

    MCAR deletes exactly `round(rate * observed)` cells uniformly. MAR and
    MNAR delete each observed cell with a logistic probability in
    `strength * z + b`, where `z` is the standardized driver column (MAR)
    or the cell's own column (MNAR) and `b` is chosen so the expected share
    equals `rate`; the realized count is random.

    Args:
        frame: The table to copy.
        columns: Columns to delete from.
        rate: Target share of each column's observed cells.
        seed: Run seed.
        index: Repeat counter for the random stream.
        mechanism: `mcar`, `mar` or `mnar`.
        driver: Numeric column MAR depends on; must not be in `columns`.
        strength: Logit slope per standard deviation, nonzero.

    Returns:
        The injection; `mask` marks the cells that became missing.

    Raises:
        ValueError: On an invalid mechanism, driver, column or rate.
    """
    check_frame(frame)
    names = check_columns(frame, columns)
    check_rate(rate)
    scale = missing_strength(mechanism, strength)
    check_missing_driver(frame, names, mechanism, driver)
    rng = stream(seed, "missing", index)
    out = frame.copy()
    mask = blank_mask(out)
    for name in names:
        observed = numpy.flatnonzero(frame[name].notna().to_numpy())
        if mechanism == "mcar":
            chosen = rng.choice(
                observed,
                cell_count(rate, len(observed), "cell"),
                replace=False,
            )
        else:
            source = frame[driver] if mechanism == "mar" else frame[name]
            z = standardized(source)[observed]
            probability = calibrated_probabilities(z, scale, rate)
            chosen = observed[rng.random(len(observed)) < probability]
            if len(chosen) < 1:
                raise ValueError("no cell was selected; raise the rate")
        chosen = numpy.sort(chosen)
        series = out[name]
        if pandas.api.types.is_integer_dtype(series):
            series = series.astype("float64")
        series = series.copy()
        series.iloc[chosen] = numpy.nan
        out[name] = series
        mark(mask, name, chosen)
    return finish(
        out,
        mask,
        f"missing_{mechanism}",
        "missing",
        seed,
        index,
        {
            "mechanism": mechanism,
            "driver": driver,
            "strength": scale,
            "rate": rate,
            "cells": int(mask.to_numpy().sum()),
        },
    )


def inject_sentinels(
    frame: pandas.DataFrame,
    columns: Sequence[str],
    rate: float,
    seed: int,
    index: int = 0,
    sentinel: int = -999,
    text_sentinel: str = TEXT_SENTINEL,
) -> Injection:
    """Overwrites observed cells with a code that stands in for missing.

    Numeric columns receive `sentinel`, text columns `text_sentinel`. Cells
    that already hold the code are not candidates, so every chosen cell
    really changes.

    Args:
        frame: The table to copy.
        columns: Numeric or text columns.
        rate: Share of each column's candidate cells to overwrite.
        seed: Run seed.
        index: Repeat counter for the random stream.
        sentinel: Numeric code, such as -999.
        text_sentinel: Text code, such as "N/A".

    Returns:
        The injection; `mask` marks the overwritten cells.

    Raises:
        ValueError: On an unsupported column type or a code that does not
            fit an unsigned integer column.
    """
    check_frame(frame)
    names = check_columns(frame, columns)
    check_rate(rate)
    rng = stream(seed, "sentinel", index)
    out = frame.copy()
    mask = blank_mask(out)
    for name in names:
        series = out[name]
        if is_numeric(series):
            if (
                pandas.api.types.is_unsigned_integer_dtype(series)
                and sentinel < 0
            ):
                raise ValueError(
                    f"'{name}' is unsigned; use a non-negative code"
                )
            code: Any = (
                int(sentinel)
                if pandas.api.types.is_integer_dtype(series)
                else float(sentinel)
            )
        elif pandas.api.types.is_string_dtype(series) or series.dtype == object:
            code = text_sentinel
        else:
            raise ValueError(f"'{name}' has an unsupported type for sentinels")
        candidates = numpy.flatnonzero(
            (series.notna() & (series != code)).to_numpy()
        )
        chosen = numpy.sort(
            rng.choice(
                candidates,
                cell_count(rate, len(candidates), "cell"),
                replace=False,
            )
        )
        changed = series.copy()
        changed.iloc[chosen] = code
        out[name] = changed
        mark(mask, name, chosen)
    return finish(
        out,
        mask,
        "sentinel",
        "sentinel",
        seed,
        index,
        {"sentinel": sentinel, "text_sentinel": text_sentinel},
    )


def has_default_index(frame: pandas.DataFrame) -> bool:
    """Whether the index is exactly 0..n-1.

    Args:
        frame: The table.

    Returns:
        True for a `RangeIndex` that starts at 0 with step 1.
    """
    index = frame.index
    return (
        isinstance(index, pandas.RangeIndex)
        and index.start == 0
        and index.step == 1
    )


def inject_duplicates(
    frame: pandas.DataFrame,
    rate: float,
    seed: int,
    index: int = 0,
    near: bool = False,
    scale: float = 1e-3,
) -> Injection:
    """Appends copies of existing rows.

    Args:
        frame: The table to copy; its index must be `RangeIndex(n)`.
        rate: Appended rows as a share of the original rows.
        seed: Run seed.
        index: Repeat counter for the random stream.
        near: If True, jitter float columns by `scale` standard deviations
            so the copies are close but not equal.
        scale: Jitter size in column standard deviations, positive.

    Returns:
        The injection; `mask` marks every cell of each appended row, and
        `info["source_rows"]` lists the source row of each copy.

    Raises:
        ValueError: If the index is not `RangeIndex(n)`, or near copies are
            requested without float columns that have spread.
    """
    check_frame(frame)
    if not has_default_index(frame):
        raise ValueError("duplicates need a default RangeIndex; reset_index")
    check_rate(rate)
    rng = stream(seed, "duplicate", index)
    rows = len(frame)
    count = cell_count(rate, rows, "row")
    sources = rng.choice(rows, count, replace=False)
    copies = frame.iloc[sources].reset_index(drop=True)
    if near:
        jitter = qualification.finite(scale, "scale")
        if jitter <= 0.0:
            raise ValueError("scale must be positive")
        floats = [
            name
            for name in frame.columns
            if pandas.api.types.is_float_dtype(frame[name])
        ]
        if not floats:
            raise ValueError("near duplicates need float columns")
        for name in floats:
            spread = float(frame[name].std())
            if not math.isfinite(spread) or spread == 0.0:
                raise ValueError(f"'{name}' has no spread to jitter")
            noise = rng.normal(size=count)
            noise[noise == 0.0] = 1.0
            copies[name] = copies[name] + noise * jitter * spread
    out = pandas.concat([frame, copies], ignore_index=True)
    mask = blank_mask(out)
    mask.iloc[rows:, :] = True
    return finish(
        out,
        mask,
        "duplicate_near" if near else "duplicate_exact",
        "duplicate",
        seed,
        index,
        {"near": near, "source_rows": tuple(int(s) for s in sources)},
    )


def robust_scale(values: numpy.ndarray) -> float:
    """Normal-consistent median absolute deviation; std if that is zero.

    Args:
        values: The sample.

    Returns:
        The scaled median absolute deviation, or the standard deviation when
        the deviation is zero.
    """
    centre = float(numpy.median(values))
    mad = MAD_SCALE * float(numpy.median(numpy.abs(values - centre)))
    if mad > 0.0:
        return mad
    return float(numpy.std(values))


def inject_outliers(
    frame: pandas.DataFrame,
    columns: Sequence[str],
    rate: float,
    seed: int,
    index: int = 0,
    magnitude: float = 8.0,
) -> Injection:
    """Moves observed cells far from the column centre.

    Each chosen cell moves by `magnitude` robust scales (normal-consistent
    MAD) in a random direction.

    Args:
        frame: The table to copy.
        columns: Numeric columns; integer columns become float.
        rate: Share of each column's observed cells to move.
        seed: Run seed.
        index: Repeat counter for the random stream.
        magnitude: Distance in robust scales, positive.

    Returns:
        The injection; `mask` marks the moved cells.

    Raises:
        ValueError: If a column is not numeric or has no spread.
    """
    check_frame(frame)
    names = check_columns(frame, columns)
    require_numeric(frame, names)
    check_rate(rate)
    distance = qualification.finite(magnitude, "magnitude")
    if distance <= 0.0:
        raise ValueError("magnitude must be positive")
    rng = stream(seed, "outlier", index)
    out = frame.copy()
    mask = blank_mask(out)
    for name in names:
        values = out[name].astype("float64")
        observed = numpy.flatnonzero(values.notna().to_numpy())
        spread = robust_scale(values.to_numpy()[observed])
        if not math.isfinite(spread) or spread == 0.0:
            raise ValueError(f"'{name}' has no spread")
        chosen = numpy.sort(
            rng.choice(
                observed,
                cell_count(rate, len(observed), "cell"),
                replace=False,
            )
        )
        sign = rng.choice([-1.0, 1.0], size=len(chosen))
        array = values.to_numpy(dtype=float, copy=True)
        array[chosen] = array[chosen] + sign * distance * spread
        out[name] = pandas.Series(array, index=values.index)
        mark(mask, name, chosen)
    return finish(
        out,
        mask,
        "outlier",
        "outlier",
        seed,
        index,
        {"magnitude": distance},
    )


def is_classification(series: pandas.Series) -> bool:
    """Whether a target should be treated as class labels.

    Args:
        series: The target column.

    Returns:
        True for a non-numeric column or one with at most
        `MAX_LABEL_CLASSES` distinct values.
    """
    return (not is_numeric(series)) or series.nunique() <= MAX_LABEL_CLASSES


def noisy_leak(
    series: pandas.Series, rng: numpy.random.Generator, noise: float
) -> tuple[pandas.Series, float]:
    """A noisy copy of the target: flipped classes or added Gaussian noise.

    Args:
        series: The target column.
        rng: Random generator.
        noise: Flip probability in (0, 1) for classes, or standard
            deviations of Gaussian noise for a continuous target.

    Returns:
        The noisy column and the finite noise level.

    Raises:
        ValueError: On a bad noise level, too few classes or a target
            without spread.
    """
    level = qualification.finite(noise, "noise")
    if is_classification(series):
        if not 0.0 < level < 1.0:
            raise ValueError("noise must be within (0, 1) for classes")
        classes = list(pandas.unique(series.dropna()))
        if len(classes) < MINIMUM_CLASSES:
            raise ValueError("a noisy label copy needs two classes")
        codes = pandas.Index(classes).get_indexer(series)
        flip = rng.random(len(series)) < level
        flip &= codes >= 0
        new_codes = codes.copy()
        new_codes[flip] = other_classes(rng, codes[flip], len(classes))
        added = series.copy()
        added.iloc[numpy.flatnonzero(flip)] = [
            classes[code] for code in new_codes[flip]
        ]
        return added, level
    if level <= 0.0:
        raise ValueError("noise must be positive")
    spread = float(series.std())
    if not math.isfinite(spread) or spread == 0.0:
        raise ValueError("target has no spread")
    added = series.astype("float64") + rng.normal(
        scale=level * spread, size=len(series)
    )
    return added, level


def class_gated_leak(
    series: pandas.Series, rng: numpy.random.Generator, missing_class: Any
) -> tuple[pandas.Series, Any]:
    """Random values recorded only when the target is not `missing_class`.

    Args:
        series: The target column.
        rng: Random generator.
        missing_class: Class whose rows are left blank; `None` means the
            first class seen.

    Returns:
        The new column and the class whose rows are blank.

    Raises:
        ValueError: If the target has fewer than two classes or lacks the
            class.
    """
    classes = list(pandas.unique(series.dropna()))
    if len(classes) < MINIMUM_CLASSES:
        raise ValueError("missing_for_class needs at least two classes")
    blank = classes[0] if missing_class is None else missing_class
    if blank not in classes:
        raise ValueError(f"class {blank!r} is not in the target")
    added = pandas.Series(
        rng.normal(size=len(series)), index=series.index, dtype="float64"
    )
    added[(series == blank).to_numpy()] = numpy.nan
    return added, blank


def inject_leak(
    frame: pandas.DataFrame,
    target: str,
    kind: str,
    seed: int,
    index: int = 0,
    name: str | None = None,
    noise: float = 0.1,
    missing_class: Any = None,
) -> Injection:
    """Adds a column that carries the target.

    Kinds: `copy` (the target itself); `noisy_copy` (class labels flipped
    with probability `noise`, or Gaussian noise of `noise` standard
    deviations for a continuous target); `missing_for_class` (random values
    that are recorded only when the target is not `missing_class`, so being
    recorded reveals the outcome, as with a Titanic lifeboat number).

    Args:
        frame: The table to copy.
        target: Name of the label column.
        kind: One of `LEAK_KINDS`.
        seed: Run seed.
        index: Repeat counter for the random stream.
        name: Name of the new column; defaults to `leak_<kind>`.
        noise: Noise level in (0, 1) for classes, positive for a continuous
            target.
        missing_class: Class whose rows are left blank; defaults to the
            first class seen.

    Returns:
        The injection; `mask` marks the whole new column.

    Raises:
        ValueError: On an unknown kind, a name clash or a bad target.
    """
    check_frame(frame)
    check_columns(frame, [target])
    if kind not in LEAK_KINDS:
        raise ValueError(f"kind must be one of {LEAK_KINDS}")
    column = name or f"leak_{kind}"
    if column in frame.columns:
        raise ValueError(f"column '{column}' already exists")
    series = frame[target]
    rng = stream(seed, "leak", index)
    info: dict[str, Any] = {"column": column, "kind": kind}
    if kind == "copy":
        added = series.copy()
    elif kind == "noisy_copy":
        added, info["noise"] = noisy_leak(series, rng, noise)
    else:
        added, info["missing_class"] = class_gated_leak(
            series, rng, missing_class
        )
    out = frame.copy()
    out[column] = added
    mask = blank_mask(out)
    mask[column] = True
    return finish(out, mask, f"leak_{kind}", "leak", seed, index, info)


def shift_covariates(
    frame: pandas.DataFrame,
    columns: Sequence[str],
    fraction: float,
    delta: float,
    seed: int,
    index: int = 0,
) -> Injection:
    """Shifts numeric columns in a random share of rows.

    Used to build a second set that differs from the first.

    Args:
        frame: The table to copy.
        columns: Numeric columns; integer columns become float.
        fraction: Share of rows to shift.
        delta: Shift in standard deviations of the whole column, nonzero.
        seed: Run seed.
        index: Repeat counter for the random stream.

    Returns:
        The injection; `mask` marks shifted observed cells and
        `info["rows"]` lists the shifted rows.

    Raises:
        ValueError: On a non-numeric column, zero delta or no spread.
    """
    check_frame(frame)
    names = check_columns(frame, columns)
    require_numeric(frame, names)
    check_rate(fraction, "fraction")
    shift = qualification.finite(delta, "delta")
    if shift == 0.0:
        raise ValueError("delta must be nonzero")
    rng = stream(seed, "covariate_shift", index)
    rows = numpy.sort(
        rng.choice(
            len(frame), cell_count(fraction, len(frame), "row"), replace=False
        )
    )
    out = frame.copy()
    mask = blank_mask(out)
    for name in names:
        values = out[name].astype("float64")
        spread = float(values.std())
        if not math.isfinite(spread) or spread == 0.0:
            raise ValueError(f"'{name}' has no spread")
        observed = rows[values.iloc[rows].notna().to_numpy()]
        array = values.to_numpy(dtype=float, copy=True)
        array[observed] = array[observed] + shift * spread
        out[name] = pandas.Series(array, index=values.index)
        mark(mask, name, observed)
    return finish(
        out,
        mask,
        "covariate_shift",
        "covariate_shift",
        seed,
        index,
        {"delta": shift, "rows": tuple(int(r) for r in rows)},
    )


CATALOGUE: Mapping[str, Callable[..., Injection]] = {
    "label_flip": flip_labels,
    "missing": inject_missing,
    "sentinel": inject_sentinels,
    "duplicate": inject_duplicates,
    "outlier": inject_outliers,
    "leak": inject_leak,
    "covariate_shift": shift_covariates,
}
