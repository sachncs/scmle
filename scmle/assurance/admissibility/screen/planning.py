"""Row choice, the certificate store and the run wiring of a screen."""

import dataclasses
import functools
import math
from pathlib import Path

import numpy
import pandas

from scmle.assurance.admissibility import leakage
from scmle.assurance.admissibility.screen.report import halt
from scmle.assurance.admissibility.screen.report import Screen
from scmle.assurance.admissibility.screen.settings import MAX_ROWS
from scmle.assurance.admissibility.screen.settings import MECHANISM_DEFECTS
from scmle.assurance.admissibility.screen.settings import STORE_PATH
from scmle.assurance.admissibility.screen.settings import TIME_BLOCKS
from scmle.assurance.detectors import certificates
from scmle.runtime import randomness


def subsample(
    n: int,
    groups: numpy.ndarray | None,
    block_size: int | None,
    seed: int,
) -> tuple[numpy.ndarray, int | None]:
    """Row positions to use, at most about `MAX_ROWS`, and the block size.

    The choice does not look at the outcome. Groups are kept whole, time
    blocks keep their order (every k-th row), otherwise rows are drawn at
    random.

    Args:
        n: Number of rows available.
        groups: Group label per row, or None.
        block_size: Consecutive rows per time block, or None.
        seed: Run seed of the draw.

    Returns:
        A pair (sorted row positions, block size to use). Everything is
        kept when `n` is at most `MAX_ROWS`. With groups, whole groups are
        drawn until `MAX_ROWS` rows are reached; with time blocks, every
        k-th row is kept and the block size shrinks by k (not below 2).
    """
    if n <= MAX_ROWS:
        return numpy.arange(n), block_size
    rng = randomness.component_rng(seed, "leakage.screen.rows", 0)
    if groups is not None:
        labels = pandas.unique(groups)
        chosen: list[numpy.ndarray] = []
        size = 0
        for label in labels[rng.permutation(len(labels))]:
            where = numpy.flatnonzero(groups == label)
            chosen.append(where)
            size += len(where)
            if size >= MAX_ROWS:
                break
        return numpy.sort(numpy.concatenate(chosen)), block_size
    if block_size is not None:
        stride = math.ceil(n / MAX_ROWS)
        return numpy.arange(0, n, stride), max(2, block_size // stride)
    return numpy.sort(rng.choice(n, MAX_ROWS, replace=False)), block_size


def load_store(
    path: Path | None = None,
) -> certificates.Store | None:
    """Loads the certificates of the screen.

    Args:
        path: Certificate file; defaults to the one shipped with the
            package.

    Returns:
        The store, or None if the file is missing or unreadable.
    """
    target = path or STORE_PATH
    if not target.is_file():
        return None
    try:
        return certificates.Store.load(target)
    except (ValueError, KeyError, OSError):
        # handled: documented None for missing or unreadable store
        return None


@functools.cache
def bundled() -> certificates.Store | None:
    """Loads the certificates shipped with the package, once.

    Returns:
        The store, or None if it is missing or unreadable. The result is
        cached for the process.
    """
    return load_store()


@dataclasses.dataclass(frozen=True)
class Demand:
    """What one scan will ask for, counted before any permutation.

    Attributes:
        rows: Rows the scan sees.
        family: Hypotheses adjusted together.
        permutations: Permutations it will draw.
        skill: Columns with a skill hypothesis.
        work: Cross-fitted fits.
        columns: Columns with a hypothesis, in order (`unchecked` if the
            scan is declined).
        seconds: Predicted seconds (`cost`).
    """

    rows: int
    family: int
    permutations: int
    skill: int
    work: int
    columns: tuple[str, ...]
    seconds: float


def decline(
    cfg: leakage.Config, demand: Demand, limit: float, budget_seconds: float
) -> Screen:
    """Builds the report of a real scan that does not fit its share.

    Args:
        cfg: Settings.
        demand: What the scan would ask for.
        limit: Seconds it may take.
        budget_seconds: The declared time budget.

    Returns:
        A report listing every tested column as unchecked.
    """
    return halt(
        cfg,
        dict.fromkeys(MECHANISM_DEFECTS, certificates.Authority.GATING),
        demand.rows,
        demand.family,
        (
            f"{demand.family} hypotheses need about {demand.work} "
            f"cross-fitted fits ({demand.permutations} permutations for "
            f"{demand.skill} columns on {demand.rows} rows), predicted "
            f"{demand.seconds:.1f} s; the limit is {limit:.1f} s. Nothing "
            "was flagged (power, not size, is lost): screen fewer columns"
        ),
        demand.columns,
        f"work limit for a {budget_seconds:g} s time budget",
    )


def arrange(
    frame: pandas.DataFrame, group_column: str | None, time_column: str | None
) -> tuple[pandas.DataFrame, pandas.Series | None, int | None]:
    """Orders the rows for screening and fixes their structure.

    Groups keep entities whole. A time column orders the rows (stable sort)
    and defines consecutive blocks of `len(frame) // TIME_BLOCKS` rows, at
    least 2; groups win when both are given.

    Args:
        frame: Training rows including any structure columns.
        group_column: Entity column, or None. Ignored when it is not a
            column of `frame`.
        time_column: Time column, or None. Ignored when it is not a column
            of `frame` or when `group_column` is used.

    Returns:
        The rows in screening order, their group labels (or None) and the
        rows per time block (or None).
    """
    groups = (
        frame[group_column].astype(str)
        if group_column and group_column in frame.columns
        else None
    )
    block_size = None
    order: numpy.ndarray = numpy.arange(len(frame))
    if groups is None and time_column and time_column in frame.columns:
        order = frame[time_column].argsort(kind="stable").to_numpy()
        block_size = max(2, len(frame) // TIME_BLOCKS)
    grouping = None if groups is None else groups.iloc[order]
    return frame.iloc[order], grouping, block_size


def disclose(
    report: Screen,
    found: dict[str, bool],
    published: str,
    left: float | None = None,
    limit: float | None = None,
) -> Screen:
    """Attaches what the probe found and the published evidence to a report.

    Args:
        report: The real screen's report.
        found: Leak type to whether its plant was found, from `probe`;
            empty when the planted copy did not fit the time left.
        published: The sentence from `evidence`.
        left: Seconds that were left of the screen's share for the probe,
            or None for no cap.
        limit: The screen's share of the time budget, or None for no cap.

    Returns:
        The report with `probe` set and two notes added: the published
        evidence, and the planted leaks that were not found, if any. When
        the probe did not run, `probe` is not set and the second note says
        why.
    """
    if not found:
        reason = (
            "probe not run: the planted copy did not fit the "
            f"{max(left or 0.0, 0.0):.1f} s left of the screen's "
            f"{limit or 0.0:.1f} s share of the time budget"
        )
        return dataclasses.replace(
            report, notes=(*report.notes, published, reason)
        )
    missed = [d for d, hit in found.items() if not hit]
    notes = [*report.notes, published]
    if missed:
        notes.append(
            "on these rows the screen did not find the planted "
            + ", ".join(missed)
            + ": a leak of that kind may remain undetected"
        )
    return dataclasses.replace(report, notes=tuple(notes), probe=found)
