"""Row choice, scan demand, the certificate store and run wiring of a screen."""

from collections.abc import Hashable
from collections.abc import Sequence
import dataclasses
import functools
import math
from pathlib import Path

import numpy
import pandas

from scmle.assurance.admissibility import leakage
from scmle.assurance.admissibility import plants
from scmle.assurance.admissibility.screen.fitting import encode
from scmle.assurance.admissibility.screen.fitting import enough
from scmle.assurance.admissibility.screen.fitting import Problem
from scmle.assurance.admissibility.screen.hypotheses import Pending
from scmle.assurance.admissibility.screen.hypotheses import propose
from scmle.assurance.admissibility.screen.hypotheses import sift
from scmle.assurance.admissibility.screen.hypotheses import tested
from scmle.assurance.admissibility.screen.permutation import resamples
from scmle.assurance.admissibility.screen.records import candidates
from scmle.assurance.admissibility.screen.records import duplication
from scmle.assurance.admissibility.screen.records import encodings
from scmle.assurance.admissibility.screen.report import halt
from scmle.assurance.admissibility.screen.report import Screen
from scmle.assurance.admissibility.screen.settings import BINARY_CLASSES
from scmle.assurance.admissibility.screen.settings import cost
from scmle.assurance.admissibility.screen.settings import HYPOTHESES
from scmle.assurance.admissibility.screen.settings import MAX_PERMUTATIONS
from scmle.assurance.admissibility.screen.settings import MAX_ROWS
from scmle.assurance.admissibility.screen.settings import MECHANISM_DEFECTS
from scmle.assurance.admissibility.screen.settings import MIN_CLASSES
from scmle.assurance.admissibility.screen.settings import prepare
from scmle.assurance.admissibility.screen.settings import SCAN_SECONDS
from scmle.assurance.admissibility.screen.settings import SCREEN_SHARE
from scmle.assurance.admissibility.screen.settings import SKILL
from scmle.assurance.admissibility.screen.settings import STORE_PATH
from scmle.assurance.admissibility.screen.settings import TIME_BLOCKS
from scmle.assurance.detectors import certificates
from scmle.assurance.ledger import budget
from scmle.contracts import errors
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


def admit(
    frame: pandas.DataFrame,
    target: str,
    task: str,
    cfg: leakage.Config,
    names: Sequence[Hashable],
    groups: pandas.Series | None,
    block_size: int | None,
    limit: float | None,
) -> Screen | None:
    """Declines a real scan whose work before any permutation cannot fit.

    Args:
        frame: Rows to screen, with the target.
        target: Target column.
        task: `classification` or `regression`.
        cfg: Settings.
        names: Candidate columns.
        groups: Group label per row, or None.
        block_size: Rows per time block, or None.
        limit: The screen's share of the time budget (`share`), or None.

    Returns:
        The declined report (`forgo`) when `foresee` predicts more than
        `limit`; None when the scan may go on, or stops before proposing
        (and so says why itself, or raises as `screen` does).
    """
    if limit is None:
        return None
    before = foresee(frame, target, task, cfg, names, groups, block_size)
    if before is None or before <= limit:
        return None
    return forgo(cfg, before, limit, limit / SCREEN_SHARE, names)


def confine(
    cfg: leakage.Config, demand: Demand, limit: float | None
) -> Screen | None:
    """Declines a scan whose formed family cannot fit its seconds.

    Args:
        cfg: Settings.
        demand: The scan's work (`reckon`).
        limit: Seconds the scan may take, or None for no cap.

    Returns:
        The declined report (`decline`) when the demand is over `limit`,
        else None.
    """
    if limit is None or demand.seconds <= limit:
        return None
    return decline(cfg, demand, limit, limit / SCREEN_SHARE)


def forgo(
    cfg: leakage.Config,
    seconds: float,
    limit: float,
    budget_seconds: float,
    columns: Sequence[Hashable],
) -> Screen:
    """Builds the report of a real scan declined before it proposed anything.

    Args:
        cfg: Settings.
        seconds: Predicted seconds before any permutation (`foresee`).
        limit: Seconds the scan may take.
        budget_seconds: The declared time budget.
        columns: Candidate columns, all unchecked.

    Returns:
        A report listing every candidate column as unchecked.
    """
    return halt(
        cfg,
        dict.fromkeys(MECHANISM_DEFECTS, certificates.Authority.GATING),
        0,
        0,
        (
            f"proposing the hypotheses of {len(columns)} columns and their "
            f"target-mean tests is predicted to take {seconds:.1f} s before "
            f"any permutation; the limit is {limit:.1f} s. Nothing was "
            "flagged (power, not size, is lost): screen fewer columns"
        ),
        tuple(str(c) for c in columns),
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


def bound(rows: int, columns: int, classes: int) -> float:
    """Bounds the seconds of the real scan and the probe before any data.

    Every column is assumed to raise the most hypotheses a column can, so
    the bound is not below what the scans ask for; in a run `share` caps it.

    Args:
        rows: Rows to be screened (before `subsample`).
        columns: Candidate columns.
        classes: Classes of the target; below two means regression.

    Returns:
        The predicted seconds of both scans, with no cap applied.
    """
    task = "classification" if classes >= MIN_CLASSES else "regression"
    seen = min(rows, MAX_ROWS)
    split = classes if classes > BINARY_CLASSES else 1
    alpha = budget.exact(leakage.CERTIFIED.alpha, "alpha")
    total = 0.0
    for width in (columns, columns + len(plants.DEFECTS)):
        draws = min(
            max(
                leakage.CERTIFIED.permutations,
                resamples(HYPOTHESES * width + 1, alpha),
            ),
            MAX_PERMUTATIONS,
        )
        live = (HYPOTHESES - 1) * width
        levels = max(classes, leakage.TARGET_BINS)
        total += prepare(width, split, seen * width * width) + cost(
            task,
            seen,
            draws * width * split,
            draws,
            draws * live * levels,
            live + width,
        )
    return total


def ready(
    frame: pandas.DataFrame,
    target: str,
    task: str,
    cfg: leakage.Config,
    names: Sequence[Hashable],
    groups: pandas.Series | None,
    block_size: int | None,
) -> tuple[pandas.DataFrame, numpy.ndarray | None, int | None, Problem] | None:
    """Chooses the rows as `screen` does and encodes their outcome.

    Args:
        frame: Rows to screen, with the target.
        target: Target column.
        task: `classification` or `regression`.
        cfg: Settings.
        names: Candidate columns.
        groups: Group label per row, or None.
        block_size: Rows per time block, or None.

    Returns:
        The chosen rows, their groups, the block size and the encoded
        outcome; None when the scan will stop before proposing (a missing
        target or candidate column, which `screen` refuses, an invalid
        outcome or too few rows), which costs no more than `SCAN_SECONDS`.
    """
    known = target in frame.columns
    if not known or any(c not in frame.columns or c == target for c in names):
        return None
    values = None if groups is None else groups.to_numpy()
    rows, blocks = subsample(len(frame), values, block_size, cfg.run_seed)
    sample = frame.iloc[rows].reset_index(drop=True)
    grouping = None if values is None else values[rows]
    # The scan abstains where `encode` fails: no folds (one group, or fewer
    # rows than folds) or an invalid outcome (InputError).
    single = grouping is not None and len(pandas.unique(grouping)) <= 1
    if single or (grouping is None and len(sample) < cfg.folds):
        return None
    try:
        problem = encode(sample[target], task, cfg, grouping, blocks)
    except errors.InputError:
        # handled: the scan abstains on the same error before drawing
        return None
    if not enough(problem.y, task):
        return None
    return sample, grouping, blocks, problem


def foresee(
    frame: pandas.DataFrame,
    target: str,
    task: str,
    cfg: leakage.Config,
    names: Sequence[Hashable],
    groups: pandas.Series | None,
    block_size: int | None,
) -> float | None:
    """Predicts the seconds of a scan before its first statistic.

    Read from the table's shape alone (`prepare`): proposing every column
    and the target-mean test, the work a scan does whatever its
    permutations, so a scan that cannot fit is declined before any of it.

    Args:
        frame: Rows to screen, with the target.
        target: Target column.
        task: `classification` or `regression`.
        cfg: Settings.
        names: Candidate columns.
        groups: Group label per row, or None.
        block_size: Rows per time block, or None.

    Returns:
        `SCAN_SECONDS` plus the proposing and record seconds; None when the
        scan stops before proposing (`ready`), which costs no more than
        `SCAN_SECONDS`.
    """
    found = ready(frame, target, task, cfg, names, groups, block_size)
    if found is None:
        return None
    sample, problem = found[0], found[3]
    cells = len(sample) * len(candidates(sample, names, problem)) * len(names)
    return SCAN_SECONDS + prepare(len(names), ways(problem), cells)


def ways(problem: Problem) -> int:
    """Classes of the target when more than two (one-vs-rest), else 1."""
    return problem.classes if problem.classes > BINARY_CLASSES else 1


def reckon(
    pending: Sequence[Pending],
    problem: Problem,
    frame: pandas.DataFrame,
    names: Sequence[Hashable],
    cfg: leakage.Config,
) -> Demand:
    """Counts the work of a scan whose hypotheses are proposed and sifted.

    Args:
        pending: The family, after `sift`.
        problem: The encoded outcome.
        frame: The chosen rows.
        names: Candidate columns.
        cfg: Settings.

    Returns:
        The demand; its seconds are those of the whole scan: proposing and
        records (`prepare`) and the permutations, fits, readings and
        bounds (`cost`).
    """
    alpha = budget.exact(cfg.alpha, "alpha")
    family = len(pending)
    total = max(cfg.permutations, resamples(family, alpha))
    live = [h for h in pending if h.statistic is not None and h.active]
    skill = sum(1 for h in live if h.mechanism == SKILL)
    work = total * skill * ways(problem)
    draws = total if live else 0
    reads = (
        draws * (len(live) - skill) * max(problem.classes, leakage.TARGET_BINS)
    )
    rows = len(frame)
    cells = rows * len(candidates(frame, names, problem)) * len(names)
    seconds = prepare(len(names), ways(problem), cells) + cost(
        problem.task, rows, work, draws, reads, len(live)
    )
    columns = tested(pending, names)
    return Demand(rows, family, total, skill, work, columns, seconds)


def measure(
    frame: pandas.DataFrame,
    target: str,
    task: str,
    cfg: leakage.Config,
    names: Sequence[Hashable],
    groups: pandas.Series | None,
    block_size: int | None,
) -> Demand | None:
    """Counts the work of `screen` on these rows without drawing any.

    It chooses the rows as `screen` does and forms the same hypotheses, so
    the count equals the scan's; forming them is itself the work `foresee`
    predicts.

    Args:
        frame: Rows to screen, with the target.
        target: Target column.
        task: `classification` or `regression`.
        cfg: Settings.
        names: Candidate columns.
        groups: Group label per row, or None.
        block_size: Rows per time block, or None.

    Returns:
        The demand (`reckon`); None when the scan will stop before drawing
        (`ready`, nothing to test, or the permutation limit), which costs no
        more than `SCAN_SECONDS`.
    """
    found = ready(frame, target, task, cfg, names, groups, block_size)
    if found is None:
        return None
    sample, grouping, blocks, problem = found
    classes = leakage.bucket(sample[target], task)
    codes = pandas.factorize(classes, sort=True)[0].astype("int64")
    totals = numpy.bincount(codes).astype("float64")
    structured = grouping is not None or blocks is not None
    pending = propose(
        sample,
        target,
        names,
        problem,
        cfg,
        codes,
        totals,
        structured=structured,
    )[0]
    pending += encodings(sample, names, problem, cfg)
    copied = duplication(frame, names, target, cfg) if pending else None
    pending += [] if copied is None else [copied]
    live = sift(pending, cfg)
    demand = reckon(pending, problem, sample, names, cfg)
    if not pending or (live and demand.permutations > MAX_PERMUTATIONS):
        return None
    return demand
