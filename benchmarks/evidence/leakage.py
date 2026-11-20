"""Claim C2: leakage is caught and clean data is not blocked.

Each case runs the product exactly as a run does up to its leakage gate:
`scmle.api.project.Project(..., leakage_policy="block").plan()` ingests the
table, splits it, profiles the development rows (where
`Profiler.leakage_screen` runs the permutation screen) and applies
`Profiling.enforce_leakage_policy`. A case is *blocked by the screen*
when that raises `BlockedError` carrying the screen's `leakage` evidence.

Two corpora, both of real public tables (scikit-learn's bundled sets and
OpenML tables in scikit-learn's local cache; `refusal.fetch` loads them
without the network, at most `refusal.ROWS` rows):

* `clean`: tables as published, with the task a practitioner would declare
  and the documented leak columns of five tables removed (`DOCUMENTED` for
  the one table the rule below adds). A block is a false block; any other
  refusal or a crash also counts against the row, as
  docs/preregistration/c2.md words it ("does not reach a plan for any
  reason"): a table the split gate refuses (a class too rare for the test
  rows) is a false block of the row, though not of the screen, and the
  row's `outcomes` count the two apart (`refused`, `blocked`). The first 78
  tables were chosen by hand (75 once the sealed copies below are out).
  The rest follow a rule fixed before any of them was screened: every
  real table of the C14 catalog (`refusal.EXTRA`, then `refusal.SUITE`:
  four published OpenML suites and the tables cached before them) not
  already here, with the task C14 declares, except a
  table that is one already here under another source (`SAME`). One of
  them has documented leaks: Tecator (OpenML 505), whose description says
  its moisture and protein, like fat, the outcome its spectra are to
  predict, were "determined by analytic chemistry"; they are removed here
  and are leak cases below. C14's other documented leak, NYC green taxi
  `total_amount`, is already here with the column removed. The rule adds
  84 tables, so the 86 clean tables the first Holm step needs are reached
  (159 in all). No table is a copy of the sealed TabArena-v0.1 hold-out
  (`benchmarks/sota/split.json`), which no evidence run may read: the
  tables C14's rule 5 names (`refusal.SEALED`: credit-g, diabetes and
  diamonds, chosen here by hand, and Amazon_employee_access, Diabetes130US
  and KDDCup09_appetency, added by the rule) are left out, and a planted
  base among them is replaced as `BASES` states.
* `leak`: two provenances, stated on every case.
  `documented real leak`: the real leaked dataset itself (Titanic `boat`
  and `body`, recorded after the sinking; bank marketing `V12`, the call
  duration the UCI page says is known only after the outcome; NYC green
  taxi `total_amount`, which includes the tip being predicted; Tecator
  `moisture` and `protein`, lab measurements made with the fat they
  predict), each with the table's other leak columns removed.
  `planted on real data`: a real table with one realistic leak added:
  `target_encoding` (an unsmoothed target mean over quantile bins of the
  values of the numeric column with the most values, about five rows per bin
  and tied values in one bin, computed on all rows), `post_outcome` (a field
  recorded after the outcome: the label with a tenth of entries garbled, or
  the target plus noise), `duplicates` (three in ten rows copied, so copies
  straddle the split) or `future_timestamp` (a closing date that falls later
  for the positive class or a larger target). A leak is caught only when the
  screen blocks the run. A closing date is screened as its user would run it:
  that user scores records as of a known moment, the snapshot the closing
  dates fall after (`START`), and declares it (`cutoff`); without that
  declaration a closing date cannot be told from a legitimate trend.
  Likewise a documented real leak whose documentation says when the column
  becomes known is screened with that declared (`known`, `Dataset.known`),
  as a user who read it would run it, and never from knowing the answer:
  bank marketing `V12`, of which the UCI page (archive.ics.uci.edu/dataset/
  222/bank+marketing) says "the duration is not known before a call is
  performed. Also, after the end of the call y is obviously known"; NYC
  green taxi `total_amount`, "the total amount charged to passengers" in
  the TLC data dictionary (data_dictionary_trip_records_green.pdf), which
  holds `tip_amount` and so is charged only once the tip is (the OpenML
  page drops it because "otherwise the target could be predicted
  deterministically"); Tecator `moisture` and `protein`, which the
  source's description (OpenML 505, from StatLib) lists as outputs, not
  inputs ("real_in=122", the spectrum and its principal components;
  "real_out=3", the three contents "determined by analytic chemistry";
  "predict the fat content ... on the basis of its near infrared
  absorbance spectrum"), so they are known with the outcome, from the same
  analysis. Titanic's `boat` and `body` are not declared: their
  documentation names them without saying when they become known
  relative to the prediction.

Rows decided (`acceptance.toml`): recall, the one-sided Clopper-Pearson
lower bound of caught over leak cases, at least `leak_recall_min` (Floor);
false blocks, the upper bound over clean cases, at most `false_block_max`
(Ceiling). Both must pass. Each bound is taken at its Holm-adjusted `alpha`
(`holm`) over the `FAMILY` stage-D rows of docs/preregistration/c2.md (C2
recall and false blocks, C14 refusal rate and precision); the C14 rows are
decided elsewhere and enter here as not rejected, which can only lower
the C2 levels. A row with fewer cases than its rule's minimum at the first
Holm step cannot pass. A case whose table cannot be loaded counts against
its row (a miss, or a false block), never out of it.

NOT measured: entity-level leaks, leaks labelled by people not working on
the screen, a sealed corpus, tables over `refusal.ROWS` rows, task kinds
other than binary, multiclass and regression.

    python -m benchmarks.evidence.leakage --write
    python -m benchmarks.evidence.leakage --check
"""

import argparse
from collections.abc import Callable
from concurrent import futures
import dataclasses
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
import warnings

import numpy
import pandas
from scipy import stats

from benchmarks import acceptance
from benchmarks.evidence import refusal
from scmle.assurance.admissibility import screen
import scmle.assurance.admissibility.screen.association
import scmle.assurance.admissibility.screen.fitting
import scmle.assurance.admissibility.screen.hypotheses
import scmle.assurance.admissibility.screen.permutation
import scmle.assurance.admissibility.screen.planning
import scmle.assurance.admissibility.screen.records
import scmle.assurance.admissibility.screen.report
import scmle.assurance.admissibility.screen.settings
from scmle.assurance.detectors import certificates
from scmle.contracts import errors
import scmle.evaluation.coverage
from scmle.pipeline import profiler
from scmle.pipeline import stages

PATH = Path(__file__).with_suffix(".json")
RECALL = "leak_recall_min"
BLOCKS = "false_block_max"
# Two worker processes: the machine is shared.
WORKERS = 2
CLASSIFY = "classification"
REGRESS = "regression"
BINARY = 2
KINDS = ("target_encoding", "post_outcome", "duplicates", "future_timestamp")
REAL = "documented real leak"
PLANTED = "planted on real data"
# Rows per level of the planted target encoding.
LEVEL = 5
# Share of planted post-outcome labels replaced by a random label.
GARBLE = 0.1
# Noise of a planted post-outcome value, in target standard deviations.
NOISE = 0.25
# Share of rows copied by the planted duplicates.
COPIES = 0.3
# Days the planted closing date moves with the outcome, plus exponential
# noise of this mean.
SHIFT = 30
JITTER = 5.0
START = pandas.Timestamp("2020-01-01")
# Rows in the stage-D Holm family (docs/preregistration/c2.md, c14.md).
FAMILY = 4
# C14 tables that are a clean table here under another source: OpenML id to
# the source used here.
SAME = {61: "load_iris"}
# Documented leak columns of the tables the C14 rule adds, removed as the
# hand-chosen tables' are: OpenML id to columns.
DOCUMENTED = {505: ("moisture", "protein")}
NOT_MEASURED = (
    "entity-level leaks",
    "a corpus labelled by people not working on the screen",
    "a sealed corpus",
    "tables over the row cap (refusal.ROWS)",
    "task kinds other than binary, multiclass and regression",
)


@dataclasses.dataclass(frozen=True)
class Table:
    """One real public table and the task a practitioner would declare.

    Attributes:
        name: Unique name.
        source: OpenML data id or a scikit-learn `load_*` name.
        target: Target column.
        task: `classification` or `regression`.
        drop: Columns removed first (documented leaks).
    """

    name: str
    source: int | str
    target: str
    task: str
    drop: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class Case:
    """One screened run.

    Attributes:
        name: Unique name.
        corpus: `clean` or `leak`.
        table: The real table.
        kind: The leak kind: one of `KINDS`, the leak column of a
            documented case, or empty for a clean case.
        provenance: `REAL`, `PLANTED` or `clean real data`.
        column: The leaking column, empty for duplicates and clean cases.
        cutoff: The prediction time its user declares (`Dataset.cutoff`),
            empty for none.
        known: Columns its user declares known only after the outcome
            (`Dataset.known`), from the data's documentation.
    """

    name: str
    corpus: str
    table: Table
    kind: str = ""
    provenance: str = "clean real data"
    column: str = ""
    cutoff: str = ""
    known: tuple[str, ...] = ()


def clean() -> list[Table]:
    """Every clean table, with documented leak columns removed.

    Returns:
        The tables.
    """
    classes = (
        (3, "class"), (6, "class"), (11, "class"), (12, "class"),
        (14, "class"), (15, "Class"), (16, "class"), (18, "class"),
        (22, "class"), (23, "Contraceptive_method_used"), (24, "class"),
        (28, "class"), (29, "class"), (31, "class"), (32, "class"),
        (36, "class"), (37, "class"), (38, "Class"), (44, "class"),
        (46, "Class"), (50, "Class"), (54, "Class"), (150, "class"),
        (151, "class"), (179, "class"), (182, "class"), (307, "Class"),
        (458, "Author"), (991, "binaryClass"), (1049, "c"), (1050, "c"),
        (1063, "problems"), (1067, "defects"), (1068, "defects"),
        (1120, "class:"), (1169, "Delay"), (1462, "Class"), (1464, "Class"),
        (1480, "Class"), (1489, "Class"), (1494, "Class"), (1510, "Class"),
        (1597, "Class"), (4532, "class"), (4534, "Result"),
        (6332, "band_type"), (23381, "Class"), (40498, "Class"),
        (40499, "Class"), (40670, "class"), (40691, "class"),
        (40701, "class"), (40982, "target"), (40983, "class"),
        (40994, "outcome"),
    )  # fmt: skip
    numbers = (
        (183, "Class_number_of_rings"), (189, "y"), (195, "price"),
        (197, "usr"), (227, "usr"), (507, "ln(VOTES/POP)"),
        (537, "median_house_value"), (574, "price"), (41700, "runtime"),
        (42225, "price"), (42563, "SalePrice"), (42712, "count"),
        (42731, "price"), (43939, "median_house_value"),
    )  # fmt: skip
    found = [Table(f"openml_{i}", i, t, CLASSIFY) for i, t in classes]
    found += [Table(f"openml_{i}", i, t, REGRESS) for i, t in numbers]
    found += [
        Table("breast_cancer", "load_breast_cancer", "target", CLASSIFY),
        Table("wine", "load_wine", "target", CLASSIFY),
        Table("iris", "load_iris", "target", CLASSIFY),
        Table("digits", "load_digits", "target", CLASSIFY),
        Table("diabetes", "load_diabetes", "target", REGRESS),
        Table("bank_marketing", 1461, "Class", CLASSIFY, ("V12",)),
        Table("energy_heating", 1472, "y1", REGRESS, ("y2",)),
        Table("titanic", 40945, "survived", CLASSIFY, ("boat", "body")),
        Table("green_taxi", 42729, "tip_amount", REGRESS, ("total_amount",)),
    ]
    have = {t.source for t in found}
    found += [
        Table(
            f"openml_{source}",
            source,
            target,
            task,
            DOCUMENTED.get(source, ()),
        )
        for source, (_, _, target, task) in {
            **refusal.EXTRA,
            **refusal.SUITE,
        }.items()
        if source not in have and source not in SAME
    ]
    return [t for t in found if t.source not in refusal.SEALED]


# Real tables that receive planted leaks: binary and regression targets, so
# a target mean and a target-ordered date are well defined.
# A base that is a copy of the sealed hold-out (`refusal.SEALED`) is
# replaced, in its place (so every other base keeps its seed), by the first
# clean table of its kind in `clean()` order that is not a base and not
# sealed, and whose numeric column with the most values (where the target
# encoding is planted, `plant`) has at least one value per `LEVEL` rows, so
# its bins of `LEVEL` rows are bins of its values rather than of row
# positions among tied values: credit-g (31) by credit-approval (29),
# diabetes (37) by adult (179), diamonds (42225) by auto_price (195).
BASES = (
    "openml_29", "openml_179", "openml_1464", "openml_1489", "openml_44",
    "openml_40701", "openml_1067", "openml_1120", "breast_cancer",
    "openml_151", "openml_183", "openml_189", "openml_197", "openml_43939",
    "openml_195", "diabetes",
)  # fmt: skip


def catalog() -> dict[str, Case]:
    """Every case, clean and leaking.

    Returns:
        Case name to case.
    """
    tables = {t.name: t for t in clean()}
    cases = [Case(f"clean/{t.name}", "clean", t) for t in tables.values()]
    real = (
        ("titanic_boat", tables["titanic"], ("body",), "boat", ()),
        ("titanic_body", tables["titanic"], ("boat",), "body", ()),
        (
            "tecator_moisture",
            tables["openml_505"],
            ("protein",),
            "moisture",
            ("moisture",),
        ),
        (
            "tecator_protein",
            tables["openml_505"],
            ("moisture",),
            "protein",
            ("protein",),
        ),
        ("bank_duration", tables["bank_marketing"], (), "V12", ("V12",)),
        (
            "taxi_total",
            tables["green_taxi"],
            (),
            "total_amount",
            ("total_amount",),
        ),
    )
    cases += [
        Case(
            f"leak/{n}",
            "leak",
            dataclasses.replace(t, drop=drop),
            c,
            REAL,
            c,
            known=known,
        )
        for n, t, drop, c, known in real
    ]
    for base in BASES:
        for kind in KINDS:
            column = "" if kind == "duplicates" else name(kind)
            moment = START.isoformat() if kind == "future_timestamp" else ""
            cases.append(
                Case(
                    f"leak/{base}/{kind}",
                    "leak",
                    tables[base],
                    kind,
                    PLANTED,
                    column,
                    moment,
                )
            )
    return {c.name: c for c in cases}


def name(kind: str) -> str:
    """The column a planted leak adds.

    Args:
        kind: One of `KINDS` other than `duplicates`.

    Returns:
        The column name.
    """
    return {
        "target_encoding": "level_target_mean",
        "post_outcome": "outcome_recorded",
        "future_timestamp": "closed_at",
    }[kind]


def load(table: Table) -> pandas.DataFrame:
    """Reads a real table from the local caches.

    Args:
        table: The table.

    Returns:
        The frame, its documented leak columns removed and a regression
        target as numbers.

    Raises:
        InputError: If an OpenML table is not cached.
    """
    frame = refusal.fetch(table.source).drop(columns=list(table.drop))
    if table.task == "regression":
        frame[table.target] = pandas.to_numeric(frame[table.target])
    return frame


def score(frame: pandas.DataFrame, table: Table) -> numpy.ndarray:
    """The target as numbers: the value, or 1 for the second sorted class.

    Args:
        frame: The table.
        table: Its description.

    Returns:
        One float per row.

    Raises:
        InputError: If a classification target has other than two classes.
    """
    target = frame[table.target]
    if table.task == "regression":
        return target.to_numpy(dtype=float)
    labels = sorted(target.astype(str).unique())
    if len(labels) != BINARY:
        raise errors.InputError(f"{table.name} is not binary")
    ones: numpy.ndarray = (target.astype(str) == labels[1]).to_numpy(float)
    return ones


def plant(
    frame: pandas.DataFrame, table: Table, kind: str, seed: int
) -> pandas.DataFrame:
    """Adds one realistic leak to a real table.

    Args:
        frame: The real table; not modified.
        table: Its description.
        kind: One of `KINDS`.
        seed: Seed of the planted noise.

    Returns:
        The leaking table.

    Raises:
        InputError: For an unknown kind.
    """
    rng = numpy.random.default_rng(seed)
    values = score(frame, table)
    frame = frame.copy()
    if kind == "target_encoding":
        features = frame.drop(columns=[table.target])
        numeric = features.select_dtypes("number")
        source = numeric.nunique().idxmax()
        # Quantile bins of the values, not of row ranks: tied values share
        # a level, so the encoding is a function of the column, as a
        # fitted target encoding is (a bin of ranks cut a value by row
        # position, which no column holds).
        bins = max(2, len(frame) // LEVEL)
        levels = pandas.qcut(
            numeric[source], bins, labels=False, duplicates="drop"
        )
        means = (
            pandas.Series(values).groupby(levels.to_numpy()).transform("mean")
        )
        frame[name(kind)] = means.to_numpy()
    elif kind == "post_outcome":
        target = frame[table.target]
        if table.task == "regression":
            spread = NOISE * float(numpy.std(values))
            frame[name(kind)] = values + rng.normal(0, spread, len(frame))
        else:
            garbled = rng.random(len(frame)) < GARBLE
            picks = rng.choice(target.unique(), len(frame))
            frame[name(kind)] = numpy.where(garbled, picks, target)
    elif kind == "duplicates":
        copies = frame.sample(frac=COPIES, random_state=seed)
        frame = pandas.concat([frame, copies]).sample(
            frac=1.0, random_state=seed
        )
        frame = frame.reset_index(drop=True)
    elif kind == "future_timestamp":
        ranks = pandas.Series(values).rank(pct=True).to_numpy()
        days = SHIFT * ranks + rng.exponential(JITTER, len(frame))
        frame[name(kind)] = START + pandas.to_timedelta(days, unit="D")
    else:
        raise errors.InputError(f"unknown leak kind {kind!r}")
    return frame


def build(case: Case) -> pandas.DataFrame:
    """The table a case screens.

    Args:
        case: The case.

    Returns:
        The real table, with its planted leak if it has one.
    """
    frame = load(case.table)
    if case.provenance == PLANTED:
        seed = BASES.index(case.table.name) * len(KINDS)
        frame = plant(
            frame, case.table, case.kind, seed + KINDS.index(case.kind)
        )
    return frame


def attempt(case: Case) -> dict[str, Any]:
    """Screens one case as a run does and records how it ended.

    Args:
        case: The case.

    Returns:
        The case's labels and declarations, the outcome (`blocked` by the
        screen, `passed`, `refused` by another gate, `missing` data or
        `failed`), the screen's blocking, suspect and unchecked columns, and
        seconds.
    """
    import scmle.api.project

    started = time.monotonic()
    record: dict[str, Any] = {
        "name": case.name,
        "corpus": case.corpus,
        "table": case.table.name,
        "source": case.table.source,
        "kind": case.kind,
        "provenance": case.provenance,
        "column": case.column,
        "known": list(case.known),
    }
    try:
        frame = build(case)
    except errors.InputError as error:
        record.update(outcome="missing", message=str(error))
        return record
    record.update(rows=len(frame), columns=frame.shape[1] - 1)
    # A declaration is passed only when made, so a case without one runs
    # exactly as before the declaration existed.
    declared: dict[str, Any] = (
        {"known": dict.fromkeys(case.known, "outcome")} if case.known else {}
    )
    try:
        with tempfile.TemporaryDirectory() as folder:
            project = scmle.api.project.Project(
                frame,
                case.table.target,
                task=case.table.task,  # type: ignore[arg-type]
                leakage="block",
                cutoff=case.cutoff or None,
                directory=folder,
                **declared,
            )
            project.plan()
            profile = project.profile
            evidence = {} if profile is None else dict(profile.leakage)
            record["outcome"] = "passed"
    except errors.BlockedError as error:
        evidence = dict(error.details.get("leakage", {}))
        record.update(
            outcome="blocked" if "leakage" in error.details else "refused",
            message=str(error)[:300],
        )
    except Exception as error:  # noqa: BLE001 -- every ending is recorded
        evidence = {}
        record.update(
            outcome="failed",
            error=type(error).__name__,
            message=str(error)[:300],
        )
    for key in ("blocking", "suspects", "unchecked"):
        record[key] = sorted({str(c) for c in evidence.get(key, ())})
    record["named"] = bool(case.column) and case.column in record["blocking"]
    record["seconds"] = round(time.monotonic() - started, 1)
    return record


def screened(name: str) -> dict[str, Any]:
    """Runs one case by name in a worker process.

    Args:
        name: A key of `catalog`.

    Returns:
        What `attempt` returned.
    """
    logging.disable(logging.WARNING)
    warnings.simplefilter("ignore")
    return attempt(catalog()[name])


def tally(
    records: list[dict[str, Any]],
    corpus: str,
    counted: Callable[..., bool],
    level: float | None = None,
) -> dict[str, Any]:
    """Counts one corpus and decides its row at one level.

    A case whose table is missing counts against the row: a leak it hid is
    not caught (`caught`), and a clean table it held did not reach a plan
    (`blocked`).

    Args:
        records: Every case's record.
        corpus: `clean` or `leak`.
        counted: Whether a record counts as a hit (a catch, or a block).
        level: The one-sided level of the bound; None is `alpha` over
            `FAMILY` (the first Holm step; `decide` sets each row's own).

    Returns:
        Hits, trials, the outcome counts, the level, the minimum cases at
        the first Holm step, the bound and whether it passes.
    """
    variables = acceptance.load()
    alpha = variables["alpha"].value
    variable = RECALL if corpus == "leak" else BLOCKS
    rule = variables[variable].rule
    if rule is None:
        raise errors.InputError(f"{variable} must have a rule")
    chosen = [r for r in records if r["corpus"] == corpus]
    hits = sum(counted(r) for r in chosen)
    outcomes: dict[str, int] = {}
    for r in chosen:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    used = alpha / FAMILY if level is None else level
    return {
        "variable": variable,
        "threshold": variables[variable].value,
        "hits": hits,
        "trials": len(chosen),
        "rate": hits / len(chosen) if chosen else None,
        "outcomes": dict(sorted(outcomes.items())),
        "level": used,
        "minimum": rule.minimum(alpha / FAMILY),
        "bound": bound(hits, len(chosen), used, corpus == "leak"),
        "passed": rule.passes(hits, len(chosen), used),
    }


def chance(row: dict[str, Any]) -> float:
    """The exact one-sided p-value of a row against its threshold.

    The Clopper-Pearson bound passes at level `a` exactly when this is at
    most `a`, so Holm can order the rows by it.

    Args:
        row: A row from `tally`.

    Returns:
        `P(X >= hits)` under `Binomial(trials, threshold)` for recall (a
        floor), `P(X <= hits)` for false blocks (a ceiling); 1 with no
        trials.
    """
    if row["trials"] == 0:
        return 1.0
    if row["variable"] == RECALL:
        tail = stats.binom.sf(row["hits"] - 1, row["trials"], row["threshold"])
    else:
        tail = stats.binom.cdf(row["hits"], row["trials"], row["threshold"])
    return float(tail)


def holm(chances: list[float], alpha: float) -> list[tuple[float, bool]]:
    """Holm's step-down levels for the rows measured here.

    The family has `FAMILY` rows; those not measured here (the C14 rows)
    enter as not rejected (p = 1), which only lowers these levels. The row
    of rank `i` (0 first, by p-value) is decided at `alpha / (FAMILY - i)`,
    and only when every row ranked before it passed.

    Args:
        chances: The p-value of each row, from `chance`.
        alpha: The family-wise level.

    Returns:
        Per row, in the given order: its level, and whether every row
        ranked before it passed.
    """
    found = [(0.0, False)] * len(chances)
    reached = True
    order = sorted(range(len(chances)), key=chances.__getitem__)
    for rank, index in enumerate(order):
        level = alpha / (FAMILY - rank)
        found[index] = (level, reached)
        reached = reached and chances[index] <= level
    return found


def decide(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Both rows, each at its Holm-adjusted level.

    Args:
        records: Every case's record.

    Returns:
        The recall row and the false-block row; a row passes only when its
        own bound passes and every row Holm ranks before it passed.
    """
    alpha = acceptance.load()["alpha"].value
    plan = (("leak", caught), ("clean", blocked))
    first = [tally(records, corpus, counted) for corpus, counted in plan]
    steps = holm([chance(r) for r in first], alpha)
    rows = []
    for (corpus, counted), (level, reached) in zip(plan, steps, strict=True):
        row = tally(records, corpus, counted, level)
        row["passed"] = bool(row["passed"] and reached)
        rows.append(row)
    return rows[0], rows[1]


def bound(hits: int, trials: int, alpha: float, lower: bool) -> float | None:
    """One-sided Clopper-Pearson bound at `alpha`.

    Args:
        hits: Events.
        trials: Trials.
        alpha: One-sided error rate.
        lower: The lower bound when True, else the upper.

    Returns:
        The bound, or None with no trials.
    """
    if trials == 0:
        return None
    pair = scmle.evaluation.coverage.clopper_pearson(
        hits, trials, 1 - 2 * alpha
    )
    return float(pair[0] if lower else pair[1])


def caught(record: dict[str, Any]) -> bool:
    """Whether the screen blocked a leaking case; a missing table is not."""
    return bool(record["outcome"] == "blocked")


def blocked(record: dict[str, Any]) -> bool:
    """Whether a clean case did not get through: blocked, refused, failed."""
    return bool(record["outcome"] != "passed")


def breakdown(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Caught and total leak cases per kind and per provenance.

    Args:
        records: Every case's record.

    Returns:
        Group label to `{"caught", "cases"}`; descriptive, decides nothing.
    """
    groups: dict[str, dict[str, int]] = {}
    for r in records:
        if r["corpus"] != "leak":
            continue
        label = r["kind"] if r["provenance"] == PLANTED else "documented"
        for key in (label, r["provenance"]):
            entry = groups.setdefault(key, {"caught": 0, "cases": 0})
            entry["caught"] += caught(r)
            entry["cases"] += 1
    return dict(sorted(groups.items()))


def digest() -> str:
    """Hash of the code measured and the code measuring it.

    Returns:
        The `code_hash` of the product's whole screen package (`code_hash`
        hashes only the objects it is given, not what they call, so every
        module of the package is named: an edit to any check the screen runs
        makes the stored result stale), the gate, and this module's corpus and
        trial.
    """
    return certificates.fingerprint(
        screen,
        scmle.assurance.admissibility.screen.association,
        scmle.assurance.admissibility.screen.fitting,
        scmle.assurance.admissibility.screen.hypotheses,
        scmle.assurance.admissibility.screen.permutation,
        scmle.assurance.admissibility.screen.planning,
        scmle.assurance.admissibility.screen.records,
        scmle.assurance.admissibility.screen.report,
        scmle.assurance.admissibility.screen.settings,
        profiler.Profiler.leakage_screen,
        stages.Profiling.enforce_leakage_policy,
        refusal.fetch,
        Table,
        Case,
        clean,
        catalog,
        load,
        score,
        plant,
        build,
        attempt,
        tally,
        chance,
        holm,
        decide,
        caught,
        blocked,
    )


def measure(workers: int = WORKERS) -> dict[str, Any]:
    """Screens every case and builds the stored document.

    Args:
        workers: Worker processes.

    Returns:
        The JSON-ready document: both rows, the per-kind breakdown, every
        case's record, what was not measured, the code hash and seconds.
    """
    started = time.monotonic()
    os.environ["OMP_NUM_THREADS"] = "1"
    names = list(catalog())
    with futures.ProcessPoolExecutor(workers) as pool:
        records = list(pool.map(screened, names))
    variables = acceptance.load()
    recall, blocks = decide(records)
    return {
        "claim": "C2",
        "alpha": variables["alpha"].value,
        "method": (
            "Project(leakage_policy='block').plan(): ingest, split, profile "
            "the development rows, enforce the leakage policy; default seed"
        ),
        "recall": recall,
        "false_blocks": blocks,
        "kinds": breakdown(records),
        "cases": records,
        "not_measured": list(NOT_MEASURED),
        "code": digest(),
        "seconds": round(time.monotonic() - started, 1),
    }


def read(document: dict[str, Any] | None) -> dict[str, Any] | None:
    """The given document, or the stored one.

    Args:
        document: A document, or None to read `PATH`.

    Returns:
        The document, or None when nothing is stored.
    """
    if document is not None:
        return document
    if not PATH.exists():
        return None
    loaded: dict[str, Any] = json.loads(PATH.read_text(encoding="utf-8"))
    return loaded


def problems(document: dict[str, Any] | None = None) -> list[str]:
    """Why the document cannot decide the rows.

    Args:
        document: The document; read from `PATH` when None.

    Returns:
        Messages for a missing file, code that changed, a threshold other
        than acceptance.toml's, or a row with fewer cases than its rule's
        minimum at the first Holm step; empty when it decides. A missing
        table does not stop the decision: it counts against its row.
    """
    found = read(document)
    if found is None:
        return [f"{PATH.name} is missing"]
    issues = []
    if found.get("code") != digest():
        issues.append("leakage: measured with other code")
    variables = acceptance.load()
    alpha = variables["alpha"].value
    for key, variable in (("recall", RECALL), ("false_blocks", BLOCKS)):
        row = found.get(key) or {}
        rule = variables[variable].rule
        if rule is None:
            issues.append(f"leakage: {variable} has no rule")
            continue
        if row.get("threshold") != variables[variable].value:
            issues.append(
                f"leakage: {variable} {row.get('threshold')} is not "
                f"{variables[variable].value}"
            )
        least = rule.minimum(alpha / FAMILY)
        if row.get("trials", 0) < least:
            issues.append(
                f"leakage: {key} has {row.get('trials', 0)} cases; none can "
                f"pass below {least}"
            )
    return issues


def summary(document: dict[str, Any] | None = None) -> tuple[str, str]:
    """The scorecard's value and verdict for the row.

    Args:
        document: The document; read from `PATH` when None.

    Returns:
        The value text (both counts and bounds, and how many leaks are
        planted on real data) and `pass` when both rows pass, else `fail`;
        `not measured` when nothing is stored or `problems` finds any.
    """
    found = read(document)
    if found is None or "recall" not in found:
        return "no measurement", "not measured"
    recall, blocks = found["recall"], found["false_blocks"]
    kinds = found.get("kinds", {})
    planted = kinds.get(PLANTED, {}).get("cases", 0)
    real = kinds.get(REAL, {}).get("cases", 0)
    text = (
        f"recall {recall['hits']}/{recall['trials']} caught, lower bound "
        f"{recall['bound'] or 0.0:.3f} ({planted} planted on real data, "
        f"{real} documented real leaks); false blocks "
        f"{blocks['hits']}/{blocks['trials']} clean real tables, upper "
        f"bound {blocks['bound'] if blocks['bound'] is not None else 1.0:.3f}"
    )
    if problems(found):
        return text, "not measured"
    passed = recall["passed"] and blocks["passed"]
    return text, "pass" if passed else "fail"


def main(arguments: list[str] | None = None) -> int:
    """Writes (`--write`) or checks (`--check`) the stored measurement.

    Args:
        arguments: Command-line arguments; None reads `sys.argv`.

    Returns:
        0 after `--write`; for `--check`, 0 when `problems` is empty and 1
        otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    options = parser.parse_args(arguments)
    if options.write:
        document = measure()
        PATH.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary(document)))
        return 0
    issues = problems()
    print(json.dumps({"ok": not issues, "problems": issues}, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
