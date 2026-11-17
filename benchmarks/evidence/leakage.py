"""Claim C2: leakage is caught and clean data is not blocked.

Each case runs the product exactly as a run does up to its leakage gate:
`scmle.api.project.Project(..., leakage_policy="block").plan()` ingests the
table, splits it, profiles the development rows (where
`Profiler.leakage_screen` runs the permutation screen) and applies
`ProfileStage.enforce_leakage_policy`. A case is *blocked by the screen*
when that raises `BlockedError` carrying the screen's `leakage` evidence.

Two corpora, both of real public tables (scikit-learn's bundled sets and
OpenML tables in scikit-learn's local cache; `refusal.fetch` loads them
without the network, at most `refusal.ROWS` rows):

* `clean`: tables as published, with the task a practitioner would declare
  and the documented leak columns of four tables removed. A block is a
  false block; any other refusal or a crash also counts against the row.
* `leak`: two provenances, stated on every case.
  `documented real leak`: the real leaked dataset itself (Titanic `boat`
  and `body`, recorded after the sinking; bank marketing `V12`, the call
  duration the UCI page says is known only after the outcome; NYC green
  taxi `total_amount`, which includes the tip being predicted).
  `planted on real data`: a real table with one realistic leak added:
  `target_encoding` (an unsmoothed target mean over about five rows per
  level, computed on all rows), `post_outcome` (a field recorded after the
  outcome: the label with a tenth of entries garbled, or the target plus
  noise), `duplicates` (three in ten rows copied, so copies straddle the
  split) or `future_timestamp` (a closing date that falls later for the
  positive class or a larger target). A leak is caught only when the
  screen blocks the run.

Rows decided (`acceptance.toml`): recall, the one-sided Clopper-Pearson
lower bound of caught over leak cases, at least `leak_recall_min` (Floor);
false blocks, the upper bound over clean cases, at most `false_block_max`
(Ceiling). Both must pass. A row with fewer cases than its rule's minimum
cannot pass.

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

from benchmarks import acceptance
from benchmarks.evidence import refusal
from scmle.contracts import errors
from scmle.core import stages
from scmle.data.admissibility import screen
from scmle.data.detectors import certificates
from scmle.data.profiling import profiler
import scmle.evaluation.coverage

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
    """

    name: str
    corpus: str
    table: Table
    kind: str = ""
    provenance: str = "clean real data"
    column: str = ""


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
    return found


# Real tables that receive planted leaks: binary and regression targets, so
# a target mean and a target-ordered date are well defined.
BASES = (
    "openml_31", "openml_37", "openml_1464", "openml_1489", "openml_44",
    "openml_40701", "openml_1067", "openml_1120", "breast_cancer",
    "openml_151", "openml_183", "openml_189", "openml_197", "openml_43939",
    "openml_42225", "diabetes",
)  # fmt: skip


def catalog() -> dict[str, Case]:
    """Every case, clean and leaking.

    Returns:
        Case name to case.
    """
    tables = {t.name: t for t in clean()}
    cases = [Case(f"clean/{t.name}", "clean", t) for t in tables.values()]
    real = (
        ("titanic_boat", tables["titanic"], ("body",), "boat"),
        ("titanic_body", tables["titanic"], ("boat",), "body"),
        ("bank_duration", tables["bank_marketing"], (), "V12"),
        ("taxi_total", tables["green_taxi"], (), "total_amount"),
    )
    cases += [
        Case(
            f"leak/{n}",
            "leak",
            dataclasses.replace(t, drop=drop),
            c,
            REAL,
            c,
        )
        for n, t, drop, c in real
    ]
    for base in BASES:
        for kind in KINDS:
            column = "" if kind == "duplicates" else name(kind)
            cases.append(
                Case(
                    f"leak/{base}/{kind}",
                    "leak",
                    tables[base],
                    kind,
                    PLANTED,
                    column,
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
        bins = max(2, len(frame) // LEVEL)
        levels = pandas.qcut(
            numeric[source].rank(method="first"), bins, labels=False
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
        The case's labels, the outcome (`blocked` by the screen, `passed`,
        `refused` by another gate, `missing` data or `failed`), the screen's
        blocking, suspect and unchecked columns, and seconds.
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
    }
    try:
        frame = build(case)
    except errors.InputError as error:
        record.update(outcome="missing", message=str(error))
        return record
    record.update(rows=len(frame), columns=frame.shape[1] - 1)
    try:
        with tempfile.TemporaryDirectory() as folder:
            project = scmle.api.project.Project(
                frame,
                case.table.target,
                task=case.table.task,  # type: ignore[arg-type]
                leakage_policy="block",
                output_dir=folder,
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
    records: list[dict[str, Any]], corpus: str, counted: Callable[..., bool]
) -> dict[str, Any]:
    """Counts one corpus and decides its row.

    Cases whose table is missing are left out of the count (and reported
    by `problems`).

    Args:
        records: Every case's record.
        corpus: `clean` or `leak`.
        counted: Whether a record counts as a hit (a catch, or a block).

    Returns:
        Hits, trials, the outcome counts, the bound and whether it passes.
    """
    variables = acceptance.load()
    alpha = variables["alpha"].value
    variable = RECALL if corpus == "leak" else BLOCKS
    rule = variables[variable].rule
    if rule is None:
        raise errors.InputError(f"{variable} must have a rule")
    chosen = [
        r
        for r in records
        if r["corpus"] == corpus and r["outcome"] != "missing"
    ]
    hits = sum(counted(r) for r in chosen)
    outcomes: dict[str, int] = {}
    for r in chosen:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    return {
        "variable": variable,
        "threshold": variables[variable].value,
        "hits": hits,
        "trials": len(chosen),
        "rate": hits / len(chosen) if chosen else None,
        "outcomes": dict(sorted(outcomes.items())),
        "minimum": rule.minimum(alpha),
        "bound": bound(hits, len(chosen), alpha, corpus == "leak"),
        "passed": rule.passes(hits, len(chosen), alpha),
    }


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
    """Whether the screen blocked a leaking case."""
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
        if r["corpus"] != "leak" or r["outcome"] == "missing":
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
        The `code_hash` of the product's screen and gate and this module's
        corpus and trial.
    """
    return certificates.code_hash(
        screen.screen_table,
        profiler.Profiler.leakage_screen,
        stages.ProfileStage.enforce_leakage_policy,
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
    return {
        "claim": "C2",
        "alpha": variables["alpha"].value,
        "method": (
            "Project(leakage_policy='block').plan(): ingest, split, profile "
            "the development rows, enforce the leakage policy; default seed"
        ),
        "recall": tally(records, "leak", caught),
        "false_blocks": tally(records, "clean", blocked),
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
        than acceptance.toml's, a case whose table was missing, or a row
        with fewer cases than its rule's minimum; empty when it decides.
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
        least = rule.minimum(alpha)
        if row.get("trials", 0) < least:
            issues.append(
                f"leakage: {key} has {row.get('trials', 0)} cases; none can "
                f"pass below {least}"
            )
    missing = [
        c["name"] for c in found.get("cases", []) if c["outcome"] == "missing"
    ]
    if missing:
        issues.append(f"leakage: data missing for {', '.join(missing)}")
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
