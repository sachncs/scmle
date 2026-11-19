"""Claim C5: data are read and joined correctly.

Two measurements.

Awkward files. `corpus` lists files a user is likely to hand over: other
encodings (UTF-8 with a byte-order mark, latin-1, UTF-16), other
delimiters, quoted newlines and delimiters, mixed types, empty columns,
duplicate headers, huge and long integers, leading zeros, dates in several
formats, and truncated files. Each has its expected table, written by hand,
or an expected refusal. Every file is read with the product's reader,
`scmle.data.readers.tabular.read_table`, and the outcome is one of:

* `correct`: the table equals the expected one;
* `refused`: the reader raised `DataError` (acceptable for every file; a
  refusal of a file with an expected table is counted as over-refusal);
* `wrong`: a table came back that differs from the expected one, or a
  table came back where a refusal was expected. Any one fails the row.

A cell expected as text must come back as that exact text; an integer must
come back exact (as a number or as its own digits); a float within
`reference_rel_tol`; a date either as its own text or as the same instant.

Point in time. On `WORLDS` random parent/child/leaf schemas (ties, gaps,
late and backfilled rows, missing recorded times, windows, lags, nested
aggregates; `benchmarks.qualification.ingest_checks.random_schema` under
its own component name), the product's vectorised synthesis
(`scmle.assurance.ingest.synthesize.evaluate` over `asof_view.View`) is compared
with the row-by-row reference `benchmarks.qualification.ingest_naive`,
within `reference_rel_tol` relative and absolute (`acceptance.toml`).

Real relational data. `relational.DATASETS` lists at least `REAL_DATASETS`
public, freely licensed multi-table datasets pinned by URL and SHA-256;
each is read by the product and, independently, by the standard library,
and every primitive and nested feature of the grammar at hard cutoffs (ties
with event and recorded times, window edges) is compared with
`ingest_naive` within the same tolerance (see `relational`). A dataset that
cannot be fetched, matched to its checksum or read counts against the row,
as does any feature that differs.

Split hygiene. No tied timestamp may sit on both sides of any split, nor
an entity where the product promised to keep it apart, nor a duplicate
record of one declared entity. Identical rows are independent draws and may
cross, as in scikit-learn, OpenML and the practitioner baseline (owner
decision, 2026-11-19); C2 flags a duplicate rate above the table's natural
rate as leakage. Duplicate crossings are counted and reported, not failed.
`tables` lists the tables split: the child table of every relational
dataset under each of `DECLARATIONS` (nothing declared, the entity as
`group_column`, the event time as `time_column`, both, both under
`validation="rolling"`, and the last two again with `entities="unseen"`),
and every real table of `benchmarks.evidence.refusal` with the
declarations its task needs. Each is
split as a user run splits it (`hygiene`): `Project(...)` builds the
specification, rows without a target are set aside, `TaskResolver`
resolves the task and `Splitter(spec, task).split` makes the holdout and
the folds. Every pair of sides (development against test, and each fold's
training against its validation rows) is checked for a shared value of
the declared `group_column` (entity), a shared duplicate record (the
product's own `duplicates.duplicate_groups` over the splitter's feature
columns; reported only), a shared duplicate record of one declared entity
(the same duplicate group and the same `group_column` value on both sides;
`copy`) and a shared value of the declared `time_column`. A column that is
not declared is not checked: the product cannot know it. An entity on
both sides of a temporal or rolling split is what a forecast of known
entities asks for (its past trains, its future is held out), so there an
entity or a `copy` crossing is a violation only where the run declared
`entities="unseen"`; under any other strategy a declared entity column,
and so a copy of one entity, must never cross. The summary
reports how many runs with an entity column were declared each way. A split the
product refuses (`BlockedError`, `DataError`) puts nothing on both sides
and is reported; a table that cannot be loaded counts against the row.

Verdict. `fail` when any file is silently wrong, any world differs, any
real dataset fails to agree, or any split shares a timestamp, a duplicate
record of one declared entity, or an entity where the product promised to
keep it apart, across its sides; `pass` otherwise; `not measured` when
`problems` finds any.

NOT measured: Excel, Parquet and Feather files, the streaming reader
(`streaming.Csv`), database connectors other than SQLite.

    python -m benchmarks.evidence.reading --write
    python -m benchmarks.evidence.reading --check
"""

import argparse
from concurrent import futures
import dataclasses
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy
import pandas

from benchmarks import acceptance
from benchmarks.evidence import refusal
from benchmarks.evidence import relational
from benchmarks.qualification import ingest_checks
from benchmarks.qualification import ingest_naive
import scmle.api.project
from scmle.assurance.detectors import certificates
from scmle.assurance.ingest import asof_view
from scmle.assurance.ingest import synthesize
import scmle.assurance.ingest.synthesize.evaluation
import scmle.assurance.ingest.synthesize.grammar
import scmle.assurance.ingest.synthesize.selection
from scmle.contracts import errors
from scmle.data.readers import tabular
from scmle.data.splitting import duplicates
from scmle.data.splitting import splitters
import scmle.data.validation.target

PATH = Path(__file__).with_suffix(".json")
VARIABLE = "reference_rel_tol"
# Generated worlds the plan's C5 criterion asks for (plan section 5).
WORLDS = 1000
# Real relational datasets the criterion asks for (plan section 5).
REAL_DATASETS = 5
# Component name of the world stream; differs from ingest_certify's, so
# these worlds are not the ones the qualification harness measured.
STREAM = "evidence.reading.schema"
WORKERS = 2
# Rows of the mixed-type file: more than pandas' low-memory chunk, so a
# column typed per chunk shows up.
CHUNKED_ROWS = 300_000
REFUSE = "refuse"
# What a practitioner declares for a relational child table, one split each.
DECLARATIONS = (
    "rows",
    "entity",
    "time",
    "both",
    "rolling",
    "unseen",
    "unseen_rolling",
)
# What may not sit on both sides of a split: a declared entity, a duplicate
# record of one declared entity, a tied timestamp.
CROSSINGS = ("entity", "copy", "time")
# What is counted on both sides of a split but is not a violation: identical
# rows are independent draws (owner decision, 2026-11-19).
REPORTED = ("duplicate",)
# Strategies under which the entity declaration decides whether an entity
# may cross: a known entity's past trains and its future is held out.
TIMED = ("temporal", "rolling")
NOT_MEASURED = (
    "Excel, Parquet and Feather files",
    (
        "the streaming reader (streaming.Csv), database connectors other "
        "than SQLite"
    ),
)


@dataclasses.dataclass(frozen=True)
class Date:
    """A date cell: its text in the file and the instant it means.

    Attributes:
        text: The cell as written.
        instant: ISO 8601 form of what it means.
    """

    text: str
    instant: str


@dataclasses.dataclass(frozen=True)
class Case:
    """One awkward file and what reading it must give.

    Attributes:
        name: Identifier.
        group: What is awkward about it.
        suffix: File suffix, which picks the product's reader.
        payload: The file's bytes.
        columns: Expected column names, or None when only a refusal is
            correct.
        rows: Expected cells, row by row: str, int, float, Date or None
            (missing).
    """

    name: str
    group: str
    suffix: str
    payload: bytes
    columns: tuple[str, ...] | None = None
    rows: tuple[tuple[Any, ...], ...] = ()


def text(
    name: str,
    group: str,
    body: str,
    columns: tuple[str, ...] | None,
    rows: tuple[tuple[Any, ...], ...] = (),
    suffix: str = ".csv",
) -> Case:
    """A UTF-8 case.

    Args:
        name: Identifier.
        group: What is awkward about it.
        body: The file's text.
        columns: Expected column names, or None for a refusal.
        rows: Expected cells.
        suffix: File suffix.

    Returns:
        The case.
    """
    return Case(name, group, suffix, body.encode("utf-8"), columns, rows)


def chunked() -> Case:
    """A column of digits whose later rows include a word.

    Returns:
        The case: the token "1" must read as one value throughout.
    """
    body = "v\n" + "1\n" * (CHUNKED_ROWS - 1) + "x\n1\n"
    rows = ((str(v),) for v in ["1"] * (CHUNKED_ROWS - 1) + ["x", "1"])
    return Case(
        "mixed_chunks",
        "mixed types",
        ".csv",
        body.encode(),
        ("v",),
        tuple(rows),
    )


def truncated_gzip() -> Case:
    """A gzip CSV cut in the middle.

    Returns:
        The case; only a refusal is correct.
    """
    whole = gzip.compress(b"a,b\n" + b"1,2\n" * 5000, mtime=0)
    return Case("gzip_cut", "truncated", ".csv.gz", whole[: len(whole) // 2])


def corpus() -> list[Case]:
    """The awkward-file corpus with hand-written expectations.

    Returns:
        Every case, in a fixed order.
    """
    big = 9007199254740993  # 2**53 + 1: not a float64
    huge = 123456789012345678901234567890
    return [
        text(
            "plain",
            "baseline",
            "a,b\n1,x\n2,y\n",
            ("a", "b"),
            ((1, "x"), (2, "y")),
        ),
        text("bom", "encoding", "﻿id,name\n1,é\n", ("id", "name"), ((1, "é"),)),
        text(
            "bom_quoted",
            "encoding",
            '﻿"id",name\n1,a\n',
            ("id", "name"),
            ((1, "a"),),
        ),
        Case(
            "latin1",
            "encoding",
            ".csv",
            "id,name\n1,café\n".encode("latin-1"),
            ("id", "name"),
            ((1, "café"),),
        ),
        Case(
            "utf16",
            "encoding",
            ".csv",
            "id,name\n1,ü\n".encode("utf-16"),
            ("id", "name"),
            ((1, "ü"),),
        ),
        text(
            "semicolon",
            "delimiter",
            "a;b;c\n1;2;3\n4;5;6\n",
            ("a", "b", "c"),
            ((1, 2, 3), (4, 5, 6)),
        ),
        text(
            "semicolon_decimal_comma",
            "delimiter",
            "a;b\n1,5;2\n3,5;4\n",
            ("a", "b"),
            ((1.5, 2), (3.5, 4)),
        ),
        text("pipe", "delimiter", "a|b\n1|2\n", ("a", "b"), ((1, 2),)),
        text(
            "tab",
            "delimiter",
            "a\tb\n1\tx,y\n",
            ("a", "b"),
            ((1, "x,y"),),
            suffix=".tsv",
        ),
        text(
            "quoted_newline",
            "quoting",
            'id,note\n1,"line one\nline two"\n2,z\n',
            ("id", "note"),
            ((1, "line one\nline two"), (2, "z")),
        ),
        text(
            "quoted_delimiter",
            "quoting",
            'id,note\n1,"a, ""b"""\n',
            ("id", "note"),
            ((1, 'a, "b"'),),
        ),
        text(
            "crlf",
            "line endings",
            "a,b\r\n1,2\r\n3,4\r\n",
            ("a", "b"),
            ((1, 2), (3, 4)),
        ),
        text(
            "cr_only",
            "line endings",
            "a,b\r1,2\r3,4\r",
            ("a", "b"),
            ((1, 2), (3, 4)),
        ),
        text(
            "mixed_small",
            "mixed types",
            "v\n1\nx\n2.5\n",
            ("v",),
            (("1",), ("x",), ("2.5",)),
        ),
        chunked(),
        text(
            "empty_column",
            "empty columns",
            "a,b,c\n1,,3\n2,,4\n",
            ("a", "b", "c"),
            ((1, None, 3), (2, None, 4)),
        ),
        text("duplicate_header", "duplicate headers", "a,a,b\n1,2,3\n", None),
        text(
            "duplicate_header_tsv",
            "duplicate headers",
            "a\ta\n1\t2\n",
            None,
            suffix=".tsv",
        ),
        text(
            "beyond_int64",
            "huge numbers",
            f"v\n{huge}\n1\n",
            ("v",),
            ((huge,), (1,)),
        ),
        text(
            "uint64_max",
            "huge numbers",
            "v\n18446744073709551615\n0\n",
            ("v",),
            ((18446744073709551615,), (0,)),
        ),
        text(
            "long_int_with_missing",
            "huge numbers",
            f"k,v\n1,{big}\n2,\n",
            ("k", "v"),
            ((1, big), (2, None)),
        ),
        text(
            "long_int_with_float",
            "huge numbers",
            f"v\n{big}\n1.5\n",
            ("v",),
            ((big,), (1.5,)),
        ),
        text(
            "tiny_and_huge_floats",
            "huge numbers",
            "v\n1e-300\n1e300\n",
            ("v",),
            ((1e-300,), (1e300,)),
        ),
        text(
            "leading_zeros",
            "mixed types",
            "zip\n02134\n10001\n",
            ("zip",),
            (("02134",), ("10001",)),
        ),
        text(
            "dates_iso",
            "dates",
            "d\n2024-01-02\n2024-12-31\n",
            ("d",),
            (
                (Date("2024-01-02", "2024-01-02"),),
                (Date("2024-12-31", "2024-12-31"),),
            ),
        ),
        text(
            "dates_mixed",
            "dates",
            "d\n2024-01-02\n03/01/2024\nJan 4 2024\n2024-01-05T06:07:08Z\n",
            ("d",),
            (
                (Date("2024-01-02", "2024-01-02"),),
                (Date("03/01/2024", "2024-03-01"),),
                (Date("Jan 4 2024", "2024-01-04"),),
                (Date("2024-01-05T06:07:08Z", "2024-01-05T06:07:08+00:00"),),
            ),
        ),
        text("truncated_row", "truncated", "a,b,c\n1,2,3\n4,5", None),
        text("truncated_quote", "truncated", 'a,b\n1,"abc', None),
        truncated_gzip(),
        text("extra_field", "truncated", "a,b\n1,2\n3,4,5\n", None),
        text("header_only", "truncated", "a,b\n", None),
        text("empty_file", "truncated", "", None),
        text(
            "jsonl_long_int",
            "huge numbers",
            f'{{"v": {big}}}\n{{"v": 1}}\n',
            ("v",),
            ((big,), (1,)),
            suffix=".jsonl",
        ),
        text(
            "jsonl_truncated",
            "truncated",
            '{"v": 1}\n{"v": 2',
            None,
            suffix=".jsonl",
        ),
    ]


def fingerprint(cases: list[Case]) -> str:
    """Hash of the corpus: names, bytes and expectations.

    Args:
        cases: The corpus.

    Returns:
        A SHA-256 hex digest.
    """
    digest = hashlib.sha256()
    for case in cases:
        digest.update(case.name.encode() + case.suffix.encode())
        digest.update(hashlib.sha256(case.payload).digest())
        digest.update(repr((case.columns, case.rows)).encode())
    return digest.hexdigest()


def matches(cell: Any, expected: Any, tolerance: float) -> bool:
    """Whether one read cell is the expected one.

    Args:
        cell: What the reader returned.
        expected: str, int, float, Date or None.
        tolerance: Relative tolerance for floats.

    Returns:
        True when the cell carries the expected value without loss.
    """
    missing = bool(pandas.api.types.is_scalar(cell) and pandas.isna(cell))
    if expected is None or missing:
        return expected is None and missing
    if isinstance(expected, Date):
        if isinstance(cell, pandas.Timestamp):
            return cell == pandas.Timestamp(expected.instant)
        return isinstance(cell, str) and cell == expected.text
    if isinstance(expected, str):
        return isinstance(cell, str) and cell == expected
    if isinstance(cell, str):
        return cell.strip() == str(expected)
    return number(cell, expected, tolerance)


def number(cell: Any, expected: float, tolerance: float) -> bool:
    """Whether a numeric cell is the expected number.

    Args:
        cell: A non-missing, non-text cell.
        expected: The exact integer or the float meant.
        tolerance: Relative tolerance for floats.

    Returns:
        True for the exact integer, or a float within `tolerance`; False
        for booleans and anything not a number.
    """
    if isinstance(cell, bool | numpy.bool_):
        return False
    if not isinstance(cell, int | float | numpy.integer | numpy.floating):
        return False
    if isinstance(expected, int):
        if isinstance(cell, int | numpy.integer):
            return int(cell) == expected
        value = float(cell)
        return value.is_integer() and int(value) == expected
    return math.isclose(float(cell), expected, rel_tol=tolerance)


def judge(case: Case, frame: pandas.DataFrame, tolerance: float) -> str:
    """Why a returned table is wrong, or the empty string when correct.

    Args:
        case: The case read.
        frame: What the reader returned.
        tolerance: Relative tolerance for floats.

    Returns:
        A short reason, or "" when the table equals the expected one.
    """
    if case.columns is None:
        return f"expected a refusal, got {frame.shape} {list(frame.columns)}"
    if [str(c) for c in frame.columns] != list(case.columns):
        return f"columns {list(frame.columns)}"
    if not isinstance(frame.index, pandas.RangeIndex):
        return f"index {list(frame.index[:3])} taken from the data"
    if len(frame) != len(case.rows):
        return f"{len(frame)} rows, expected {len(case.rows)}"
    cells = frame.astype(object).to_numpy()
    for row, (got, want) in enumerate(zip(cells, case.rows, strict=True)):
        for column, (cell, expected) in enumerate(zip(got, want, strict=True)):
            if not matches(cell, expected, tolerance):
                return (
                    f"row {row} column {case.columns[column]}: "
                    f"{cell!r} ({type(cell).__name__}), expected {expected!r}"
                )
    return ""


def read(case: Case, directory: Path, tolerance: float) -> dict[str, Any]:
    """Reads one case with the product's reader and classifies it.

    Args:
        case: The case.
        directory: Where the file is written.
        tolerance: Relative tolerance for floats.

    Returns:
        The case's name, group, whether a table was expected, the outcome
        (`correct`, `refused` or `wrong`) and the reason.
    """
    path = directory / f"{case.name}{case.suffix}"
    path.write_bytes(case.payload)
    outcome = {
        "name": case.name,
        "group": case.group,
        "expects_table": case.columns is not None,
    }
    try:
        frame = tabular.read_table(path)
    except errors.DataError as error:
        return outcome | {"outcome": "refused", "reason": str(error)[:200]}
    reason = judge(case, frame, tolerance)
    return outcome | {
        "outcome": "wrong" if reason else "correct",
        "reason": reason,
    }


def world(seed: int, tolerance: float) -> dict[str, Any]:
    """One random schema: the product's feature against the naive one.

    Args:
        seed: World seed.
        tolerance: Relative and absolute tolerance.

    Returns:
        The seed, whether every target agrees, the largest absolute gap and
        the targets compared.
    """
    schema = ingest_checks.random_schema(seed, STREAM)
    fast = ingest_checks.fast_feature(schema).to_numpy(dtype=float)
    naive = numpy.array(ingest_checks.naive_feature(schema), dtype=float)
    equal = bool(
        fast.shape == naive.shape
        and numpy.allclose(
            fast, naive, rtol=tolerance, atol=tolerance, equal_nan=True
        )
    )
    both = ~(numpy.isnan(fast) | numpy.isnan(naive))
    gap = float(numpy.max(numpy.abs(fast - naive)[both], initial=0.0))
    return {
        "seed": seed,
        "equal": equal,
        "gap": gap,
        "targets": len(fast),
        "nested": schema.nested,
    }


def tables() -> list[tuple[str, str]]:
    """The tables split for the hygiene criterion.

    Returns:
        `(table, declared)` pairs: `<dataset>/<declaration>` for each
        relational dataset and each of `DECLARATIONS`, then
        `refusal/<task>` with `declared` "task" for each real refusal task.
    """
    found = [
        (f"{d.name}/{declared}", declared)
        for d in relational.DATASETS
        for declared in DECLARATIONS
    ]
    found += [
        (f"refusal/{name}", "task")
        for name, task in refusal.catalog().items()
        if task.group == "real"
    ]
    return found


def declare(dataset: relational.Dataset, declared: str) -> dict[str, Any]:
    """The `Project` arguments of one declaration on a child table.

    Args:
        dataset: The relational dataset.
        declared: One of `DECLARATIONS`.

    Returns:
        The keyword arguments.
    """
    child = dataset.child
    entity: dict[str, Any] = {"group_column": child.key}
    time: dict[str, Any] = {"time_column": child.event}
    rolling: dict[str, Any] = {"validation": "rolling"}
    unseen: dict[str, Any] = {"entities": "unseen"}
    options: dict[str, dict[str, Any]] = {
        "rows": {},
        "entity": entity,
        "time": time,
        "both": entity | time,
        "rolling": entity | time | rolling,
        "unseen": entity | time | unseen,
        "unseen_rolling": entity | time | rolling | unseen,
    }
    return options[declared]


def shared(left: numpy.ndarray, right: numpy.ndarray) -> int:
    """Values found on both sides, missing values ignored.

    Args:
        left: One side's values.
        right: The other side's values.

    Returns:
        The number of distinct values on both sides.
    """
    one = {v for v in left.tolist() if not pandas.isna(v)}
    return len(one & {v for v in right.tolist() if not pandas.isna(v)})


def hygiene(
    frame: pandas.DataFrame, target: Any, options: dict[str, Any]
) -> dict[str, Any]:
    """Splits a table as a user run does and counts what crosses.

    Args:
        frame: The table.
        target: The target column.
        options: Further `Project` arguments (the declarations).

    Returns:
        The strategy, the number of splits checked (the holdout and each
        fold), the number of splits sharing an entity, a duplicate record,
        a duplicate record of one declared entity (`copy`) and a timestamp
        across their sides, the refusal message (empty
        when the product split the table), an empty error, the declared
        `entities` and whether an entity column was declared (`grouped`).
    """
    with tempfile.TemporaryDirectory() as folder:
        spec = scmle.api.project.Project(
            frame, target, output_dir=folder, **options
        ).spec
    data = spec.data
    rows = frame.loc[frame[data.target].notna()].reset_index(drop=True)
    record: dict[str, Any] = dict.fromkeys(CROSSINGS + REPORTED, 0) | {
        "strategy": "",
        "splits": 0,
        "refused": "",
        "error": "",
        "entities": spec.validation.entities,
        "grouped": bool(data.group_column),
    }
    try:
        task = scmle.data.validation.target.TaskResolver(spec).resolve(
            rows[data.target]
        )
        splitter = splitters.Splitter(spec, task)
        record["strategy"] = splitter.strategy_name
        manifest = splitter.split(rows).manifest
    except (errors.BlockedError, errors.DataError) as error:
        record["refused"] = f"{type(error).__name__}: {error}"[:300]
        return record
    record["strategy"] = manifest.strategy
    copies = duplicates.duplicate_groups(rows, splitter.feature_columns(rows))
    copies = numpy.where(copies == duplicates.NOT_DUPLICATED, None, copies)
    columns = {
        "entity": data.group_column,
        "time": data.time_column,
    }
    entity = (
        rows[data.group_column].to_numpy()
        if data.group_column
        else numpy.full(len(rows), None)
    )
    owned = numpy.array(
        [
            None if c is None or pandas.isna(e) else repr((c, e))
            for c, e in zip(copies, entity, strict=True)
        ],
        dtype=object,
    )
    development = numpy.asarray(manifest.train_ids)
    sides = [(development, numpy.asarray(manifest.test_ids))] + [
        (development[list(fit)], development[list(held)])
        for fit, held in manifest.folds
    ]
    for left, right in sides:
        record["duplicate"] += bool(shared(copies[left], copies[right]))
        record["copy"] += bool(shared(owned[left], owned[right]))
        for name, column in columns.items():
            if column:
                values = rows[column].to_numpy()
                record[name] += bool(shared(values[left], values[right]))
    record["splits"] = len(sides)
    return record


def split(table: str, declared: str) -> dict[str, Any]:
    """Loads one table of `tables` and checks its split.

    Args:
        table: The table's name from `tables`.
        declared: Its declaration.

    Returns:
        `hygiene`'s record with the table and declaration; on a failure to
        load or split, the error.
    """
    record: dict[str, Any] = {"table": table, "declared": declared}
    try:
        source, name = table.split("/", 1)
        if source == "refusal":
            frame, target, options = refusal.catalog()[name].build()
        else:
            dataset = next(d for d in relational.DATASETS if d.name == source)
            frame = relational.product(
                dataset, relational.fetch(dataset), dataset.child
            )
            target = dataset.child.measures[0]
            options = declare(dataset, name)
        return record | hygiene(frame, target, options)
    except Exception as error:  # noqa: BLE001 -- counted against the row
        return (
            record
            | dict.fromkeys(CROSSINGS + REPORTED, 0)
            | {
                "strategy": "",
                "splits": 0,
                "refused": "",
                "error": f"{type(error).__name__}: {error}"[:300],
                "entities": "",
                "grouped": False,
            }
        )


def breached(record: dict[str, Any], crossing: str) -> bool:
    """Whether one kind of crossing in a split record is a violation.

    Args:
        record: One `split` record.
        crossing: One of `CROSSINGS`.

    Returns:
        True when it crossed, except an entity crossing, or a duplicate
        record of one entity crossing (`copy`), in a temporal or rolling
        split of a run that did not declare `entities="unseen"`: there the
        entity is known and its past may train while its future is held
        out. A timestamp is a violation under every strategy. A record
        measured before `copy` was counted has none.
    """
    count = record.get(crossing, 0)
    if crossing in ("entity", "copy") and record.get("strategy") in TIMED:
        return bool(count) and record.get("entities") == "unseen"
    return bool(count)


def crossed(record: dict[str, Any]) -> bool:
    """Whether a split record fails the criterion.

    Args:
        record: One `split` record.

    Returns:
        True when a crossing is a violation (`breached`) or the table could
        not be split.
    """
    return bool(record["error"]) or any(breached(record, c) for c in CROSSINGS)


def digest() -> str:
    """Hash of the code measured, the code measuring it and the corpus.

    Returns:
        A SHA-256 hex digest.
    """
    code = certificates.code_hash(
        tabular,
        synthesize,
        scmle.assurance.ingest.synthesize.evaluation,
        scmle.assurance.ingest.synthesize.grammar,
        scmle.assurance.ingest.synthesize.selection,
        asof_view,
        ingest_naive,
        ingest_checks,
        relational,
        splitters.splits,
        splitters.temporal,
        splitters.Splitter,
        duplicates,
        tables,
        declare,
        shared,
        hygiene,
        split,
        breached,
        crossed,
        corpus,
        matches,
        number,
        judge,
        read,
        world,
    )
    return hashlib.sha256(
        (code + fingerprint(corpus())).encode("ascii")
    ).hexdigest()


def tolerance() -> float:
    """The comparison tolerance from acceptance.toml.

    Returns:
        `reference_rel_tol`.
    """
    return acceptance.load()[VARIABLE].value


def measure(worlds: int = WORLDS, workers: int = WORKERS) -> dict[str, Any]:
    """Reads the corpus, runs the worlds and builds the stored document.

    Args:
        worlds: Random schemas compared.
        workers: Worker processes for the worlds.

    Returns:
        The JSON-ready document.

    Raises:
        InputError: For fewer than one world or worker.
    """
    if worlds < 1 or workers < 1:
        raise errors.InputError("worlds and workers must be at least 1")
    started = time.monotonic()
    tol = tolerance()
    cases = corpus()
    with tempfile.TemporaryDirectory() as folder:
        files = [read(c, Path(folder), tol) for c in cases]
    with futures.ProcessPoolExecutor(workers) as pool:
        real = pool.map(
            relational.compare,
            relational.DATASETS,
            [tol] * len(relational.DATASETS),
        )
        done = list(
            pool.map(world, range(worlds), [tol] * worlds, chunksize=25)
        )
        results = list(real)
        listed = tables()
        hygienic = list(pool.map(split, *zip(*listed, strict=True)))
    counts = {
        k: sum(f["outcome"] == k for f in files)
        for k in ("correct", "refused", "wrong")
    }
    return {
        "claim": "C5",
        "variable": VARIABLE,
        "tolerance": tol,
        "files": files,
        "corpus": counts
        | {
            "cases": len(files),
            "over_refused": sum(
                f["outcome"] == "refused" and f["expects_table"] for f in files
            ),
            "wrong_names": [
                f["name"] for f in files if f["outcome"] == "wrong"
            ],
        },
        "point_in_time": {
            "worlds": len(done),
            "equal": sum(w["equal"] for w in done),
            "nested": sum(w["nested"] for w in done),
            "targets": sum(w["targets"] for w in done),
            "max_gap": max((w["gap"] for w in done), default=0.0),
            "unequal_seeds": [w["seed"] for w in done if not w["equal"]],
            "stream": STREAM,
            "reference": "benchmarks/qualification/ingest_naive.py",
        },
        "real": results,
        "real_datasets": sum(relational.agrees(r) for r in results),
        "splits": hygienic,
        "not_measured": list(NOT_MEASURED),
        "code": digest(),
        "seconds": round(time.monotonic() - started, 1),
    }


def load(document: dict[str, Any] | None) -> dict[str, Any] | None:
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
    """Why the document cannot decide the row.

    Args:
        document: The document; read from `PATH` when None.

    Returns:
        Messages for a missing file, other code or corpus, a tolerance other
        than acceptance.toml's, missing corpus cases, fewer worlds than
        `WORLDS`, real datasets not run or tables not split; empty when the
        document is current.
    """
    found = load(document)
    if found is None:
        return [f"{PATH.name} is missing"]
    issues = []
    if found.get("code") != digest():
        issues.append("reading: measured with other code or corpus")
    tol = tolerance()
    if found.get("tolerance") != tol:
        issues.append(
            f"reading: tolerance {found.get('tolerance')} is not {tol}"
        )
    names = {f.get("name") for f in found.get("files", [])}
    missing = [c.name for c in corpus() if c.name not in names]
    if missing:
        issues.append(f"reading: cases not read: {missing}")
    ran = {r.get("name") for r in found.get("real", [])}
    absent = [d.name for d in relational.DATASETS if d.name not in ran]
    if absent or len(relational.DATASETS) < REAL_DATASETS:
        issues.append(f"reading: real datasets not run: {absent}")
    if "splits" not in found:
        issues.append("reading: split criterion not measured")
    else:
        done = {r.get("table") for r in found["splits"]}
        unsplit = [t for t, _ in tables() if t not in done]
        if unsplit:
            issues.append(f"reading: tables not split: {unsplit}")
    count = found.get("point_in_time", {}).get("worlds", 0)
    if count < WORLDS:
        issues.append(
            f"reading: {count} worlds, fewer than the {WORLDS} required"
        )
    return issues


def summary(document: dict[str, Any] | None = None) -> tuple[str, str]:
    """The scorecard's value and verdict for the row.

    Args:
        document: The document; read from `PATH` when None.

    Returns:
        The value text and `fail` when any file is silently wrong, any
        world differs from the reference, any real dataset fails to agree
        (or to be read), or any split shares a timestamp or a duplicate
        record of one declared entity across its sides, or an entity where
        `breached` says it may not (or a table could not be split);
        otherwise `pass`; `not measured` when `problems` finds any. A
        duplicate record on both sides is reported, not failed. The value
        counts the runs with an entity column declared `known` and
        `unseen`.
    """
    found = load(document)
    if found is None or "corpus" not in found:
        return "no measurement", "not measured"
    files = found["corpus"]
    pit = found.get("point_in_time", {})
    unequal = pit.get("worlds", 0) - pit.get("equal", 0)
    real = found.get("real", [])
    agree = sum(relational.agrees(r) for r in real)
    hygienic = found.get("splits", [])
    bad = [r for r in hygienic if crossed(r)]
    counts = ", ".join(
        f"{c} {sum(breached(r, c) for r in hygienic)}" for c in CROSSINGS
    )
    copied = sum(bool(r.get("duplicate")) for r in hygienic)
    grouped = [r for r in hygienic if r.get("grouped")]
    declared = ", ".join(
        f"{sum(r.get('entities') == way for r in grouped)} {way}"
        for way in ("known", "unseen")
    )
    value = (
        f"awkward files {files['cases']}: {files['correct']} correct, "
        f"{files['refused']} refused ({files['over_refused']} readable), "
        f"{files['wrong']} silently wrong; point in time "
        f"{pit.get('equal', 0)}/{pit.get('worlds', 0)} simulated worlds "
        f"equal the naive reference within {found.get('tolerance')}; "
        f"real relational datasets {agree} of {len(real)} equal the naive "
        f"reference (at least {REAL_DATASETS} required); splits: "
        f"{len(hygienic)} table runs, "
        f"{sum(r['splits'] for r in hygienic)} splits checked, "
        f"{sum(bool(r['refused']) for r in hygienic)} refused, "
        f"{sum(bool(r['error']) for r in hygienic)} not loaded; "
        f"{len(bad)} cross (table runs with a split sharing {counts}); "
        f"{copied} runs share a duplicate record (reported, not a "
        "violation); "
        f"entity declared on {len(grouped)} runs: {declared}"
    )
    if problems(found):
        return value, "not measured"
    if files["wrong"] or unequal or agree < len(real) or bad:
        return value, "fail"
    return value, "pass"


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
