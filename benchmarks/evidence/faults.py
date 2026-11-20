"""Claim C1: no silent wrong outcome; every registered gate blocks its fault.

Criterion (plan section 5, `acceptance.toml`): every gate registered in
`gates.toml` (loaded by `benchmarks.registry`) has a pass, a block and an
override path, and of the injected failures the share that blocks or ends
in a recorded override is at least `gate_fault_block_min`.

What is run. For each registered gate this module injects the condition
the gate guards (`INJECTIONS`) and records how the call ended:

* `blocked`: it raised `BlockedError` or `LeakageError`; the gate that
  raised is the innermost traceback frame `gates.toml` registers
  (`benchmarks.evidence.refusal.locate`), so a block by another gate is
  visible as such;
* `override`: the documented override was applied and the run continued
  with the override recorded (a logged warning, a curation note, a
  decision-log entry, or the setting in the run's `config.json`);
* `silent`: it continued and nothing records the fault;
* `failed`: another exception ended it;
* `not injected`: no injection was built; the reason is stored.

An injection is either `run` level, one public entry point executed end
to end (`Project(...).run()`, `Project.resume` or a `scmle` command on
files) with fast settings (no tuning,
ensembles or deep learning), or `site` level, a direct call of the gate's
function (or the command or stage function that contains it). Each gate's
pass path is run too (`control`) where a clean input is cheap to build; it
decides nothing. A path whose gate sits behind an optional extra this
interpreter cannot import (`EXTRAS`: the ONNX divergence gate has no
onnxruntime wheel on free-threaded 3.14t) runs under `INTERPRETER` with the
extra installed, and its record names that interpreter.

Two parts are measured and kept apart:

* `real` (`REAL`): the fault is planted in a public table read from the
  local caches (`refusal.fetch`: scikit-learn's bundled sets and the
  OpenML cache; nothing is downloaded) and reaches the gate through a run.
  Survival and uplift faults use the real churn table (OpenML 40701:
  account length and leaving; the international plan as the arm), ranking
  faults the real house sales with each zip code a query (OpenML 42731),
  and the ingest gates a SQLite source of those sales by zip code. Faults
  of the test-set custody are planted by a stage enrolled through the
  public plugin registry. A gate no run reaches has no real injection: it
  is recorded `not injected` with the reason and counts against the claim;
* `simulated` (`INJECTIONS`): the faults planted in simulated 400-row
  tables and synthetic inputs, through runs and direct gate calls.

The share counts one unit per gate (its fault) and one per gate whose
override is not `none` (its fault under the override). A unit not injected
counts against the claim, and in the real part so does a unit not injected
through a run. A fault blocked by another gate is listed (`elsewhere`) and
does not count for the gate injected. The claim is about real use, so its
injected-failure clause is decided on the real part: the row passes when
the real share and the simulated share are both at least the floor and
every registered gate was attempted in both parts; with no real part it
is `not measured`. `acceptance.toml` gives this policy variable no
statistical rule, so the minimum sample is the whole registry.

NOT measured: faults reached through every code path to a gate (one
injection per gate and part), tables published as survival, ranking or
uplift data and sources published as relational (none is cached).

    python -m benchmarks.evidence.faults --write
    python -m benchmarks.evidence.faults --check
"""

import argparse
from collections.abc import Callable
from concurrent import futures
import contextlib
import dataclasses
from fractions import Fraction
import functools
import hashlib
import importlib
import io
import json
import logging
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import types
from typing import Any

import numpy
import pandas

from benchmarks import acceptance
from benchmarks import registry
from benchmarks.evidence import refusal
from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import tasks

PATH = Path(__file__).with_suffix(".json")
ROOT = Path(__file__).resolve().parents[2]
VARIABLE = "gate_fault_block_min"
# Policy: seconds of budget per run. Preparation of a real charter run took
# 82 to 130 s on a loaded laptop, which a 60 s budget refuses through
# `train.first_trial` before the injected gate is reached.
BUDGET = 180
WORKERS = 2
ROWS = 400
SEED = 7
OUTCOMES = ("blocked", "override", "silent", "failed", "not injected")
SATISFIED = frozenset({"blocked", "override"})
# The real part first: it decides the claim's injected-failure clause.
PARTS = ("real", "simulated")
NOT_MEASURED = (
    "more than one injection per gate and part or per code path to a gate",
    (
        "tables published as survival, uplift or ranking data (none is "
        "cached; the real churn and house sales tables serve those kinds)"
    ),
    (
        "a source published as relational (the ingest gates read the real "
        "house sales split into zip codes and sales)"
    ),
)
# Optional extras a gate's paths need, by extra: the modules that must
# import for a path to reach its gate rather than fail on the import.
EXTRAS = {"onnx": ("skl2onnx", "onnxruntime")}
# The interpreter that runs a path whose extra this one cannot import: the
# newest CPython every extra has wheels for (onnxruntime has none for the
# free-threaded 3.14t), the one the CI `onnx` job runs on.
INTERPRETER = "3.13"
# Seconds to build that interpreter's isolated environment, on top of the
# path's own `BUDGET`: 42 s measured cold on 2026-11-19, about three times.
SETUP = 120
FAST = {
    "search": tasks.Search(top_models=0),
    "strategies": tasks.Strategies(top_models=0),
    "ensemble": tasks.Stacking(size=0),
    "models": tasks.Candidates(
        exclude_families=["bagging", "deep_learning", "foundation"]
    ),
    "enforcement": "cooperative",
}


class Warnings(logging.Handler):
    """Keeps the text of every warning logged while it is attached."""

    def __init__(self) -> None:
        """Initializes an empty record."""
        super().__init__(logging.WARNING)
        self.texts: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Keeps one record's message.

        Args:
            record: The log record.
        """
        self.texts.append(record.getMessage())


@contextlib.contextmanager
def listening() -> Any:
    """Collects warnings logged by the product while the block runs.

    Yields:
        The handler; its `texts` hold the messages.
    """
    handler = Warnings()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield handler
    finally:
        root.removeHandler(handler)


def launch(
    frame: pandas.DataFrame,
    target: Any,
    adjust: Callable[[tasks.Config], tasks.Config] | None = None,
    **options: Any,
) -> Any:
    """Runs one project end to end with fast settings.

    Args:
        frame: The table.
        target: Its target column or columns.
        adjust: Edits the built specification (for settings `Project` does
            not take as arguments), or None.
        **options: Further `Project` arguments; `output_dir` defaults to
            a new temporary directory.

    Returns:
        The completed run; its folder is under a temporary directory that
        lives until the process ends.
    """
    import scmle.api.project

    folder = tempfile.mkdtemp(prefix="faults-")
    project = scmle.api.project.Project(
        frame,
        target,
        time_budget=BUDGET,
        **{"output_dir": folder, **FAST, **options},
    )
    if adjust is not None:
        project.spec = adjust(project.spec)
        project.orchestrator.spec = project.spec
    return project.run()


def configured(run: Any) -> dict[str, Any]:
    """The configuration a run saved.

    Args:
        run: A completed run.

    Returns:
        The parsed `config.json` of its folder.
    """
    loaded: dict[str, Any] = json.loads(
        (Path(run.directory) / "config.json").read_text(encoding="utf-8")
    )
    return loaded


def text(run: Any, name: str) -> str:
    """The text of one file of a run's folder, or ''.

    Args:
        run: A completed run.
        name: Path inside the run folder.

    Returns:
        The file's text, empty when it is absent.
    """
    path = Path(run.directory) / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


def spec(task: tasks.TaskRequest = "auto", **fields: Any) -> tasks.Config:
    """A minimal specification for a direct call of a gate.

    Args:
        task: The task request.
        **fields: `Dataset` fields; `strategy`, `folds`, `horizon_rows`,
            `embargo_rows` and `entities` go to `Validation` instead.

    Returns:
        The specification, with target `y` unless `target` is given.
    """
    keys = ("strategy", "folds", "horizon_rows", "embargo_rows", "entities")
    validation = {k: fields.pop(k) for k in keys if k in fields}
    fields.setdefault("target", "y")
    return tasks.Config(
        data=tasks.Dataset(uri="memory://faults", **fields),
        task=task,
        validation=tasks.Validation(**validation),
    )


Task = tuple[pandas.DataFrame, Any, dict[str, Any]]

# Real tables by task kind, read from the local caches by `refusal.fetch`:
# OpenML 31 (credit-g), OpenML 54 (vehicle), scikit-learn's diabetes,
# OpenML 40691 (red wine quality) and OpenML 40589 (emotions).
TABLES: dict[str, Callable[[], Task]] = {
    "binary": refusal.table(31, "class", "classification"),
    "multiclass": refusal.table(54, "Class", "classification"),
    "regression": refusal.table("load_diabetes", "target", "regression"),
    "ordinal": refusal.table(40691, "class", "ordinal"),
    "multilabel": refusal.table(
        40589,
        [
            "amazed.suprised",
            "happy.pleased",
            "relaxing.calm",
            "quiet.still",
            "sad.lonely",
            "angry.aggresive",
        ],
        "multilabel",
    ),
}
# OpenML 42731 (King County house sales): real sale dates.
SALES = 42731
# Rows of a real table kept for a run, so it fits the short budget.
SAMPLE = 1000


def sales(rows: int = SAMPLE) -> pandas.DataFrame:
    """Real house sales with their sale date as one column.

    Args:
        rows: Rows kept, drawn with a fixed seed; 0 keeps every sale.

    Returns:
        The sales, oldest first, with a `date` column and the price.
    """
    frame = refusal.fetch(SALES, rows or 10**6)
    parts = frame[["date_year", "date_month", "date_day"]]
    parts = parts.set_axis(["year", "month", "day"], axis=1)
    frame["date"] = pandas.to_datetime(parts)
    frame = frame.drop(columns=["date_year", "date_month", "date_day"])
    return frame.sort_values("date", kind="stable").reset_index(drop=True)


# OpenML 40701 (telecom churn): each customer's account length in days and
# whether the customer left, the duration and event of a survival table; the
# international plan as the arm and leaving as the outcome of an uplift
# table. The phone number identifies the customer and is dropped.
CHURN = 40701


def churn(kind: str) -> Task:
    """The real churn table as a survival or an uplift task.

    Args:
        kind: `survival` (duration `account_length`, event `event`, 1 when
            the customer left) or `uplift` (outcome `churned`, arm
            `treated`, the international plan).

    Returns:
        `SAMPLE` rows, the target and the `Project` arguments.
    """
    frame = refusal.fetch(CHURN, SAMPLE).drop(columns="phone_number")
    left = pandas.to_numeric(frame.pop("class")).astype(int)
    if kind == "survival":
        frame["event"] = left
        options = {"task": "survival", "event_column": "event"}
        return frame, "account_length", options
    frame["churned"] = left
    frame["treated"] = pandas.to_numeric(frame.pop("international_plan"))
    frame["treated"] = frame["treated"].astype(int)
    return frame, "churned", {"task": "uplift", "treatment_column": "treated"}


def listings() -> Task:
    """The real house sales as a ranking task.

    Each zip code is a query (`query`) and each sale a document whose
    relevance is its view grade (`view`, 0 to 4).

    Returns:
        `SAMPLE` sales, the target and the `Project` arguments.
    """
    frame = sales().drop(columns="date")
    frame["query"] = frame.pop("zipcode").astype(str)
    return frame, "view", {"task": "ranking", "query_column": "query"}


def clean(kind: str, part: str = "simulated") -> Task:
    """A clean task of one kind.

    Args:
        kind: As for `refusal.simulate`.
        part: `simulated` (a generated table of `ROWS` rows) or `real` (a
            cached public table of that kind; for `forecast` the daily
            mean price of the real house sales, for `survival` and
            `uplift` the churn table, for `ranking` the house sales).

    Returns:
        The table, target and `Project` arguments.

    Raises:
        InputError: For a real kind no cached table serves.
    """
    if part == "simulated":
        return refusal.simulate(kind, ROWS, SEED)
    if part == "real" and kind in ("survival", "uplift"):
        return churn(kind)
    if part == "real" and kind == "ranking":
        return listings()
    if kind == "forecast":
        daily = sales(0).groupby("date")["price"].mean().reset_index()
        options = {
            "task": "forecast",
            "horizon": 4,
            "time_column": "date",
            "seasonal_period": 7,
        }
        return daily, "price", options
    if part != "real" or kind not in TABLES:
        raise errors.InputError(f"no {part} table of kind {kind!r}")
    frame, target, options = TABLES[kind]()
    if len(frame) > SAMPLE:
        frame = frame.sample(SAMPLE, random_state=SEED)
    return frame.reset_index(drop=True), target, options


def planted(kind: str, defect: str, part: str = "simulated") -> Any:
    """Runs a refusal-benchmark planted defect end to end.

    Args:
        kind: The task kind.
        defect: As for `refusal.spoil`.
        part: Where the clean table comes from (see `clean`).

    Returns:
        The completed run, if the product did not refuse.
    """
    frame, target, options = clean(kind, part)
    refusal.spoil(frame, target, defect)
    options = {**options, **refusal.OVERRIDES.get(defect, {})}
    return launch(frame, target, **options)


def database(folder: str) -> str:
    """Writes a SQLite source of customers and their orders.

    Args:
        folder: Where to write it.

    Returns:
        The path of the database file.
    """
    path = str(Path(folder) / "source.sqlite")
    keeper = sqlite3.connect(path)
    pandas.DataFrame(
        {"customer_id": range(10), "signup_month": [1] * 10}
    ).to_sql("customers", keeper, index=False)
    pandas.DataFrame(
        {"order_id": range(20), "customer_id": [k % 10 for k in range(20)]}
    ).to_sql("orders", keeper, index=False)
    keeper.commit()
    keeper.close()
    return path


def market(folder: str | Path) -> str:
    """Writes a SQLite source of the real house sales by zip code.

    Two tables, the relational shape an ingest plan reads: `zipcodes` (one
    row per zip code of OpenML 42731) and `sales` (every sale, with its
    zip code, its day counted from the first sale, the day it was recorded
    and its price).

    Args:
        folder: Where to write it.

    Returns:
        The path of the database file.
    """
    sold = sales(0)
    day = (sold["date"] - sold["date"].min()).dt.days.astype(float)
    path = str(Path(folder) / "market.sqlite")
    keeper = sqlite3.connect(path)
    zipcodes = sorted(sold["zipcode"].astype(int).unique())
    pandas.DataFrame({"zipcode": zipcodes}).to_sql(
        "zipcodes", keeper, index=False
    )
    pandas.DataFrame(
        {
            "sale_id": range(len(sold)),
            "zipcode": sold["zipcode"].astype(int),
            "day": day,
            "recorded": day,
            "price": sold["price"],
        }
    ).to_sql("sales", keeper, index=False)
    keeper.commit()
    keeper.close()
    return path


# Days between prediction moments, and the first and last moment, of the
# real zip-code task: one row per zip code and month of the sales' year.
MONTH = 30
MOMENTS = range(60, 390, MONTH)


def demand(path: str) -> pandas.DataFrame:
    """The real zip-code task: does a zip code sell more than usual?

    One row per zip code and prediction moment; `busy` is 1 when the zip
    code records more sales in the next `MONTH` days than the median zip
    code and month does.

    Args:
        path: The database `market` wrote.

    Returns:
        The rows, keyed by `zipcode` and `moment`, with the label `busy`.
    """
    keeper = sqlite3.connect(path)
    sold = pandas.read_sql("SELECT zipcode, day FROM sales", keeper)
    keeper.close()
    rows: list[tuple[int, float, int]] = []
    for moment in MOMENTS:
        ahead = sold[(sold["day"] > moment) & (sold["day"] <= moment + MONTH)]
        counts = ahead.groupby("zipcode").size()
        rows.extend(
            (int(zipcode), float(moment), counts.get(zipcode, 0))
            for zipcode in sorted(sold["zipcode"].unique())
        )
    frame = pandas.DataFrame(rows, columns=["zipcode", "moment", "sold"])
    frame["busy"] = (frame["sold"] > frame["sold"].median()).astype(int)
    return frame.drop(columns="sold")


def recipe(path: str, folder: Path, unit: str) -> Path:
    """Writes the ingest plan of the real zip-code task.

    Args:
        path: The database `market` wrote.
        folder: Where to write the plan.
        unit: The unit the plan declares; `zipcode` is the grain of the
            `zipcodes` table.

    Returns:
        The plan file.
    """
    note = {
        "name_or_id": "A. Owner",
        "role": "data owner",
        "statement": "a sale is recorded on the day it closes",
        "evidence": "OpenML 42731 sale dates",
        "date": "2026-01-01",
        "scope": "sales",
        "expires": "2099-12-31",
    }
    plan = {
        "database": path,
        "key": "zipcode",
        "moment": "moment",
        "tables": ["zipcodes", "sales"],
        "unit": unit,
        "target_table": "zipcodes",
        "views": [
            {
                "name": "sales",
                "table": "sales",
                "order_by": "sale_id",
                "key": "zipcode",
                "target_key": "zipcode",
                "role": "child",
                "event_time": "day",
                "rule": {
                    "mode": "transaction",
                    "known_column": "recorded",
                    "delta_rec_days": 0.0,
                    "attestation": note,
                },
            }
        ],
    }
    written = folder / "plan.json"
    written.write_text(json.dumps(plan), encoding="utf-8")
    return written


def totals(path: str, short: int) -> Any:
    """The sales totals the source system declares, for reconciliation.

    Args:
        path: The database `market` wrote.
        short: Rows the extract is short of the declared count; 0 declares
            the true totals.

    Returns:
        The charter's `reconcile` block.
    """
    from scmle.assurance.charter import ingest
    from scmle.assurance.ingest import declared

    keeper = sqlite3.connect(path)
    sold = pandas.read_sql("SELECT sale_id, price FROM sales", keeper)
    keeper.close()
    return ingest.Expectation(
        (
            ingest.Totals(
                "sales",
                rows=len(sold) + short,
                sums=(ingest.Total("price", float(sold["price"].sum())),),
                column="sale_id",
                checksum=declared.sha(sold["sale_id"]),
            ),
        )
    )


def ingesting(
    unit: str = "zipcode", short: int | None = None, **options: Any
) -> Any:
    """Runs a signed charter whose features come from the real sales source.

    The charter is the C8 search's (`bypass.CHARTER`) with an ingest block
    (windows of 30 and 90 days, a lag of one day) and, when `short` is
    given, a `reconcile` block; the table is `demand`'s.

    Args:
        unit: The unit of the ingest plan; `zipcode-month` is coarser than
            the grain of the `zipcodes` table (the fault of the stage).
        short: Rows the declared sales count exceeds the source by (0
            declares the true totals); None declares no reconciliation.
        **options: Further `Project` arguments; a synthesized feature
            named as the `group_column` leaves the ledger's family.

    Returns:
        The completed run.
    """
    from benchmarks.evidence import bypass
    from scmle.assurance.charter import ingest

    lab = bypass.Lab(Path(tempfile.mkdtemp(prefix="faults-")))
    path = market(lab.folder)
    block = ingest.Ingestion(
        features=40,
        windows=(30.0, 90.0),
        lag=1.0,
        reconcile=None if short is None else totals(path, short),
    )
    bare = lab.folder / "charter.json"
    bare.write_text(
        json.dumps({**bypass.CHARTER, "ingest": block.to_dict()}),
        encoding="utf-8",
    )
    key = lab.folder / "key.txt"
    key.write_text(bypass.KEY, encoding="utf-8")
    signed = lab.folder / "signed.json"
    bypass.cli(
        [
            "charter",
            "sign",
            str(bare),
            "--output",
            str(signed),
            "--key-file",
            str(key),
        ]
    )
    with bypass.environ(SCMLE_SIGNING_KEY=bypass.KEY):
        return launch(
            demand(path),
            "busy",
            task="binary_classification",
            charter=signed,
            ingest_plan=recipe(path, lab.folder, unit),
            id_columns=["zipcode", "moment"],
            output_dir=lab.folder / "runs",
            **options,
        )


def command(unit: str, part: str = "simulated") -> Any:
    """Runs `scmle ingest` on a SQLite source with a charter unit.

    Args:
        unit: The unit text; `customer-month` (simulated) or
            `zipcode-month` (real) is coarser than the grain.
        part: `simulated` (ten customers and their orders) or `real` (the
            house sales by zip code, see `market`).

    Returns:
        The command's exit status.
    """
    from scmle import cli

    with tempfile.TemporaryDirectory() as folder:
        if part == "simulated":
            source, tables, target = (
                database(folder),
                "customers,orders",
                "customers",
            )
        else:
            source, tables, target = (
                market(folder),
                "zipcodes,sales",
                "zipcodes",
            )
        parser = argparse.ArgumentParser()
        step = cli.IngestCommand()
        step.configure(parser)
        arguments = parser.parse_args(
            [
                "--sqlite",
                source,
                "--tables",
                tables,
                "--unit",
                unit,
                "--target-table",
                target,
                "--output",
                str(Path(folder) / "out"),
                "--max-queries",
                "1000",
            ]
        )
        with contextlib.redirect_stdout(io.StringIO()):
            return step.execute(arguments)


def stage(unit: str) -> Any:
    """Runs the ingest stage function with a charter unit.

    Args:
        unit: The unit text; `customer-month` is coarser than the grain.

    Returns:
        The stage output.
    """
    from scmle.assurance.charter import ingest
    from scmle.pipeline import ingest_stage

    with tempfile.TemporaryDirectory() as folder:
        document = {
            "key": "customer_id",
            "moment": "moment",
            "database": database(folder),
            "tables": ["customers", "orders"],
            "unit": unit,
            "target_table": "customers",
            "views": [],
        }
        frame = pandas.DataFrame({"customer_id": range(10), "moment": 1.0})
        return ingest_stage.run(
            document, ingest.Ingestion(), frame, Path(folder) / "run"
        )


def reviewer(approve: bool) -> Any:
    """One reviewer at the planning state.

    Args:
        approve: Its verdict.

    Returns:
        The checkpoint.
    """
    from scmle.pipeline import policies
    from scmle.runtime import state

    def verdict(review: Any) -> Any:
        del review
        return policies.Approval(approve, "reviewer", "fault")

    return policies.Callback(
        verdict,
        gates=frozenset({state.Progress.PLANNING}),
    )


def checkpoint(approve: bool, part: str = "simulated") -> Any:
    """Runs a project with one reviewer at the planning state.

    Args:
        approve: The reviewer's verdict.
        part: Where the table comes from (see `clean`).

    Returns:
        The completed run.
    """
    frame, target, options = clean("binary", part)
    return launch(frame, target, checkpoints=(reviewer(approve),), **options)


def leak(policy: str, part: str = "simulated") -> str | None:
    """Runs a planted target copy under a leakage policy.

    Args:
        policy: `block` (the fault) or `warn` (the override).
        part: `simulated` (a 600-row binary table) or `real` (the real
            regression table, whose numeric target can be copied).

    Returns:
        The recorded warning under `warn`, or None if nothing names it.
    """
    if part == "simulated":
        frame, target, options = refusal.plant("binary", 600, "leak")()
    else:
        frame, target, options = clean("regression", part)
        refusal.spoil(frame, target, "leak")
    options["leakage_policy"] = policy
    with listening() as heard:
        launch(frame, target, **options)
    found = [t for t in heard.texts if "leak" in t.lower()]
    return found[0][:200] if found else None


def ignore(*arguments: Any) -> None:
    """Accepts and drops a call a stand-in context receives.

    Args:
        *arguments: Ignored.
    """
    del arguments


def admissibility(attest: bool) -> str | None:
    """Calls the admissibility stage on a table whose only feature leaks.

    The stage runs the real `qualify.run` (leakage screen, certificates,
    attestations) on an exploration view whose single feature is a copy of
    the target; the run context and charter are minimal stand-ins holding
    only what the stage reads.

    Args:
        attest: Whether a valid attestation file keeps the column (the
            documented override).

    Returns:
        The ledger check that records the attestation, or None.
    """
    from scmle.assurance.admissibility import attestations
    from scmle.assurance.admissibility import provenance
    from scmle.assurance.admissibility import qualify
    from scmle.pipeline import stages
    from scmle.pipeline.stages import screening
    from scmle.pipeline.stages import surveying

    scope = "c" * 64
    rng = numpy.random.default_rng(SEED)
    target = pandas.Series(rng.integers(0, 2, 500), name="y")
    features = pandas.DataFrame({"copy": target.to_numpy() * 1.0})
    with tempfile.TemporaryDirectory() as folder:
        path = None
        if attest:
            item = attestations.Affidavit(
                "copy",
                provenance.Attestation(
                    "A. Owner",
                    "data owner",
                    "recorded before the moment",
                    "ticket FAULT-1",
                    "2026-01-01",
                    scope,
                    "2099-12-31",
                ),
                3600.0,
            )
            path = Path(folder) / "attestations.json"
            path.write_text(
                json.dumps(
                    {"format_version": 1, "attestations": [item.to_dict()]}
                ),
                encoding="utf-8",
            )
        spec = types.SimpleNamespace(
            leakage_attestations=None if path is None else str(path),
            data=types.SimpleNamespace(
                group_column=None, time_column=None, cutoff=None, known={}
            ),
        )
        made: dict[type, Any] = {
            screening.Screen: types.SimpleNamespace(ingest=None),
            surveying.Portrait: types.SimpleNamespace(
                profile=types.SimpleNamespace(
                    feature_plan=data.Blueprint(("copy",), ())
                )
            ),
        }
        context: Any = types.SimpleNamespace(
            services=types.SimpleNamespace(spec=spec),
            journal=types.SimpleNamespace(write=ignore, emit=ignore),
            get=made.__getitem__,
        )
        charter: Any = types.SimpleNamespace(
            estimand=types.SimpleNamespace(kind="predictive", causal=None),
            seed=0,
            digest=scope,
        )
        view = types.SimpleNamespace(features=features, target=target)
        found = stages.Profiling().qualify(
            context, tasks.Task.BINARY, charter, view, ()
        )
    rows = [c for s, c in found.checks if s == qualify.ATTESTATION_SOURCE]
    if not rows:
        return None
    return f"ledger check {rows[0].identifier}, tier {rows[0].tier.value}"


def unlabeled(policy: str, part: str = "simulated") -> str | None:
    """Runs a regression with a tenth of the target missing.

    Args:
        policy: `block` (the fault) or `ignore` (the override).
        part: Where the table comes from (see `clean`).

    Returns:
        The curation note that records the set-aside rows, or None.
    """
    frame, target, options = clean("regression", part)
    refusal.spoil(frame, target, "unlabeled")
    run = launch(frame, target, unlabeled=policy, **options)
    if "set aside" not in text(run, "curation_report.json"):
        return None
    return f"curation_report.json: rows set aside (policy {policy})"


def serving(
    megabytes: float, pareto: bool = False, part: str = "simulated"
) -> str | None:
    """Runs a binary task under a model-size limit.

    Args:
        megabytes: The limit; 1e-6 no model meets.
        pareto: Whether to choose on the Pareto front.
        part: Where the table comes from (see `clean`).

    Returns:
        The decision-log entry that records the choice under the limit.
    """
    frame, target, options = clean("binary", part)
    limit = tasks.Selection(
        max_model_mb=megabytes, max_candidates_checked=3, pareto=pareto
    )
    run = launch(frame, target, selection=limit, **options)
    log = text(run, "diagnostics/decision_log.json")
    for phrase in ("within serving limits", "fastest within"):
        if phrase in log:
            return f"decision_log.json: {phrase} (max_model_mb {megabytes})"
    return None


def reconcile(drop: int) -> None:
    """Reconciles an extract against declared totals.

    Args:
        drop: Rows dropped from the extract; 0 passes.
    """
    from scmle.assurance.charter import ingest
    from scmle.assurance.ingest import declared

    orders = pandas.DataFrame(
        {"order_id": range(1, 101), "amount": [50.0] * 100}
    )
    expected = ingest.Expectation(
        (
            ingest.Totals(
                "orders",
                rows=100,
                sums=(ingest.Total("amount", 5000.0),),
                column="order_id",
                checksum=declared.sha(orders["order_id"]),
            ),
        )
    )
    frame = orders.iloc[: len(orders) - drop]
    declared.require(declared.reconcile({"orders": frame}, expected, 0.0))


def reuse(asked: int) -> None:
    """Asks a reusable holdout `asked` queries; its terms allow three.

    Args:
        asked: Queries asked; 3 passes, 4 asks one past the session's
            answers.
    """
    from scmle.assurance.ledger.confirm import thresholdout

    allowed = 3
    size = thresholdout.minimum(queries=allowed, budget=2)
    rng = numpy.random.default_rng(5)
    rows = pandas.DataFrame({"x": rng.normal(size=size)})
    labels = pandas.Series(rng.integers(0, 2, size), name="y")
    terms = thresholdout.Terms(size, queries=allowed, budget=2)
    guard = thresholdout.Thresholdout((rows, labels), (rows, labels), terms)
    for _ in range(asked):
        guard.query(
            lambda frame, target: ((frame["x"] > 0) == (target > 0)).astype(
                float
            )
        )


def rejournal(foreign: bool) -> None:
    """Resumes a reusable holdout from a journal of other rows, or its own.

    Args:
        foreign: True plants a journal written for other holdout rows;
            False resumes from the session's own journal.
    """
    from scmle.assurance.ledger.confirm import thresholdout

    size = thresholdout.minimum()
    rng = numpy.random.default_rng(6)
    rows = pandas.DataFrame({"x": rng.normal(size=size)})
    labels = pandas.Series(rng.integers(0, 2, size), name="y")
    other = pandas.Series(1 - labels, name="y")
    with tempfile.TemporaryDirectory() as folder:
        journal = Path(folder) / "journal.json"
        writer = (rows, other) if foreign else (rows, labels)
        thresholdout.Thresholdout(
            (rows, labels), writer, journal=journal
        ).query(
            lambda frame, target: ((frame["x"] > 0) == (target > 0)).astype(
                float
            )
        )
        thresholdout.Thresholdout((rows, labels), (rows, labels)).resume(
            journal
        )


# OpenML 1461 (bank marketing, 45211 calls): the cached binary table large
# enough for a reusable holdout under Thresholdout's default terms.
BANK = 1461
# The confirmation set's share of a charter run's rows: a quarter (the
# charter's `confirmation_fraction`) of the four fifths not held for test;
# 4322 of the 21613 house sales in a measured run.
HELD = 0.2
# Slack over the expected size for the stratified rounding of both splits.
SLACK = 1.05


def planter(fault: str, other: Task) -> type:
    """A plugin stage that leaves a reuse journal in the run directory.

    It runs right after the reservation, before the exploration opens `C`:
    the place a stale `thresholdout_journal.json` of another session would
    be found by the run's reuse session.

    Args:
        fault: `foreign` (a journal written for `other`, rows the run does
            not hold) or `none` (writes nothing).
        other: Rows of the same table outside the run, and their target.

    Returns:
        The `Stage` subclass.
    """
    from scmle.assurance.ledger import governed
    from scmle.assurance.ledger.confirm import thresholdout
    from scmle.pipeline import stages
    from scmle.pipeline.stages import reservation
    import scmle.runtime.state

    class Plant(stages.Stage):
        """Leaves a reuse journal of other rows before `C` opens."""

        state = scmle.runtime.state.Progress.VALIDATING
        requires = (reservation.Reservation,)

        def describe(self, context: Any) -> str:
            """Names the stage.

            Args:
                context: The run context; not consulted.

            Returns:
                The stage's reason text.
            """
            del context
            return f"plugin stage (journal {fault})"

        def execute(self, board: Any) -> None:
            """Writes one answered session on `other` as the run's journal.

            Args:
                board: The run's board holding the run's store.
            """
            if fault != "foreign":
                return
            frame, target, _ = other
            labels = pandas.Series(frame[target] == "2", name=target)
            rows = (frame.drop(columns=[target]), labels)
            terms = thresholdout.Terms(len(frame), queries=1, budget=1)
            folder = Path(board.services.store.directory)
            thresholdout.Thresholdout(
                rows, rows, terms, journal=folder / governed.JOURNAL
            ).query(lambda *scored: numpy.zeros(len(scored[1])))

    return Plant


def replanted(fault: str) -> Any:
    """Runs a signed `thresholdout` charter on real bank calls with `planter`.

    The charter is the C8 search's (`bypass.CHARTER`) with rules whose
    failure policy is `thresholdout`, so the confirmation also runs the
    reuse session `governed.replication` journals in the run directory. The
    run keeps enough calls for a confirmation set of `thresholdout.minimum`
    rows; the calls it leaves out are the other rows of the planted journal.

    Args:
        fault: As for `planter`.

    Returns:
        The completed run.
    """
    import math

    from benchmarks.evidence import bypass
    from scmle.assurance.charter import rules
    from scmle.assurance.ledger import confirm
    from scmle.assurance.ledger.confirm import thresholdout
    from scmle.runtime import plugins

    calls = refusal.fetch(BANK, 10**6)
    kept = math.ceil(SLACK * thresholdout.minimum() / HELD)
    frame = calls.sample(kept, random_state=SEED)
    rest = calls.drop(index=frame.index).reset_index(drop=True)
    lab = bypass.Lab(Path(tempfile.mkdtemp(prefix="faults-")))
    procedure = dataclasses.replace(
        rules.Procedure(), failure=confirm.THRESHOLDOUT
    )
    bare = lab.folder / "charter.json"
    bare.write_text(
        json.dumps({**bypass.CHARTER, "rules": procedure.to_dict()}),
        encoding="utf-8",
    )
    key = lab.folder / "key.txt"
    key.write_text(bypass.KEY, encoding="utf-8")
    path = lab.folder / "signed.json"
    bypass.cli(
        [
            "charter",
            "sign",
            str(bare),
            "--output",
            str(path),
            "--key-file",
            str(key),
        ]
    )
    name = f"faults-journal-{fault}"
    plugins.REGISTRY.enrol(
        plugins.Kind.STAGES, name, planter(fault, (rest, "Class", {}))
    )
    try:
        with bypass.environ(SCMLE_SIGNING_KEY=bypass.KEY):
            return launch(
                frame.reset_index(drop=True),
                "Class",
                task="classification",
                charter=path,
            )
    finally:
        plugins.REGISTRY.withdraw(plugins.Kind.STAGES, name)


def confirmation(times: int) -> None:
    """Runs the single confirmation on a real ledger `times` times.

    Args:
        times: 1 passes; 2 injects a second use of the confirmation set.
    """
    from scmle.assurance.custody import sealed
    from scmle.assurance.ledger import confirm
    from scmle.assurance.ledger import controls
    from scmle.assurance.ledger import explore
    from scmle.assurance.ledger import online_fdr
    import scmle.assurance.ledger.ledger
    from scmle.data.splitting import splitters

    def world(rows: int, seed: int) -> tuple[pandas.DataFrame, pandas.Series]:
        rng = numpy.random.default_rng(seed)
        features = pandas.DataFrame(
            {f"n{i}": rng.normal(size=rows) for i in range(6)}
        )
        target = 0.6 * features["n0"] + rng.normal(size=rows)
        return features, pandas.Series(target, name="y")

    alpha = Fraction(1, 20)
    ledger = scmle.assurance.ledger.ledger.Ledger.create(
        None,
        charter_hash="c" * 64,
        config=online_fdr.Settings(online_fdr.ELOND, Fraction(1, 10)),
        alpha_confirm=alpha,
        seed=3,
    )
    features, target = world(600, 1)
    explore.explore(
        ledger,
        features,
        target,
        explore.hypothesize(features, explore.NUMERIC),
    )
    ledger.negatives(
        controls.summarize("permuted_target", "strata", 6, (0,) * 20).to_dict()
    )
    rows, labels = world(500, 2)
    locked = splitters.lockbox.Resident(rows, labels, tuple(range(len(rows))))
    source = sealed.Locker("C", locked, sealed.Journal())
    for _ in range(times):
        confirm.confirm(ledger, source, alpha_confirm=alpha)


def rerun(again: bool) -> Any:
    """Runs a signed charter on a real binary table, once or twice.

    The charter and its signature are those of the C8 search
    (`bypass.charter`, signed by the `scmle charter sign` command). Both
    runs write to the same output directory, the scope in which the
    product records a confirmation set's use (`scmle.assurance.ledger.usage`
    documents that another output directory does not see the record).

    Args:
        again: Whether to run the same charter on the same rows a second
            time, the second use of its confirmation set (the fault).

    Returns:
        The last completed run.
    """
    from benchmarks.evidence import bypass

    frame, target, options = clean("binary", "real")
    lab = bypass.Lab(Path(tempfile.mkdtemp(prefix="faults-")))
    path = bypass.charter(lab)
    options = {**options, "output_dir": lab.folder / "runs"}
    with bypass.environ(SCMLE_SIGNING_KEY=bypass.KEY):
        result = launch(frame, target, charter=path, **options)
        if again:
            result = launch(frame, target, charter=path, **options)
    return result


def monotonic(column: str, part: str = "simulated") -> Any:
    """Runs a binary task with a monotonic constraint on one column.

    Args:
        column: A categorical column (the fault: `region` simulated,
            `purpose` real) or a numeric one (`x1`, `duration`).
        part: Where the table comes from (see `clean`; the simulated table
            has the mixed shape).

    Returns:
        The completed run.
    """
    if part == "simulated":
        frame, target, options = refusal.simulate("binary", ROWS, SEED, "mixed")
    else:
        frame, target, options = clean("binary", part)
    return launch(frame, target, monotonic_constraints={column: 1}, **options)


def featureless(part: str = "simulated") -> Any:
    """Runs a binary task whose only feature is constant.

    Args:
        part: Where the target comes from (see `clean`).

    Returns:
        The completed run.
    """
    frame, target, options = clean("binary", part)
    frame = pandas.DataFrame({"constant": 1.0, target: frame[target]})
    return launch(frame, target, **options)


def starved(seconds: float, part: str = "simulated") -> Any:
    """Runs a binary task under a time budget.

    Args:
        seconds: The budget; 1 s cannot fit the minimum run (the fault),
            `BUDGET` can.
        part: Where the table comes from (see `clean`).

    Returns:
        The completed run.
    """
    frame, target, options = clean("binary", part)

    def adjust(config: tasks.Config) -> tasks.Config:
        resources = config.resources.model_copy(
            update={"time_budget_seconds": seconds}
        )
        return config.model_copy(update={"resources": resources})

    return launch(frame, target, adjust=adjust, **options)


def overrun(spent: bool, part: str = "simulated") -> Any:
    """Runs a binary task whose preparation may have closed the trial window.

    Args:
        spent: Whether training starts with the run's start moved back by
            the whole budget, as after a preparation far longer than its
            estimate (the fault).
        part: Where the table comes from (see `clean`).

    Returns:
        The completed run.
    """
    import scmle.api.project
    from scmle.pipeline import stages

    class Late(stages.Training):
        """Training that first spends the whole budget when `spent`."""

        def execute(self, board: Any) -> Any:
            """Moves the run's start back, then trains.

            Args:
                board: The run's board.

            Returns:
                What the training stage returns.
            """
            if spent:
                board.regulator.started_at -= BUDGET
            return super().execute(board)

    frame, target, options = clean("binary", part)
    project = scmle.api.project.Project(
        frame,
        target,
        time_budget=BUDGET,
        output_dir=tempfile.mkdtemp(prefix="faults-"),
        **{**FAST, **options},
    )
    project.orchestrator.stages = tuple(
        Late() if isinstance(stage, stages.Training) else stage
        for stage in project.orchestrator.stages
    )
    return project.run()


def survival(censored: int | None) -> Any:
    """Runs a survival task with a set number of censored rows.

    Args:
        censored: Rows kept censored (the rest are events), or None to keep
            the simulated censoring.

    Returns:
        The completed run.
    """
    frame, target, options = clean("survival")
    if censored is not None:
        frame["event"] = 1
        frame.loc[: censored - 1, "event"] = 0
    return launch(frame, target, **options)


def folds(kind: str, rare: int) -> Any:
    """Calls a stratified splitter's folds with a rare stratum among `dev`.

    Args:
        kind: `stratified`, `events` or `treatment`.
        rare: Development rows of the rarest stratum; 1 injects the fault.

    Returns:
        The folds.
    """
    from scmle.data.splitting import splitters

    rows = 60
    frame = pandas.DataFrame({"x": numpy.arange(rows, dtype=float)})
    labels = numpy.array([0] * (rows // 2) + [1] * (rows // 2))
    labels[rows // 2 + rare :] = 0
    if kind == "stratified":
        frame["y"] = labels
        split: Any = splitters.splits.Stratified(spec(), tasks.Task.BINARY)
    elif kind == "events":
        frame["y"] = numpy.arange(rows, dtype=float) + 1
        frame["event"] = labels
        split = splitters.splits.Events(
            spec("survival", event_column="event"), tasks.Task.SURVIVAL
        )
    else:
        frame["y"] = numpy.tile([0, 1], rows // 2)
        frame["treated"] = labels
        frame.loc[frame["treated"] == 1, "y"] = 1
        frame.loc[frame.index[-1], "y"] = 0
        frame.loc[frame.index[-2], "y"] = 1
        split = splitters.splits.Treatment(
            spec("uplift", treatment_column="treated"), tasks.Task.UPLIFT
        )
    if kind == "treatment" and rare == 1:
        cells = split.cells(frame)
        keep = cells[cells == cells.value_counts().idxmin()].index[1:]
        frame = frame.drop(index=keep).reset_index(drop=True)
    return split.fold(frame, numpy.arange(len(frame)), [])


def scarce(rare: int | None) -> Any:
    """Runs the real binary task with its minority class cut to a few rows.

    Half the rows are locked for the test, so the stratified holdout keeps
    one of two minority rows among the development rows.

    Args:
        rare: Rows of the minority class kept; None keeps them all.

    Returns:
        The completed run.
    """
    frame, target, options = clean("binary", "real")
    if rare is not None:
        label = frame[str(target)]
        minority = (label == label.value_counts().idxmin()).to_numpy()
        frame = frame.drop(index=frame.index[minority][rare:])
        frame = frame.reset_index(drop=True)
    return launch(
        frame, target, validation="stratified", test_fraction=0.5, **options
    )


def unscored(rare: int | None, part: str = "simulated") -> Any:
    """Runs a binary task whose minority class is cut to two rows.

    With the default test fraction the stratified holdout keeps both
    minority rows among the development rows, so the test rows hold one
    class: ROC AUC is undefined and F1 scores a single class (the C1
    measurement of 2026-11-19).

    Args:
        rare: Rows of the minority class kept; None keeps them all.
        part: Where the table comes from (see `clean`).

    Returns:
        The completed run.
    """
    frame, target, options = clean("binary", part)
    if rare is not None:
        label = frame[str(target)]
        minority = (label == label.value_counts().idxmin()).to_numpy()
        frame = frame.drop(index=frame.index[minority][rare:])
        frame = frame.reset_index(drop=True)
    return launch(frame, target, **options)


def grouped(classes: bool, part: str = "simulated") -> Any:
    """Runs a binary task with a group column.

    Args:
        classes: Whether the positive class spreads over many groups; when
            False, it lives in a single group (the simulated label is
            rewritten to group 0; the real minority class is moved into
            group 0, its label kept).
        part: Where the table comes from (see `clean`).

    Returns:
        The completed run.
    """
    frame, target, options = clean("binary", part)
    frame["entity"] = numpy.arange(len(frame)) % 40
    if not classes and part == "simulated":
        frame[target] = (frame["entity"] == 0).astype(int)
    elif not classes:
        minority = frame[target] == frame[target].value_counts().idxmin()
        frame.loc[minority, "entity"] = 0
    return launch(frame, target, group_column="entity", **options)


def strategy(name: str, part: str = "simulated") -> Any:
    """Runs a binary task with a validation strategy.

    Args:
        name: `stratified` passes; `stratified_events` cannot serve binary.
        part: Where the table comes from (see `clean`).

    Returns:
        The completed run.
    """
    frame, target, options = clean("binary", part)
    return launch(frame, target, validation=name, **options)


def resident(fault: str) -> None:
    """Misuses or uses the in-process test set once.

    Args:
        fault: `bind` (bind twice), `open` (a wrong token) or `none`.
    """
    from scmle.data.splitting import splitters

    locked = splitters.lockbox.Resident(
        pandas.DataFrame({"a": [1.0, 2.0]}), pandas.Series([0, 1]), (0, 1)
    )
    locked.bind("selection")
    if fault == "bind":
        locked.bind("another")
    token = "wrong" if fault == "open" else locked.mint()
    locked.open(token, "selection")


def vault(fault: str) -> None:
    """Misuses or uses the sealed test-set process once.

    Args:
        fault: `bind` (bind twice), `open` (a wrong token) or `none`.
    """
    import scmle.assurance.custody.vault

    locked = scmle.assurance.custody.vault.SealedTestSet(
        pandas.DataFrame({"a": [1.0, 2.0, 3.0]}),
        pandas.Series([0, 1, 0]),
        (10, 11, 12),
    )
    try:
        locked.bind("selection")
        if fault == "bind":
            locked.bind("another")
        token = "wrong" if fault == "open" else locked.mint()
        locked.open(token, "selection")
    finally:
        locked.close()


def custody(fault: str) -> None:
    """Resumes a run against its custody journal once.

    Args:
        fault: `opened` (the test set was opened, then resumed), `edited`
            (a verdict's reviewer was rewritten, then read) or `none`.
    """
    from scmle.assurance.custody import journal

    folder = Path(tempfile.mkdtemp(prefix="faults-"))
    record = journal.Journal(folder / "custody" / "run_x.jsonl", "run_x")
    record.start("hash")
    record.decide("PLANNING", approved=False, reviewer="alice", reason="no")
    if fault == "opened":
        record.witness()
    if fault == "edited":
        lines = record.path.read_text(encoding="utf-8").splitlines()
        lines[1] = lines[1].replace("alice", "bob")
        record.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        record.read()
        return
    record.admit()


def resumed(fault: str) -> None:
    """Reads a run's stage records back once, as a resume does.

    Args:
        fault: `edited` (a written record's seconds were rewritten, then
            read) or `none`.
    """
    from scmle.pipeline import archive
    from scmle.runtime import state

    folder = Path(tempfile.mkdtemp(prefix="faults-"))
    records = archive.Directory(folder, "hash")
    records.record(0, "Profile", state.Progress.PROFILING, 1.0)
    path = records.path(0, "Profile")
    if fault == "edited":
        body = json.loads(path.read_text(encoding="utf-8"))
        body["seconds"] = 0.5
        path.write_text(json.dumps(body), encoding="utf-8")
    records.read()


def revisit(fault: str) -> Any:
    """Resumes a real binary run through `Project.resume`.

    The run is first stopped: a reviewer rejects its plan, or (for
    `opened`) it completes and its evaluation and completion marker are
    deleted. The resume brings a reviewer that approves the plan.

    Args:
        fault: `opened` (the test set was opened before the resume),
            `journal` (the plan's verdict in the custody journal was
            rewritten), `records` (a stage record's seconds were rewritten)
            or `none`.

    Returns:
        The resumed run.

    Raises:
        InputError: For an unknown fault, or a stopped run that did not
            leave exactly one run directory.
    """
    import scmle.api.project

    if fault not in ("opened", "journal", "records", "none"):
        raise errors.InputError(f"unknown resume fault {fault!r}")
    frame, target, options = clean("binary", "real")
    folder = Path(tempfile.mkdtemp(prefix="faults-"))
    approve = fault == "opened"
    project = scmle.api.project.Project(
        frame,
        target,
        time_budget=BUDGET,
        output_dir=folder,
        checkpoints=(reviewer(approve=approve),),
        **{**FAST, **options},
    )
    with contextlib.suppress(errors.BlockedError):
        project.run()
    found = [p for p in folder.iterdir() if p.name.startswith("run_")]
    if len(found) != 1:
        raise errors.InputError(f"expected one run in {folder}: {found}")
    run = found[0]
    if fault == "opened":
        (run / "evaluation.json").unlink()
        (run / "complete.json").unlink()
    if fault == "journal":
        path = folder / "custody" / f"{run.name}.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        index = max(i for i, t in enumerate(lines) if "PLANNING" in t)
        lines[index] = lines[index].replace('"fault"', '"edited"')
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if fault == "records":
        path = min((run / "stages").glob("*.json"))
        body = json.loads(path.read_text(encoding="utf-8"))
        body["seconds"] = body["seconds"] + 1.0
        path.write_text(json.dumps(body), encoding="utf-8")
    return scmle.api.project.Project.resume(
        run, data=frame, checkpoints=(reviewer(approve=True),)
    )


def rolling(horizon: int, part: str = "simulated") -> str | None:
    """Runs a forecast under rolling validation with a horizon.

    Args:
        horizon: Rows per validation window; one longer than the series
            injects the fault.
        part: `simulated` (a 200-day series) or `real` (the daily mean
            price of the real house sales, see `clean`).

    Returns:
        The saved horizon, the record of the shortened setting.
    """
    if part == "simulated":
        frame, target, options = refusal.simulate("forecast", 200, SEED)
    else:
        frame, target, options = clean("forecast", part)
    run = launch(
        frame, target, validation="rolling", horizon_rows=horizon, **options
    )
    saved = configured(run)["validation"]["rolling"]["horizon"]
    return f"config.json: validation.rolling.horizon {saved}"


def chronology(fault: str, entities: str = "known") -> Any:
    """Calls the temporal partition on a small dated table.

    Args:
        fault: `gap` (a missing time), `short` (too few rows), `tied` (one
            timestamp for every row), `few` (three entities in column `g`,
            fewer than five folds need) or `none`.
        entities: `validation.entities`; `unseen` with `few` plants the
            entity fault, `known` is its override.

    Returns:
        The partition, or under `few` and `known` the note that records
        the declaration the split followed.
    """
    from scmle.data.splitting import splitters

    rows = 6 if fault == "short" else 200
    frame = pandas.DataFrame(
        {
            "t": pandas.date_range("2024-01-01", periods=rows),
            "g": numpy.arange(rows) % 3,
            "x": numpy.arange(rows, dtype=float),
            "y": numpy.arange(rows) % 2,
        }
    )
    if fault == "gap":
        frame.loc[3, "t"] = pandas.NaT
    if fault == "tied":
        frame["t"] = pandas.Timestamp("2024-01-01")
    fields: dict[str, Any] = {"time_column": "t", "entities": entities}
    if fault == "few":
        fields["group_column"] = "g"
    split = splitters.Chronology(spec(**fields), tasks.Task.BINARY)
    partition = split.partition(frame)
    if fault == "few" and entities == "known":
        return "; ".join(partition.notes[-1:])
    return partition


def dated(fault: str, entities: str = "known") -> Any:
    """Runs a regression on the real house sales ordered by sale date.

    Args:
        fault: `gap` (one sale date missing), `short` (six sales), `tied`
            (the sales of the latest date alone), `few` (the sales of three
            zip codes, each zip code an entity) or `none`.
        entities: `validation.entities`; `unseen` with `few` plants the
            entity fault, `known` is its override.

    Returns:
        The completed run, or under `few` and `known` the saved
        declaration, the record of the override.
    """
    frame = sales()
    options: dict[str, Any] = {}
    if fault == "short":
        frame = frame.iloc[:6]
    if fault == "gap":
        frame.loc[3, "date"] = pandas.NaT
    if fault == "tied":
        latest = frame["date"].max()
        frame = frame[frame["date"] == latest].reset_index(drop=True)
    if fault == "few":
        kept = sorted(frame["zipcode"].unique())[:3]
        frame = frame[frame["zipcode"].isin(kept)].reset_index(drop=True)
        options = {"group_column": "zipcode", "entities": entities}
    run = launch(
        frame, "price", task="regression", time_column="date", **options
    )
    if fault == "few" and entities == "known":
        saved = configured(run)["validation"]["entities"]
        return f"config.json: validation.entities {saved}"
    return run


def uplift(rare: bool) -> Any:
    """Runs an uplift task, optionally with a single treated success.

    Args:
        rare: Whether only one treated row has outcome 1.

    Returns:
        The completed run.
    """
    frame, target, options = clean("uplift")
    if rare:
        treated = frame.index[frame["treated"] == 1]
        frame.loc[treated, target] = 0
        frame.loc[treated[0], target] = 1
    return launch(frame, target, **options)


def stayers(kept: int | None, **settings: Any) -> Any:
    """Runs the real churn survival task with its censored rows cut.

    Args:
        kept: Customers who stayed (censored rows) kept; None keeps all.
        **settings: Further `Project` arguments (`test_fraction` 0.5 leaves
            one of two censored rows among the development rows).

    Returns:
        The completed run.
    """
    frame, target, options = clean("survival", "real")
    if kept is not None:
        stayed = frame.index[frame["event"] == 0]
        frame = frame.drop(index=stayed[kept:]).reset_index(drop=True)
    return launch(frame, target, **options, **settings)


def churners(kept: int | None, **settings: Any) -> Any:
    """Runs the real churn uplift task with one arm-outcome cell cut.

    Args:
        kept: Customers on the international plan who left (the treated
            successes) kept; None keeps all.
        **settings: Further `Project` arguments (`test_fraction` 0.5 leaves
            one of two such rows among the development rows).

    Returns:
        The completed run.
    """
    frame, target, options = clean("uplift", "real")
    if kept is not None:
        succeeded = (frame["treated"] == 1) & (frame[str(target)] == 1)
        cell = frame.index[succeeded.to_numpy()]
        frame = frame.drop(index=cell[kept:]).reset_index(drop=True)
    return launch(frame, target, **options, **settings)


def plugin(fault: str) -> type:
    """A plugin stage that reaches for the locked test rows.

    It is enrolled through the public plugin registry (kind `stages`) and
    runs after model selection, before the final evaluation: the place a
    third-party stage would read the test rows to tune on them.

    Args:
        fault: `bind` (binds the test set for its own use, so the final
            evaluation binds it a second time), `open` (opens it with a
            token it guessed) or `none` (touches nothing).

    Returns:
        The `Stage` subclass.
    """
    from scmle.pipeline import stages
    from scmle.pipeline.stages import profiling
    from scmle.pipeline.stages import selecting
    import scmle.runtime.state

    class Reach(stages.Stage):
        """Reaches for the locked test rows after selection."""

        state = scmle.runtime.state.Progress.SELECTING
        requires = (selecting.Selection,)

        def describe(self, context: Any) -> str:
            """Names the stage.

            Args:
                context: The run context; not consulted.

            Returns:
                The stage's reason text.
            """
            del context
            return f"plugin stage ({fault})"

        def execute(self, board: Any) -> None:
            """Binds or opens the locked test rows.

            Args:
                board: The run's board holding the partition and its split.
            """
            locked = board.get(profiling.Partition).split.test
            if fault == "bind":
                locked.bind("plugin")
            if fault == "open":
                locked.open("0" * 32, None)

    return Reach


def peek(fault: str, chartered: bool = False) -> Any:
    """Runs the real binary task with a `plugin` stage enrolled.

    Args:
        fault: As for `plugin`.
        chartered: Whether a signed charter governs the run, which seals
            the test rows into a vault process (`vault.*` gates); without
            it they stay in this process (`split.resident.*` gates).

    Returns:
        The completed run.
    """
    from benchmarks.evidence import bypass
    from scmle.runtime import plugins

    frame, target, options = clean("binary", "real")
    name = f"faults-peek-{fault}"
    plugins.REGISTRY.enrol(plugins.Kind.STAGES, name, plugin(fault))
    try:
        if not chartered:
            return launch(frame, target, **options)
        lab = bypass.Lab(Path(tempfile.mkdtemp(prefix="faults-")))
        path = bypass.charter(lab)
        with bypass.environ(SCMLE_SIGNING_KEY=bypass.KEY):
            return launch(frame, target, charter=path, **options)
    finally:
        plugins.REGISTRY.withdraw(plugins.Kind.STAGES, name)


def resolve(
    task: tasks.TaskRequest, target: pandas.Series, **fields: Any
) -> Any:
    """Resolves a task request against a target column.

    Args:
        task: The request.
        target: The target values.
        **fields: Further `Dataset` fields.

    Returns:
        The resolved task.
    """
    import scmle.data.validation.target

    resolver = scmle.data.validation.target.Resolver(spec(task, **fields))
    return resolver.resolve(target)


def blank(empty: bool) -> Any:
    """Runs the real regression, optionally with every target value missing.

    Args:
        empty: Whether to blank the whole target.

    Returns:
        The completed run.
    """
    frame, target, options = clean("regression", "real")
    if empty:
        frame[target] = numpy.nan
    return launch(frame, target, **options)


def worded() -> Any:
    """Runs regression declared on the real binary table's text label.

    Returns:
        The completed run.
    """
    frame, target, options = clean("binary", "real")
    return launch(frame, target, **{**options, "task": "regression"})


def ambiguous(task: str, part: str = "simulated") -> str | None:
    """Runs a task on a small-cardinality integer target.

    Args:
        task: `auto` (the fault) or `regression` (the override).
        part: `simulated` (a regression target cut into five integers) or
            `real` (the real wine quality score, integers 3 to 8).

    Returns:
        The saved task request, the record of the explicit choice.
    """
    frame, target, options = clean(
        "regression" if part == "simulated" else "ordinal", part
    )
    label = frame[str(target)]
    if part == "simulated":
        frame[target] = pandas.qcut(label, 5, labels=False).astype(int)
    else:
        frame[target] = pandas.to_numeric(label).astype(int)
        options = {k: v for k, v in options.items() if k != "ordinal_order"}
    options["task"] = task
    run = launch(frame, target, **options)
    return f"config.json: task {configured(run)['task']}"


def ordinal(complete: bool, part: str = "simulated") -> Any:
    """Runs an ordinal task whose listed order omits a label.

    Args:
        complete: Whether the order lists every label.
        part: Where the table comes from (see `clean`).

    Returns:
        The completed run.
    """
    frame, target, options = clean("ordinal", part)
    if not complete:
        options["ordinal_order"] = list(options["ordinal_order"])[:-1]
    return launch(frame, target, **options)


def ranking(negative: bool, part: str = "simulated") -> Any:
    """Runs a ranking task, optionally with one negative grade.

    Args:
        negative: Whether to plant a negative grade.
        part: Where the table comes from (see `clean`).

    Returns:
        The completed run.
    """
    frame, target, options = clean("ranking", part)
    if negative:
        frame.loc[0, target] = -1
    return launch(frame, target, **options)


def durations(negative: bool, part: str = "simulated") -> Any:
    """Runs a survival task, optionally with a negative duration.

    Args:
        negative: Whether to plant a negative duration.
        part: Where the table comes from (see `clean`).

    Returns:
        The completed run.
    """
    frame, target, options = clean("survival", part)
    if negative:
        frame.loc[0, target] = -1.0
    return launch(frame, target, **options)


def outcome(textual: bool, part: str = "simulated") -> Any:
    """Runs an uplift task, optionally with a text outcome.

    Args:
        textual: Whether the outcome is text.
        part: Where the table comes from (see `clean`).

    Returns:
        The completed run.
    """
    frame, target, options = clean("uplift", part)
    if textual:
        frame[target] = numpy.where(frame[target] == 1, "yes", "no")
    return launch(frame, target, **options)


def gpu(policy: str, part: str = "simulated") -> str | None:
    """Runs a binary task under a GPU policy.

    Args:
        policy: `required` (the fault on a machine without a usable GPU
            adapter) or `auto` (the override).
        part: Where the table comes from (see `clean`).

    Returns:
        The saved policy, the record of the override.
    """
    frame, target, options = clean("binary", part)

    def adjust(config: tasks.Config) -> tasks.Config:
        resources = config.resources.model_copy(update={"gpu": policy})
        return config.model_copy(update={"resources": resources})

    run = launch(frame, target, adjust=adjust, **options)
    return f"config.json: resources.gpu {configured(run)['resources']['gpu']}"


def multiclass(
    constraints: dict[str, int], part: str = "simulated"
) -> str | None:
    """Runs a multiclass task with monotonic constraints.

    Args:
        constraints: The constraints; any on multiclass is the fault, none
            is the override.
        part: Where the table comes from (see `clean`).

    Returns:
        The saved constraints, the record of the relaxed setting.
    """
    frame, target, options = clean("multiclass", part)
    run = launch(frame, target, monotonic_constraints=constraints, **options)
    saved = configured(run)["data"]["monotonic"]
    return f"config.json: data.monotonic {saved}"


def estimate(missing: bool, part: str = "simulated") -> Any:
    """Runs `scmle estimate` on a regression slope, end to end.

    The table is written to a CSV file and the declaration (a gaussian
    linear slope of the target on one feature, the default `reject` policy
    for missing values) to JSON; the command reads both, as a user's call
    does.

    Args:
        missing: Whether one value of the feature is missing (the fault).
        part: `simulated` (feature `x1`) or `real` (the diabetes table,
            feature `bmi`).

    Returns:
        The command's exit status.
    """
    from scmle import cli

    frame, target, _ = clean("regression", part)
    feature = "x1" if part == "simulated" else "bmi"
    frame = frame[[feature, target]].copy()
    if missing:
        frame.loc[0, feature] = numpy.nan
    folder = Path(tempfile.mkdtemp(prefix="faults-"))
    data = folder / "data.csv"
    frame.to_csv(data, index=False)
    declared = folder / "estimand.json"
    declared.write_text(
        json.dumps(
            {
                "estimand": {
                    "name": "slope",
                    "outcome": target,
                    "population": "the declared dataset",
                    "likelihood": "gaussian",
                    "features": [feature],
                    "assumptions": ["the conditional mean is linear"],
                },
                "policy": {"estimator": "linear"},
            }
        ),
        encoding="utf-8",
    )
    parser = argparse.ArgumentParser()
    step = cli.Estimation()
    step.configure(parser)
    arguments = parser.parse_args(
        [
            str(data),
            "--estimation",
            str(declared),
            "--output-dir",
            str(folder / "out"),
        ]
    )
    return step.execute(arguments)


def quarantined(attest: bool) -> str | None:
    """Runs a signed charter whose only feature copies the real label.

    The table keeps the real binary label and one feature, its 0/1 copy
    with normal noise of scale 0.01 (as `refusal.spoil` plants a leak), so
    rows are not exact duplicates; the charter is the C8 search's
    (`bypass.charter`).

    Args:
        attest: Whether a valid attestation for the copy, scoped to the
            charter's hash, is passed (the documented override).

    Returns:
        The run file that records the attestation, or None.
    """
    from benchmarks.evidence import bypass
    from scmle.assurance.admissibility import attestations
    from scmle.assurance.admissibility import provenance
    from scmle.assurance.charter import sealing

    frame, target, options = clean("binary", "real")
    first = frame[target].iloc[0]
    noise = numpy.random.default_rng(SEED).normal(0, 0.01, len(frame))
    exact = (frame[target] == first).astype(float)
    copied = pandas.DataFrame({"copy": exact + noise, target: frame[target]})
    lab = bypass.Lab(Path(tempfile.mkdtemp(prefix="faults-")))
    path = bypass.charter(lab)
    if attest:
        scope = sealing.read(path, bypass.KEY).digest
        item = attestations.Affidavit(
            "copy",
            provenance.Attestation(
                "A. Owner",
                "data owner",
                "recorded before the moment",
                "ticket FAULT-1",
                "2026-01-01",
                scope,
                "2099-12-31",
            ),
            3600.0,
        )
        filed = lab.folder / "attestations.json"
        filed.write_text(
            json.dumps({"format_version": 1, "attestations": [item.to_dict()]}),
            encoding="utf-8",
        )
        options = {**options, "leakage_attestations": filed}
    with bypass.environ(SCMLE_SIGNING_KEY=bypass.KEY):
        run = launch(copied, target, charter=path, **options)
    for written in sorted(Path(run.directory).rglob("*.json")):
        body = written.read_text(encoding="utf-8")
        if "attest" in body and '"copy"' in body:
            return f"{written.name}: attestation of column copy"
    return None


@dataclasses.dataclass(frozen=True)
class Injection:
    """How one gate's guarded condition is planted.

    Attributes:
        level: `run` (an end-to-end project or resume), `site` (a direct
            call) or `none` (no injection is built for this part).
        cause: The planted condition.
        fault: Plants it.
        control: Runs the pass path, or None when it is not built.
        override: Plants the fault under the documented override, or None.
        reason: Why a path is not injected, or ''.
        extra: The optional extra (a key of `EXTRAS`) its paths need, or
            ''; without it they run under `INTERPRETER` (`delegate`).
    """

    level: str
    cause: str
    fault: Callable[[], Any] | None
    control: Callable[[], Any] | None = None
    override: Callable[[], Any] | None = None
    reason: str = ""
    extra: str = ""


# Rows held out from the model for the ONNX divergence gate: at least the
# gate's minimum (`onnx_export.ROWS`, 299), the rest trains the model.
HELD_OUT = 300


def exported(fault: str, part: str = "simulated") -> Any:
    """Exports a logistic model of a binary table as an ONNX graph.

    The model is fitted on the rows not held out and saved as a bundle;
    the simulated part exports it through `Bundle.export_onnx`, the real
    part through the `scmle export-onnx` command on the saved bundle.

    Args:
        fault: `unchecked` (no held-out rows given: the fault the default
            verification blocks), `checked` (`HELD_OUT` held-out rows: the
            pass path) or `unverified` (the check skipped: the override).
        part: `simulated` (a generated table of `ROWS` plus `HELD_OUT`
            rows) or `real` (`CREDIT`, see `clean`).

    Returns:
        The export manifest's record of the skipped check for
        `unverified`; otherwise the path written or the exit status.
    """
    import scmle.artifacts.bundle
    from scmle.cli import delivery
    import scmle.models.registry
    from scmle.preparation import preprocessing

    if part == "simulated":
        frame, target, _ = refusal.simulate("binary", ROWS + HELD_OUT, SEED)
    else:
        frame, target, _ = clean("binary", part)
    features = frame.drop(columns=[target])
    numeric = tuple(features.select_dtypes("number").columns)
    plan = data.Blueprint(
        numeric, tuple(c for c in features.columns if c not in numeric)
    )
    training, held = features.iloc[HELD_OUT:], features.iloc[:HELD_OUT]
    fitted = preprocessing.instantiate(
        plan,
        scmle.models.registry.Catalog.stock().get("logistic_regression"),
        tasks.Task.BINARY,
        SEED,
    ).fit(training, frame[target].iloc[HELD_OUT:].astype(str))
    bundle = scmle.artifacts.bundle.Bundle.create(
        fitted,
        plan,
        training,
        target,
        tasks.Task.BINARY.value,
        "roc_auc",
        "logistic_regression",
    )
    folder = Path(tempfile.mkdtemp(prefix="faults-"))
    graph = folder / "model.onnx"
    check = held if fault == "checked" else None
    if part == "simulated":
        written: Any = bundle.export_onnx(
            graph, check=check, unverified=fault == "unverified"
        )
    else:
        bundle.save(folder / "bundle")
        held.to_csv(folder / "held.csv", index=False)
        options = {
            "unchecked": [],
            "checked": ["--check", str(folder / "held.csv")],
            "unverified": ["--unverified"],
        }[fault]
        parser = argparse.ArgumentParser()
        step = delivery.Export()
        step.configure(parser)
        arguments = parser.parse_args(
            [
                str(folder / "bundle"),
                "--trusted",
                "--output",
                str(graph),
                *options,
            ]
        )
        written = step.execute(arguments)
    if fault != "unverified":
        return written
    manifest = json.loads(
        graph.with_name(graph.name + ".manifest.json").read_text(
            encoding="utf-8"
        )
    )
    return f"{graph.name}.manifest.json: verified {manifest['verified']}"


def series(values: Any) -> pandas.Series:
    """A target column named `y`.

    Args:
        values: Its values.

    Returns:
        The column.
    """
    return pandas.Series(values, name="y")


INTEGERS = series(numpy.arange(ROWS) % 4)


def bind(
    function: Callable[..., Any], *arguments: Any, **keywords: Any
) -> functools.partial[Any]:
    """Fixes the leading arguments of a fault or control call.

    Args:
        function: The call to fix.
        *arguments: Its leading positional arguments.
        **keywords: Its keyword arguments.

    Returns:
        The call with those arguments bound.
    """
    return functools.partial(function, *arguments, **keywords)


INJECTIONS: dict[str, Injection] = {
    "export.onnx.divergence": Injection(
        "site",
        "Bundle.export_onnx with no held-out rows (verification is the "
        "default)",
        bind(exported, "unchecked"),
        bind(exported, "checked"),
        override=bind(exported, "unverified"),
        extra="onnx",
    ),
    "ingest.cli.unit_verdict": Injection(
        "site",
        "scmle ingest with unit customer-month on a customer table",
        bind(command, "customer-month"),
        bind(command, "customer"),
    ),
    "ingest.stage.run": Injection(
        "site",
        "ingest stage with unit customer-month on a customer table",
        bind(stage, "customer-month"),
        bind(stage, "customer"),
    ),
    "policy.checkpoints": Injection(
        "run",
        "a reviewer rejects the plan",
        bind(checkpoint, approve=False),
        bind(checkpoint, approve=True),
    ),
    "leakage.policy": Injection(
        "run",
        "a noisy copy of the target under leakage_policy block",
        bind(leak, "block"),
        override=bind(leak, "warn"),
    ),
    "admissibility.quarantine_all": Injection(
        "site",
        "the only exploration feature is a copy of the target",
        bind(admissibility, attest=False),
        override=bind(admissibility, attest=True),
    ),
    "data.unlabeled": Injection(
        "run",
        "a tenth of the regression target missing, unlabeled block",
        bind(unlabeled, "block"),
        override=bind(unlabeled, "ignore"),
    ),
    "select.serving_limits": Injection(
        "run",
        "max_model_mb 1e-6, three candidates checked",
        bind(serving, 1e-6),
        override=bind(serving, 1000.0),
    ),
    "select.pareto_front": Injection(
        "run",
        "Pareto selection with max_model_mb 1e-6",
        bind(serving, 1e-6, pareto=True),
        override=bind(serving, 1000.0, pareto=True),
    ),
    "ingest.reconcile": Injection(
        "site",
        "extract one row short of the declared totals",
        bind(reconcile, 1),
        bind(reconcile, 0),
    ),
    "confirm.single_use": Injection(
        "site",
        "a second confirmation on the same ledger and sealed set",
        bind(confirmation, 2),
        bind(confirmation, 1),
    ),
    "confirm.reuse_exhausted": Injection(
        "site",
        "a reusable holdout asked one query past its answers",
        bind(reuse, 4),
        bind(reuse, 3),
    ),
    "confirm.reuse_journal": Injection(
        "site",
        "a reusable holdout resumed from a journal of other holdout rows",
        bind(rejournal, foreign=True),
        bind(rejournal, foreign=False),
    ),
    "profile.monotonic_columns": Injection(
        "run",
        "monotonic constraint on a categorical column",
        bind(monotonic, "region"),
        bind(monotonic, "x1"),
    ),
    "profile.target_and_features": Injection(
        "run",
        "the only feature is constant",
        featureless,
    ),
    "split.event.folds": Injection(
        "site",
        "one censored row among the development rows",
        bind(folds, "events", 1),
        bind(folds, "events", 10),
    ),
    "split.event.stratify": Injection(
        "run",
        "survival table with a single censored row",
        bind(survival, 1),
        bind(survival, None),
    ),
    "split.group.class_complete_folds": Injection(
        "run",
        "the positive class lives in a single group",
        bind(grouped, classes=False),
        bind(grouped, classes=True),
    ),
    "split.group.partition": Injection(
        "run",
        "ranking table with only three queries",
        bind(planted, "ranking", "queries"),
    ),
    "split.task_supported": Injection(
        "run",
        "binary task with validation stratified_events",
        bind(strategy, "stratified_events"),
        bind(strategy, "stratified"),
    ),
    "split.resident.bind": Injection(
        "site",
        "in-process test set bound twice",
        bind(resident, "bind"),
        bind(resident, "none"),
    ),
    "split.resident.open": Injection(
        "site",
        "in-process test set opened with a wrong token",
        bind(resident, "open"),
        bind(resident, "none"),
    ),
    "split.resident.reopen": Injection(
        "site",
        "run resumed after its journal recorded the test set opened",
        bind(custody, "opened"),
        bind(custody, "none"),
    ),
    "split.resident.journal": Injection(
        "site",
        "custody journal with a verdict's reviewer rewritten",
        bind(custody, "edited"),
        bind(custody, "none"),
    ),
    "split.resident.records": Injection(
        "site",
        "stage record with its seconds rewritten, then read on resume",
        bind(resumed, "edited"),
        bind(resumed, "none"),
    ),
    "split.rolling.partition": Injection(
        "run",
        "rolling validation with horizon_rows longer than the series",
        bind(rolling, 500),
        override=bind(rolling, 10),
    ),
    "split.stratified.folds": Injection(
        "site",
        "one minority row among the development rows",
        bind(folds, "stratified", 1),
        bind(folds, "stratified", 10),
    ),
    "split.test_support": Injection(
        "run",
        "a binary class of two rows, both kept for development",
        bind(unscored, 2),
        bind(unscored, None),
    ),
    "split.stratified.stratify": Injection(
        "run",
        "a multiclass class of a single row",
        bind(planted, "multiclass", "singleton"),
    ),
    "split.temporal.holdout": Injection(
        "site",
        "every dated row on one timestamp",
        bind(chronology, "tied"),
        bind(chronology, "none"),
    ),
    "split.temporal.times": Injection(
        "site",
        "a missing time stamp",
        bind(chronology, "gap"),
        bind(chronology, "none"),
    ),
    "split.temporal.entities": Injection(
        "site",
        "three entities declared unseen for five folds",
        bind(chronology, "few", "unseen"),
        bind(chronology, "none"),
        bind(chronology, "few", "known"),
    ),
    "split.temporal.partition": Injection(
        "site",
        "six dated rows for five folds",
        bind(chronology, "short"),
        bind(chronology, "none"),
    ),
    "split.treatment.folds": Injection(
        "site",
        "a treatment-outcome cell of one development row",
        bind(folds, "treatment", 1),
        bind(folds, "treatment", 10),
    ),
    "split.treatment.stratify": Injection(
        "run",
        "uplift table with a single treated success",
        bind(uplift, rare=True),
        bind(uplift, rare=False),
    ),
    "task.resolve": Injection(
        "site",
        "a target with no non-missing value",
        bind(resolve, "auto", series([numpy.nan] * 10)),
        bind(resolve, "auto", series([0, 1] * 5)),
    ),
    "task.auto": Injection(
        "run",
        "auto task on a five-valued integer target",
        bind(ambiguous, "auto"),
        override=bind(ambiguous, "regression"),
    ),
    "task.classification": Injection(
        "run",
        "binary_classification declared on three classes",
        bind(planted, "multiclass", "binary"),
    ),
    "task.ordinal": Injection(
        "run",
        "ordinal_order omits a label of the data",
        bind(ordinal, complete=False),
        bind(ordinal, complete=True),
    ),
    "task.ranking": Injection(
        "run",
        "a negative relevance grade",
        bind(ranking, negative=True),
        bind(ranking, negative=False),
    ),
    "task.regression": Injection(
        "run",
        "regression declared on a text target",
        bind(planted, "regression", "text"),
    ),
    "task.survival": Injection(
        "run",
        "a negative duration",
        bind(durations, negative=True),
        bind(durations, negative=False),
    ),
    "task.uplift": Injection(
        "run",
        "an uplift outcome stored as text",
        bind(outcome, textual=True),
        bind(outcome, textual=False),
    ),
    "target.columns": Injection(
        "run",
        "a multi-label column that is not 0/1",
        bind(planted, "multilabel", "labels"),
    ),
    "target.events": Injection(
        "run",
        "an event column holding 0 and 2",
        bind(planted, "survival", "events"),
    ),
    "target.treatment": Injection(
        "run",
        "every row treated",
        bind(planted, "uplift", "arm"),
    ),
    "vault.bind": Injection(
        "site",
        "sealed test set bound twice",
        bind(vault, "bind"),
        bind(vault, "none"),
    ),
    "vault.open": Injection(
        "site",
        "sealed test set opened with a wrong token",
        bind(vault, "open"),
        bind(vault, "none"),
    ),
    "plan.gpu_required": Injection(
        "run",
        "resources.gpu required",
        bind(gpu, "required"),
        override=bind(gpu, "auto"),
    ),
    "plan.monotonic": Injection(
        "run",
        "monotonic constraints on a multiclass task",
        bind(multiclass, {"x1": 1}),
        override=bind(multiclass, {}),
    ),
    "budget.minimum_run": Injection(
        "run",
        "a 1 s time budget, below the estimated minimum run",
        bind(starved, 1),
        bind(starved, BUDGET),
    ),
    "estimate.refusal": Injection(
        "run",
        "scmle estimate of a slope whose feature has a missing value",
        bind(estimate, missing=True),
        bind(estimate, missing=False),
    ),
    "train.first_trial": Injection(
        "run",
        "training starts after preparation used up the trial window",
        bind(overrun, spent=True),
        bind(overrun, spent=False),
    ),
}


def absent(cause: str, reason: str) -> Injection:
    """A real-data injection that is not built, with the reason.

    Args:
        cause: The condition the gate guards.
        reason: Why no run on real data plants it.

    Returns:
        The injection; every path of it is recorded `not injected`.
    """
    return Injection("none", cause, None, reason=reason)


CREDIT = "OpenML 31 credit-g"
CHURNED = "OpenML 40701 churn"
LISTED = "OpenML 42731 house sales by zip code"
BANKED = "OpenML 1461 bank marketing"
MARKET = "a SQLite source of the OpenML 42731 house sales by zip code"

REAL: dict[str, Injection] = {
    "export.onnx.divergence": Injection(
        "run",
        f"scmle export-onnx of a logistic model of {CREDIT} with no "
        f"held-out rows (verification is the default)",
        bind(exported, "unchecked", part="real"),
        bind(exported, "checked", part="real"),
        override=bind(exported, "unverified", part="real"),
        extra="onnx",
    ),
    "ingest.cli.unit_verdict": Injection(
        "run",
        f"scmle ingest with unit zipcode-month on the zip-code table "
        f"({MARKET})",
        bind(command, "zipcode-month", part="real"),
        bind(command, "zipcode", part="real"),
    ),
    "ingest.stage.run": Injection(
        "run",
        f"signed charter run whose ingest plan declares unit zipcode-month "
        f"on the zip-code table ({MARKET})",
        bind(ingesting, "zipcode-month"),
        bind(ingesting, "zipcode"),
    ),
    "policy.checkpoints": Injection(
        "run",
        f"a reviewer rejects the plan ({CREDIT})",
        bind(checkpoint, approve=False, part="real"),
        bind(checkpoint, approve=True, part="real"),
    ),
    "leakage.policy": Injection(
        "run",
        "a noisy copy of the diabetes target under leakage_policy block",
        bind(leak, "block", part="real"),
        override=bind(leak, "warn", part="real"),
    ),
    "admissibility.quarantine_all": Injection(
        "run",
        f"signed charter run whose only feature copies the {CREDIT} label",
        bind(quarantined, attest=False),
        override=bind(quarantined, attest=True),
    ),
    "data.unlabeled": Injection(
        "run",
        "a tenth of the diabetes target missing, unlabeled block",
        bind(unlabeled, "block", part="real"),
        override=bind(unlabeled, "ignore", part="real"),
    ),
    "select.serving_limits": Injection(
        "run",
        f"max_model_mb 1e-6, three candidates checked ({CREDIT})",
        bind(serving, 1e-6, part="real"),
        override=bind(serving, 1000.0, part="real"),
    ),
    "select.pareto_front": Injection(
        "run",
        f"Pareto selection with max_model_mb 1e-6 ({CREDIT})",
        bind(serving, 1e-6, pareto=True, part="real"),
        override=bind(serving, 1000.0, pareto=True, part="real"),
    ),
    "ingest.reconcile": Injection(
        "run",
        f"signed charter run whose reconcile block declares one sale more "
        f"than the source holds ({MARKET})",
        bind(ingesting, short=1),
        bind(ingesting, short=0),
    ),
    "confirm.single_use": Injection(
        "run",
        f"the same signed charter run twice on {CREDIT}",
        bind(rerun, again=True),
        bind(rerun, again=False),
    ),
    "confirm.reuse_exhausted": absent(
        "a charter run's reusable holdout asked past its answers",
        "a run's analyst (`governed.replication`) asks at most one query "
        "per survivor and stops when the budget is spent, its terms are "
        "the module defaults (no charter field tightens them), and the "
        "session records a refusal as status `exhausted` instead of "
        "raising; a session journal that spends the budget first would "
        "have to be sealed for C's rows, which only opening C gives; the "
        "gate is planted only by a direct call",
    ),
    "confirm.reuse_journal": Injection(
        "run",
        f"a plugin stage leaves a reuse journal of other calls in the run "
        f"directory of a signed thresholdout charter run ({BANKED})",
        bind(replanted, "foreign"),
        bind(replanted, "none"),
    ),
    "profile.monotonic_columns": Injection(
        "run",
        f"monotonic constraint on categorical purpose ({CREDIT})",
        bind(monotonic, "purpose", part="real"),
        bind(monotonic, "duration", part="real"),
    ),
    "profile.target_and_features": Injection(
        "run",
        f"the {CREDIT} label with a constant as its only feature",
        bind(featureless, part="real"),
    ),
    "split.event.folds": Injection(
        "run",
        f"two customers who stayed, half the rows locked for the test "
        f"({CHURNED} survival)",
        bind(stayers, 2, test_fraction=0.5),
        bind(stayers, None, test_fraction=0.5),
    ),
    "split.event.stratify": Injection(
        "run",
        f"a single customer who stayed ({CHURNED} survival)",
        bind(stayers, 1),
        bind(stayers, None),
    ),
    "split.group.class_complete_folds": Injection(
        "run",
        f"the {CREDIT} minority class moved into a single group",
        bind(grouped, classes=False, part="real"),
        bind(grouped, classes=True, part="real"),
    ),
    "split.group.partition": Injection(
        "run",
        f"the sales relabelled into three queries ({LISTED} ranking)",
        bind(planted, "ranking", "queries", part="real"),
    ),
    "split.task_supported": Injection(
        "run",
        f"binary task with validation stratified_events ({CREDIT})",
        bind(strategy, "stratified_events", part="real"),
        bind(strategy, "stratified", part="real"),
    ),
    "split.resident.bind": Injection(
        "run",
        f"a plugin stage binds the in-process test set before the final "
        f"evaluation does ({CREDIT})",
        bind(peek, "bind"),
        bind(peek, "none"),
    ),
    "split.resident.open": Injection(
        "run",
        f"a plugin stage opens the in-process test set with a guessed "
        f"token ({CREDIT})",
        bind(peek, "open"),
        bind(peek, "none"),
    ),
    "split.resident.reopen": Injection(
        "run",
        f"a completed {CREDIT} run, its evaluation and marker deleted, resumed",
        bind(revisit, "opened"),
        bind(revisit, "none"),
    ),
    "split.resident.journal": Injection(
        "run",
        f"a rejected {CREDIT} run resumed after its custody journal's plan "
        "verdict was rewritten",
        bind(revisit, "journal"),
        bind(revisit, "none"),
    ),
    "split.resident.records": Injection(
        "run",
        f"a rejected {CREDIT} run resumed after a stage record's seconds "
        "were rewritten",
        bind(revisit, "records"),
        bind(revisit, "none"),
    ),
    "split.rolling.partition": Injection(
        "run",
        "rolling validation with horizon_rows longer than the daily mean "
        "price series of OpenML 42731 house sales",
        bind(rolling, 500, part="real"),
        override=bind(rolling, 10, part="real"),
    ),
    "split.stratified.folds": Injection(
        "run",
        f"the {CREDIT} minority class cut to two rows",
        bind(scarce, 2),
        bind(scarce, None),
    ),
    "split.test_support": Injection(
        "run",
        f"the {CREDIT} minority class cut to two rows, both kept for "
        "development under the default test fraction",
        bind(unscored, 2, part="real"),
        bind(unscored, None, part="real"),
    ),
    "split.stratified.stratify": Injection(
        "run",
        "an OpenML 54 vehicle class of a single row",
        bind(planted, "multiclass", "singleton", part="real"),
    ),
    "split.temporal.holdout": Injection(
        "run",
        "the OpenML 42731 house sales of a single sale date",
        bind(dated, "tied"),
        bind(dated, "none"),
    ),
    "split.temporal.times": Injection(
        "run",
        "a missing sale date in OpenML 42731 house sales",
        bind(dated, "gap"),
        bind(dated, "none"),
    ),
    "split.temporal.entities": Injection(
        "run",
        "the OpenML 42731 house sales of three zip codes, each zip code an "
        "entity declared unseen",
        bind(dated, "few", "unseen"),
        bind(dated, "none"),
        bind(dated, "few", "known"),
    ),
    "split.temporal.partition": Injection(
        "run",
        "six dated OpenML 42731 house sales",
        bind(dated, "short"),
        bind(dated, "none"),
    ),
    "split.treatment.folds": Injection(
        "run",
        f"two treated customers who left, half the rows locked for the "
        f"test ({CHURNED} uplift)",
        bind(churners, 2, test_fraction=0.5),
        bind(churners, None, test_fraction=0.5),
    ),
    "split.treatment.stratify": Injection(
        "run",
        f"a single treated customer who left ({CHURNED} uplift)",
        bind(churners, 1),
        bind(churners, None),
    ),
    "task.resolve": Injection(
        "run",
        "the diabetes target with no non-missing value",
        bind(blank, empty=True),
        bind(blank, empty=False),
    ),
    "task.auto": Injection(
        "run",
        "auto task on the OpenML 40691 wine quality score (integers 3 to 8)",
        bind(ambiguous, "auto", part="real"),
        override=bind(ambiguous, "regression", part="real"),
    ),
    "task.classification": Injection(
        "run",
        "binary_classification declared on the four OpenML 54 vehicle classes",
        bind(planted, "multiclass", "binary", part="real"),
    ),
    "task.ordinal": Injection(
        "run",
        "ordinal_order omits the top OpenML 40691 wine quality score",
        bind(ordinal, complete=False, part="real"),
        bind(ordinal, complete=True, part="real"),
    ),
    "task.ranking": Injection(
        "run",
        f"a negative view grade ({LISTED} ranking)",
        bind(ranking, negative=True, part="real"),
        bind(ranking, negative=False, part="real"),
    ),
    "task.regression": Injection(
        "run",
        f"regression declared on the {CREDIT} text label good/bad",
        worded,
    ),
    "task.survival": Injection(
        "run",
        f"a negative account length ({CHURNED} survival)",
        bind(durations, negative=True, part="real"),
        bind(durations, negative=False, part="real"),
    ),
    "task.uplift": Injection(
        "run",
        f"the churn outcome stored as yes/no text ({CHURNED} uplift)",
        bind(outcome, textual=True, part="real"),
        bind(outcome, textual=False, part="real"),
    ),
    "target.columns": Injection(
        "run",
        "OpenML 40589 emotions labels shifted by 0.5 (not 0/1)",
        bind(planted, "multilabel", "labels", part="real"),
    ),
    "target.events": Injection(
        "run",
        f"the event column doubled to 0 and 2 ({CHURNED} survival)",
        bind(planted, "survival", "events", part="real"),
    ),
    "target.treatment": Injection(
        "run",
        f"every customer on the international plan ({CHURNED} uplift)",
        bind(planted, "uplift", "arm", part="real"),
    ),
    "vault.bind": Injection(
        "run",
        f"a plugin stage binds the sealed test set of a signed charter run "
        f"before the final evaluation does ({CREDIT})",
        bind(peek, "bind", chartered=True),
        bind(peek, "none", chartered=True),
    ),
    "vault.open": Injection(
        "run",
        f"a plugin stage opens the sealed test set of a signed charter run "
        f"with a guessed token ({CREDIT})",
        bind(peek, "open", chartered=True),
        bind(peek, "none", chartered=True),
    ),
    "plan.gpu_required": Injection(
        "run",
        f"resources.gpu required ({CREDIT})",
        bind(gpu, "required", part="real"),
        override=bind(gpu, "auto", part="real"),
    ),
    "plan.monotonic": Injection(
        "run",
        "monotonic constraints on the OpenML 54 vehicle multiclass task",
        bind(multiclass, {"COMPACTNESS": 1}, part="real"),
        override=bind(multiclass, {}, part="real"),
    ),
    "budget.minimum_run": Injection(
        "run",
        f"a 1 s time budget, below the estimated minimum run ({CREDIT})",
        bind(starved, 1, part="real"),
        bind(starved, BUDGET, part="real"),
    ),
    "estimate.refusal": Injection(
        "run",
        "scmle estimate of the diabetes target's slope on bmi, one bmi "
        "value missing",
        bind(estimate, missing=True, part="real"),
        bind(estimate, missing=False, part="real"),
    ),
    "train.first_trial": Injection(
        "run",
        f"training starts after preparation used up the trial window "
        f"({CREDIT}; the training stage moves the run's start back)",
        bind(overrun, spent=True, part="real"),
        bind(overrun, spent=False, part="real"),
    ),
}


def injections(part: str) -> dict[str, Injection]:
    """The injections of one part.

    Args:
        part: `real` or `simulated`.

    Returns:
        `REAL` or `INJECTIONS`.

    Raises:
        InputError: For an unknown part.
    """
    if part not in PARTS:
        raise errors.InputError(f"unknown part {part!r}")
    return REAL if part == "real" else INJECTIONS


def overridable(gate: registry.Gate) -> bool:
    """Whether the registry documents a way to proceed past the gate.

    Args:
        gate: The registered gate.

    Returns:
        False when its override text starts with `none`.
    """
    return not gate.override.strip().lower().startswith("none")


def attempt(name: str, path: str, part: str = "simulated") -> dict[str, Any]:
    """Runs one path of one gate's injection and records how it ended.

    Args:
        name: The gate.
        path: `fault`, `override` or `control`.
        part: `simulated` or `real`.

    Returns:
        The gate, path, part, level, cause, outcome, the gate that blocked
        (if any) with its message, the override's record, the error of a
        failure, seconds; a path with nothing built is `not injected` with
        the reason.

    Raises:
        InputError: For an unknown gate, path or part.
    """
    table = injections(part)
    if name not in table or path not in ("fault", "override", "control"):
        raise errors.InputError(f"no injection {name!r} / {path!r}")
    injection = table[name]
    action = getattr(injection, path)
    record: dict[str, Any] = {
        "gate": name,
        "path": path,
        "part": part,
        "level": injection.level,
        "cause": injection.cause,
    }
    if action is None:
        reason = injection.reason or f"no {path} path built for this gate"
        record.update(outcome="not injected", reason=reason)
        return record
    started = time.monotonic()
    try:
        returned = action()
    except (errors.BlockedError, errors.LeakageError) as error:
        gate, site = refusal.locate(error, refusal.registry())
        record.update(
            outcome="blocked",
            by=gate,
            site=site,
            error=type(error).__name__,
            message=str(error)[:300],
        )
    except Exception as error:  # noqa: BLE001 -- every ending is recorded
        record.update(
            outcome="failed",
            error=type(error).__name__,
            message=str(error)[:300],
        )
    else:
        recorded = returned if path == "override" else None
        if path == "override" and isinstance(recorded, str) and recorded:
            record.update(outcome="override", record=recorded)
        elif path == "control":
            record["outcome"] = "passed"
        else:
            record["outcome"] = "silent"
    record["seconds"] = round(time.monotonic() - started, 2)
    return record


def jobs(gates: dict[str, registry.Gate]) -> list[tuple[str, str, str]]:
    """Every path to run, part by part and gate by gate.

    Args:
        gates: The registry.

    Returns:
        (gate, path, part) triples: in each part the fault of every gate,
        the override of every overridable gate and the control where one
        is built.
    """
    found = []
    for part in PARTS:
        table = injections(part)
        for name, gate in gates.items():
            found.append((name, "fault", part))
            if overridable(gate):
                found.append((name, "override", part))
            injection = table.get(name)
            if injection is not None and injection.control is not None:
                found.append((name, "control", part))
    return found


def installed(extra: str) -> bool:
    """Whether every module of an optional extra imports here.

    Args:
        extra: A key of `EXTRAS`.

    Returns:
        True when each of its modules imports.
    """
    for name in EXTRAS[extra]:
        try:
            importlib.import_module(name)
        except ImportError:
            # handled: the extra is reported absent and the path delegated
            return False
    return True


def delegate(name: str, path: str, part: str, extra: str) -> dict[str, Any]:
    """Runs one path under `INTERPRETER` with its extra installed.

    The path runs in an isolated environment of this project with the
    extra (`uv run --isolated`), through this module's `--attempt` mode,
    and its record comes back as the last line printed.

    Args:
        name: The gate.
        path: The path.
        part: The part.
        extra: The extra the path needs.

    Returns:
        The path's record with the interpreter and extra it ran under, or
        `failed` with the reason when that environment could not run it.
    """
    command = [
        "uv",
        "run",
        "--isolated",
        "--python",
        INTERPRETER,
        "--extra",
        extra,
        "python",
        "-m",
        "benchmarks.evidence.faults",
        "--attempt",
        name,
        path,
        part,
    ]
    environment = {
        k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"
    } | {"OMP_NUM_THREADS": "1"}
    record: dict[str, Any] = {
        "gate": name,
        "path": path,
        "part": part,
        "level": injections(part)[name].level,
        "cause": injections(part)[name].cause,
    }
    started = time.monotonic()
    try:
        done = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=BUDGET + SETUP,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        record.update(
            outcome="failed",
            error=type(error).__name__,
            message=f"{INTERPRETER} with the {extra} extra: {error}"[:300],
            seconds=round(time.monotonic() - started, 2),
        )
        return record
    lines = done.stdout.strip().splitlines()
    try:
        returned = json.loads(lines[-1]) if done.returncode == 0 else None
    except (IndexError, json.JSONDecodeError):
        returned = None
    if not isinstance(returned, dict):
        tail = (done.stderr or done.stdout).strip()[-300:]
        record.update(
            outcome="failed",
            error="DelegationError",
            message=f"{INTERPRETER} with the {extra} extra exited "
            f"{done.returncode}: {tail}"[:300],
            seconds=round(time.monotonic() - started, 2),
        )
        return record
    return returned | {"interpreter": INTERPRETER, "extra": extra}


def run(name: str, path: str, part: str = "simulated") -> dict[str, Any]:
    """`attempt`, with a missing gate recorded and missing extras delegated.

    A path whose injection needs an optional extra this interpreter cannot
    import runs under `INTERPRETER` with the extra (`delegate`), so it
    reaches its gate instead of failing on the import.

    Args:
        name: The gate.
        path: The path.
        part: The part.

    Returns:
        The record.
    """
    injection = injections(part).get(name)
    if injection is None:
        return {
            "gate": name,
            "path": path,
            "part": part,
            "level": "none",
            "outcome": "not injected",
            "reason": "no injection built for this gate",
        }
    if injection.extra and not installed(injection.extra):
        return delegate(name, path, part, injection.extra)
    return attempt(name, path, part)


def prepare() -> None:
    """Quiets the product's logs in a worker and pins one thread."""
    os.environ["OMP_NUM_THREADS"] = "1"
    logging.getLogger("scmle").setLevel(logging.WARNING)


def satisfies(record: dict[str, Any]) -> bool:
    """Whether one fault or override unit meets the claim.

    Args:
        record: The unit's record.

    Returns:
        True when it was blocked by the gate injected (a block by another
        gate, such as the time budget on a loaded machine, shows nothing
        about this gate) or ended in a recorded override, and, in the real
        part, it was injected through a run.
    """
    if record["outcome"] not in SATISFIED:
        return False
    if record["outcome"] == "blocked" and record.get("by") != record["gate"]:
        return False
    return record.get("part") != "real" or record.get("level") == "run"


def select(records: list[dict[str, Any]], part: str) -> list[dict[str, Any]]:
    """The records of one part.

    Args:
        records: Records of both parts; one without a part is simulated.
        part: `real` or `simulated`.

    Returns:
        Those of `part`.
    """
    return [r for r in records if r.get("part", "simulated") == part]


def tally(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts the units the claim is decided on.

    Args:
        records: One record per path run, of one part (`select`).

    Returns:
        Units (faults and overrides), those satisfied (`satisfies`), units
        injected through a run and by a direct call, counts by outcome,
        blocks attributed to another gate, overrides that still blocked,
        controls that did not pass.
    """
    units = [r for r in records if r["path"] in ("fault", "override")]
    outcomes = {o: sum(r["outcome"] == o for r in units) for o in OUTCOMES}
    satisfied = [r for r in units if satisfies(r)]
    return {
        "units": len(units),
        "satisfied": len(satisfied),
        "share": len(satisfied) / len(units) if units else None,
        "faults": sum(r["path"] == "fault" for r in units),
        "overrides": sum(r["path"] == "override" for r in units),
        "runs": sum(r.get("level") == "run" for r in units),
        "sites": sum(r.get("level") == "site" for r in units),
        "outcomes": outcomes,
        "unsatisfied": sorted(
            f"{r['gate']} ({r['path']}: {r['outcome']}, "
            f"{r.get('level', 'none')})"
            for r in units
            if not satisfies(r)
        ),
        "elsewhere": sorted(
            f"{r['gate']} blocked by {r['by']}"
            for r in units
            if r["outcome"] == "blocked" and r.get("by") != r["gate"]
        ),
        "override_blocked": sorted(
            r["gate"]
            for r in units
            if r["path"] == "override" and r["outcome"] == "blocked"
        ),
        "controls": {
            "run": sum(r["path"] == "control" for r in records),
            "passed": sum(
                r["path"] == "control" and r["outcome"] == "passed"
                for r in records
            ),
            "not_passed": sorted(
                f"{r['gate']} ({r['outcome']}: {r.get('by', r.get('error'))})"
                for r in records
                if r["path"] == "control" and r["outcome"] != "passed"
            ),
        },
    }


def sites() -> list[Any]:
    """The function of every registered gate.

    Returns:
        Each gate's raising function, resolved from its `module:qualname`.
    """
    found = []
    for gate in registry.load().values():
        module, qualname = gate.site.split(":")
        item: Any = importlib.import_module(module)
        for part in qualname.split("."):
            item = getattr(item, part)
        found.append(item)
    return found


def digest() -> str:
    """Hash of the gates measured, the registry and this module's code.

    Returns:
        SHA-256 over the source of every gate function, `gates.toml` and
        the functions of this module that inject and decide.
    """
    from scmle.assurance.detectors import certificates

    module = sys.modules[__name__]
    own = certificates.fingerprint(module)
    gates = certificates.fingerprint(*sites())
    table = hashlib.sha256(registry.PATH.read_bytes()).hexdigest()
    return hashlib.sha256(f"{own}{gates}{table}".encode()).hexdigest()


def measure(workers: int = WORKERS) -> dict[str, Any]:
    """Runs every path of every registered gate in both parts.

    Args:
        workers: Worker processes.

    Returns:
        The JSON-ready document: the records, the tally of each part, the
        floor, what was not measured, the code hash and the duration.
    """
    started = time.monotonic()
    gates = registry.load()
    todo = jobs(gates)
    with futures.ProcessPoolExecutor(workers, initializer=prepare) as pool:
        records = list(pool.map(run, *zip(*todo, strict=True)))
    return {
        "claim": "C1",
        "variable": VARIABLE,
        "floor": acceptance.load()[VARIABLE].value,
        "gates": sorted(gates),
        "records": records,
        "tally": {part: tally(select(records, part)) for part in PARTS},
        "not_measured": list(NOT_MEASURED),
        "data": {
            "real": "public tables from the local caches (refusal.fetch), "
            "faults planted and injected through Project.run, "
            "Project.resume or a scmle command",
            "simulated": "simulated tables and synthetic inputs, through "
            "runs and direct gate calls",
        },
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
    """Why the document cannot decide the row.

    Args:
        document: The document; read from `PATH` when None.

    Returns:
        Messages for a missing file, code or registry that changed, a floor
        other than acceptance.toml's, a part with no records, or a
        registered gate whose fault (or documented override) was not
        attempted in a part; empty when it decides.
    """
    found = read(document)
    if found is None:
        return [f"{PATH.name} is missing"]
    issues = []
    if found.get("code") != digest():
        issues.append("faults: measured with other code or another registry")
    floor = acceptance.load()[VARIABLE].value
    if found.get("floor") != floor:
        issues.append(f"faults: floor {found.get('floor')} is not {floor}")
    records = found.get("records", [])
    for part in PARTS:
        chosen = select(records, part)
        if not chosen:
            issues.append(f"faults: {part} part not measured")
        done = {(r.get("gate"), r.get("path")) for r in chosen}
        for name, gate in registry.load().items():
            if (name, "fault") not in done:
                issues.append(f"faults: {name} fault not attempted ({part})")
            if overridable(gate) and (name, "override") not in done:
                issues.append(f"faults: {name} override not attempted ({part})")
    return issues


def describe(part: str, counted: dict[str, Any]) -> str:
    """One part's value text.

    Args:
        part: `real` or `simulated`.
        counted: Its tally.

    Returns:
        Satisfied units over units, how many went through runs, by outcome.
    """
    outcomes = counted["outcomes"]
    label = "real data" if part == "real" else "simulated inputs"
    return (
        f"{label}: {counted['satisfied']}/{counted['units']} injected "
        f"faults blocked or ended in a recorded override "
        f"({counted['faults']} gate faults, {counted['overrides']} "
        f"overrides; {counted['runs']} through runs, {counted['sites']} "
        f"direct gate calls); silent {outcomes['silent']}, failed "
        f"{outcomes['failed']}, not injected {outcomes['not injected']}"
    )


def verdicts(
    document: dict[str, Any] | None = None,
) -> dict[str, tuple[str, str]]:
    """Each part's value text and verdict.

    The real part decides the claim's injected-failure clause; a real unit
    counts only when it went through a run (`satisfies`).

    Args:
        document: The document; read from `PATH` when None.

    Returns:
        Part to (value text, `pass` or `fail` by the floor); a part with no
        records, or any part when `problems` finds any, is `not measured`.
    """
    found = read(document)
    records = [] if found is None else found.get("records", [])
    stale = found is None or bool(problems(found))
    floor = acceptance.load()[VARIABLE].value
    result = {}
    for part in PARTS:
        chosen = select(records, part)
        if not chosen:
            label = "real data" if part == "real" else "simulated inputs"
            result[part] = (f"{label}: no measurement", "not measured")
            continue
        counted = tally(chosen)
        text = describe(part, counted)
        if stale:
            result[part] = (text, "not measured")
            continue
        share = counted["share"] or 0.0
        result[part] = (text, "pass" if share >= floor else "fail")
    return result


def summary(document: dict[str, Any] | None = None) -> tuple[str, str]:
    """The scorecard's value and verdict for the row.

    Args:
        document: The document; read from `PATH` when None.

    Returns:
        The value text of both parts (real first) and the verdict: `not
        measured` when nothing is stored, `problems` finds any or the real
        part is missing; otherwise `pass` only when the real part and the
        simulated part both pass, else `fail`.
    """
    found = read(document)
    if found is None or not found.get("records"):
        return "no measurement", "not measured"
    parts = verdicts(found)
    text = "; ".join(parts[part][0] for part in PARTS)
    states = {parts[part][1] for part in PARTS}
    if "not measured" in states:
        return text, "not measured"
    return text, "fail" if "fail" in states else "pass"


def main(arguments: list[str] | None = None) -> int:
    """Writes (`--write`) or checks (`--check`) the stored measurement.

    `--attempt GATE PATH PART` runs one path here and prints its record as
    the last line (what `delegate` reads).

    Args:
        arguments: Command-line arguments; None reads `sys.argv`.

    Returns:
        0 after `--write` or `--attempt`; for `--check`, 0 when `problems`
        is empty and 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--attempt", nargs=3, metavar=("GATE", "PATH", "PART"))
    options = parser.parse_args(arguments)
    if options.attempt:
        prepare()
        print(json.dumps(attempt(*options.attempt), sort_keys=True))
        return 0
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
