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

An injection is either `run` level, one `Project(...).run()` through the
public API with fast settings (no tuning, ensembles or deep learning), or
`site` level, a direct call of the gate's function (or the command or
stage function that contains it) when earlier gates or a charter run would
otherwise stand between the data and the gate. Each gate's pass path is run
too (`control`) where a clean input is cheap to build; it decides nothing.

The share counts one unit per gate (its fault) and one per gate whose
override is not `none` (its fault under the override). A unit not injected
counts against the claim. The row passes when the share is at least the
floor and every registered gate was attempted; `acceptance.toml` gives this
policy variable no statistical rule, so the minimum sample is the whole
registry.

NOT measured: faults reached through every code path to a gate (one
injection per gate), real data (the faults are planted in simulated
tables).

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
VARIABLE = "gate_fault_block_min"
BUDGET = 60
WORKERS = 2
ROWS = 400
SEED = 7
OUTCOMES = ("blocked", "override", "silent", "failed", "not injected")
SATISFIED = frozenset({"blocked", "override"})
NOT_MEASURED = (
    "more than one injection per gate or per code path to a gate",
    "real datasets (faults are planted in simulated tables)",
)
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
        **options: Further `Project` arguments.

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
        output_dir=folder,
        **{**FAST, **options},
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
        (Path(run.run_dir) / "config.json").read_text(encoding="utf-8")
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
    path = Path(run.run_dir) / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


def spec(task: tasks.TaskRequest = "auto", **fields: Any) -> tasks.Config:
    """A minimal specification for a direct call of a gate.

    Args:
        task: The task request.
        **fields: `Dataset` fields; `strategy`, `folds`, `horizon_rows` and
            `embargo_rows` go to `Validation` instead.

    Returns:
        The specification, with target `y` unless `target` is given.
    """
    keys = ("strategy", "folds", "horizon_rows", "embargo_rows")
    validation = {k: fields.pop(k) for k in keys if k in fields}
    fields.setdefault("target", "y")
    return tasks.Config(
        data=tasks.Dataset(uri="memory://faults", **fields),
        task=task,
        validation=tasks.Validation(**validation),
    )


def clean(kind: str) -> tuple[pandas.DataFrame, Any, dict[str, Any]]:
    """A clean simulated task of one kind.

    Args:
        kind: As for `refusal.simulate`.

    Returns:
        The table, target and `Project` arguments.
    """
    return refusal.simulate(kind, ROWS, SEED)


def planted(kind: str, defect: str) -> Any:
    """Runs a refusal-benchmark planted defect end to end.

    Args:
        kind: The task kind.
        defect: As for `refusal.spoil`.

    Returns:
        The completed run, if the product did not refuse.
    """
    frame, target, options = refusal.plant(kind, ROWS, defect)()
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


def command(unit: str) -> Any:
    """Runs `scmle ingest` on the SQLite source with a charter unit.

    Args:
        unit: The unit text; `customer-month` is coarser than the grain.

    Returns:
        The command's exit status.
    """
    from scmle import cli

    with tempfile.TemporaryDirectory() as folder:
        parser = argparse.ArgumentParser()
        step = cli.IngestCommand()
        step.configure(parser)
        arguments = parser.parse_args(
            [
                "--sqlite",
                database(folder),
                "--tables",
                "customers,orders",
                "--unit",
                unit,
                "--target-table",
                "customers",
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
    from scmle.core import ingest_stage
    from scmle.data.charter import ingest

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


def plurality(declared: int) -> None:
    """Calls the family-size check with two synthesized columns.

    Args:
        declared: The declared family size; 2 passes.
    """
    from scmle.core import ingest_stage

    ingest_stage.plurality(declared, ("a", "b"), {"a", "b"}, set())


def checkpoint(approve: bool) -> Any:
    """Runs a project with one reviewer at the planning state.

    Args:
        approve: The reviewer's verdict.

    Returns:
        The completed run.
    """
    from scmle.core import policies
    from scmle.runtime import state

    def verdict(review: Any) -> Any:
        del review
        return policies.Approval(approve, "reviewer", "fault")

    reviewer = policies.Callback(
        verdict,
        gates=frozenset({state.Progress.PLANNING}),
    )
    frame, target, options = clean("binary")
    return launch(frame, target, checkpoints=(reviewer,), **options)


def robustness() -> None:
    """Calls the robustness report with a stability stage that never ran."""
    from scmle.core import stages

    context: Any = types.SimpleNamespace()
    stage: Any = types.SimpleNamespace(found=None)
    stages.ProfileStage().assess_robustness(context, stage)


def leak(policy: str) -> str | None:
    """Runs a planted target copy under a leakage policy.

    Args:
        policy: `block` (the fault) or `warn` (the override).

    Returns:
        The recorded warning under `warn`, or None if nothing names it.
    """
    frame, target, options = refusal.plant("binary", 600, "leak")()
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
    from scmle.core import stages
    from scmle.data.admissibility import attestations
    from scmle.data.admissibility import provenance
    from scmle.data.admissibility import qualify

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
        context: Any = types.SimpleNamespace(
            spec=types.SimpleNamespace(
                leakage_attestations=None if path is None else str(path),
                data=types.SimpleNamespace(group_column=None, time_column=None),
            ),
            ingest=None,
            profile=types.SimpleNamespace(
                feature_plan=data.Blueprint(("copy",), ())
            ),
            write_json=ignore,
            emit=ignore,
        )
        charter: Any = types.SimpleNamespace(
            estimand=types.SimpleNamespace(kind="predictive", causal=None),
            seed=0,
            charter_hash=scope,
        )
        view = types.SimpleNamespace(features=features, target=target)
        found = stages.ProfileStage().qualify_exploration(
            context, tasks.Task.BINARY, charter, view, ()
        )
    rows = [c for s, c in found.checks if s == qualify.ATTESTATION_SOURCE]
    if not rows:
        return None
    return f"ledger check {rows[0].check_id}, tier {rows[0].tier.value}"


def unlabeled(policy: str) -> str | None:
    """Runs a regression with a tenth of the target missing.

    Args:
        policy: `block` (the fault) or `ignore` (the override).

    Returns:
        The curation note that records the set-aside rows, or None.
    """
    frame, target, options = refusal.plant("regression", ROWS, "unlabeled")()
    run = launch(frame, target, unlabeled=policy, **options)
    if "set aside" not in text(run, "curation_report.json"):
        return None
    return f"curation_report.json: rows set aside (policy {policy})"


def serving(megabytes: float, pareto: bool = False) -> str | None:
    """Runs a binary task under a model-size limit.

    Args:
        megabytes: The limit; 1e-6 no model meets.
        pareto: Whether to choose on the Pareto front.

    Returns:
        The decision-log entry that records the choice under the limit.
    """
    frame, target, options = clean("binary")
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
    from scmle.data.charter import ingest
    from scmle.data.ingest import declared

    orders = pandas.DataFrame(
        {"order_id": range(1, 101), "amount": [50.0] * 100}
    )
    expected = ingest.Expectation(
        (
            ingest.TableTotals(
                "orders",
                rows=100,
                sums=(ingest.Total("amount", 5000.0),),
                checksum_column="order_id",
                checksum=declared.sha(orders["order_id"]),
            ),
        )
    )
    frame = orders.iloc[: len(orders) - drop]
    declared.require(declared.reconcile({"orders": frame}, expected, 0.0))


def confirmation(times: int) -> None:
    """Runs the single confirmation on a real ledger `times` times.

    Args:
        times: 1 passes; 2 injects a second use of the confirmation set.
    """
    from scmle.assurance.ledger import confirm
    from scmle.assurance.ledger import controls
    from scmle.assurance.ledger import explore
    from scmle.assurance.ledger import online_fdr
    import scmle.assurance.ledger.ledger
    from scmle.data.custody import sealed
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
    explore.run_exploration(
        ledger,
        features,
        target,
        explore.hypothesize(features, explore.NUMERIC),
    )
    ledger.negatives(
        controls.summarize("permuted_target", "strata", 6, (0,) * 20).to_dict()
    )
    rows, labels = world(500, 2)
    locked = splitters.ResidentTestSet(rows, labels, tuple(range(len(rows))))
    source = sealed.Locker("C", locked, sealed.Journal())
    for _ in range(times):
        confirm.confirm(ledger, source, alpha_confirm=alpha)


def monotonic(column: str) -> Any:
    """Runs a binary task with a monotonic constraint on one column.

    Args:
        column: `region` (categorical, the fault) or `x1` (numeric).

    Returns:
        The completed run.
    """
    frame, target, options = refusal.simulate("binary", ROWS, SEED, "mixed")
    return launch(frame, target, monotonic_constraints={column: 1}, **options)


def featureless() -> Any:
    """Runs a binary task whose only feature is constant.

    Returns:
        The completed run.
    """
    frame, target, options = clean("binary")
    frame = pandas.DataFrame({"constant": 1.0, target: frame[target]})
    return launch(frame, target, **options)


def starved(seconds: float) -> Any:
    """Runs a binary task under a time budget.

    Args:
        seconds: The budget; 1 s cannot fit the minimum run (the fault),
            `BUDGET` can.

    Returns:
        The completed run.
    """
    frame, target, options = clean("binary")

    def adjust(config: tasks.Config) -> tasks.Config:
        resources = config.resources.model_copy(
            update={"time_budget_seconds": seconds}
        )
        return config.model_copy(update={"resources": resources})

    return launch(frame, target, adjust=adjust, **options)


def overrun(spent: bool) -> Any:
    """Runs a binary task whose preparation may have closed the trial window.

    Args:
        spent: Whether training starts with the run's start moved back by
            the whole budget, as after a preparation far longer than its
            estimate (the fault).

    Returns:
        The completed run.
    """
    import scmle.api.project
    from scmle.core import stages

    class Late(stages.TrainStage):
        """Training that first spends the whole budget when `spent`."""

        def execute(self, context: Any) -> None:
            """Moves the run's start back, then trains.

            Args:
                context: The run context.
            """
            if spent:
                context.regulator.started_at -= BUDGET
            super().execute(context)

    frame, target, options = clean("binary")
    project = scmle.api.project.Project(
        frame,
        target,
        time_budget=BUDGET,
        output_dir=tempfile.mkdtemp(prefix="faults-"),
        **{**FAST, **options},
    )
    project.orchestrator.stages = tuple(
        Late() if isinstance(stage, stages.TrainStage) else stage
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
        split: Any = splitters.StratifiedSplit(spec(), tasks.Task.BINARY)
    elif kind == "events":
        frame["y"] = numpy.arange(rows, dtype=float) + 1
        frame["event"] = labels
        split = splitters.EventStratifiedSplit(
            spec("survival", event_column="event"), tasks.Task.SURVIVAL
        )
    else:
        frame["y"] = numpy.tile([0, 1], rows // 2)
        frame["treated"] = labels
        frame.loc[frame["treated"] == 1, "y"] = 1
        frame.loc[frame.index[-1], "y"] = 0
        frame.loc[frame.index[-2], "y"] = 1
        split = splitters.TreatmentStratifiedSplit(
            spec("uplift", treatment_column="treated"), tasks.Task.UPLIFT
        )
    if kind == "treatment" and rare == 1:
        cells = split.cells(frame)
        keep = cells[cells == cells.value_counts().idxmin()].index[1:]
        frame = frame.drop(index=keep).reset_index(drop=True)
    return split.cross_validation_folds(frame, numpy.arange(len(frame)), [])


def grouped(classes: bool) -> Any:
    """Runs a binary task with a group column.

    Args:
        classes: Whether the positive class spreads over many groups; when
            False, it lives in a single group.

    Returns:
        The completed run.
    """
    frame, target, options = clean("binary")
    frame["entity"] = numpy.arange(len(frame)) % 40
    if not classes:
        frame[target] = (frame["entity"] == 0).astype(int)
    return launch(frame, target, group_column="entity", **options)


def strategy(name: str) -> Any:
    """Runs a binary task with a validation strategy.

    Args:
        name: `stratified` passes; `stratified_events` cannot serve binary.

    Returns:
        The completed run.
    """
    frame, target, options = clean("binary")
    return launch(frame, target, validation=name, **options)


def resident(fault: str) -> None:
    """Misuses or uses the in-process test set once.

    Args:
        fault: `bind` (bind twice), `open` (a wrong token) or `none`.
    """
    from scmle.data.splitting import splitters

    locked = splitters.ResidentTestSet(
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
    import scmle.governance.vault

    locked = scmle.governance.vault.SealedTestSet(
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
    from scmle.data.custody import journal

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


def rolling(horizon: int) -> str | None:
    """Runs a forecast under rolling validation with a horizon.

    Args:
        horizon: Rows per validation window; one longer than the series
            injects the fault.

    Returns:
        The saved horizon, the record of the shortened setting.
    """
    frame, target, options = refusal.simulate("forecast", 200, SEED)
    run = launch(
        frame, target, validation="rolling", horizon_rows=horizon, **options
    )
    saved = configured(run)["validation"]["horizon_rows"]
    return f"config.json: validation.horizon_rows {saved}"


def chronology(fault: str) -> Any:
    """Calls the temporal partition on a small dated table.

    Args:
        fault: `gap` (a missing time), `short` (too few rows) or `none`.

    Returns:
        The partition.
    """
    from scmle.data.splitting import splitters

    rows = 6 if fault == "short" else 200
    frame = pandas.DataFrame(
        {
            "t": pandas.date_range("2024-01-01", periods=rows),
            "x": numpy.arange(rows, dtype=float),
            "y": numpy.arange(rows) % 2,
        }
    )
    if fault == "gap":
        frame.loc[3, "t"] = pandas.NaT
    split = splitters.Chronology(spec(time_column="t"), tasks.Task.BINARY)
    return split.partition(frame)


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

    resolver = scmle.data.validation.target.TaskResolver(spec(task, **fields))
    return resolver.resolve(target)


def ambiguous(task: str) -> str | None:
    """Runs a regression on a small-cardinality integer target.

    Args:
        task: `auto` (the fault) or `regression` (the override).

    Returns:
        The saved task request, the record of the explicit choice.
    """
    frame, target, options = clean("regression")
    frame[target] = pandas.qcut(frame[target], 5, labels=False).astype(int)
    options["task"] = task
    run = launch(frame, target, **options)
    return f"config.json: task {configured(run)['task']}"


def ordinal(complete: bool) -> Any:
    """Runs an ordinal task whose listed order omits a label.

    Args:
        complete: Whether the order lists every label.

    Returns:
        The completed run.
    """
    frame, target, options = clean("ordinal")
    if not complete:
        options["ordinal_order"] = list(options["ordinal_order"])[:-1]
    return launch(frame, target, **options)


def ranking(negative: bool) -> Any:
    """Runs a ranking task, optionally with one negative grade.

    Args:
        negative: Whether to plant a negative grade.

    Returns:
        The completed run.
    """
    frame, target, options = clean("ranking")
    if negative:
        frame.loc[0, target] = -1
    return launch(frame, target, **options)


def durations(negative: bool) -> Any:
    """Runs a survival task, optionally with a negative duration.

    Args:
        negative: Whether to plant a negative duration.

    Returns:
        The completed run.
    """
    frame, target, options = clean("survival")
    if negative:
        frame.loc[0, target] = -1.0
    return launch(frame, target, **options)


def outcome(textual: bool) -> Any:
    """Runs an uplift task, optionally with a text outcome.

    Args:
        textual: Whether the outcome is text.

    Returns:
        The completed run.
    """
    frame, target, options = clean("uplift")
    if textual:
        frame[target] = numpy.where(frame[target] == 1, "yes", "no")
    return launch(frame, target, **options)


def gpu(policy: str) -> str | None:
    """Runs a binary task under a GPU policy.

    Args:
        policy: `required` (the fault on a machine without a usable GPU
            adapter) or `auto` (the override).

    Returns:
        The saved policy, the record of the override.
    """
    frame, target, options = clean("binary")

    def adjust(config: tasks.Config) -> tasks.Config:
        resources = config.resources.model_copy(update={"gpu": policy})
        return config.model_copy(update={"resources": resources})

    run = launch(frame, target, adjust=adjust, **options)
    return f"config.json: resources.gpu {configured(run)['resources']['gpu']}"


def multiclass(constraints: dict[str, int]) -> str | None:
    """Runs a multiclass task with monotonic constraints.

    Args:
        constraints: The constraints; any on multiclass is the fault, none
            is the override.

    Returns:
        The saved constraints, the record of the relaxed setting.
    """
    frame, target, options = clean("multiclass")
    run = launch(frame, target, monotonic_constraints=constraints, **options)
    saved = configured(run)["data"]["monotonic_constraints"]
    return f"config.json: data.monotonic_constraints {saved}"


@dataclasses.dataclass(frozen=True)
class Injection:
    """How one gate's guarded condition is planted.

    Attributes:
        level: `run` (an end-to-end project) or `site` (a direct call).
        cause: The planted condition.
        fault: Plants it.
        control: Runs the pass path, or None when it is not built.
        override: Plants the fault under the documented override, or None.
        reason: Why a path is not injected, or ''.
    """

    level: str
    cause: str
    fault: Callable[[], Any] | None
    control: Callable[[], Any] | None = None
    override: Callable[[], Any] | None = None
    reason: str = ""


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
    "ingest.cli.unit_verdict": Injection(
        "site",
        "scmle ingest with unit customer-month on a customer table",
        bind(command, "customer-month"),
        bind(command, "customer"),
    ),
    "ingest.multiplicity": Injection(
        "site",
        "declared family size 3, two synthesized columns",
        bind(plurality, 3),
        bind(plurality, 2),
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
    "robustness.assessment_present": Injection(
        "site",
        "stability stage without an assessment (found is None)",
        robustness,
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
    "split.stratified.stratify": Injection(
        "run",
        "a multiclass class of a single row",
        bind(planted, "multiclass", "singleton"),
    ),
    "split.temporal.holdout": Injection(
        "site",
        "a missing time stamp",
        bind(chronology, "gap"),
        bind(chronology, "none"),
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
    "train.first_trial": Injection(
        "run",
        "training starts after preparation used up the trial window",
        bind(overrun, spent=True),
        bind(overrun, spent=False),
    ),
}


def overridable(gate: registry.Gate) -> bool:
    """Whether the registry documents a way to proceed past the gate.

    Args:
        gate: The registered gate.

    Returns:
        False when its override text starts with `none`.
    """
    return not gate.override.strip().lower().startswith("none")


def attempt(name: str, path: str) -> dict[str, Any]:
    """Runs one path of one gate's injection and records how it ended.

    Args:
        name: The gate.
        path: `fault`, `override` or `control`.

    Returns:
        The gate, path, outcome, the gate that blocked (if any) with its
        message, the override's record, the error of a failure, seconds.

    Raises:
        InputError: For an unknown gate or path.
    """
    if name not in INJECTIONS or path not in ("fault", "override", "control"):
        raise errors.InputError(f"no injection {name!r} / {path!r}")
    injection = INJECTIONS[name]
    action = getattr(injection, path)
    record: dict[str, Any] = {"gate": name, "path": path}
    if action is None:
        record.update(outcome="not injected", reason=injection.reason)
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


def jobs(gates: dict[str, registry.Gate]) -> list[tuple[str, str]]:
    """Every path to run, gate by gate.

    Args:
        gates: The registry.

    Returns:
        (gate, path) pairs: the fault of every gate, the override of every
        overridable gate and the control where one is built.
    """
    found = []
    for name, gate in gates.items():
        found.append((name, "fault"))
        if overridable(gate):
            found.append((name, "override"))
        injection = INJECTIONS.get(name)
        if injection is not None and injection.control is not None:
            found.append((name, "control"))
    return found


def run(name: str, path: str) -> dict[str, Any]:
    """`attempt`, with a gate missing from `INJECTIONS` recorded.

    Args:
        name: The gate.
        path: The path.

    Returns:
        The record.
    """
    if name not in INJECTIONS:
        return {
            "gate": name,
            "path": path,
            "outcome": "not injected",
            "reason": "no injection built for this gate",
        }
    return attempt(name, path)


def prepare() -> None:
    """Quiets the product's logs in a worker and pins one thread."""
    os.environ["OMP_NUM_THREADS"] = "1"
    logging.getLogger("scmle").setLevel(logging.WARNING)


def tally(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts the units the claim is decided on.

    Args:
        records: One record per path run.

    Returns:
        Units (faults and overrides), those satisfied (blocked or recorded
        override), counts by outcome, blocks attributed to another gate,
        overrides that still blocked, controls that did not pass.
    """
    units = [r for r in records if r["path"] in ("fault", "override")]
    outcomes = {o: sum(r["outcome"] == o for r in units) for o in OUTCOMES}
    satisfied = [r for r in units if r["outcome"] in SATISFIED]
    return {
        "units": len(units),
        "satisfied": len(satisfied),
        "share": len(satisfied) / len(units) if units else None,
        "faults": sum(r["path"] == "fault" for r in units),
        "overrides": sum(r["path"] == "override" for r in units),
        "outcomes": outcomes,
        "unsatisfied": sorted(
            f"{r['gate']} ({r['path']}: {r['outcome']})"
            for r in units
            if r["outcome"] not in SATISFIED
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
    from scmle.data.detectors import certificates

    module = sys.modules[__name__]
    own = certificates.code_hash(module)
    gates = certificates.code_hash(*sites())
    table = hashlib.sha256(registry.PATH.read_bytes()).hexdigest()
    return hashlib.sha256(f"{own}{gates}{table}".encode()).hexdigest()


def measure(workers: int = WORKERS) -> dict[str, Any]:
    """Runs every path of every registered gate and builds the document.

    Args:
        workers: Worker processes.

    Returns:
        The JSON-ready document: the records, the tally, the floor, what
        was not measured, the code hash and the duration.
    """
    started = time.monotonic()
    gates = registry.load()
    todo = jobs(gates)
    with futures.ProcessPoolExecutor(workers, initializer=prepare) as pool:
        records = list(pool.map(run, *zip(*todo, strict=True)))
    for record in records:
        injection = INJECTIONS.get(record["gate"])
        if injection is not None:
            record.update(level=injection.level, cause=injection.cause)
    return {
        "claim": "C1",
        "variable": VARIABLE,
        "floor": acceptance.load()[VARIABLE].value,
        "gates": sorted(gates),
        "records": records,
        "tally": tally(records),
        "not_measured": list(NOT_MEASURED),
        "data": "simulated tables and synthetic inputs only",
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
        other than acceptance.toml's, or a registered gate whose fault (or
        documented override) was not attempted; empty when it decides.
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
    done = {(r.get("gate"), r.get("path")) for r in found.get("records", [])}
    for name, gate in registry.load().items():
        if (name, "fault") not in done:
            issues.append(f"faults: {name} fault not attempted")
        if overridable(gate) and (name, "override") not in done:
            issues.append(f"faults: {name} override not attempted")
    return issues


def summary(document: dict[str, Any] | None = None) -> tuple[str, str]:
    """The scorecard's value and verdict for the row.

    Args:
        document: The document; read from `PATH` when None.

    Returns:
        The value text (satisfied units over units, by outcome) and `pass`
        or `fail` by the floor; `not measured` when nothing is stored or
        `problems` finds any.
    """
    found = read(document)
    if found is None or not found.get("records"):
        return "no measurement", "not measured"
    counted = tally(found["records"])
    outcomes = counted["outcomes"]
    text = (
        f"{counted['satisfied']}/{counted['units']} injected faults blocked "
        f"or ended in a recorded override ({counted['faults']} gate faults, "
        f"{counted['overrides']} overrides); silent {outcomes['silent']}, "
        f"failed {outcomes['failed']}, not injected "
        f"{outcomes['not injected']}; simulated inputs, not real data"
    )
    if problems(found):
        return text, "not measured"
    floor = acceptance.load()[VARIABLE].value
    share = counted["share"] or 0.0
    return text, "pass" if share >= floor else "fail"


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
