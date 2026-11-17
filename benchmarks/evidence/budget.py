"""Claim C7 in part: runs keep their time budget and survive a hard kill.

Three measurements, each run in child processes started from this module:

* Budget compliance on real tables (the deciding family). Every table of
  `tables()`: the 15 OpenML tasks cached for C4 (`quality.load`, training
  rows only, C4's metric) and the 31 OpenML tables of C14's `EXTRA` list
  (`refusal.table`, at most 4,000 rows), run through `Project(...).run()`
  with hard enforcement, each in its own process with `OMP_NUM_THREADS=1`,
  `WORKERS` at a time. The tables are run in passes, as many as start
  inside `REAL_SECONDS`; pass `k` gives a table the seed
  `scmle_runner.seed_for(id, 0, k)` (pass 0 is C4's seed), so every pass is
  a distinct run, and the seed is filed with the run. The budget is set
  from the table's size by `allot`: `BASE_SECONDS` up to `BASE_CELLS`
  cells, doubled for every tenfold more, because tree-model search time
  grows with the cells while a user's budget grows much more slowly. The
  clock starts before `Project` is built. A run is compliant when it
  completes within its allowance, the budget plus `limits.SLACK_SECONDS`
  (the poll interval plus the kill time: the only excess the product
  allows itself, since it reserves the refit, diagnostics and packaging
  inside the budget). A run the product refuses before starting
  (`BlockedError`: its minimum cannot fit the budget) is filed as refused:
  neither compliant nor over, and left out of the floor's count. A run
  still going at `TIMEOUT_FACTOR` times its allowance is killed and
  counted as over; one that ends with any other error is counted as over
  (crashed). A run starts only if its kill time falls inside
  `REAL_SECONDS`, so the measurement cannot overrun its window; runs of a
  started pass that did not fit are listed.
* Budget compliance on the practice project, reported but never deciding.
  `RUNS_MAX` small runs (`reliability.fast_project` on
  `reliability.binary_data`, a different data seed each, hard enforcement)
  with a `BUDGET_SECONDS` time budget, one after another, each in its own
  process with the environment inherited unchanged, compliant within the
  same allowance. The stored runs were measured under the old rule (budget
  plus a grace of `max(0.1 B, 1 s)`, 2 s at 20 s, since removed from the
  product); `--write` keeps them and files their rule beside them, and
  only `--write --practice` remeasures them under the current rule.

The row is the `budget_compliance_min` floor of `acceptance.toml`, decided
by its `Floor` rule at `alpha` on the real family alone. Fewer real-table
runs (refusals excluded) than the rule's minimum is "not measured", never
a pass, whatever the practice project shows.
* Kill and resume. One uninterrupted run with a budget large enough that
  every candidate runs (`RESUME_BUDGET_SECONDS`), then `len(KILL_POINTS)`
  runs of the same project, each killed with SIGKILL (its whole process
  group, so trial workers die too) once `k` trial checkpoints exist for
  `k` in `KILL_POINTS`, and continued in a new process with
  `Project.resume`. Each resumed result is compared field by field with the
  uninterrupted one: selection, scores, test metrics and the trial list,
  and its event log must show exactly one creation and one resume.
  These children run with `OMP_NUM_THREADS=1` so the run is quick on a
  shared machine; the comparison does not depend on timing.

Every run's one-minute load average is filed, because the laptop running
this is shared and compliance depends on it.

NOT measured: the memory budget, budgets other than the stated ones, real
tables above 52,035 training rows, kills outside the training stage, and
journal and lease operations.

`--write` reruns the real family only and keeps the stored practice runs;
`--write --practice` reruns everything (about 70 minutes more).

    python -m benchmarks.evidence.budget --write
    python -m benchmarks.evidence.budget --check
"""

import argparse
from concurrent import futures
import contextlib
import functools
import json
import math
import os
from pathlib import Path
import signal
import subprocess  # nosec B404
import sys
import tempfile
import time
from typing import Any

from benchmarks import acceptance
from benchmarks import reliability
from benchmarks.evidence import quality
from benchmarks.evidence import refusal
from benchmarks.sota import scmle_runner
import scmle.api.project
from scmle.contracts import errors
from scmle.core import stages
from scmle.data.detectors import certificates
from scmle.runtime import limits

PATH = Path(__file__).with_suffix(".json")
ROOT = Path(__file__).resolve().parents[2]
VARIABLE = "budget_compliance_min"
MODULE = "benchmarks.evidence.budget"
# Small enough to repeat, large enough that an unloaded laptop finishes the
# preparation, the search and the test inside it (about 8 s on 600 rows).
BUDGET_SECONDS = 20
# Wall time spent on compliance runs before the module stops starting new
# ones. The floor needs at least 299 runs at alpha 0.05; an unloaded run
# takes about 10 s, so 299 runs need about 3,000 s, and 4,200 s leaves room
# to reach RUNS_MAX. A write takes about 70 minutes on an unloaded laptop.
COMPLIANCE_SECONDS = 4200.0
RUNS_MAX = 400
# Past the allowance a run is already over; three times the allowance tells
# a slow run from a stalled one without waiting for a stall to end.
TIMEOUT_FACTOR = 3.0
# Every candidate of the quick project runs well inside this budget, so the
# result does not depend on how fast the machine is.
RESUME_BUDGET_SECONDS = 600
KILL_POINTS = (1, 2, 3)
RESUME_SEED = 0
POLL_SECONDS = 0.02
CHILD_SECONDS = 900.0
# a child mode word and its operands
RUN_ARGUMENTS = 4
RESUME_ARGUMENTS = 3
TABLE_ARGUMENTS = 5
# Budget of a real table: 30 s up to 10,000 cells (rows times features),
# doubled for every tenfold more. The practice project (600 rows, 20 s)
# sits on this line; C4's 60 s budget is its second step. Fixed before any
# real-table run of this module and not tuned to the outcomes.
BASE_SECONDS = 30
BASE_CELLS = 10_000
# The real family's window. A run starts only when its kill time
# (`TIMEOUT_FACTOR` times its allowance) still falls inside the window, so
# the measurement itself cannot overrun it. One pass takes about 1,400 s
# at two workers, so an hour holds two passes and part of a third.
REAL_SECONDS = 3600.0
WORKERS = 2
# Passes queued over the real tables; the window, not this, ends the
# measurement. 46 tables per pass; the floor needs 299 runs, so 7 full
# passes with distinct seeds would be needed to decide it.
PASSES = 8
# Rule of the practice runs stored before the grace was removed, and of
# runs measured now.
OLDRULE = "budget + max(0.1 B, 1 s) grace (removed from the product)"
RULE = "budget + limits.SLACK_SECONDS"


def allowance(budget: float) -> float:
    """The latest wall time a compliant run may take.

    Args:
        budget: The declared time budget in seconds.

    Returns:
        The budget plus the product's enforcement slack, the poll interval
        plus the kill time (`limits.SLACK_SECONDS`).
    """
    return budget + limits.SLACK_SECONDS


def settings() -> dict[str, Any]:
    """The fixed design of the measurement.

    Returns:
        Every constant that changes what is measured.
    """
    return {
        "budget_seconds": BUDGET_SECONDS,
        "allowance_seconds": allowance(BUDGET_SECONDS),
        "slack_seconds": limits.SLACK_SECONDS,
        "diagnostic_allowance_seconds": stages.DIAGNOSTIC_MIN_SECONDS,
        "compliance_seconds": COMPLIANCE_SECONDS,
        "runs_max": RUNS_MAX,
        "timeout_factor": TIMEOUT_FACTOR,
        "resume_budget_seconds": RESUME_BUDGET_SECONDS,
        "kill_points": list(KILL_POINTS),
        "resume_seed": RESUME_SEED,
    }


def project(
    directory: Path, budget: int, seed: int
) -> scmle.api.project.Project:
    """The quick project every child runs.

    Args:
        directory: Output directory of the run.
        budget: Time budget in seconds.
        seed: Seed of the generated data.

    Returns:
        The configured project, hard enforcement.
    """
    return reliability.fast_project(
        reliability.binary_data(seed=seed),
        "y",
        directory,
        time_budget=budget,
        enforcement="hard",
    )


def clean(value: Any) -> Any:
    """Replaces NaN, which never equals itself, by None, recursively.

    Args:
        value: A JSON-ready value.

    Returns:
        The value with every NaN float replaced by None.
    """
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [clean(item) for item in value]
    return value


def outcome(run: Any) -> dict[str, Any]:
    """The parts of a finished run a resume must reproduce.

    Args:
        run: A completed `results.Run`.

    Returns:
        Selection, scores, test metrics and the trial list.
    """
    evaluation = run.evaluation
    found: dict[str, Any] = clean(
        {
            "selected_model": run.selected_model,
            "selected_params": evaluation["selected_params"],
            "selected_cv_score": evaluation["selected_cv_score"],
            "test_metrics": evaluation["test_metrics"],
            "trials": [
                [t.trial_id, t.model_name, str(t.status), t.score_mean]
                for t in run.trials
            ],
        }
    )
    return found


def child(arguments: list[str]) -> int:
    """Runs inside a child process and prints one JSON line.

    Args:
        arguments: `run DIRECTORY BUDGET SEED` or `resume RUN_DIRECTORY
            SEED`.

    Returns:
        0 once the line is printed.

    Raises:
        InputError: For an unknown mode or the wrong number of arguments.
    """
    if arguments[:1] == ["run"] and len(arguments) == RUN_ARGUMENTS:
        directory, budget, seed = arguments[1:]
        built = project(Path(directory), int(budget), int(seed))
        started = time.monotonic()
        run = built.run()
    elif arguments[:1] == ["resume"] and len(arguments) == RESUME_ARGUMENTS:
        frame = reliability.binary_data(seed=int(arguments[2]))
        started = time.monotonic()
        run = scmle.api.project.Project.resume(arguments[1], data=frame)
    else:
        raise errors.InputError(f"unknown child arguments {arguments}")
    seconds = time.monotonic() - started
    print(json.dumps({"seconds": seconds, "outcome": outcome(run)}))
    return 0


def spawn(arguments: list[str], *, quiet: bool) -> subprocess.Popen[str]:
    """Starts this module as a child in its own process group.

    Args:
        arguments: The child's arguments.
        quiet: Whether to cap the child's math libraries at one thread.

    Returns:
        The started process, its output piped.
    """
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), environment.get("PYTHONPATH", "")]
    )
    if quiet:
        environment["OMP_NUM_THREADS"] = "1"
    command = [sys.executable, "-m", MODULE, "--child", *arguments]
    return subprocess.Popen(  # nosec B603
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def kill(process: subprocess.Popen[str]) -> None:
    """Kills a child and every process it started, then reaps it.

    Args:
        process: A child started by `spawn`.
    """
    # the group may already have exited
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.communicate()


def finish(process: subprocess.Popen[str], timeout: float) -> dict[str, Any]:
    """Waits for a child and reads its line.

    Args:
        process: A child started by `spawn`.
        timeout: Seconds to wait before killing it.

    Returns:
        `seconds` and `outcome` from the child, or `error` with the tail
        of its output, and `timed_out`.
    """
    try:
        out, err = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill(process)
        return {"timed_out": True, "error": f"killed after {timeout:.0f} s"}
    lines = [line for line in out.splitlines() if line.startswith("{")]
    if process.returncode != 0 or not lines:
        return {"timed_out": False, "error": err[-600:]}
    found: dict[str, Any] = json.loads(lines[-1])
    found["timed_out"] = False
    return found


def comply(seed: int, directory: Path) -> dict[str, Any]:
    """One budget-compliance run.

    Args:
        seed: Data seed of the run.
        directory: Scratch directory for the run.

    Returns:
        The seed, wall time, load average, outcome and whether it is
        compliant.
    """
    limit = allowance(BUDGET_SECONDS)
    load = os.getloadavg()[0]
    started = time.monotonic()
    process = spawn(
        ["run", str(directory), str(BUDGET_SECONDS), str(seed)], quiet=False
    )
    found = finish(process, TIMEOUT_FACTOR * limit)
    seconds = found.get("seconds")
    return {
        "seed": seed,
        "load": round(load, 1),
        "process_seconds": round(time.monotonic() - started, 2),
        "run_seconds": None if seconds is None else round(seconds, 2),
        "completed": "outcome" in found,
        "timed_out": found["timed_out"],
        "error": found.get("error", ""),
        "compliant": seconds is not None and seconds <= limit,
    }


def compliance(scratch: Path) -> dict[str, Any]:
    """Runs compliance runs until the time or run limit.

    Args:
        scratch: Directory for the runs.

    Returns:
        The runs, the counts and the allowance.
    """
    started = time.monotonic()
    runs: list[dict[str, Any]] = []
    while len(runs) < RUNS_MAX:
        if time.monotonic() - started >= COMPLIANCE_SECONDS:
            break
        seed = len(runs)
        runs.append(comply(seed, scratch / f"comply_{seed}"))
        print(f"budget run {seed}: {runs[-1]['run_seconds']}", file=sys.stderr)
    limit = allowance(BUDGET_SECONDS)
    lenient = limit + stages.DIAGNOSTIC_MIN_SECONDS
    return {
        "allowance_seconds": limit,
        "trials": len(runs),
        "hits": sum(r["compliant"] for r in runs),
        "within_allowance_plus_diagnostics": sum(
            r["run_seconds"] is not None and r["run_seconds"] <= lenient
            for r in runs
        ),
        "runs": runs,
    }


def checkpoints(directory: Path) -> int:
    """Trial checkpoints written so far under an output directory.

    Args:
        directory: The output directory of one run.

    Returns:
        The number of `trial_*.json` files.
    """
    return len(list(directory.glob("run_*/trials/trial_*.json")))


def differences(
    reference: dict[str, Any], resumed: dict[str, Any]
) -> list[str]:
    """Fields of a resumed outcome that differ from the reference.

    Args:
        reference: The uninterrupted run's outcome.
        resumed: The resumed run's outcome.

    Returns:
        One message per differing field; empty when identical.
    """
    keys = sorted(set(reference) | set(resumed))
    return [
        f"{key}: {reference.get(key)!r} != {resumed.get(key)!r}"
        for key in keys
        if reference.get(key) != resumed.get(key)
    ]


def interrupt(
    point: int, directory: Path, reference: dict[str, Any]
) -> dict[str, Any]:
    """Kills one run after `point` checkpoints and resumes it.

    Args:
        point: Trial checkpoints to wait for before the kill.
        directory: Scratch output directory of this run.
        reference: The uninterrupted run's outcome.

    Returns:
        How many checkpoints existed at the kill, whether the run had
        already finished, the resume's error if any, the differences and
        whether the resumed result is identical.
    """
    process = spawn(
        [
            "run",
            str(directory),
            str(RESUME_BUDGET_SECONDS),
            str(RESUME_SEED),
        ],
        quiet=True,
    )
    deadline = time.monotonic() + CHILD_SECONDS
    while checkpoints(directory) < point and process.poll() is None:
        if time.monotonic() > deadline:
            break
        time.sleep(POLL_SECONDS)
    early = process.poll() is not None
    kill(process)
    killed = checkpoints(directory)
    runs = sorted(directory.glob("run_*"))
    if early or not runs:
        return {
            "point": point,
            "checkpoints_at_kill": killed,
            "finished_before_kill": early,
            "error": "the run ended before the kill point",
            "differences": [],
            "identical": False,
        }
    found = finish(
        spawn(["resume", str(runs[-1]), str(RESUME_SEED)], quiet=True),
        CHILD_SECONDS,
    )
    differ = (
        differences(reference, found["outcome"]) if "outcome" in found else []
    )
    created = history(runs[-1])
    if created != ["run created", "run resumed"]:
        differ.append(f"run history {created!r}")
    return {
        "point": point,
        "checkpoints_at_kill": killed,
        "finished_before_kill": False,
        "error": found.get("error", ""),
        "created": created,
        "differences": differ,
        "identical": "outcome" in found and not differ,
    }


def history(run: Path) -> list[str]:
    """The reasons of every `state.CREATED` event of a run.

    Args:
        run: A run directory.

    Returns:
        The reasons in order; a killed and resumed run has `run created`
        then `run resumed`.
    """
    path = run / "events.jsonl"
    if not path.exists():
        return []
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [e["reason"] for e in events if e["kind"] == "state.CREATED"]


def resumption(scratch: Path) -> dict[str, Any]:
    """The uninterrupted reference run and every kill and resume.

    Args:
        scratch: Directory for the runs.

    Returns:
        The reference outcome, the cases and the count of identical ones.

    Raises:
        ScmleError: If the uninterrupted reference run fails.
    """
    reference = finish(
        spawn(
            [
                "run",
                str(scratch / "reference"),
                str(RESUME_BUDGET_SECONDS),
                str(RESUME_SEED),
            ],
            quiet=True,
        ),
        CHILD_SECONDS,
    )
    if "outcome" not in reference:
        raise errors.ScmleError(
            "the uninterrupted reference run failed",
            {"error": reference.get("error", "")},
        )
    cases = [
        interrupt(point, scratch / f"kill_{point}", reference["outcome"])
        for point in KILL_POINTS
    ]
    return {
        "reference": reference["outcome"],
        "cases": cases,
        "identical": sum(c["identical"] for c in cases),
        "trials": len(cases),
    }


def current() -> str:
    """Hash of the code that decides what is measured.

    Returns:
        The combined source hash.
    """
    return certificates.code_hash(
        allowance,
        settings,
        project,
        clean,
        outcome,
        child,
        finish,
        comply,
        compliance,
        checkpoints,
        differences,
        interrupt,
        history,
        resumption,
        reliability.fast_project,
        reliability.binary_data,
    )


def allot(cells: int) -> int:
    """The budget a table of the real family is given.

    Args:
        cells: Training rows times feature columns.

    Returns:
        `BASE_SECONDS` up to `BASE_CELLS` cells, doubled for every tenfold
        more cells (module docstring).
    """
    steps = max(0, math.ceil(math.log10(max(cells, 1) / BASE_CELLS)))
    return int(BASE_SECONDS * 2**steps)


def tables() -> list[tuple[str, int]]:
    """Every table of the real family, in the order they are run.

    Returns:
        (source, id) pairs: `quality` for the OpenML tasks cached for C4,
        `refusal` for the OpenML tables of C14's `EXTRA` list.
    """
    found = [("quality", t) for ids in quality.TASKS.values() for t in ids]
    return found + [("refusal", t) for t in refusal.EXTRA]


def seeded(source: str, ident: int, number: int) -> int:
    """The seed a table is run with in one pass.

    Args:
        source: `quality` or `refusal`.
        ident: The table's id.
        number: The pass, from 0.

    Returns:
        `scmle_runner.seed_for(ident, 0, number)`, so pass 0 of a quality
        table is C4's seed and every pass is a distinct run.
    """
    del source  # both sources draw from the same seed line
    return scmle_runner.seed_for(ident, 0, number)


def load(source: str, ident: int) -> tuple[Any, str, dict[str, Any]]:
    """One real table with its target and declared task.

    Args:
        source: `quality` or `refusal`.
        ident: The OpenML task id (quality) or data id (refusal).

    Returns:
        The training frame, the target and the `Project` arguments that
        declare the task (and, for quality tables, C4's metric).

    Raises:
        InputError: For an unknown source.
    """
    if source == "quality":
        frame, _, target = quality.load(ident)
        name = quality.kind(ident)
        options = {
            "task": "regression" if name == "regression" else "classification",
            "metric": scmle_runner.METRICS[name],
        }
        return frame, target, options
    if source == "refusal":
        _, _, column, task = refusal.EXTRA[ident]
        frame, target, options = refusal.table(ident, column, task)()
        return frame, target, options
    raise errors.InputError(f"unknown table source {source!r}")


@functools.cache
def size(source: str, ident: int) -> tuple[int, int]:
    """Rows and cells of one real table, loaded once per process.

    Args:
        source: `quality` or `refusal`.
        ident: The table's id.

    Returns:
        Training rows, and rows times feature columns.
    """
    frame, _, _ = load(source, ident)
    return len(frame), len(frame) * (frame.shape[1] - 1)


def table(arguments: list[str]) -> int:
    """Runs one real table inside a child process and prints one line.

    The clock starts before `Project` is built: what the user waits for. A
    run the product refuses before starting prints `blocked` and its
    message instead of a model.

    Args:
        arguments: `SOURCE ID DIRECTORY BUDGET SEED`.

    Returns:
        0 once the line is printed.

    Raises:
        InputError: For the wrong number of arguments.
    """
    if len(arguments) != TABLE_ARGUMENTS:
        raise errors.InputError(f"unknown table arguments {arguments}")
    source, ident, directory, budget, seed = arguments
    frame, target, options = load(source, int(ident))
    options["seed"] = int(seed)
    started = time.monotonic()
    try:
        run = scmle.api.project.Project(
            frame,
            target,
            time_budget=int(budget),
            enforcement="hard",
            output_dir=Path(directory),
            **options,
        ).run()
    except errors.BlockedError as caught:
        seconds = time.monotonic() - started
        print(json.dumps({"seconds": seconds, "blocked": str(caught)}))
        return 0
    seconds = time.monotonic() - started
    print(json.dumps({"seconds": seconds, "model": run.selected_model}))
    return 0


def judge(found: dict[str, Any], limit: float) -> str:
    """The outcome of one real-table run.

    Args:
        found: What `finish` read from the child.
        limit: The run's allowance in seconds.

    Returns:
        `compliant` (completed within the allowance), `refused` (blocked
        before starting), `over` (completed past the allowance), `killed`
        (at the timeout) or `crashed` (any other end without a result).
    """
    if "blocked" in found:
        return "refused"
    if "model" in found:
        return "compliant" if found["seconds"] <= limit else "over"
    return "killed" if found["timed_out"] else "crashed"


def attempt(
    number: int, source: str, ident: int, directory: Path, end: float
) -> dict[str, Any]:
    """One real-table run, started only if its kill time fits the window.

    Args:
        number: The pass, from 0.
        source: `quality` or `refusal`.
        ident: The table's id.
        directory: Scratch output directory.
        end: Monotonic time by which every run must have ended.

    Returns:
        The pass, table, seed, size, budget, allowance, wall time, load
        average and outcome; `started` False when it did not fit.
    """
    rows, cells = size(source, ident)
    budget = allot(cells)
    limit = allowance(budget)
    seed = seeded(source, ident, number)
    record: dict[str, Any] = {
        "pass": number,
        "source": source,
        "id": ident,
        "seed": seed,
        "rows": rows,
        "cells": cells,
        "budget": budget,
        "allowance": limit,
        "started": time.monotonic() + TIMEOUT_FACTOR * limit <= end,
    }
    if not record["started"]:
        return record
    record["load"] = round(os.getloadavg()[0], 1)
    begun = time.monotonic()
    process = spawn(
        ["table", source, str(ident), str(directory), str(budget), str(seed)],
        quiet=True,
    )
    found = finish(process, TIMEOUT_FACTOR * limit)
    seconds = found.get("seconds")
    result = judge(found, limit)
    record.update(
        process_seconds=round(time.monotonic() - begun, 2),
        run_seconds=None if seconds is None else round(seconds, 2),
        over_seconds=(None if seconds is None else round(seconds - limit, 2)),
        outcome=result,
        completed="model" in found,
        refused=result == "refused",
        timed_out=found["timed_out"],
        error=found.get("error", found.get("blocked", "")),
        compliant=result == "compliant",
    )
    print(json.dumps(record), file=sys.stderr, flush=True)
    return record


def realistic(scratch: Path) -> dict[str, Any]:
    """Runs the real family in passes, `WORKERS` at a time, in the window.

    Args:
        scratch: Directory for the runs.

    Returns:
        The runs that started, the counts (refusals apart) and the runs of
        a started pass that did not fit.
    """
    order = [(n, *t) for n in range(PASSES) for t in tables()]
    for _, source, ident in order[: len(tables())]:
        size(source, ident)
    end = time.monotonic() + REAL_SECONDS
    with futures.ThreadPoolExecutor(WORKERS) as pool:
        records = list(
            pool.map(
                lambda item: attempt(
                    item[1][0],
                    item[1][1],
                    item[1][2],
                    scratch / f"real_{item[0]}",
                    end,
                ),
                enumerate(order),
            )
        )
    runs = [r for r in records if r["started"]]
    passes = sorted({r["pass"] for r in runs})
    decided = [r for r in runs if not r["refused"]]
    return {
        "passes": len(passes),
        "complete_passes": sum(
            sum(r["pass"] == n for r in runs) == len(tables()) for n in passes
        ),
        "trials": len(decided),
        "hits": sum(r["compliant"] for r in decided),
        "refused": len(runs) - len(decided),
        "skipped": [
            [r["pass"], r["source"], r["id"]]
            for r in records
            if not r["started"] and r["pass"] in passes
        ],
        "runs": runs,
    }


def realsettings() -> dict[str, Any]:
    """The fixed design of the real family.

    Returns:
        Every constant that changes what the real family measures.
    """
    return {
        "base_seconds": BASE_SECONDS,
        "base_cells": BASE_CELLS,
        "slack_seconds": limits.SLACK_SECONDS,
        "rule": RULE,
        "real_seconds": REAL_SECONDS,
        "workers": WORKERS,
        "passes": PASSES,
        "timeout_factor": TIMEOUT_FACTOR,
        "tables": [list(t) for t in tables()],
    }


def realcode() -> str:
    """Hash of the code that decides what the real family measures.

    Returns:
        The combined source hash.
    """
    return certificates.code_hash(
        allowance,
        allot,
        tables,
        seeded,
        load,
        size,
        table,
        judge,
        attempt,
        realistic,
        realsettings,
        finish,
        spawn,
        quality.load,
        refusal.table,
        refusal.fetch,
        scmle_runner.seed_for,
    )


def measure(
    practice: bool = False, stored: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Runs the real family and, when asked, the practice measurements.

    Args:
        practice: Whether to rerun the practice family and kill and resume
            under the current rule (about 70 minutes); otherwise they are
            reused from `stored` with the rule they were measured under.
        stored: The stored document, read from `PATH` when None.

    Returns:
        The JSON-ready document: both families, kill and resume, the
        practice runs' settings, code and rule, what was not measured, the
        code hashes and the duration.

    Raises:
        InputError: If the practice runs are to be reused but the stored
            ones were measured under the current rule with other code or
            settings.
    """
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="scmle-budget-") as name:
        scratch = Path(name)
        if practice:
            found = compliance(scratch)
            resumed = resumption(scratch)
            fixed, code, rule = settings(), current(), RULE
        else:
            old = read(stored)
            rule = old.get("practicerule", OLDRULE)
            if rule == RULE and (
                old.get("code") != current()
                or old.get("settings") != settings()
            ):
                raise errors.InputError(
                    "the stored practice runs are stale; pass --practice"
                )
            found, resumed = old["compliance"], old["resume"]
            fixed, code = old["settings"], old["code"]
        real = realistic(scratch)
    return {
        "version": 3,
        "claim": "C7",
        "variable": VARIABLE,
        "settings": fixed,
        "practicerule": rule,
        "compliance": found,
        "resume": resumed,
        "realsettings": realsettings(),
        "real": real,
        "not_measured": [
            "the memory budget",
            "budgets other than the stated ones",
            "real tables above 52,035 training rows",
            "kills outside the training stage",
            "journal and lease operations",
        ],
        "code": code,
        "realcode": realcode(),
        "seconds": round(time.monotonic() - started, 1),
    }


def read(document: dict[str, Any] | None) -> dict[str, Any]:
    """The given document, or the stored one.

    Args:
        document: A measurement document, or None to read `PATH`.

    Returns:
        The document.

    Raises:
        InputError: If `PATH` is missing when no document is given.
    """
    if document is not None:
        return document
    if not PATH.exists():
        raise errors.InputError(f"{PATH.name} is missing")
    loaded: dict[str, Any] = json.loads(PATH.read_text(encoding="utf-8"))
    return loaded


def problems(document: dict[str, Any] | None = None) -> list[str]:
    """Why the stored measurements are not for the current code.

    Practice runs filed under the old rule are not checked against the
    current code: they are kept as they were measured and never decide.

    Args:
        document: The measurements; read from `PATH` when None.

    Returns:
        A message for a missing file, other code or other settings; empty
        when current.
    """
    if document is None and not PATH.exists():
        return [f"{PATH.name} is missing"]
    document = read(document)
    found = []
    if document.get("practicerule", OLDRULE) == RULE:
        if document.get("code") != current():
            found.append("budget: measurements are for other code")
        if document.get("settings") != settings():
            found.append("budget: measurements are for other settings")
    if "real" not in document:
        found.append("budget: the real family is not measured")
        return found
    if document.get("realcode") != realcode():
        found.append("budget: real tables were measured with other code")
    if document.get("realsettings") != realsettings():
        found.append("budget: real tables were measured with other settings")
    return found


def counts(family: dict[str, Any]) -> tuple[int, int]:
    """Hits and trials of one family, checked.

    Args:
        family: The `compliance` or `real` part of a document.

    Returns:
        Runs within their allowance, and runs (refusals apart).

    Raises:
        InputError: If the counts are missing or inconsistent.
    """
    try:
        hits, trials = int(family["hits"]), int(family["trials"])
    except (KeyError, TypeError, ValueError) as caught:
        raise errors.InputError(
            f"malformed budget document: {caught}"
        ) from None
    if not 0 <= hits <= trials:
        raise errors.InputError("counts must lie between 0 and the trials")
    return hits, trials


def overruns(family: dict[str, Any]) -> str:
    """The runs of a family over their allowance, worst first.

    Args:
        family: The `real` part of a document.

    Returns:
        `id +seconds` for each over run, `id killed` for one killed at the
        timeout and `id crashed` for one that ended without a result,
        joined by commas, worst first; refused runs are not over. A run
        of a pass after the first is named `id/pass`. Empty when none.
    """
    over = [
        r
        for r in family.get("runs", [])
        if not r.get("compliant") and not r.get("refused")
    ]
    over.sort(key=lambda r: -(r.get("over_seconds") or math.inf))
    names = [
        f"{r['id']}/{r['pass']}" if r.get("pass") else f"{r['id']}"
        for r in over
    ]
    return ", ".join(
        f"{name} +{r['over_seconds']:g} s"
        if r.get("over_seconds") is not None
        else f"{name} killed"
        if r.get("timed_out")
        else f"{name} crashed"
        for name, r in zip(names, over, strict=True)
    )


def summary(document: dict[str, Any] | None = None) -> tuple[str, str]:
    """The scorecard's value and verdict for the measured part of C7.

    A kill whose resume differs from the uninterrupted run fails the claim
    whatever the compliance counts. Otherwise the real family alone is
    decided by the `budget_compliance_min` floor; refused runs are counted
    apart and fewer real runs than the floor's minimum is "not measured".
    The practice family is reported with its rule and never decides.

    Args:
        document: The measurements; read from `PATH` when None.

    Returns:
        The value text and "pass", "fail" or "not measured".

    Raises:
        InputError: If the document lacks the counts.
    """
    if document is None and problems():
        return ("; ".join(problems()), "not measured")
    document = read(document)
    hits, trials = counts(document.get("compliance", {}))
    real = document.get("real", {"hits": 0, "trials": 0})
    good, runs = counts(real)
    try:
        same = int(document["resume"]["identical"])
        cases = int(document["resume"]["trials"])
        budget = float(document["settings"]["budget_seconds"])
        refused = int(real.get("refused", 0))
    except (KeyError, TypeError, ValueError) as caught:
        raise errors.InputError(
            f"malformed budget document: {caught}"
        ) from None
    if not 0 <= same <= cases:
        raise errors.InputError("counts must lie between 0 and the trials")
    variables = acceptance.load()
    alpha = variables["alpha"].value
    rule = variables[VARIABLE].rule
    if not isinstance(rule, acceptance.Floor):
        raise errors.InputError(f"{VARIABLE} must be a floor")
    least = rule.minimum(alpha)
    over = overruns(real)
    old = document.get("practicerule", OLDRULE) != RULE
    value = (
        f"real tables: {good} of {runs} runs within budget + "
        f"{limits.SLACK_SECONDS:g} s"
        + (f" (over: {over})" if over else "")
        + f", {refused} refused before starting"
        + f"; practice project ({'old rule, ' if old else ''}not deciding): "
        f"{hits} of {trials} within {budget:g} s + "
        + (
            f"{float(document['settings'].get('grace_seconds', 0)):g} s grace"
            if old
            else f"{limits.SLACK_SECONDS:g} s"
        )
        + f"; kill and resume identical in {same} of {cases}"
    )
    if cases == 0 or same < cases:
        return (value, "fail")
    if runs < least:
        return (
            (
                f"{value}; {least - runs} more real runs needed "
                f"(minimum {least} at alpha {alpha:g})"
            ),
            "not measured",
        )
    return (value, "pass" if rule.passes(good, runs, alpha) else "fail")


def main(arguments: list[str] | None = None) -> int:
    """Writes (`--write`) or checks (`--check`) the measurements.

    Args:
        arguments: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        0 for `--write`; for `--check`, 0 when the measurements are for the
        current code and 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--child", nargs="+", help=argparse.SUPPRESS)
    parser.add_argument(
        "--practice",
        action="store_true",
        help="with --write, also rerun the practice family (70 minutes)",
    )
    options = parser.parse_args(arguments)
    if options.child and options.child[0] == "table":
        return table(options.child[1:])
    if options.child:
        return child(options.child)
    if options.write:
        document = measure(practice=options.practice)
        PATH.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        value, verdict = summary(document)
        print(json.dumps({"value": value, "verdict": verdict}))
        return 0
    issues = problems()
    print(json.dumps({"ok": not issues, "problems": issues}, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
