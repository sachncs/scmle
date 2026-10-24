"""Runs SCMLE on TabArena tasks, one record per task split and budget.

The task's own OpenML train / test indices are used as stored; SCMLE sees the
training rows only, and the test rows are scored here with the task's metric
(binary `roc_auc`, multiclass `log_loss`, regression `rmse`, TabArena's
defaults). Only development tasks run unless `--open-holdout` is given.

Usage:
    python -m benchmarks.sota.scmle_runner --tasks 363621 363685 \
        --budgets 60 --output /tmp/rel/scmle.jsonl --runs-dir /tmp/rel/runs
"""

import argparse
import hashlib
import json
import logging
import pathlib
import platform
import sys
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

from benchmarks.sota import records
from benchmarks.sota import split
import scmle

LOGGER = logging.getLogger(__name__)

SYSTEM = "scmle"
CPUS = 12  # Every system gets the same cores (PROTOCOL.md, amendment 1).

OVERRUN = 1.10
METRICS = {
    "binary": "roc_auc",
    "multiclass": "log_loss",
    "regression": "rmse",
}


def load_split() -> dict[str, Any]:
    """Reads the committed development / hold-out assignment."""
    return json.loads(split.OUTPUT.read_text(encoding="utf-8"))


def check_task_ids(task_ids: list[int], open_holdout: bool = False) -> None:
    """Refuses hold-out tasks unless explicitly opened.

    Args:
        task_ids: OpenML task ids requested.
        open_holdout: True only for the one final evaluation.

    Raises:
        PermissionError: If a hold-out task is requested while sealed.
        ValueError: If a task is in neither set.
    """
    assignment = load_split()
    sealed = set() if open_holdout else set(assignment["holdout"])
    known = set(assignment["development"]) | set(assignment["holdout"])
    unknown = sorted(set(task_ids) - known)
    if unknown:
        raise ValueError(f"tasks not in the protocol: {unknown}")
    blocked = sorted(set(task_ids) & sealed)
    if blocked:
        raise PermissionError(
            f"hold-out tasks are sealed: {blocked}; "
            "pass --open-holdout only for the final evaluation"
        )


def problem_type_of(task_id: int) -> str:
    """The task's problem type, from the pinned task table."""
    for row in split.load_tasks():
        if int(row["task_id"]) == task_id:
            return row["problem_type"]
    raise ValueError(f"unknown task {task_id}")


def seed_for(task_id: int, fold: int, repeat: int) -> int:
    """A deterministic 31-bit SCMLE seed for one task split."""
    key = f"scmle-sota:{task_id}:{fold}:{repeat}".encode()
    return int(hashlib.sha256(key).hexdigest(), 16) % 2**31


def load_task(
    task_id: int, fold: int, repeat: int
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Loads an OpenML task and its stored train / test rows.

    Args:
        task_id: OpenML task id.
        fold: Fold index.
        repeat: Repeat index.

    Returns:
        The train frame, the test frame (both with the target column), the
        target name and the problem type (binary, multiclass, regression).
    """
    import openml  # Optional `sota` extra.

    task: Any = openml.tasks.get_task(task_id, download_splits=True)
    target_name: str = task.target_name
    features, target, _, _ = task.get_dataset().get_data(
        target=target_name, dataset_format="dataframe"
    )
    frame = features.assign(**{target_name: target})
    train_rows, test_rows = task.get_train_test_split_indices(
        repeat=repeat, fold=fold, sample=0
    )
    return (
        frame.iloc[train_rows].reset_index(drop=True),
        frame.iloc[test_rows].reset_index(drop=True),
        target_name,
        problem_type_of(task_id),
    )


def score(result: Any, test: pd.DataFrame, target: str, kind: str) -> float:
    """Scores the fitted project on the held-out test rows."""
    truth = test[target]
    rows = test.drop(columns=[target])
    if kind == "regression":
        predicted = np.asarray(result.predict(rows), dtype=float)
        return float(
            sk_metrics.root_mean_squared_error(truth.astype(float), predicted)
        )
    classes = list(result.bundle.schema.classes)
    proba = np.asarray(result.predict_proba(rows), dtype=float)
    if kind == "binary":
        return float(sk_metrics.roc_auc_score(truth == classes[1], proba[:, 1]))
    return float(sk_metrics.log_loss(truth, proba, labels=classes))


def run_frames(
    task_id: int,
    fold: int,
    repeat: int,
    frames: tuple[pd.DataFrame, pd.DataFrame, str, str],
    budget_s: int,
    runs_dir: pathlib.Path,
    workers: int = 1,
) -> records.RunRecord:
    """Runs SCMLE on given train / test frames; never raises.

    The clock covers everything SCMLE does: profiling, search, ensembling,
    packaging and predicting the test rows. Loading the data is excluded.

    Args:
        task_id: OpenML task id, for the record and the seed.
        fold: Fold index.
        repeat: Repeat index.
        frames: Train frame, test frame, target name, problem type.
        budget_s: Wall-clock budget in seconds.
        runs_dir: Parent directory for SCMLE run directories.
        workers: Concurrent trial processes.

    Returns:
        The record; a crash is a `failed` record carrying the error text.
    """
    train, test, target, kind = frames
    record = records.RunRecord(
        system=SYSTEM,
        budget_s=budget_s,
        task_id=task_id,
        fold=fold,
        repeat=repeat,
        metric=METRICS[kind],
        score=None,
        seconds=0.0,
        status="failed",
        error="did not finish",
        versions={
            "scmle": scmle.__version__,
            "python": platform.python_version(),
            "cpus": str(workers),
        },
    )
    started = time.monotonic()
    try:
        project = scmle.MLProject(
            train,
            target,
            task="regression" if kind == "regression" else "classification",
            metric=record.metric,
            time_budget=budget_s,
            workers=workers,
            seed=seed_for(task_id, fold, repeat),
            output_dir=runs_dir / f"t{task_id}_f{fold}_r{repeat}_b{budget_s}",
        )
        result = project.run()
        record.score = score(result, test, target, kind)
        record.status, record.error = "ok", ""
    except Exception:  # Every task yields a record, even a crash.
        record.status = "failed"
        record.error = traceback.format_exc(limit=4)
    record.seconds = time.monotonic() - started
    if record.status == "ok" and record.seconds > OVERRUN * budget_s:
        record.error = (
            f"over budget: {record.seconds:.1f} s used of {budget_s} s"
        )
    return record


def main(argv: list[str] | None = None) -> int:
    """Runs the requested tasks and appends records to a JSON-lines file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, nargs="+", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[60])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--workers", type=int, default=CPUS)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--runs-dir", type=pathlib.Path, required=True)
    parser.add_argument("--open-holdout", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    check_task_ids(args.tasks, args.open_holdout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = records.completed(args.output)
    failures = 0
    for task_id in args.tasks:
        try:
            frames = load_task(task_id, args.fold, args.repeat)
        except Exception:  # A task that cannot load is still recorded.
            frames = None
            error = traceback.format_exc(limit=4)
        for budget in args.budgets:
            if (SYSTEM, budget, task_id, args.fold, args.repeat) in done:
                LOGGER.info("task %d budget %d: already done", task_id, budget)
                continue
            if frames is None:
                record = records.RunRecord(
                    system=SYSTEM,
                    budget_s=budget,
                    task_id=task_id,
                    fold=args.fold,
                    repeat=args.repeat,
                    metric=METRICS[problem_type_of(task_id)],
                    score=None,
                    seconds=0.0,
                    status="failed",
                    error=error,
                )
            else:
                record = run_frames(
                    task_id,
                    args.fold,
                    args.repeat,
                    frames,
                    budget,
                    args.runs_dir,
                    args.workers,
                )
            records.append(args.output, record)
            failures += record.status != "ok"
            LOGGER.info(
                "task %d budget %d: %s score=%s %.1fs %s",
                task_id,
                budget,
                record.status,
                record.score,
                record.seconds,
                record.error.splitlines()[-1] if record.error else "",
            )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
