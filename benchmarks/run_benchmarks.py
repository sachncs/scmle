"""Runs magicml end to end on open datasets and reports the results.

Usage:
    python -m benchmarks.run_benchmarks [--offline] [--only NAME ...]
        [--budget SECONDS] [--output benchmarks/results.json]
"""

import argparse
import dataclasses
import json
import logging
import pathlib
import sys
import tempfile
import time
import traceback
from typing import Any

from benchmarks import datasets
from benchmarks import reference
import mml

LOGGER = logging.getLogger(__name__)

HIGHER_IS_BETTER = {"roc_auc", "pr_auc", "r2", "accuracy"}


@dataclasses.dataclass
class BenchmarkRecord:
    """Outcome of one dataset.

    Attributes:
        name: Dataset name.
        status: `ok` or `error`.
        rows: Row count.
        task: Resolved task.
        metric: Primary metric.
        selected: Selected model and strategy.
        cv_score: Cross-validation score of the selected candidate.
        baseline_cv: Dummy baseline cross-validation score.
        test_score: Final locked-test score of the selected candidate.
        reference_test: Plain HistGradientBoosting score on the same split.
        trials: Number of trials run.
        seconds: Wall-clock time.
        findings: Quality and leakage findings raised by profiling.
        error: Error text when failed.
    """

    name: str
    status: str = "ok"
    rows: int = 0
    task: str = ""
    metric: str = ""
    selected: str = ""
    cv_score: float = float("nan")
    baseline_cv: float | None = None
    test_score: float = float("nan")
    reference_test: float = float("nan")
    trials: int = 0
    seconds: float = 0.0
    findings: list[str] = dataclasses.field(default_factory=list)
    error: str = ""


def run_one(
    dataset: datasets.BenchmarkDataset,
    budget: int,
    workers: int,
    root: pathlib.Path,
) -> BenchmarkRecord:
    """Runs the full workflow on one dataset.

    Args:
        dataset: Benchmark dataset.
        budget: Time budget in seconds.
        workers: Concurrent trial processes.
        root: Directory for run artifacts.

    Returns:
        The record; failures are captured, not raised.
    """
    record = BenchmarkRecord(dataset.name)
    started = time.monotonic()
    try:
        frame = dataset.load()
        project = mml.MLProject(
            frame,
            dataset.target,
            task=dataset.task,
            metric=dataset.metric,
            time_budget=budget,
            workers=workers,
            output_dir=root / dataset.name,
        )
        result = project.run()
        evaluation = result.evaluation
        record.rows = len(frame)
        record.task = evaluation["task"]
        record.metric = evaluation["metric"]
        record.selected = (
            f"{result.selected_model}/{evaluation['selected_strategy']}"
        )
        record.cv_score = evaluation["selected_cv_score"]
        record.baseline_cv = evaluation["baseline_cv_score"]
        record.test_score = evaluation["test_metrics"][record.metric]
        record.trials = len(result.trials)
        record.findings = list(
            result.profile.quality_findings + result.profile.risk_findings
        )
        record.reference_test = reference.reference_score(
            frame,
            dataset.target,
            result.run_dir,
            record.task,
            record.metric,
            record.task != "regression",
        )
    except Exception:  # Report every dataset, even when one fails.
        record.status = "error"
        record.error = traceback.format_exc(limit=4)
    record.seconds = time.monotonic() - started
    return record


COLUMNS = (
    "dataset",
    "rows",
    "task",
    "metric",
    "selected",
    "baseline cv",
    "cv",
    "test",
    "reference test",
    "trials",
    "seconds",
)


def format_number(value: float | None) -> str:
    """Formats a score, leaving missing values blank."""
    return "" if value is None else f"{value:.4f}"


def to_markdown(records: list[BenchmarkRecord]) -> str:
    """Renders records as a Markdown table."""
    lines = [
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "|".join(" --- " for column in COLUMNS) + "|",
    ]
    for record in records:
        if record.status != "ok":
            cells = [record.name, "", "ERROR"] + [""] * 7
            cells.append(f"{record.seconds:.0f}")
        else:
            cells = [
                record.name,
                str(record.rows),
                record.task,
                record.metric,
                record.selected,
                format_number(record.baseline_cv),
                format_number(record.cv_score),
                format_number(record.test_score),
                format_number(record.reference_test),
                str(record.trials),
                f"{record.seconds:.0f}",
            ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Runs the selected benchmarks and writes the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="skip downloads")
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", default="benchmarks/results.json")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING, format="%(message)s", stream=sys.stdout
    )
    LOGGER.setLevel(logging.INFO)

    chosen = [
        d
        for d in datasets.ALL
        if (not args.offline or not d.online)
        and (not args.only or d.name in args.only)
    ]
    records: list[BenchmarkRecord] = []
    with tempfile.TemporaryDirectory() as scratch:
        for dataset in chosen:
            LOGGER.info("running %s ...", dataset.name)
            records.append(
                run_one(
                    dataset, args.budget, args.workers, pathlib.Path(scratch)
                )
            )
    LOGGER.info("%s", to_markdown(records))
    payload: dict[str, Any] = {
        "results": [dataclasses.asdict(r) for r in records]
    }
    pathlib.Path(args.output).write_text(json.dumps(payload, indent=2))
    for record in records:
        if record.status != "ok":
            LOGGER.error("%s failed:\n%s", record.name, record.error)
    return 0 if all(r.status == "ok" for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
