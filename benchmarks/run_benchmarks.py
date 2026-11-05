"""Runs scmle end to end on open datasets and reports the results.

Usage:
    python -m benchmarks.run_benchmarks [--offline] [--only NAME ...]
        [--budget SECONDS] [--output benchmarks/results.json]
"""

import argparse
import dataclasses
import json
import logging
import pathlib
import platform
import sys
import tempfile
import time
import traceback
from typing import Any

from benchmarks import datasets
from benchmarks import intervals
from benchmarks import reference
import scmle
from scmle.core import limits

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
        peak_memory_gb: Observed peak resident memory.
        machine: Hardware and operating-system identity.
        software: Interpreter identity used for the benchmark.
        completed_seeds: Number of repetitions that completed.
        failure_rate: Fraction of requested repetitions that failed.
        findings: Quality and leakage findings raised by profiling.
        error: Error text when failed.
        seeds: Repetitions with different seeds (split and models).
        test_scores: Locked-test score of each repetition.
        reference_scores: Reference score of each repetition.
        advantage: Mean, 95% interval and verdict of how much better the
            selected model is than the reference (positive = better).
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
    peak_memory_gb: float = 0.0
    machine: str = ""
    software: str = ""
    completed_seeds: int = 0
    failure_rate: float = 1.0
    findings: list[str] = dataclasses.field(default_factory=list)
    error: str = ""
    seeds: int = 1
    test_scores: list[float] = dataclasses.field(default_factory=list)
    reference_scores: list[float] = dataclasses.field(default_factory=list)
    advantage: dict[str, Any] = dataclasses.field(default_factory=dict)


def run_one(
    dataset: datasets.BenchmarkDataset,
    budget: int,
    workers: int,
    root: pathlib.Path,
    seeds: int = 1,
) -> BenchmarkRecord:
    """Runs the full workflow on one dataset, once per seed.

    Args:
        dataset: Benchmark dataset.
        budget: Time budget in seconds, per seed.
        workers: Concurrent trial processes.
        root: Directory for run artifacts.
        seeds: Repetitions; each seed changes the split and the models.

    Returns:
        The record, averaged over seeds; failures are captured, not raised.
    """
    if seeds < 1:
        raise ValueError("seeds must be at least 1")
    record = BenchmarkRecord(dataset.name, seeds=seeds)
    started = time.monotonic()
    completed_seeds = 0
    try:
        frame = dataset.load()
        cv_scores: list[float] = []
        for seed in range(seeds):
            project = scmle.MLProject(
                frame,
                dataset.target,
                task=dataset.task,
                metric=dataset.metric,
                time_budget=budget,
                workers=workers,
                time_column=dataset.time_column,
                # The benchmark measures model quality on public tables whose
                # strong legitimate predictors the leakage screen flags.
                leakage_policy="warn",
                time_as_feature=dataset.time_as_feature,
                seed=seed,
                output_dir=root / f"{dataset.name}-{seed}",
            )
            result = project.run()
            evaluation = result.evaluation
            if seed == 0:
                record.rows = len(frame)
                record.task = evaluation["task"]
                record.metric = evaluation["metric"]
                record.selected = (
                    f"{result.selected_model}/{evaluation['selected_strategy']}"
                )
                record.baseline_cv = evaluation["baseline_cv_score"]
                record.findings = list(
                    result.profile.quality_findings
                    + result.profile.risk_findings
                )
            record.trials += len(result.trials)
            cv_scores.append(evaluation["selected_cv_score"])
            record.test_scores.append(evaluation["test_metrics"][record.metric])
            record.reference_scores.append(
                reference.reference_score(
                    frame,
                    dataset.target,
                    result.run_dir,
                    record.task,
                    record.metric,
                    record.task != "regression",
                )
            )
            completed_seeds += 1
        record.cv_score = sum(cv_scores) / seeds
        record.test_score = sum(record.test_scores) / seeds
        record.reference_test = sum(record.reference_scores) / seeds
        gains = [
            intervals.advantage(record.metric, score, base_score)
            for score, base_score in zip(
                record.test_scores, record.reference_scores, strict=True
            )
        ]
        mean, low, high = intervals.mean_interval(gains)
        record.advantage = {
            "mean": mean,
            "low": low,
            "high": high,
            "verdict": intervals.verdict(low, high, seeds),
        }
    except Exception:  # Report every dataset, even when one fails.
        record.status = "error"
        record.error = traceback.format_exc(limit=4)
    finally:
        record.completed_seeds = completed_seeds
        record.failure_rate = (seeds - completed_seeds) / seeds
        record.peak_memory_gb = limits.peak_memory_gb()
        record.machine = platform.platform()
        record.software = (
            f"{platform.python_implementation()} {platform.python_version()}"
        )
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
    "advantage (95% CI)",
    "verdict",
    "trials",
    "peak memory GB",
    "failure rate",
    "seconds",
)


def format_number(value: float | None) -> str:
    """Formats a score, leaving missing values blank."""
    return "" if value is None else f"{value:.4f}"


def describe_advantage(record: BenchmarkRecord) -> str:
    """The advantage over the reference with its interval, if there is one."""
    found = record.advantage
    if record.seeds < 2:
        return f"{found['mean']:+.4f}"
    return f"{found['mean']:+.4f} [{found['low']:+.4f}, {found['high']:+.4f}]"


def to_markdown(records: list[BenchmarkRecord]) -> str:
    """Renders records as a Markdown table."""
    lines = [
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "|".join(" --- " for column in COLUMNS) + "|",
    ]
    for record in records:
        if record.status != "ok":
            cells = [record.name, "", "ERROR"] + [""] * 11
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
                describe_advantage(record),
                record.advantage["verdict"],
                str(record.trials),
                f"{record.peak_memory_gb:.4f}",
                f"{record.failure_rate:.3f}",
                f"{record.seconds:.0f}",
            ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Runs the selected benchmarks and writes the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="skip downloads")
    parser.add_argument(
        "--suite",
        choices=sorted(datasets.SUITES),
        default="core",
        help="which datasets: the core nine, the extra OpenML tables, the "
        "Hugging Face tables, or everything",
    )
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="repeat each dataset with this many seeds and report intervals",
    )
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
        for d in datasets.SUITES[args.suite]
        if (not args.offline or not d.online)
        and (not args.only or d.name in args.only)
    ]
    records: list[BenchmarkRecord] = []
    with tempfile.TemporaryDirectory() as scratch:
        for dataset in chosen:
            LOGGER.info("running %s ...", dataset.name)
            records.append(
                run_one(
                    dataset,
                    args.budget,
                    args.workers,
                    pathlib.Path(scratch),
                    args.seeds,
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
