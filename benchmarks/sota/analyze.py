"""Turns result records into the numbers the protocol's claim is judged on.

Follows PROTOCOL.md section 5: per-task normalized scores over the compared
systems, the median over tasks, a bootstrap interval over tasks for the gap
between the best reference system and SCMLE, and the two-part claim test.
A failed or timed-out run takes the worst value on its task (it is counted,
never dropped).

Usage:
    python -m benchmarks.sota.analyze RESULTS_DIR [--output summary.json]
"""

import argparse
import collections
import hashlib
import itertools
import json
import logging
import pathlib
import statistics
import sys
from typing import Any

import numpy as np
from scipy import stats

from benchmarks.sota import records

LOGGER = logging.getLogger(__name__)

SCMLE = "scmle"
LOWER_IS_BETTER = {"log_loss", "rmse"}
BOOTSTRAP_SEED = "scmle-sota-bootstrap-v1"
RESAMPLES = 10000
MEDIAN_GAP = 0.02
UPPER_GAP = 0.05
ALPHA = 0.05


def load(directory: pathlib.Path) -> list[records.RunRecord]:
    """Reads every results file in a directory.

    Args:
        directory: Directory holding `*.jsonl` results files.

    Returns:
        The records of all files, in file-name order.
    """
    found: list[records.RunRecord] = []
    for path in sorted(directory.glob("*.jsonl")):
        found.extend(records.read(path))
    return found


def oriented(record: records.RunRecord) -> float | None:
    """The score, signed so that higher is always better.

    Args:
        record: The run record.

    Returns:
        The score, negated for lower-is-better metrics; None when the run
        did not finish `ok` or has no score.
    """
    if record.status != "ok" or record.score is None:
        return None
    sign = -1.0 if record.metric in LOWER_IS_BETTER else 1.0
    return sign * record.score


def task_scores(
    found: list[records.RunRecord], budget: int
) -> dict[int, dict[str, float | None]]:
    """Per task and system, the mean oriented score over the splits run.

    Args:
        found: All run records.
        budget: Only records with this budget, in seconds, are used.

    Returns:
        Mapping of task id to system name to the mean oriented score of its
        `ok` runs; None when the system has no `ok` run on that task.
    """
    cells: dict[tuple[int, str], list[float | None]] = collections.defaultdict(
        list
    )
    for record in found:
        if record.budget_s == budget:
            cells[(record.task_id, record.system)].append(oriented(record))
    table: dict[int, dict[str, float | None]] = collections.defaultdict(dict)
    for (task, system), values in cells.items():
        good = [v for v in values if v is not None]
        table[task][system] = statistics.fmean(good) if good else None
    return table


def normalize(
    table: dict[int, dict[str, float | None]], systems: list[str]
) -> dict[int, dict[str, float]]:
    """Min-max normalization per task; a failure scores the worst (0).

    Tasks with no scored system, or on which some compared system has no
    record at all, are left out.

    Args:
        table: Output of `task_scores`.
        systems: The systems to compare.

    Returns:
        Mapping of task id to system name to a score in [0, 1]; on a task
        where all scored systems tie they score 1.0, and a failed system 0.0.
    """
    normalized: dict[int, dict[str, float]] = {}
    for task, row in table.items():
        values = [row[s] for s in systems if row.get(s) is not None]
        if not values or any(s not in row for s in systems):
            continue  # a system with no record at all for this task
        worst, best = min(values), max(values)  # type: ignore[type-var]
        normalized[task] = {
            system: (
                0.0
                if row[system] is None
                else 1.0
                if best == worst
                else (row[system] - worst) / (best - worst)  # type: ignore[operator]
            )
            for system in systems
        }
    return normalized


def bootstrap_gap(
    reference: list[float], ours: list[float]
) -> tuple[float, float, float]:
    """Median gap (reference - ours) and its 95% bootstrap interval.

    The tasks are resampled with replacement 10000 times from a fixed seed
    derived from `BOOTSTRAP_SEED`, so the interval is reproducible.

    Args:
        reference: Normalized scores of the reference system, one per task.
        ours: Normalized scores of SCMLE on the same tasks, in the same order.

    Returns:
        The observed gap between the medians, and the 2.5th and 97.5th
        percentiles of the bootstrap gap.
    """
    seed = int(hashlib.sha256(BOOTSTRAP_SEED.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    a, b = np.asarray(reference), np.asarray(ours)
    picks = rng.integers(0, len(a), size=(RESAMPLES, len(a)))
    gaps = np.median(a[picks], axis=1) - np.median(b[picks], axis=1)
    low, high = np.percentile(gaps, [2.5, 97.5])
    return float(np.median(a) - np.median(b)), float(low), float(high)


def holm(pvalues: dict[tuple[str, str], float]) -> dict[tuple[str, str], bool]:
    """Holm step-down: which pairwise differences are significant.

    Args:
        pvalues: Raw p-value of each system pair.

    Returns:
        Mapping of each pair to True when its null hypothesis is rejected at
        level `ALPHA` after the Holm correction.
    """
    order = sorted(pvalues.items(), key=lambda item: item[1])
    verdict: dict[tuple[str, str], bool] = {}
    rejecting = True
    for rank, (pair, value) in enumerate(order):
        rejecting = rejecting and value <= ALPHA / (len(order) - rank)
        verdict[pair] = rejecting
    return verdict


def comparison_tests(
    normalized: dict[int, dict[str, float]], systems: list[str]
) -> dict[str, Any]:
    """Friedman across all systems, Wilcoxon with Holm for each pair.

    Args:
        normalized: Output of `normalize`.
        systems: The compared systems.

    Returns:
        A dict with `friedman` (statistic and p-value; present only with at
        least three systems and three tasks) and `pairs`, a list with the
        Wilcoxon p-value and the Holm verdict of each pair. A pair with
        identical scores on every task gets p = 1.
    """
    columns = {
        s: [normalized[t][s] for t in sorted(normalized)] for s in systems
    }
    out: dict[str, Any] = {}
    if len(systems) >= 3 and len(normalized) >= 3:
        statistic, p = stats.friedmanchisquare(*columns.values())
        out["friedman"] = {"statistic": float(statistic), "p": float(p)}
    pvalues: dict[tuple[str, str], float] = {}
    for first, second in itertools.combinations(systems, 2):
        differences = np.asarray(columns[first]) - np.asarray(columns[second])
        if np.allclose(differences, 0):
            pvalues[(first, second)] = 1.0
        else:
            pvalues[(first, second)] = float(
                stats.wilcoxon(columns[first], columns[second]).pvalue
            )
    significant = holm(pvalues)
    out["pairs"] = [
        {
            "a": a,
            "b": b,
            "p": pvalues[(a, b)],
            "significant": significant[(a, b)],
        }
        for (a, b) in pvalues
    ]
    return out


def claim(median_gap: float, upper: float) -> bool:
    """The protocol's two-part claim test (section 1).

    Args:
        median_gap: Observed median gap between the best reference and SCMLE.
        upper: Upper end of the gap's bootstrap interval.

    Returns:
        True when the median gap is at most `MEDIAN_GAP` and the upper bound
        at most `UPPER_GAP`.
    """
    return median_gap <= MEDIAN_GAP and upper <= UPPER_GAP


def summarize(found: list[records.RunRecord], budget: int) -> dict[str, Any]:
    """Everything the protocol reports for one budget.

    Args:
        found: All run records.
        budget: The budget in seconds to summarize.

    Returns:
        The systems, task count and failure count per system. When there are
        normalized tasks and SCMLE is among the systems it also holds the
        median normalized score per system and the comparison tests; with at
        least one reference system, the best reference, the gap and its
        interval, whether the claim is met, and SCMLE's wins, losses and ties
        against the best reference.
    """
    systems = sorted({r.system for r in found if r.budget_s == budget})
    table = task_scores(found, budget)
    normalized = normalize(table, systems)
    summary: dict[str, Any] = {
        "budget_s": budget,
        "systems": systems,
        "tasks": len(normalized),
        "failures": {
            s: sum(
                1
                for r in found
                if r.system == s and r.budget_s == budget and r.status != "ok"
            )
            for s in systems
        },
    }
    if not normalized or SCMLE not in systems:
        return summary
    medians = {
        s: statistics.median(normalized[t][s] for t in normalized)
        for s in systems
    }
    summary["median_normalized"] = medians
    references = [s for s in systems if s != SCMLE]
    if references:
        best = max(references, key=lambda s: medians[s])
        tasks = sorted(normalized)
        gap, low, high = bootstrap_gap(
            [normalized[t][best] for t in tasks],
            [normalized[t][SCMLE] for t in tasks],
        )
        summary["best_reference"] = best
        summary["gap"] = {"median": gap, "low": low, "high": high}
        summary["claim_met"] = claim(gap, high)
        summary["wins_losses_ties"] = {
            "wins": sum(
                normalized[t][SCMLE] > normalized[t][best] for t in tasks
            ),
            "losses": sum(
                normalized[t][SCMLE] < normalized[t][best] for t in tasks
            ),
            "ties": sum(
                normalized[t][SCMLE] == normalized[t][best] for t in tasks
            ),
        }
    summary["tests"] = comparison_tests(normalized, systems)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Prints and optionally saves the summary for every budget present.

    Args:
        argv: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        Process exit status, always 0.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    found = load(args.directory)
    budgets = sorted({r.budget_s for r in found})
    result = {"summaries": [summarize(found, b) for b in budgets]}
    text = json.dumps(result, indent=2)
    LOGGER.info("%s", text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
