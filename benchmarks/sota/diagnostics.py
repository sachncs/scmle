"""Facts about one SCMLE run, read from the files it wrote.

The facts answer where the budget went: how long the run took against its
budget, how many trials ended in each status, which stages the budget cut
short, and which model families and strategies actually ran.

Usage:
    python -m benchmarks.sota.diagnostics RUN_DIR [RUN_DIR ...]
"""

import argparse
import collections
import datetime
import json
import pathlib
import sys
from typing import Any

import pandas

STATUSES = ("ok", "failed", "pruned", "skipped")


def read_events(run_dir: pathlib.Path) -> list[dict[str, Any]]:
    """Reads the run's event log.

    Args:
        run_dir: A run directory holding `events.jsonl`.

    Returns:
        The events in file order, one dict per non-blank line.
    """
    path = run_dir / "events.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def seconds_between(first: str, last: str) -> float:
    """Seconds between two ISO timestamps.

    Args:
        first: The earlier ISO 8601 timestamp.
        last: The later ISO 8601 timestamp.

    Returns:
        The elapsed time in seconds (negative if `last` is earlier).
    """
    start = datetime.datetime.fromisoformat(first)
    return (datetime.datetime.fromisoformat(last) - start).total_seconds()


def summarize_run(run_dir: pathlib.Path) -> dict[str, Any]:
    """Summarizes one run directory.

    Args:
        run_dir: A `runs/run_*` directory written by SCMLE.

    Returns:
        A dict with:
            budget_s: Time budget from the run's config.
            seconds_events: First to last event, a lower bound on the run
                (it excludes data loading before the run directory exists).
            fraction_used: `seconds_events / budget_s`.
            trials: Count of trials per status (ok, failed, pruned, skipped).
            fit_seconds: Summed trial fit time, all trials.
            skipped_stages: Events ending in `.skipped`, each with its
                reason.
            models: Number of ok trials per model name.
            strategies: Number of ok trials per data strategy.
            failures: Error text of failed trials, by trial id.
            selected: Name of the selected trial's model, or "".
    """
    run_dir = pathlib.Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    budget = float(config["resources"]["time_budget_seconds"])
    events = read_events(run_dir)
    elapsed = seconds_between(events[0]["timestamp"], events[-1]["timestamp"])
    board = pandas.read_parquet(run_dir / "leaderboard.parquet")
    counts = collections.Counter(board["status"])
    finished = board[board["status"] == "ok"]
    selected = board[board["selected"]]
    return {
        "budget_s": budget,
        "seconds_events": round(elapsed, 2),
        "fraction_used": round(elapsed / budget, 3),
        "trials": {status: counts.get(status, 0) for status in STATUSES},
        "fit_seconds": round(float(board["fit_seconds"].sum()), 2),
        "skipped_stages": [
            {"event": e["kind"], "reason": e["data"].get("reason", "")}
            for e in events
            if e["kind"].endswith(".skipped")
        ],
        "models": dict(collections.Counter(finished["model_name"])),
        "strategies": dict(collections.Counter(finished["strategy"])),
        "failures": {
            row.trial_id: str(row.error)
            for row in board[board["status"] == "failed"].itertuples()
        },
        "selected": "" if selected.empty else selected.iloc[0]["model_name"],
    }


def main(argv: list[str] | None = None) -> int:
    """Prints the summary of each run directory as JSON.

    Args:
        argv: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        Process exit status, always 0.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=pathlib.Path)
    args = parser.parse_args(argv)
    for run_dir in args.run_dirs:
        summary = {"run": str(run_dir), **summarize_run(run_dir)}
        sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
