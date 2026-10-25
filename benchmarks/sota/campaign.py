"""Runs a whole comparison campaign, one task at a time, and can be restarted.

For each task it runs every system in turn, so slow drift of the machine
(thermal state, background load) affects all systems alike. Each system writes
its own JSON-lines file; the runners skip runs that already finished, so
relaunching after a crash or a reboot continues where it stopped. A lock file
stops two campaigns from sharing the machine, which would corrupt timings.

Usage:
    python -m benchmarks.sota.campaign --phase development \
        --baseline-python /tmp/rel/ta/bin/python --out-dir results/dev
"""

import argparse
import json
import logging
import os
import pathlib
import subprocess  # nosec B404
import sys

from benchmarks.sota import split

LOGGER = logging.getLogger(__name__)

SCMLE = "scmle"
DEFAULT_SYSTEMS = (SCMLE, "autogluon-best", "flaml", "lightgbm")
DEFAULT_BUDGETS = (60, 600)


def phase_tasks(phase: str, open_holdout: bool) -> list[int]:
    """Task ids of a phase; the hold-out needs an explicit opt-in."""
    if phase == "holdout" and not open_holdout:
        raise PermissionError(
            "the hold-out is sealed until the protocol's final evaluation; "
            "pass --open-holdout only then"
        )
    document = json.loads(split.OUTPUT.read_text(encoding="utf-8"))
    return list(document[phase])


def command(
    system: str,
    task: int,
    budgets: tuple[int, ...],
    out_dir: pathlib.Path,
    baseline_python: str,
    open_holdout: bool,
) -> list[str]:
    """The runner invocation for one system on one task."""
    output = str(out_dir / f"{system}.jsonl")
    budget_args = [str(budget) for budget in budgets]
    if system == SCMLE:
        run = [
            sys.executable,
            "-m",
            "benchmarks.sota.scmle_runner",
            "--tasks",
            str(task),
            "--budgets",
            *budget_args,
            "--output",
            output,
            "--runs-dir",
            str(out_dir / "scmle_runs"),
        ]
    else:
        run = [
            sys.executable,
            "-m",
            "benchmarks.sota.baselines",
            "run",
            "--python",
            baseline_python,
            "--tasks",
            str(task),
            "--systems",
            system,
            "--budgets",
            *budget_args,
            "--lite",
            "--output",
            output,
        ]
    if open_holdout:
        run.append("--open-holdout")
    return run


def acquire(lock: pathlib.Path) -> None:
    """Takes the campaign lock, or fails if a live campaign holds it."""
    if lock.exists():
        text = lock.read_text(encoding="utf-8").strip()
        holder = int(text) if text.isdigit() else 0
        if holder and process_alive(holder):
            raise RuntimeError(f"campaign already running (pid {holder})")
        LOGGER.info("removing a stale lock of process %d", holder)
    lock.write_text(str(os.getpid()), encoding="utf-8")


def process_alive(pid: int) -> bool:
    """Whether a process with this id exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main(argv: list[str] | None = None) -> int:
    """Runs the campaign; the exit code is the number of failed commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("development", "holdout"), required=True
    )
    parser.add_argument("--baseline-python", required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--systems", nargs="+", default=list(DEFAULT_SYSTEMS))
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS)
    )
    parser.add_argument("--open-holdout", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    tasks = phase_tasks(args.phase, args.open_holdout)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lock = args.out_dir / "campaign.lock"
    acquire(lock)
    failures = 0
    try:
        for task in tasks:
            for system in args.systems:
                run = command(
                    system,
                    task,
                    tuple(args.budgets),
                    args.out_dir,
                    args.baseline_python,
                    args.open_holdout,
                )
                LOGGER.info("task %d system %s", task, system)
                status = subprocess.run(run, check=False).returncode  # nosec B603
                if status:
                    failures += 1
                    LOGGER.error(
                        "task %d system %s exited %d", task, system, status
                    )
    finally:
        lock.unlink(missing_ok=True)
    return failures


if __name__ == "__main__":
    sys.exit(main())
