"""Re-measures stale evidence in a clean clone and brings the results home.

Every evidence store (`benchmarks/evidence/<x>.py`) and every certificate
script (`benchmarks/qualification/*_certify.py`) that has a `--check` mode
decides for itself whether its stored JSON is stale: `--check` exits 0 when
the fingerprint matches the code and non-zero otherwise. This command runs
those checks in a clean clone at one commit and runs `--write` only for the
stale ones, in dependency order:

1. `reproducibility`, alone, since it trains in fresh processes and
   compares bytes, and others may read what it writes.
2. Every other untimed store and certificate, in parallel (`--workers`).
3. The timing-sensitive stores (`TIMING`), only with `--timing`, one at a
   time, each after the one-minute load average falls below `--load`.

Each stage re-checks staleness just before it runs, so a store that an
earlier stage made fresh is not measured again. When every stage is done,
the JSON files that changed or appeared in the clone (as `git status` lists
them) are copied to the same paths in the working tree, and the scorecard
is printed. Nothing is copied if a measurement fails. It never edits product
code and never commits; review and commit the copied files by hand.

    python scripts/evidence.py --clone /tmp/scmle-clean
    python scripts/evidence.py --clone /tmp/scmle-clean --commit 57c48f69 \
        --timing
"""

import argparse
import concurrent.futures
import dataclasses
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIRST = "reproducibility"
TIMING = ("quality", "budget", "refusal")
SOURCES = (
    ("benchmarks/evidence", "*.py"),
    ("benchmarks/qualification", "*_certify.py"),
)
SCORECARD = "benchmarks.scorecard"
PYTHON = "uv run --frozen python"
# Share of the cores the one-minute load average may occupy before a timed
# store starts. A timed store runs one measured process; with the run queue
# at most a quarter of the cores, at least three quarters stay free, so the
# measured process and its BLAS threads are not descheduled for another
# runnable task in expectation (budget.json records the load next to every
# run, so a looser or tighter value can be judged from those records).
QUIET = 0.25
# Seconds between load readings: half the 60 s time constant of the
# one-minute load average, so every material change is seen once.
POLL = 30.0
# Longest wait, in seconds, for a quiet machine before giving up.
PATIENCE = 3600.0
# Cores one parallel measurement is assumed to occupy (most stores fit
# with scikit-learn's default threading on a few cores), so the default
# worker count does not oversubscribe the machine.
CORES = 4
# Characters of a failed measurement's output kept in its error.
TAIL = 2000


class CloneError(Exception):
    """The clone is not a clean checkout of the requested commit."""


class MeasureError(Exception):
    """A store or certificate failed to write its measurement."""


class LoadError(Exception):
    """The machine stayed too busy to measure timing-sensitive stores."""


@dataclasses.dataclass(frozen=True)
class Job:
    """One store or certificate script with `--check` and `--write` modes.

    Attributes:
        module: Dotted module path, run with `python -m`.
        timing: Whether its measurement depends on machine load.
    """

    module: str
    timing: bool

    @property
    def name(self) -> str:
        """The module's last component, such as `quality`."""
        return self.module.rsplit(".", 1)[-1]

    def stale(self, clone: pathlib.Path, python: list[str]) -> bool:
        """Runs `--check`; True when the stored result is stale."""
        found = subprocess.run(
            [*python, "-m", self.module, "--check"],
            cwd=clone,
            capture_output=True,
            check=False,
        )
        return found.returncode != 0

    def write(self, clone: pathlib.Path, python: list[str]) -> None:
        """Runs `--write`.

        Raises:
            MeasureError: When the script exits non-zero.
        """
        found = subprocess.run(
            [*python, "-m", self.module, "--write"],
            cwd=clone,
            capture_output=True,
            text=True,
            check=False,
        )
        if found.returncode != 0:
            tail = (found.stdout + found.stderr)[-TAIL:]
            message = f"{self.module} --write exited {found.returncode}"
            raise MeasureError(f"{message}:\n{tail}")


def discover(root: pathlib.Path) -> list[Job]:
    """Lists the scripts under `SOURCES` that offer `--check` and `--write`.

    Args:
        root: Repository or clone root.

    Returns:
        One job per script, sorted by module.
    """
    jobs = []
    for directory, pattern in SOURCES:
        for path in sorted((root / directory).glob(pattern)):
            text = path.read_text(encoding="utf-8")
            if '"--check"' in text and '"--write"' in text:
                module = ".".join([*pathlib.Path(directory).parts, path.stem])
                jobs.append(Job(module, path.stem in TIMING))
    return sorted(jobs, key=lambda job: job.module)


def git(root: pathlib.Path, *arguments: str) -> str:
    """Runs a read-only or clone-local git command and returns its output."""
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def prepare(repo: pathlib.Path, clone: pathlib.Path, commit: str) -> None:
    """Makes `clone` a clean checkout of `commit`, or proves it is one.

    A missing directory is cloned from `repo` and checked out at `commit`;
    an existing one is only inspected, never changed.

    Raises:
        CloneError: When the clone is dirty or at another commit.
    """
    wanted = git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}").strip()
    if not clone.exists():
        git(repo, "clone", "-q", "--no-checkout", str(repo), str(clone))
        git(clone, "checkout", "-q", "--detach", wanted)
    head = git(clone, "rev-parse", "HEAD").strip()
    if head != wanted:
        raise CloneError(f"{clone} is at {head}, not {wanted}")
    if changed(clone):
        raise CloneError(f"{clone} has uncommitted changes")


def changed(clone: pathlib.Path) -> list[str]:
    """Lists paths `git status` reports as modified or new in the clone."""
    output = git(clone, "status", "--porcelain=v1", "-z", "-uall")
    return [entry[3:] for entry in output.split("\0") if entry]


def wait(threshold: float, patience: float) -> None:
    """Blocks until the one-minute load average is below `threshold`.

    Raises:
        LoadError: When it stays at or above it for `patience` seconds.
    """
    waited = 0.0
    while (load := os.getloadavg()[0]) >= threshold:
        if waited >= patience:
            message = f"load {load:.1f} stayed at or above {threshold:.1f}"
            raise LoadError(f"{message} for {waited:.0f} s")
        print(f"load {load:.1f} >= {threshold:.1f}; waiting", flush=True)
        time.sleep(POLL)
        waited += POLL


def measure(
    jobs: list[Job],
    clone: pathlib.Path,
    python: list[str],
    workers: int,
) -> dict[str, str]:
    """Writes every stale job, `workers` at a time.

    Returns:
        Each job's module and `measured` or `fresh`.

    Raises:
        MeasureError: When any write fails, after the others finish.
    """

    def one(job: Job) -> str:
        if not job.stale(clone, python):
            return "fresh"
        print(f"measuring {job.module}", flush=True)
        job.write(clone, python)
        return "measured"

    with concurrent.futures.ThreadPoolExecutor(max(1, workers)) as pool:
        futures = {job.module: pool.submit(one, job) for job in jobs}
    return {module: future.result() for module, future in futures.items()}


def run(
    repo: pathlib.Path,
    clone: pathlib.Path,
    python: list[str],
    options: argparse.Namespace,
) -> dict[str, str]:
    """Runs the three stages in the clone and copies the results to `repo`.

    Returns:
        Each job's module and `measured`, `fresh` or `skipped`.
    """
    subprocess.run([*python, "-c", ""], cwd=clone, check=True)
    jobs = discover(clone)
    first = [job for job in jobs if job.name == FIRST]
    untimed = [job for job in jobs if job.name != FIRST and not job.timing]
    timed = [job for job in jobs if job.timing]
    outcome = measure(first, clone, python, 1)
    outcome |= measure(untimed, clone, python, options.workers)
    for job in timed:
        if not options.timing:
            stale = job.stale(clone, python)
            outcome[job.module] = "skipped" if stale else "fresh"
            continue
        if job.stale(clone, python):
            wait(options.load, options.wait)
        outcome |= measure([job], clone, python, 1)
    for path in changed(clone):
        if path.endswith(".json"):
            target = repo / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(clone / path, target)
            print(f"copied {path}")
    return outcome


def main(arguments: list[str] | None = None) -> int:
    """Re-measures stale evidence and prints the scorecard.

    Returns:
        0 when every requested measurement succeeded.
    """
    cores = os.cpu_count() or 1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clone", type=pathlib.Path, required=True)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--repo", type=pathlib.Path, default=ROOT)
    parser.add_argument("--python", default=PYTHON)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--load", type=float, default=QUIET * cores)
    parser.add_argument("--wait", type=float, default=PATIENCE)
    parser.add_argument("--workers", type=int, default=max(1, cores // CORES))
    options = parser.parse_args(arguments)
    repo = options.repo.resolve()
    clone = options.clone.resolve()
    python = shlex.split(options.python)
    prepare(repo, clone, options.commit)
    outcome = run(repo, clone, python, options)
    for module, verdict in sorted(outcome.items()):
        print(f"{verdict:9} {module}")
    found = subprocess.run(
        [*python, "-m", SCORECARD],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    print(found.stdout, end="")
    print(found.stderr, end="", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
