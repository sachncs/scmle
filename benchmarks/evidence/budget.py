"""Claim C7 in part: runs keep their time budget and survive a hard kill.

Two measurements, each run in child processes started from this module:

* Budget compliance. `RUNS_MAX` small runs (`reliability.fast_project` on
  `reliability.binary_data`, a different data seed each, hard enforcement)
  with a `BUDGET_SECONDS` time budget, one after another, each in its own
  process with the environment inherited unchanged. A run is compliant when
  it completes and the wall time of `Project.run` is at most the budget plus
  the product's own grace, `max(limits.GRACE_SHARE * budget,
  limits.MIN_CEILING_SECONDS)`: the hard ceiling a trial dispatched at the
  last moment of the budget is killed at, so the latest a search can end.
  Nothing else is allowed: preparation and the post-selection work count
  against the budget as the user sees it. The diagnostics allowance the
  product grants itself after selection (`stages.DIAGNOSTIC_MIN_SECONDS`) is
  filed beside the result and does not gate. A run still going at
  `TIMEOUT_FACTOR` times the allowance is killed and counted as over. No
  new run starts after `COMPLIANCE_SECONDS`. The row is the
  `budget_compliance_min` floor of `acceptance.toml`, decided by its
  `Floor` rule at `alpha`; fewer runs than the rule's minimum is "not
  measured", never a pass.
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

NOT measured: the memory budget, budgets other than `BUDGET_SECONDS`, other
tasks, data sizes and model families, kills outside the training stage,
and journal and lease operations.

    python -m benchmarks.evidence.budget --write
    python -m benchmarks.evidence.budget --check
"""

import argparse
import contextlib
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
import scmle.api.project
from scmle.contracts import errors
from scmle.core import limits
from scmle.core import stages
from scmle.data.detectors import certificates

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


def grace(budget: float) -> float:
    """Seconds past the budget a compliant run may take.

    Args:
        budget: The declared time budget in seconds.

    Returns:
        The hard-ceiling grace of `scmle.core.limits`.
    """
    return max(limits.GRACE_SHARE * budget, limits.MIN_CEILING_SECONDS)


def settings() -> dict[str, Any]:
    """The fixed design of the measurement.

    Returns:
        Every constant that changes what is measured.
    """
    return {
        "budget_seconds": BUDGET_SECONDS,
        "grace_seconds": grace(BUDGET_SECONDS),
        "grace_share": limits.GRACE_SHARE,
        "min_ceiling_seconds": limits.MIN_CEILING_SECONDS,
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
    allowance = BUDGET_SECONDS + grace(BUDGET_SECONDS)
    load = os.getloadavg()[0]
    started = time.monotonic()
    process = spawn(
        ["run", str(directory), str(BUDGET_SECONDS), str(seed)], quiet=False
    )
    found = finish(process, TIMEOUT_FACTOR * allowance)
    seconds = found.get("seconds")
    return {
        "seed": seed,
        "load": round(load, 1),
        "process_seconds": round(time.monotonic() - started, 2),
        "run_seconds": None if seconds is None else round(seconds, 2),
        "completed": "outcome" in found,
        "timed_out": found["timed_out"],
        "error": found.get("error", ""),
        "compliant": seconds is not None and seconds <= allowance,
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
    allowance = BUDGET_SECONDS + grace(BUDGET_SECONDS)
    lenient = allowance + stages.DIAGNOSTIC_MIN_SECONDS
    return {
        "allowance_seconds": allowance,
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
        grace,
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


def measure() -> dict[str, Any]:
    """Runs both measurements.

    Returns:
        The JSON-ready document: settings, both measurements, what was not
        measured, the code hash and the duration.
    """
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="scmle-budget-") as name:
        scratch = Path(name)
        found = compliance(scratch)
        resumed = resumption(scratch)
    return {
        "version": 1,
        "claim": "C7",
        "variable": VARIABLE,
        "settings": settings(),
        "compliance": found,
        "resume": resumed,
        "not_measured": [
            "the memory budget",
            "budgets other than the one in settings",
            "other tasks, data sizes and model families",
            "kills outside the training stage",
            "journal and lease operations",
        ],
        "code": current(),
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
    if document.get("code") != current():
        found.append("budget: measurements are for other code")
    if document.get("settings") != settings():
        found.append("budget: measurements are for other settings")
    return found


def summary(document: dict[str, Any] | None = None) -> tuple[str, str]:
    """The scorecard's value and verdict for the measured part of C7.

    A kill whose resume differs from the uninterrupted run fails the claim
    whatever the compliance count. Otherwise compliance is decided by the
    `budget_compliance_min` floor; fewer runs than its minimum is "not
    measured".

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
    try:
        hits = int(document["compliance"]["hits"])
        trials = int(document["compliance"]["trials"])
        same = int(document["resume"]["identical"])
        cases = int(document["resume"]["trials"])
        budget = float(document["settings"]["budget_seconds"])
        extra = float(document["settings"]["grace_seconds"])
    except (KeyError, TypeError, ValueError) as caught:
        raise errors.InputError(
            f"malformed budget document: {caught}"
        ) from None
    if not 0 <= hits <= trials or not 0 <= same <= cases:
        raise errors.InputError("counts must lie between 0 and the trials")
    variables = acceptance.load()
    alpha = variables["alpha"].value
    rule = variables[VARIABLE].rule
    if not isinstance(rule, acceptance.Floor):
        raise errors.InputError(f"{VARIABLE} must be a floor")
    least = rule.minimum(alpha)
    value = (
        f"{hits} of {trials} runs within {budget:g} s + {extra:g} s grace; "
        f"kill and resume identical in {same} of {cases}"
    )
    if cases == 0 or same < cases:
        return (value, "fail")
    if trials < least:
        return (
            (
                f"{value}; {least - trials} more runs needed "
                f"(minimum {least} at alpha {alpha:g})"
            ),
            "not measured",
        )
    return (value, "pass" if rule.passes(hits, trials, alpha) else "fail")


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
    options = parser.parse_args(arguments)
    if options.child:
        return child(options.child)
    if options.write:
        document = measure()
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
