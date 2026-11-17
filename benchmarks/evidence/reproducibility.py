"""Claim C6: the same inputs and seed give a byte-identical bundle.

Plan section 5, claim C6 "Reproducible and traceable": same inputs, seed,
lock file and platform give a byte-identical bundle; across platforms the
metrics agree within `cross_platform_tol` (`acceptance.toml`); every bundle
traces to its data hash, configuration, commit and environment.

What is measured, on the platform that runs `--write`:

* For each task kind in `TASKS` (binary, multiclass, regression and a
  nonlinear binary task), the same small project is trained from scratch
  twice with seed `SEED`, each time in a fresh Python process
  (`subprocess`, so nothing is cached in memory between the two) and in a
  different scratch directory. `PYTHONHASHSEED` is left as the caller has
  it, so the two processes normally hash strings differently, as two real
  runs would.
* Every member of the two saved, signed bundles is compared byte for byte;
  the members that differ are listed.
* The largest absolute and relative difference between the two runs'
  locked-test metrics, against `cross_platform_tol`.
* Which of the trace fields (data hash, configuration, commit, environment)
  the bundle manifest carries.

A record is kept per platform. `--write` replaces this platform's record and
keeps the other platforms' records that were made by the same code, so a
second platform can add its own; the cross-platform comparison of metrics is
made only when two or more records exist and is otherwise "not measured".

Verdict: "fail" when any platform's bundles differ, its metrics move beyond
the tolerance or its manifest lacks a trace field; otherwise "not measured"
with one platform and "pass" with two or more whose metrics agree within the
tolerance.

    python -m benchmarks.evidence.reproducibility --write
    python -m benchmarks.evidence.reproducibility --check
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy
import pandas
import threadpoolctl

from benchmarks import acceptance
from benchmarks import reliability
import scmle
import scmle.api.project
import scmle.api.results
from scmle.artifacts import bundle
from scmle.contracts import errors
from scmle.data.detectors import certificates

PATH = pathlib.Path(__file__).with_suffix(".json")
ROOT = pathlib.Path(__file__).resolve().parents[2]
TOLERANCE = "cross_platform_tol"
SEED = 7
ROWS = 600
SIGNING_KEY = "reproducibility-evidence-key"
# seconds; ample, so the wall-clock budget, which depends on machine load,
# never decides which trials run (a binding budget is a different input)
BUDGET = 3600
# share of labels flipped in the nonlinear task
FLIP = 0.1
# records needed for a cross-platform comparison
PAIR = 2
TASKS = ("binary", "multiclass", "regression", "interaction")
LIBRARIES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "joblib",
    "pyarrow",
    "threadpoolctl",
)
# Manifest keys that would carry each trace item (ML-10); any one suffices.
TRACE = {
    "data hash": ("dataset", "data_hash", "dataset_hash"),
    "configuration": ("config_hash", "configuration", "config"),
    "commit": ("commit", "git_commit"),
    "environment": ("environment",),
}


def generate(kind: str) -> tuple[pandas.DataFrame, str]:
    """The fixed dataset of one task kind.

    Args:
        kind: One of `TASKS`.

    Returns:
        The frame and the name of its target column.

    Raises:
        InputError: If `kind` is not one of `TASKS`.
    """
    if kind == "binary":
        return reliability.binary_data(ROWS, SEED), "y"
    if kind == "regression":
        return reliability.regression_data(ROWS, SEED), "y"
    rng = numpy.random.default_rng(SEED)
    frame = pandas.DataFrame(
        {
            "a": rng.normal(size=ROWS),
            "b": rng.normal(size=ROWS),
            "c": rng.choice(["x", "y", "z"], ROWS),
        }
    )
    if kind == "multiclass":
        scores = numpy.stack(
            [
                frame.a + (frame.c == "x"),
                frame.b - 0.5 * frame.a,
                -frame.a - frame.b + (frame.c == "z"),
            ],
            axis=1,
        ) + rng.gumbel(size=(ROWS, 3))
        frame["y"] = numpy.array(["low", "mid", "high"])[scores.argmax(axis=1)]
        return frame, "y"
    if kind == "interaction":
        noise = rng.uniform(size=ROWS) < FLIP
        frame["y"] = ((frame.a * frame.b > 0) ^ noise).astype(int)
        return frame, "y"
    raise errors.InputError(f"unknown task kind {kind!r}; known: {TASKS}")


def train(kind: str, directory: pathlib.Path) -> dict[str, Any]:
    """Trains one project and saves its signed bundle (run in a child).

    Args:
        kind: One of `TASKS`.
        directory: Fresh scratch directory; the run goes to `run` and the
            bundle to `bundle` under it.

    Returns:
        The selected model, the locked-test metrics, the number of trials,
        the seconds taken and
        the BLAS and OpenMP libraries the process loaded.
    """
    started = time.monotonic()
    data, target = generate(kind)
    result = reliability.fast_project(
        data, target, directory / "run", seed=SEED, time_budget=BUDGET
    ).run()
    result.save(directory / "bundle", signing_key=SIGNING_KEY)
    return {
        "selected": result.selected_model,
        "metrics": {
            name: float(value)
            for name, value in result.evaluation["test_metrics"].items()
        },
        "trials": len(result.trials),
        "seconds": round(time.monotonic() - started, 1),
        "threads": [
            {
                key: entry.get(key)
                for key in ("internal_api", "version", "num_threads")
            }
            for entry in threadpoolctl.threadpool_info()
        ],
    }


def launch(kind: str, directory: pathlib.Path) -> dict[str, Any]:
    """Runs `train` in a fresh Python process.

    Args:
        kind: One of `TASKS`.
        directory: Fresh scratch directory for that process.

    Returns:
        What `train` returned in the child process.

    Raises:
        RuntimeError: If the child process fails.
    """
    command = [
        sys.executable,
        "-m",
        "benchmarks.evidence.reproducibility",
        "--train",
        kind,
        "--into",
        str(directory),
    ]
    finished = subprocess.run(  # fixed command: this module
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if finished.returncode != 0:
        raise RuntimeError(
            f"training {kind} failed:\n{finished.stderr[-4000:]}"
        )
    lines = finished.stdout.strip().splitlines()
    return dict(json.loads(lines[-1]))


def digest(directory: pathlib.Path) -> dict[str, str]:
    """SHA-256 of every file in a bundle.

    Args:
        directory: The bundle directory.

    Returns:
        Path relative to the bundle (POSIX form) to hex digest, sorted.
    """
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def differ(
    first: dict[str, float], second: dict[str, float]
) -> dict[str, float]:
    """Largest absolute and relative gap between two sets of metrics.

    The relative gap of a pair is the absolute gap over the larger
    magnitude, and zero when both are zero.

    Args:
        first: Metric name to value.
        second: Metric name to value, with the same names.

    Returns:
        `absolute` and `relative`, the largest gaps over the metrics.

    Raises:
        InputError: If the metric names differ.
    """
    if set(first) != set(second):
        raise errors.InputError(
            f"metrics differ in name: {sorted(first)} vs {sorted(second)}"
        )
    largest, ratio = 0.0, 0.0
    for name, value in first.items():
        gap = abs(value - second[name])
        scale = max(abs(value), abs(second[name]))
        largest = max(largest, gap)
        ratio = max(ratio, gap / scale if scale else 0.0)
    return {"absolute": largest, "relative": ratio}


def trace(manifest: dict[str, Any]) -> list[str]:
    """The trace items a bundle manifest does not carry.

    Args:
        manifest: The parsed `model_manifest.json`.

    Returns:
        The names in `TRACE` with no non-empty key in the manifest.
    """
    return [
        item
        for item, keys in TRACE.items()
        if not any(manifest.get(key) for key in keys)
    ]


def compare(kind: str, scratch: pathlib.Path) -> dict[str, Any]:
    """Trains one task twice in fresh processes and compares the results.

    Args:
        kind: One of `TASKS`.
        scratch: Directory for the two runs.

    Returns:
        The selected models, metrics, metric gaps, differing members, the
        number of members, missing trace items, and the trials and seconds
        of each run.
    """
    runs = []
    hashes = []
    for index in range(2):
        directory = scratch / f"{kind}{index}"
        runs.append(launch(kind, directory))
        hashes.append(digest(directory / "bundle"))
    members = sorted(set(hashes[0]) | set(hashes[1]))
    manifest = json.loads(
        (scratch / f"{kind}0" / "bundle" / bundle.MANIFEST_FILE).read_text(
            encoding="utf-8"
        )
    )
    return {
        "selected": [run["selected"] for run in runs],
        "metrics": runs[0]["metrics"],
        "gap": differ(runs[0]["metrics"], runs[1]["metrics"]),
        "members": len(members),
        "differing": [
            name
            for name in members
            if hashes[0].get(name) != hashes[1].get(name)
        ],
        "hashes": hashes[0],
        "untraced": trace(manifest),
        "trials": [run["trials"] for run in runs],
        "seconds": [run["seconds"] for run in runs],
        "threads": runs[0]["threads"],
    }


def describe() -> dict[str, Any]:
    """The platform, interpreter and library versions of this machine.

    Returns:
        Platform text, machine, Python version, `PYTHONHASHSEED` as set
        (None when unset) and the version of each of `LIBRARIES`.
    """
    versions = {"scmle": scmle.__version__}
    for name in LIBRARIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            # handled: an absent library is recorded as absent
            versions[name] = "absent"
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "hashseed": os.environ.get("PYTHONHASHSEED"),
        "versions": versions,
    }


def fingerprint() -> str:
    """Hash of the code whose behaviour the measurement depends on.

    Returns:
        The `certificates.code_hash` of the bundle, project and results
        modules, the quick project and data builders of
        `benchmarks.reliability` and the measuring functions here.
    """
    return certificates.code_hash(
        bundle,
        scmle.api.project,
        scmle.api.results,
        reliability.fast_project,
        reliability.binary_data,
        reliability.regression_data,
        generate,
        train,
        launch,
        digest,
        differ,
        trace,
        compare,
    )


def tolerance() -> float:
    """The relative tolerance of claim C6 from `acceptance.toml`.

    Returns:
        The value of `cross_platform_tol`.
    """
    return float(acceptance.load()[TOLERANCE].value)


def measure() -> dict[str, Any]:
    """Measures this platform and keeps current records of other platforms.

    Returns:
        The document written to `PATH`: the tolerance, the code hash and one
        record per platform, keyed by platform text.
    """
    started = time.monotonic()
    machine = describe()
    with tempfile.TemporaryDirectory(prefix="scmle-c6-") as scratch:
        results = {kind: compare(kind, pathlib.Path(scratch)) for kind in TASKS}
    code = fingerprint()
    records: dict[str, Any] = {}
    if PATH.exists():
        previous = json.loads(PATH.read_text(encoding="utf-8"))
        if previous.get("code") == code:
            records = dict(previous.get("platforms", {}))
    records[machine["platform"]] = {
        "machine": machine,
        "seed": SEED,
        "budget": BUDGET,
        "rows": ROWS,
        "tasks": results,
        "seconds": round(time.monotonic() - started, 1),
    }
    return {
        "version": 1,
        "claim": "C6",
        "tolerance": {"name": TOLERANCE, "value": tolerance()},
        "code": code,
        "platforms": dict(sorted(records.items())),
        "not_measured": [
            "platforms other than those recorded here",
            "a clean-runner rebuild from the lock file",
            (
                "model families excluded by the quick project (bagging, "
                "deep learning, foundation) and optional boosters not "
                "installed"
            ),
            "runs whose wall-clock time budget cuts the search short",
        ],
    }


def read(document: dict[str, Any] | None) -> dict[str, Any]:
    """The given document, or the stored one.

    Args:
        document: A document, or None to read `PATH`.

    Returns:
        The document.

    Raises:
        InputError: If `PATH` is missing or the document has no platforms.
    """
    if document is None:
        if not PATH.exists():
            raise errors.InputError(f"{PATH.name} is missing")
        document = json.loads(PATH.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(
        document.get("platforms"), dict
    ):
        raise errors.InputError("the document has no platform records")
    return document


def problems(document: dict[str, Any] | None = None) -> list[str]:
    """Why the stored measurements are not for the current code.

    Args:
        document: The measurements document; read from `PATH` when None.

    Returns:
        A message for a missing or malformed document, a code hash that
        differs from the current code, a tolerance that differs from
        `acceptance.toml` or a document with no platform record; empty when
        current.
    """
    try:
        found = read(document)
    except errors.InputError as error:
        return [f"reproducibility: {error}"]
    issues = []
    if found.get("code") != fingerprint():
        issues.append("reproducibility: measurements are for other code")
    if found.get("tolerance", {}).get("value") != tolerance():
        issues.append(
            f"reproducibility: {TOLERANCE} differs from acceptance.toml"
        )
    if not found["platforms"]:
        issues.append("reproducibility: no platform was measured")
    return issues


def spread(records: dict[str, Any]) -> float:
    """Largest relative metric gap between platforms over shared tasks.

    Args:
        records: Platform text to record.

    Returns:
        The largest relative gap between any two platforms' first-run
        metrics on a task they both measured (0.0 with one platform).
    """
    largest = 0.0
    names = sorted(records)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            one, two = records[left]["tasks"], records[right]["tasks"]
            for kind in sorted(set(one) & set(two)):
                gap = differ(one[kind]["metrics"], two[kind]["metrics"])
                largest = max(largest, gap["relative"])
    return largest


def summary(document: dict[str, Any] | None = None) -> tuple[str, str]:
    """The scorecard's "Value now" text and verdict for claim C6.

    Args:
        document: The measurements document; read from `PATH` when None.

    Returns:
        The value text and "pass", "fail" or "not measured". Stale or
        missing measurements are "not measured".
    """
    issues = problems(document)
    if issues:
        return "; ".join(issues), "not measured"
    found = read(document)
    limit = float(found["tolerance"]["value"])
    records = found["platforms"]
    parts, failed = [], False
    for name, record in records.items():
        tasks = record["tasks"]
        identical = [k for k, t in tasks.items() if not t["differing"]]
        worst = max(t["gap"]["relative"] for t in tasks.values())
        untraced = sorted({i for t in tasks.values() for i in t["untraced"]})
        failed = failed or len(identical) < len(tasks) or worst > limit
        failed = failed or bool(untraced)
        text = (
            f"{name}, Python {record['machine']['python']}: "
            f"{len(identical)}/{len(tasks)} tasks byte-identical over two "
            f"fresh processes; max metric relative difference {worst:.3g} "
            f"(tol {limit:g})"
        )
        for kind, task in tasks.items():
            if task["differing"]:
                text += f"; {kind} differs in {', '.join(task['differing'])}"
        if untraced:
            text += f"; manifest lacks {', '.join(untraced)}"
        parts.append(text)
    if len(records) < PAIR:
        parts[0] = "this platform only: " + parts[0]
        parts.append("cross-platform not measured")
        return "; ".join(parts), "fail" if failed else "not measured"
    across = spread(records)
    failed = failed or across > limit
    parts.append(
        f"across {len(records)} platforms max metric relative difference "
        f"{across:.3g}"
    )
    return "; ".join(parts), "fail" if failed else "pass"


def main(arguments: list[str] | None = None) -> int:
    """Writes (`--write`) or checks (`--check`) the measurements.

    `--train KIND --into DIR` is the child process `launch` starts.

    Args:
        arguments: Command-line arguments; None reads `sys.argv`.

    Returns:
        0 for `--write` and `--train`; for `--check`, 0 when the stored
        measurements are for the current code and 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--train", choices=TASKS)
    parser.add_argument("--into", type=pathlib.Path)
    options = parser.parse_args(arguments)
    if options.train:
        if options.into is None:
            raise errors.InputError("--train needs --into")
        print(json.dumps(train(options.train, options.into)))
        return 0
    if options.write:
        document = measure()
        PATH.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        value, verdict = summary(document)
        print(json.dumps({"value": value, "verdict": verdict}, indent=2))
        return 0
    issues = problems()
    print(json.dumps({"ok": not issues, "problems": issues}, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
