"""Claim C6: every bundle traces, and rebuilds on a clean runner.

Plan section 5, claim C6 "Reproducible and traceable", two clauses:

* "every bundle traces to data hash, configuration, commit and
  environment": each task kind in `reproducibility.TASKS` is trained by the
  real command line (``python -m scmle run data.csv ...``, a fresh process
  in its own folder, reading a file) and its signed bundle is checked with
  `scmle.trail.origin.trace`: each of the four fields must be present and
  verify against records other than the manifest (the data file rehashed,
  ``config.json`` rehashed, the commit looked up in this repository, the
  pins matched with the manifest's own version records). So that a pass
  means the check can fail, each `CONTROLS` tampering is applied to a copy
  of the first bundle's run and must be caught by exactly the field it
  breaks.
* "rebuild on a clean runner": each bundle is rebuilt by
  `scmle.trail.replica.rebuild` from its manifest alone (a new virtual
  environment at the manifest's Python with exactly its pins, scmle from
  its commit, a retraining in that environment) and compared byte for byte
  and by locked-test metrics within `cross_platform_tol`. `--write` does it
  on this machine in temporary environments; the ``reproducibility``
  workflow does it on a fresh CI runner (``--build`` on one runner,
  ``--rebuild`` on another) and its record is stored with ``--merge``.

Verdicts: the trace clause passes when every bundle traces and every
control is caught by its field alone. The rebuild clause passes when every
local and every clean-runner rebuild reproduces (identical bytes, or
metrics within the tolerance); with no current clean-runner record it is
"not measured", and any rebuild that does not reproduce fails it.

    python -m benchmarks.evidence.traceability --write
    python -m benchmarks.evidence.traceability --check
    python -m benchmarks.evidence.traceability --build DIR
    python -m benchmarks.evidence.traceability --rebuild DIR --record FILE
    python -m benchmarks.evidence.traceability --merge FILE
"""

import argparse
from collections.abc import Callable
import hashlib
import json
import math
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

from benchmarks import acceptance
from benchmarks import sources
from benchmarks.evidence import reproducibility
from scmle.artifacts.bundle import layout
from scmle.artifacts.bundle import seal
import scmle.cli
from scmle.contracts import errors
from scmle.trail import origin
from scmle.trail import replica

PATH = pathlib.Path(__file__).with_suffix(".json")
ROOT = pathlib.Path(__file__).resolve().parents[2]
TOLERANCE = "cross_platform_tol"
SIGNING_KEY = "traceability-evidence-key"
DATA = "data.csv"
# The families the quick projects of the other C6 measurements exclude.
FAMILIES = ("bagging", "deep_learning", "foundation")
TRACE = "trace"
REBUILD = "rebuild"
CLAUSES = (TRACE, REBUILD)
NOT_MEASURED = [
    "runs that read an in-memory frame (no file to rehash or retrain from)",
    "modalities other than tabular files read whole",
    "system libraries outside the Python environment (OpenMP, BLAS, libc)",
    (
        "whether a bundle built elsewhere came from a tree with uncommitted "
        "changes (the manifest does not record it; the bundles measured here "
        "are built only from a clean tree)"
    ),
    (
        "whether the interpreter was a free-threaded build (the manifest "
        "records the Python version only)"
    ),
]


def resign(bundle: pathlib.Path, manifest: dict[str, Any]) -> None:
    """Writes a changed manifest and signs it again with `SIGNING_KEY`.

    So only the field a control breaks can catch it, not the signature.

    Args:
        bundle: The bundle directory.
        manifest: The manifest to write.
    """
    path = bundle / layout.MANIFEST_FILE
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    signature = json.loads((bundle / seal.SIGNATURE_FILE).read_text("utf-8"))
    signature["signature"] = seal.sign(path.read_bytes(), SIGNING_KEY.encode())
    (bundle / seal.SIGNATURE_FILE).write_text(
        json.dumps(signature), encoding="utf-8"
    )


def edit(bundle: pathlib.Path, key: str, value: Any, sign: bool) -> None:
    """Changes one manifest field.

    Args:
        bundle: The bundle directory.
        key: The manifest key.
        value: Its new value.
        sign: Whether to sign the changed manifest again.
    """
    manifest = origin.load(bundle)
    manifest[key] = value
    if sign:
        resign(bundle, manifest)
    else:
        (bundle / layout.MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )


def corrupt(path: pathlib.Path) -> None:
    """Flips the last byte of a file.

    Args:
        path: The file.
    """
    content = bytearray(path.read_bytes())
    content[-1] ^= 0xFF
    path.write_bytes(bytes(content))


def perturb(folder: pathlib.Path) -> None:
    """Changes one value of the training data (the data field breaks)."""
    path = folder / DATA
    lines = path.read_text(encoding="utf-8").splitlines()
    cells = lines[1].split(",")
    cells[0] = repr(float(cells[0]) + 1.0)
    lines[1] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reseed(folder: pathlib.Path) -> None:
    """Changes the stored configuration's seed (the configuration breaks)."""
    path = run(folder) / origin.CONFIG
    config = json.loads(path.read_text(encoding="utf-8"))
    config["validation"]["seed"] += 1
    path.write_text(json.dumps(config), encoding="utf-8")


def recommit(folder: pathlib.Path) -> None:
    """Names a well-formed commit the repository does not hold."""
    edit(run(folder) / "model", "commit", "0" * 40, sign=True)


def repin(folder: pathlib.Path) -> None:
    """Pins scikit-learn at a version the manifest does not record."""
    bundle = run(folder) / "model"
    pins = dict(origin.load(bundle)["environment"])
    pins["scikit-learn"] = "0.0.1"
    edit(bundle, "environment", pins, sign=True)


def forge(folder: pathlib.Path) -> None:
    """Changes the decision threshold without signing the manifest again.

    The threshold is outside the four trace fields, so only the signature
    can catch the edit; an unsigned edit of a trace field is also caught by
    that field's own verification.
    """
    edit(run(folder) / "model", "threshold", 0.123, sign=False)


def tamper(folder: pathlib.Path) -> None:
    """Corrupts the stored pipeline (its manifest digest breaks)."""
    corrupt(run(folder) / "model" / "pipeline.joblib")


def unsign(folder: pathlib.Path) -> None:
    """Removes the signature, so an unsigned manifest edit would pass."""
    (run(folder) / "model" / seal.SIGNATURE_FILE).unlink()


INTEGRITY = "integrity"
# Each tampering and the one field that must catch it.
CONTROLS: dict[str, tuple[Callable[[pathlib.Path], None], str]] = {
    "changed data value": (perturb, "data hash"),
    "changed configuration": (reseed, "configuration"),
    "unknown commit": (recommit, "commit"),
    "pin disagrees with manifest": (repin, "environment"),
    "manifest edited without signing": (forge, INTEGRITY),
    "stored pipeline corrupted": (tamper, INTEGRITY),
    "signature removed": (unsign, INTEGRITY),
}


def run(folder: pathlib.Path) -> pathlib.Path:
    """The one run directory a task folder holds.

    Args:
        folder: The task folder.

    Returns:
        ``runs/run_*`` inside it.

    Raises:
        ArtifactError: If it holds none or several.
    """
    found = sorted((folder / "runs").glob("run_*"))
    if len(found) != 1:
        raise errors.ArtifactError(f"{folder} holds {len(found)} runs, not 1")
    return found[0]


def train(kind: str, folder: pathlib.Path) -> pathlib.Path:
    """Trains one task with the command line in a fresh process.

    Args:
        kind: One of `reproducibility.TASKS`.
        folder: A new folder; the data and the runs are written into it.

    Returns:
        The run directory.

    Raises:
        ArtifactError: If the command fails.
    """
    folder.mkdir(parents=True, exist_ok=True)
    frame, target = reproducibility.generate(kind)
    frame.to_csv(folder / DATA, index=False)
    command = [
        sys.executable,
        "-m",
        "scmle",
        "run",
        DATA,
        "--target",
        target,
        "--seed",
        str(reproducibility.SEED),
        "--budget",
        str(reproducibility.BUDGET),
        "--exclude-families",
        *FAMILIES,
        "--output-dir",
        "runs",
    ]
    variables = {**os.environ, "SCMLE_SIGNING_KEY": SIGNING_KEY}
    try:
        finished = subprocess.run(  # fixed command: the scmle command line
            command,
            cwd=folder,
            env=variables,
            capture_output=True,
            text=True,
            timeout=replica.SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise errors.ArtifactError(
            f"training {kind} failed: {error}"
        ) from error
    if finished.returncode != 0:
        raise errors.ArtifactError(
            f"training {kind} failed:\n{finished.stderr[-4000:]}"
        )
    return run(folder)


def clean() -> bool:
    """Whether this repository's tracked files match its HEAD commit.

    A bundle records HEAD as its commit, so a bundle trained from a tree
    with uncommitted changes would trace to code that did not train it.

    Returns:
        True when ``git status`` lists no changed tracked file.

    Raises:
        ArtifactError: If git fails.
    """
    status = origin.git(ROOT, "status", "--porcelain", "--untracked-files=no")
    if status.returncode != 0:
        raise errors.ArtifactError(f"git status failed: {status.stderr}")
    return not status.stdout.strip()


def build(directory: pathlib.Path) -> dict[str, pathlib.Path]:
    """Trains every task, one folder each, from a clean tree.

    Args:
        directory: The folder that holds a folder per task.

    Returns:
        Task kind to its run directory.

    Raises:
        InputError: If the working tree has uncommitted changes.
    """
    if not clean():
        raise errors.InputError(
            "the working tree has uncommitted changes; measure in a clean clone"
        )
    return {
        kind: train(kind, directory / kind) for kind in reproducibility.TASKS
    }


def check(bundle: pathlib.Path) -> dict[str, Any]:
    """The trace record of one bundle.

    Args:
        bundle: The bundle directory.

    Returns:
        `origin.Report.dump` of it, checked in this repository.
    """
    return origin.trace(bundle, repository=ROOT, key=SIGNING_KEY).dump()


def failing(record: dict[str, Any]) -> list[str]:
    """The fields of a trace record that do not verify.

    Args:
        record: An `origin.Report.dump` record.

    Returns:
        `INTEGRITY` when the manifest is not intact or its signature did not
        verify (every bundle here is signed), then each field that is absent
        or fails, sorted.
    """
    intact = not record["integrity"] and record.get("signed") is True
    found = [] if intact else [INTEGRITY]
    return found + sorted(
        name
        for name, outcome in record["checks"].items()
        if not outcome["verified"]
    )


def control(folder: pathlib.Path, scratch: pathlib.Path) -> dict[str, Any]:
    """Applies each control to a copy of one task folder and traces it.

    Args:
        folder: A trained task folder.
        scratch: A folder for the copies.

    Returns:
        Control name to the field expected to catch it, the fields that
        failed and whether exactly that field failed.
    """
    outcomes = {}
    for index, (name, (damage, field)) in enumerate(CONTROLS.items()):
        copy = scratch / f"control{index}"
        shutil.copytree(folder, copy)
        damage(copy)
        failed = failing(check(run(copy) / "model"))
        outcomes[name] = {
            "expected": field,
            "failed": failed,
            "caught": failed == [field],
        }
    return outcomes


def unbuilt(error: Exception) -> dict[str, Any]:
    """The rebuild record of a bundle that could not be rebuilt.

    Args:
        error: Why.

    Returns:
        A record that neither reproduces nor is identical, with an infinite
        gap and the error text.
    """
    return {
        "reproduced": False,
        "identical": False,
        "gap": {"absolute": math.inf, "relative": math.inf},
        "error": str(error),
    }


def untraced(error: Exception) -> dict[str, Any]:
    """The trace record of a bundle that could not be read.

    Args:
        error: Why.

    Returns:
        A record whose integrity names the error and that has no checks.
    """
    return {
        "integrity": [str(error)],
        "checks": {},
        "signed": False,
        "traced": False,
    }


def rebuild(runs: dict[str, pathlib.Path]) -> dict[str, Any]:
    """Rebuilds each bundle in a fresh temporary environment.

    One bundle's failure is recorded and does not stop the others.

    Args:
        runs: Task kind to run directory.

    Returns:
        Task kind to `replica.Rebuilt.dump`, or to `unbuilt` of the error
        for a bundle that does not trace or whose rebuild fails.
    """
    found = {}
    for kind, directory in runs.items():
        try:
            found[kind] = replica.rebuild(
                directory / "model",
                repository=ROOT,
                key=SIGNING_KEY,
                tolerance=tolerance(),
            ).dump()
        except (errors.ArtifactError, errors.InputError) as error:
            found[kind] = unbuilt(error)
    return found


def describe() -> dict[str, str]:
    """The machine a record was made on.

    Returns:
        Platform text, machine and Python version.
    """
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


def fingerprint() -> str:
    """Hash of the code and locked libraries the measurement runs.

    Returns:
        The `sources.digest` of the command line (whose imports reach every
        product module a run loads), the trace and rebuild modules and the
        measuring functions here, followed by the hash of `uv.lock`.
    """
    code = sources.digest(
        scmle.cli,
        origin,
        replica,
        reproducibility.generate,
        train,
        clean,
        build,
        check,
        failing,
        control,
        rebuild,
        perturb,
        reseed,
        recommit,
        repin,
        forge,
        tamper,
        unsign,
        root=ROOT,
    )
    lock = ROOT / "uv.lock"
    locked = (
        hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else ""
    )
    return hashlib.sha256(f"{code}\n{locked}".encode("ascii")).hexdigest()


def tolerance() -> float:
    """The relative metric tolerance from `acceptance.toml`.

    Returns:
        The value of `cross_platform_tol`.
    """
    return float(acceptance.load()[TOLERANCE].value)


def measure() -> dict[str, Any]:
    """Trains, traces, controls and rebuilds on this machine.

    Returns:
        The document written to `PATH`, keeping a stored clean-runner record
        made by the same code.

    Raises:
        InputError: If the working tree has uncommitted changes (`build`).
    """
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="scmle-c6-trace-") as scratch:
        root = pathlib.Path(scratch)
        runs = build(root / "tasks")
        traced = {kind: check(path / "model") for kind, path in runs.items()}
        controls = control(
            root / "tasks" / reproducibility.TASKS[0], root / "controls"
        )
        local = rebuild(runs)
    code = fingerprint()
    runner = None
    if PATH.exists():
        previous = json.loads(PATH.read_text(encoding="utf-8"))
        if previous.get("code") == code:
            runner = previous.get("runner")
    return {
        "version": 1,
        "claim": "C6",
        "tolerance": {"name": TOLERANCE, "value": tolerance()},
        "code": code,
        "machine": describe(),
        "traced": traced,
        "controls": controls,
        "local": local,
        "runner": runner,
        "seconds": round(time.monotonic() - started, 1),
        "not_measured": NOT_MEASURED,
    }


def record(directory: pathlib.Path) -> dict[str, Any]:
    """Rebuilds the bundles a `build` folder holds (the clean-runner job).

    Args:
        directory: A folder written by `build`, as downloaded.

    Returns:
        The clean-runner record: the code hash, this machine, and per task
        its trace record and its rebuild. A task whose run cannot be found
        or read is recorded as untraced and not rebuilt.
    """
    runs, traced, failed = {}, {}, {}
    for kind in reproducibility.TASKS:
        try:
            runs[kind] = run(directory / kind)
            traced[kind] = check(runs[kind] / "model")
        except (errors.ArtifactError, errors.InputError) as error:
            runs.pop(kind, None)
            traced[kind] = untraced(error)
            failed[kind] = unbuilt(error)
    return {
        "code": fingerprint(),
        "machine": describe(),
        "traced": traced,
        "rebuilt": {**rebuild(runs), **failed},
    }


def merge(path: pathlib.Path) -> dict[str, Any]:
    """Stores a clean-runner record in the document.

    Args:
        path: The record `--rebuild` wrote.

    Returns:
        The updated document.

    Raises:
        InputError: If there is no document, the record is not a clean-runner
            record, or it is for other code than the document.
    """
    document = read(None)
    found = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(found, dict) or not all(
        isinstance(found.get(name), dict) for name in ("traced", "rebuilt")
    ):
        raise errors.InputError(
            "the clean-runner record has no trace and rebuild records"
        )
    if found.get("code") != document.get("code"):
        raise errors.InputError(
            "the clean-runner record is for other code than the document"
        )
    document["runner"] = found
    return document


def read(document: dict[str, Any] | None) -> dict[str, Any]:
    """The given document, or the stored one.

    Args:
        document: A document, or None to read `PATH`.

    Returns:
        The document.

    Raises:
        InputError: If `PATH` is missing or the document is malformed.
    """
    if document is None:
        if not PATH.exists():
            raise errors.InputError(f"{PATH.name} is missing")
        document = json.loads(PATH.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(
        document.get("traced"), dict
    ):
        raise errors.InputError("the document has no trace records")
    return document


def problems(document: dict[str, Any] | None = None) -> list[str]:
    """Why the stored measurements are not for the current code.

    Args:
        document: The measurements document; read from `PATH` when None.

    Returns:
        A message for a missing or malformed document, a code hash or
        tolerance that differs from now, or a task not measured; empty when
        current.
    """
    try:
        found = read(document)
    except errors.InputError as error:
        return [f"traceability: {error}"]
    issues = []
    if found.get("code") != fingerprint():
        issues.append("traceability: measurements are for other code")
    if found.get("tolerance", {}).get("value") != tolerance():
        issues.append(f"traceability: {TOLERANCE} differs from acceptance.toml")
    missing = sorted(set(reproducibility.TASKS) - set(found["traced"]))
    if missing:
        issues.append(f"traceability: tasks not traced: {', '.join(missing)}")
    return issues


def tracing(found: dict[str, Any]) -> tuple[str, str]:
    """The trace clause's value text and verdict.

    Args:
        found: A current document.

    Returns:
        The text and "pass" or "fail".
    """
    traced = found["traced"]
    broken = {kind: failing(item) for kind, item in traced.items()}
    broken = {kind: fields for kind, fields in broken.items() if fields}
    controls = found["controls"]
    missed = sorted(
        name for name, item in controls.items() if not item["caught"]
    )
    text = (
        f"{len(traced) - len(broken)}/{len(traced)} bundles from real runs "
        "trace to data hash, configuration, commit and environment, each "
        f"verified; {len(controls) - len(missed)}/{len(controls)} tamperings "
        "caught by exactly the field they break"
    )
    for kind, fields in sorted(broken.items()):
        text += f"; {kind} fails {', '.join(fields)}"
    if missed:
        text += f"; missed: {', '.join(missed)}"
    passed = not broken and not missed and len(controls) == len(CONTROLS)
    return text, "pass" if passed else "fail"


def rebuilding(found: dict[str, Any]) -> tuple[str, str]:
    """The rebuild clause's value text and verdict.

    Args:
        found: A current document.

    Returns:
        The text and "pass", "fail" or "not measured" (no clean-runner
        record yet and no local failure).
    """
    limit = float(found["tolerance"]["value"])
    parts, failed, broken = [], False, []
    sides = [("this machine", found["local"])]
    runner = found.get("runner")
    if runner:
        sides.append(("a clean runner", runner["rebuilt"]))
        expected = set(reproducibility.TASKS)
        failed = bool(expected - set(runner["rebuilt"]))
        failed = failed or bool(expected - set(runner["traced"]))
        broken = sorted(
            kind for kind, item in runner["traced"].items() if failing(item)
        )
        failed = failed or bool(broken)
    for where, rebuilt in sides:
        identical = [k for k, r in rebuilt.items() if r["identical"]]
        reproduced = [k for k, r in rebuilt.items() if r["reproduced"]]
        worst = max(
            (r["gap"]["relative"] for r in rebuilt.values()), default=0.0
        )
        failed = failed or len(reproduced) < len(rebuilt) or not rebuilt
        parts.append(
            f"{where}: {len(reproduced)}/{len(rebuilt)} bundles rebuilt from "
            f"the manifest alone reproduce ({len(identical)} byte-identical; "
            f"max metric relative difference {worst:.3g}, tol {limit:g})"
        )
    if broken:
        parts.append(f"a clean runner: {', '.join(broken)} do not trace there")
    if failed:
        return "; ".join(parts), "fail"
    if not runner:
        parts.append("clean runner not measured")
        return "; ".join(parts), "not measured"
    return "; ".join(parts), "pass"


def summary(
    document: dict[str, Any] | None = None, clause: str = TRACE
) -> tuple[str, str]:
    """The scorecard's "Value now" text and verdict for one C6 clause.

    Args:
        document: The measurements document; read from `PATH` when None.
        clause: `TRACE` or `REBUILD`.

    Returns:
        The value text and "pass", "fail" or "not measured". Stale or
        missing measurements are "not measured".

    Raises:
        InputError: If `clause` is not one of `CLAUSES`.
    """
    if clause not in CLAUSES:
        raise errors.InputError(f"unknown clause {clause!r}; known: {CLAUSES}")
    issues = problems(document)
    if issues:
        return "; ".join(issues), "not measured"
    found = read(document)
    return tracing(found) if clause == TRACE else rebuilding(found)


def store(document: dict[str, Any]) -> None:
    """Writes the document to `PATH` and prints both clauses.

    Args:
        document: The measurements document.
    """
    PATH.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for clause in CLAUSES:
        value, verdict = summary(document, clause)
        print(
            json.dumps({"clause": clause, "value": value, "verdict": verdict})
        )


def main(arguments: list[str] | None = None) -> int:
    """Measures, checks, or runs one side of the clean-runner rebuild.

    Args:
        arguments: Command-line arguments; None reads `sys.argv`.

    Returns:
        0 on success. `--check` returns 1 when the stored measurements are
        not current; `--rebuild` returns 1 when a bundle does not trace or
        does not reproduce (the record is written first).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--build", type=pathlib.Path, metavar="DIR")
    mode.add_argument("--rebuild", type=pathlib.Path, metavar="DIR")
    mode.add_argument("--merge", type=pathlib.Path, metavar="FILE")
    parser.add_argument("--record", type=pathlib.Path, metavar="FILE")
    options = parser.parse_args(arguments)
    if options.build:
        for kind, directory in build(options.build).items():
            print(f"{kind}: {directory}")
        return 0
    if options.rebuild:
        if options.record is None:
            parser.error("--rebuild needs --record")
        found = record(options.rebuild)
        options.record.write_text(
            json.dumps(found, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        good = all(not failing(item) for item in found["traced"].values())
        good = good and all(r["reproduced"] for r in found["rebuilt"].values())
        return 0 if good else 1
    if options.write:
        store(measure())
        return 0
    if options.merge:
        store(merge(options.merge))
        return 0
    issues = problems()
    print(json.dumps({"ok": not issues, "problems": issues}, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
