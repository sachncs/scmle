"""Rebuilds a bundle's environment from its manifest and retrains it.

`rebuild` first requires the bundle to trace (`origin.trace`), so the data
and configuration it retrains from are the ones the manifest names. Then,
from the manifest alone, it:

1. creates a fresh virtual environment with ``uv`` at the manifest's Python
   version and installs exactly the manifest's pins (``--no-deps``: a pin
   the manifest forgot fails the rebuild rather than being resolved);
2. installs scmle editable from a fresh clone of the repository checked out
   at the manifest's commit (never the working tree), so the retrained
   bundle records that commit too;
3. retrains in a new process of that environment, isolated from the caller's
   ``sys.path``, with the run's stored configuration, the manifest's thread
   counts and the data staged where the configuration names it;
4. compares every bundle member byte for byte and the locked-test metrics
   within a relative tolerance.
"""

from collections.abc import Mapping
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess  # nosec B404
import tempfile
import time
from typing import Any

from scmle.artifacts.bundle import layout
from scmle.artifacts.bundle import seal
from scmle.contracts import errors
from scmle.trail import origin

# Policy: the relative metric tolerance when bundle bytes differ. The
# summation-order error of a float64 reduction over n terms is at most
# n * eps relative (Higham, Accuracy and Stability, 2nd ed., section 4.2);
# n = 2**32 terms bounds every metric a run computes, so 2**32 * 2.2e-16 =
# 9.5e-7, rounded up to 1e-6 (also `cross_platform_tol` in acceptance.toml).
TOLERANCE = 1e-6
# Policy: one retraining of a run may take its own time budget; installing
# and training a quick project took under five minutes when measured, so an
# hour bounds a hung process, not a slow one.
SECONDS = 3600.0
RUNS = "runs"
# Thread variables set from the manifest's thread counts, by library.
THREADS = {
    "openmp": "OMP_NUM_THREADS",
    "openblas": "OPENBLAS_NUM_THREADS",
    "mkl": "MKL_NUM_THREADS",
}
# Retrains a stored configuration with the rebuilt scmle: the arguments are
# the configuration file and the output directory.
PROGRAM = """
import pathlib, sys
from scmle.contracts import tasks
from scmle.pipeline import orchestrator
text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
run = orchestrator.Orchestrator(tasks.Config.model_validate_json(text)).run(
    sys.argv[2]
)
print(run.run_dir)
"""


def execute(
    command: list[str],
    what: str,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Runs one command of the rebuild.

    Args:
        command: The command and its arguments.
        what: What it does, for the error message.
        cwd: Its working directory.
        environment: Its environment variables; the caller's when None.

    Returns:
        Its standard output.

    Raises:
        ArtifactError: If it cannot start, times out or fails.
    """
    try:
        done = subprocess.run(  # noqa: S603 -- arguments built here  # nosec B603
            command,
            cwd=cwd,
            env=dict(environment) if environment is not None else None,
            capture_output=True,
            text=True,
            timeout=SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise errors.ArtifactError(f"{what} failed: {error}") from error
    if done.returncode != 0:
        raise errors.ArtifactError(
            f"{what} failed ({done.returncode}):\n{done.stderr[-4000:]}"
        )
    return done.stdout


def requirements(manifest: Mapping[str, Any]) -> list[str]:
    """The pins of a bundle's environment, without scmle itself.

    Args:
        manifest: The parsed bundle manifest.

    Returns:
        One ``name==version`` line per pinned distribution, sorted.

    Raises:
        ArtifactError: If the manifest records no environment.
    """
    pins = dict(manifest.get("environment") or {})
    if not pins:
        raise errors.ArtifactError("the manifest records no environment")
    return sorted(
        f"{name}=={version}"
        for name, version in pins.items()
        if origin.normalize(name) != "scmle"
    )


def interpreter(directory: Path) -> Path:
    """The Python executable of a virtual environment.

    Args:
        directory: The environment directory.

    Returns:
        ``Scripts/python.exe`` on Windows, else ``bin/python``.
    """
    if os.name == "nt":
        return directory / "Scripts" / "python.exe"
    return directory / "bin" / "python"


def checkout(commit: str, repository: Path, target: Path) -> Path:
    """Clones a repository and checks one commit out, leaving it untouched.

    Args:
        commit: The full commit id.
        repository: The repository holding it.
        target: The directory to clone into (must not exist).

    Returns:
        `target`.

    Raises:
        ArtifactError: If git is missing or the clone or checkout fails.
    """
    executable = shutil.which("git")
    if executable is None:
        raise errors.ArtifactError("git is not installed; cannot check out")
    execute(
        [
            executable,
            "clone",
            "--quiet",
            "--no-checkout",
            str(repository),
            str(target),
        ],
        "cloning the repository",
    )
    execute(
        [
            executable,
            "-C",
            str(target),
            "checkout",
            "--quiet",
            "--detach",
            commit,
        ],
        f"checking out {commit}",
    )
    return target


def provision(
    manifest: Mapping[str, Any], directory: Path, repository: Path
) -> Path:
    """Builds a fresh environment from the manifest alone.

    Args:
        manifest: The parsed bundle manifest.
        directory: A new directory for the environment, the pins and the
            checked-out source.
        repository: The repository holding the manifest's commit.

    Returns:
        The environment's Python executable.

    Raises:
        ArtifactError: If uv is missing or any step fails.
    """
    uv = shutil.which("uv")
    if uv is None:
        raise errors.ArtifactError("uv is not installed; cannot rebuild")
    environment = directory / "environment"
    execute(
        [uv, "venv", "--python", str(manifest["python"]), str(environment)],
        "creating the environment",
    )
    python = interpreter(environment)
    pins = directory / "requirements.txt"
    pins.write_text("\n".join(requirements(manifest)) + "\n", encoding="utf-8")
    execute(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            "-r",
            str(pins),
        ],
        "installing the pinned environment",
    )
    code = checkout(str(manifest["commit"]), repository, directory / "source")
    execute(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            "-e",
            str(code),
        ],
        "installing scmle at the commit",
    )
    return python


def stage(report: origin.Report, uri: str, workspace: Path) -> None:
    """Puts the verified data where the configuration names it.

    Args:
        report: The trace of the bundle, naming the verified data file.
        uri: The data source in the run's configuration.
        workspace: The working directory of the retraining.

    Raises:
        InputError: If the run read an in-memory frame, or names an absolute
            path that does not hold the verified file.
    """
    if not uri:
        raise errors.InputError(
            "the run read an in-memory frame; only a run that read a file "
            "can be retrained from its configuration"
        )
    written = Path(uri)
    if written.is_absolute():
        if written.resolve() != Path(report.data).resolve():
            raise errors.InputError(
                f"the configuration names {written}; place the verified "
                f"data ({report.data}) there to retrain"
            )
        return
    target = workspace / written
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(report.data, target)


def threads(manifest: Mapping[str, Any]) -> dict[str, str]:
    """The caller's environment with the manifest's thread counts.

    Args:
        manifest: The parsed bundle manifest.

    Returns:
        The variables for the retraining process: the caller's, without
        ``PYTHONPATH`` and ``VIRTUAL_ENV``, and one thread variable per
        library in the manifest's `threads`.
    """
    variables = {
        name: value
        for name, value in os.environ.items()
        if name not in {"PYTHONPATH", "VIRTUAL_ENV"}
    }
    for library, count in dict(manifest.get("threads") or {}).items():
        if library in THREADS:
            variables[THREADS[library]] = str(count)
    return variables


def retrain(
    python: Path,
    config: Path,
    workspace: Path,
    variables: Mapping[str, str],
) -> Path:
    """Trains the stored configuration again with the rebuilt environment.

    Args:
        python: The rebuilt environment's Python.
        config: The run's ``config.json``.
        workspace: The working directory, holding the staged data.
        variables: The process environment.

    Returns:
        The new run directory.

    Raises:
        ArtifactError: If the training fails or names no run directory.
    """
    printed = execute(
        [str(python), "-I", "-c", PROGRAM, str(config), RUNS],
        "retraining",
        cwd=workspace,
        environment=variables,
    )
    lines = printed.strip().splitlines()
    run = Path(lines[-1]) if lines else Path()
    if not run.is_absolute():
        run = workspace / run
    if not lines or not (run / "model").is_dir():
        raise errors.ArtifactError(f"the retraining named no run: {printed!r}")
    return run


def gap(
    first: Mapping[str, float], second: Mapping[str, float]
) -> dict[str, float]:
    """The largest absolute and relative gap between two sets of metrics.

    Args:
        first: Metric name to value.
        second: Metric name to value.

    Returns:
        `absolute` and `relative`; a pair's relative gap is its absolute gap
        over the larger magnitude (zero when both are zero). A metric only
        one side has, or that is not finite, gives an infinite gap.
    """
    largest, ratio = 0.0, 0.0
    for name in sorted(set(first) | set(second)):
        if name not in first or name not in second:
            return {"absolute": float("inf"), "relative": float("inf")}
        one, two = float(first[name]), float(second[name])
        if math.isnan(one) or math.isnan(two):
            if not (math.isnan(one) and math.isnan(two)):
                return {"absolute": float("inf"), "relative": float("inf")}
            continue
        difference = abs(one - two)
        scale = max(abs(one), abs(two))
        largest = max(largest, difference)
        ratio = max(ratio, difference / scale if scale else 0.0)
    return {"absolute": largest, "relative": ratio}


def metrics(run: Path) -> dict[str, float]:
    """The locked-test metrics of a run.

    Args:
        run: The run directory.

    Returns:
        Metric name to value from its ``evaluation.json``; a metric stored
        as null (undefined) is NaN.

    Raises:
        ArtifactError: If the run has no readable evaluation.
    """
    path = run / origin.EVALUATION
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise errors.ArtifactError(
            f"{path} is not readable: {error}"
        ) from error
    values = found.get("metrics") if isinstance(found, dict) else None
    if not isinstance(values, dict) or not all(
        value is None or isinstance(value, int | float)
        for value in values.values()
    ):
        raise errors.ArtifactError(f"{path} records no numeric metrics")
    return {
        name: math.nan if value is None else float(value)
        for name, value in values.items()
    }


def members(bundle: Path, manifest: Mapping[str, Any]) -> list[str]:
    """The bundle members a rebuild compares.

    Args:
        bundle: The original bundle directory.
        manifest: Its parsed manifest.

    Returns:
        Every file the manifest digests, the manifest itself and, when the
        bundle is signed, the signature.
    """
    names = {*dict(manifest.get("files", {})), layout.MANIFEST_FILE}
    if (bundle / seal.SIGNATURE_FILE).is_file():
        names.add(seal.SIGNATURE_FILE)
    return sorted(names)


def digest(path: Path) -> str:
    """The SHA-256 of a file, or empty when it is absent.

    Args:
        path: The file.

    Returns:
        The hex digest, or an empty string.
    """
    return (
        hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
    )


def compare(original: Path, rebuilt: Path, names: list[str]) -> tuple[str, ...]:
    """The members whose bytes differ between two bundles.

    Args:
        original: The original bundle directory.
        rebuilt: The rebuilt bundle directory.
        names: The members to compare (`members`).

    Returns:
        The members that differ or that only one side has, and any file the
        rebuilt bundle has beyond `names` at any depth (as a path relative
        to the bundle), sorted.
    """
    extra = {
        path.relative_to(rebuilt).as_posix()
        for path in rebuilt.rglob("*")
        if path.is_file()
    } - set(names)
    differing = {
        name
        for name in names
        if digest(original / name) != digest(rebuilt / name)
    }
    return tuple(sorted(differing | extra))


@dataclasses.dataclass(frozen=True)
class Rebuilt:
    """The outcome of rebuilding one bundle.

    Attributes:
        bundle: The original bundle directory.
        python: The rebuilt interpreter's version text.
        pins: The number of distributions installed from the manifest.
        commit: The commit scmle was installed from.
        members: The members compared.
        differing: Members whose bytes differ, or that only one side has.
        original: The original run's locked-test metrics.
        replayed: The rebuilt run's locked-test metrics.
        gap: The largest absolute and relative metric gap.
        tolerance: The relative metric tolerance.
        seconds: Wall-clock seconds the rebuild took.
    """

    bundle: str
    python: str
    pins: int
    commit: str
    members: tuple[str, ...]
    differing: tuple[str, ...]
    original: dict[str, float]
    replayed: dict[str, float]
    gap: dict[str, float]
    tolerance: float
    seconds: float

    @property
    def identical(self) -> bool:
        """Whether every compared member is byte-identical."""
        return not self.differing

    @property
    def within(self) -> bool:
        """Whether the metrics agree within the tolerance."""
        return self.gap["relative"] <= self.tolerance

    @property
    def reproduced(self) -> bool:
        """Whether the bytes are identical or the metrics agree."""
        return self.identical or self.within

    def dump(self) -> dict[str, Any]:
        """A JSON-ready record.

        Returns:
            The attributes with `identical`, `within` and `reproduced`.
        """
        record = dataclasses.asdict(self)
        record["members"] = list(self.members)
        record["differing"] = list(self.differing)
        record["identical"] = self.identical
        record["within"] = self.within
        record["reproduced"] = self.reproduced
        return record


def rebuild(
    bundle: str | Path,
    *,
    run: str | Path | None = None,
    repository: str | Path | None = None,
    data: str | Path | None = None,
    key: str | bytes | None = None,
    workspace: str | Path | None = None,
    tolerance: float = TOLERANCE,
) -> Rebuilt:
    """Rebuilds a bundle's environment from its manifest and retrains it.

    Args:
        bundle: The bundle directory.
        run: The run directory that built it (see `origin.trace`).
        repository: The git repository holding the bundle's commit.
        data: The data file, when the recorded source cannot be found.
        key: The signing key; the retraining signs with it too.
        workspace: A directory for the environment and the retraining; a
            temporary one, removed afterwards, when None.
        tolerance: The relative metric tolerance (see `TOLERANCE`).

    Returns:
        The comparison of the rebuilt bundle with the original.

    Raises:
        ArtifactError: If the bundle does not trace, or a rebuild step fails.
        InputError: If the run read an in-memory frame or its data cannot be
            staged where its configuration names it.
    """
    started = time.monotonic()
    report = origin.trace(
        bundle, run=run, repository=repository, data=data, key=key
    )
    if not report.traced:
        raise errors.ArtifactError(
            "the bundle does not trace, so it cannot be rebuilt: "
            + "; ".join(report.problems())
        )
    folder = Path(bundle)
    manifest = origin.load(folder)
    source = Path(repository) if repository is not None else origin.source()
    with tempfile.TemporaryDirectory(prefix="scmle-rebuild-") as scratch:
        root = Path(workspace) if workspace is not None else Path(scratch)
        root.mkdir(parents=True, exist_ok=True)
        python = provision(manifest, root / "build", source)
        version = execute([str(python), "-I", "-VV"], "reading the interpreter")
        config = Path(report.run) / origin.CONFIG
        uri = str(
            json.loads(config.read_text(encoding="utf-8"))["data"]["uri"] or ""
        )
        work = root / "work"
        work.mkdir(parents=True, exist_ok=True)
        stage(report, uri, work)
        variables = threads(manifest)
        if key is not None:
            variables["SCMLE_SIGNING_KEY"] = (
                key.decode() if isinstance(key, bytes) else key
            )
        replayed = retrain(python, config, work, variables)
        names = members(folder, manifest)
        differing = compare(folder, replayed / "model", names)
        original = metrics(Path(report.run))
        again = metrics(replayed)
        return Rebuilt(
            str(folder),
            version.strip(),
            len(requirements(manifest)),
            str(manifest["commit"]),
            tuple(names),
            differing,
            original,
            again,
            gap(original, again),
            tolerance,
            round(time.monotonic() - started, 1),
        )
