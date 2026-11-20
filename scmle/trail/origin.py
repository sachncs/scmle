"""Whether a bundle traces to its data, configuration, commit and environment.

A bundle's manifest names the hash of the data it was trained on, the hash
of the run configuration, the git commit of the scmle source and every
installed distribution. `trace` checks that each is present and that each
can be verified against something other than the manifest itself:

* data hash: the run's dataset manifest records the same hash, the source
  file's bytes hash to the recorded raw-source digest, and, for a tabular
  file read whole, the parsed frame rehashes to the bundle's data hash;
* configuration: the run's ``config.json`` rehashes to the bundle's
  configuration hash (and to the run provenance's);
* commit: a full commit id that exists in the repository and whose
  ``pyproject.toml`` names the scmle version the bundle records;
* environment: the pins name scmle, scikit-learn, pandas and the tracked
  libraries at the versions the manifest records separately, and equal the
  lock the run's provenance kept.

The manifest itself is trusted only when every stored file matches its
digest and, when a key is available, its signature verifies.
"""

import abc
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess  # nosec B404
import tomllib
from typing import Any

import pydantic

import scmle
from scmle.artifacts.bundle import layout
from scmle.artifacts.bundle import provenance
from scmle.artifacts.bundle import seal
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.readers import tabular

# Policy: `git` answers in milliseconds; a hung git (a lock, a network
# filesystem) must not stall a check, so give up after this long.
GIT_SECONDS = 30.0
COMMIT = re.compile(r"[0-9a-f]{40}")
CONFIG = "config.json"
DATASET = "dataset_manifest.json"
PROVENANCE = "provenance.json"
EVALUATION = "evaluation.json"
RAW = "raw-source"
# Import names of tracked libraries whose distribution name differs.
DISTRIBUTIONS = {"imblearn": "imbalanced-learn", "sklearn": "scikit-learn"}
# Manifest keys that hold a distribution's version outside `libraries`.
VERSIONS = {"scmle": "scmle", "sklearn": "scikit-learn", "pandas": "pandas"}


def normalize(name: str) -> str:
    """The normalized distribution name a bundle's pins are keyed by.

    Args:
        name: A distribution or import name.

    Returns:
        The lower-case name with `_` and `.` replaced by `-`, mapped through
        `DISTRIBUTIONS`.
    """
    plain = re.sub(r"[-_.]+", "-", name).lower()
    return DISTRIBUTIONS.get(plain, plain)


def checksum(path: Path) -> str:
    """The SHA-256 of a file's bytes.

    Args:
        path: The file.

    Returns:
        The hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Runs one read-only git command in a repository.

    Args:
        repository: The repository directory.
        *arguments: The git arguments.

    Returns:
        The finished process; a missing commit shows as a non-zero code.

    Raises:
        ArtifactError: If git is not installed or does not answer within
            `GIT_SECONDS`.
    """
    executable = shutil.which("git")
    if executable is None:
        raise errors.ArtifactError("git is not installed; cannot check commit")
    try:
        return subprocess.run(  # noqa: S603 -- fixed git arguments  # nosec B603
            [executable, "-C", str(repository), *arguments],
            capture_output=True,
            text=True,
            timeout=GIT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise errors.ArtifactError(f"git failed: {error}") from error


@dataclasses.dataclass(frozen=True)
class Records:
    """The bundle and the run records a check reads.

    Attributes:
        bundle: The bundle directory.
        manifest: The parsed ``model_manifest.json`` (written keys).
        run: The run directory that built the bundle, or None.
        repository: The git repository the commit is looked up in.
        data: A data file given in place of the run's recorded source, or
            None to find the recorded one.
    """

    bundle: Path
    manifest: Mapping[str, Any]
    run: Path | None
    repository: Path
    data: Path | None = None

    def read(self, name: str) -> dict[str, Any] | None:
        """One JSON record of the run.

        Args:
            name: The file name inside the run directory.

        Returns:
            The parsed record, or None without a run directory or that file.

        Raises:
            ArtifactError: If the file exists but is not a JSON object.
        """
        if self.run is None or not (self.run / name).is_file():
            return None
        try:
            found = json.loads((self.run / name).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise errors.ArtifactError(
                f"cannot read {name}: {error}"
            ) from error
        if not isinstance(found, dict):
            raise errors.ArtifactError(f"{name} is not a JSON object")
        return found

    def locate(self) -> Path | None:
        """The data file the run read, found from its dataset manifest.

        A relative source is looked for in each parent of the run directory
        and then in the working directory; the first whose bytes match the
        recorded raw-source digest wins, else the first that exists.

        Returns:
            `data` when given; else the found file, or None when the run
            read an in-memory frame or the file is nowhere to be found.
        """
        if self.data is not None:
            return self.data
        dataset = self.read(DATASET) or {}
        uri = str(dataset.get("uri", "")).strip()
        if not uri or uri.startswith(("<", "memory:")):
            return None
        written = Path(uri)
        if written.is_absolute():
            return written if written.is_file() else None
        roots = [*(self.run.parents if self.run else ()), Path.cwd()]
        found = [root / written for root in roots if (root / written).is_file()]
        raw = dict(dataset.get("lineage", {}).get("digests", {})).get(RAW)
        matching = [path for path in found if raw and checksum(path) == raw]
        chosen = matching or found
        return chosen[0] if chosen else None


class Link(abc.ABC):
    """One trace field of a bundle and how it is verified.

    Attributes:
        name: The field's name in reports.
        key: The manifest key that carries it.
    """

    name = ""
    key = ""

    def value(self, records: Records) -> str:
        """The field's value as the manifest records it.

        Args:
            records: The bundle and run records.

        Returns:
            Its text (a count of pins for the environment), empty if absent.
        """
        found = records.manifest.get(self.key)
        return str(found) if found else ""

    @abc.abstractmethod
    def check(self, records: Records) -> list[str]:
        """Verifies the field against records other than the manifest.

        Args:
            records: The bundle and run records.

        Returns:
            One message per failed verification; empty when verified.
        """


class Data(Link):
    """The hash of the training data."""

    name = "data hash"
    key = "dataset"

    def check(self, records: Records) -> list[str]:
        """Rehashes the source the run read and matches its manifest.

        Args:
            records: The bundle and run records.

        Returns:
            Messages for a dataset manifest that is missing or records
            another hash, a source that cannot be found, raw bytes or a
            parsed frame that hash differently, or a source that can be
            rehashed neither way.
        """
        recorded = str(records.manifest.get(self.key, ""))
        dataset = records.read(DATASET)
        if dataset is None:
            return [f"the run has no {DATASET} to match the data hash"]
        problems = []
        written = str(dict(dataset.get("digests", {})).get("data", ""))
        if written != recorded:
            problems.append(
                f"{DATASET} records data hash {written!r}, the bundle "
                f"{recorded!r}"
            )
        path = records.locate()
        if path is None:
            uri = dataset.get("uri") or "an in-memory frame"
            return [*problems, f"the training data ({uri}) cannot be found"]
        lineage = dict(dict(dataset.get("lineage", {})).get("digests", {}))
        raw = str(lineage.get(RAW, ""))
        if raw and checksum(path) != raw:
            problems.append(f"{path} does not hash to the recorded raw source")
        if self.parsable(records, dataset):
            try:
                parsed = tabular.data_hash(tabular.read(path))
            except errors.DataError as error:
                return [*problems, f"{path} cannot be read: {error}"]
            if parsed != recorded:
                problems.append(
                    f"{path} rehashes to {parsed!r}, the bundle records "
                    f"{recorded!r}"
                )
        elif not raw:
            problems.append(
                f"{path} can be rehashed neither as raw bytes nor as a table"
            )
        return problems

    def parsable(self, records: Records, dataset: Mapping[str, Any]) -> bool:
        """Whether the bundle's data hash is the hash of the file read whole.

        It is not for other modalities, for forecasting runs (the frame is
        augmented before it is hashed) and for streamed runs (the frame is a
        sample).

        Args:
            records: The bundle and run records.
            dataset: The run's dataset manifest.

        Returns:
            True for a tabular file read whole.
        """
        config = records.read(CONFIG) or {}
        evaluation = records.read(EVALUATION) or {}
        return (
            dataset.get("modality", "tabular") == "tabular"
            and not dict(config.get("data", {})).get("horizon")
            and not evaluation.get("streamed")
        )


class Configuration(Link):
    """The hash of the run configuration."""

    name = "configuration"
    key = "configuration"

    def check(self, records: Records) -> list[str]:
        """Rehashes the run's stored configuration.

        Args:
            records: The bundle and run records.

        Returns:
            Messages for a missing or invalid ``config.json``, or one (or a
            provenance record) whose hash differs from the bundle's.
        """
        recorded = str(records.manifest.get(self.key, ""))
        stored = records.read(CONFIG)
        if stored is None:
            return [f"the run has no {CONFIG} to rehash"]
        try:
            rehashed = tasks.Config.model_validate(stored).config_hash()
        except (pydantic.ValidationError, errors.ConfigError) as error:
            return [f"{CONFIG} is not a valid configuration: {error}"]
        problems = []
        if rehashed != recorded:
            problems.append(
                f"{CONFIG} rehashes to {rehashed!r}, the bundle records "
                f"{recorded!r}"
            )
        kept = (records.read(PROVENANCE) or {}).get("config_hash")
        if kept is not None and kept != recorded:
            problems.append(f"{PROVENANCE} records configuration {kept!r}")
        return problems


class Commit(Link):
    """The git commit of the scmle source that built the bundle."""

    name = "commit"
    key = "commit"

    def check(self, records: Records) -> list[str]:
        """Looks the commit up and reads the version it declares.

        Args:
            records: The bundle and run records.

        Returns:
            Messages for a value that is not a full commit id, a commit the
            repository does not hold, or one whose ``pyproject.toml``
            declares another scmle version than the bundle records.
        """
        commit = str(records.manifest.get(self.key, ""))
        if not COMMIT.fullmatch(commit):
            return [f"{commit!r} is not a full commit id"]
        found = git(
            records.repository, "cat-file", "-e", f"{commit}^{{commit}}"
        )
        if found.returncode != 0:
            return [f"commit {commit} is not in {records.repository}"]
        shown = git(records.repository, "show", f"{commit}:pyproject.toml")
        if shown.returncode != 0:
            return [f"commit {commit} has no pyproject.toml"]
        try:
            declared = tomllib.loads(shown.stdout)["project"]["version"]
        except (tomllib.TOMLDecodeError, KeyError) as error:
            return [f"commit {commit} declares no version: {error}"]
        built = records.manifest.get("scmle", "")
        if declared != built:
            return [f"commit {commit} declares {declared}, the bundle {built}"]
        return []


class Environment(Link):
    """Every distribution installed when the bundle was built."""

    name = "environment"
    key = "environment"

    def value(self, records: Records) -> str:
        """The number of pinned distributions.

        Args:
            records: The bundle and run records.

        Returns:
            For example ``105 pins``; empty when none are recorded.
        """
        pins = records.manifest.get(self.key) or {}
        return f"{len(pins)} pins" if pins else ""

    def check(self, records: Records) -> list[str]:
        """Matches the pins with the manifest's other version records.

        Args:
            records: The bundle and run records.

        Returns:
            Messages for a missing Python version, a pin that disagrees with
            (or lacks) scmle, scikit-learn, pandas or a tracked library, or a
            run lock that differs from the pins.
        """
        pins = {
            normalize(name): str(version)
            for name, version in dict(
                records.manifest.get(self.key) or {}
            ).items()
        }
        problems = []
        if not records.manifest.get("python"):
            problems.append("the manifest records no Python version")
        expected = {
            normalize(VERSIONS[key]): str(records.manifest.get(key, ""))
            for key in VERSIONS
        }
        for name, version in dict(
            records.manifest.get("libraries", {})
        ).items():
            expected[normalize(name)] = str(version)
        for name, version in sorted(expected.items()):
            if pins.get(name) != version:
                problems.append(
                    f"{name} is pinned at {pins.get(name)!r}, recorded "
                    f"{version!r}"
                )
        kept = dict((records.read(PROVENANCE) or {}).get("environment", {}))
        if "lock" in kept:
            rendered = provenance.Manifest.read(records.manifest).lock()
            if kept["lock"] != rendered:
                problems.append(
                    f"the lock in {PROVENANCE} differs from the pins"
                )
        return problems


LINKS: tuple[Link, ...] = (Data(), Configuration(), Commit(), Environment())


@dataclasses.dataclass(frozen=True)
class Check:
    """The outcome for one trace field.

    Attributes:
        name: The field (`Link.name`).
        value: What the manifest records, empty when absent.
        problems: Why it does not verify; empty when it does.
    """

    name: str
    value: str
    problems: tuple[str, ...]

    @property
    def present(self) -> bool:
        """Whether the manifest records the field."""
        return bool(self.value)

    @property
    def verified(self) -> bool:
        """Whether the field is present and every verification passed."""
        return self.present and not self.problems


@dataclasses.dataclass(frozen=True)
class Report:
    """Whether one bundle traces to its four fields.

    Attributes:
        bundle: The bundle directory.
        run: The run directory read, empty when none was found.
        data: The data file rehashed, empty when none was found.
        checks: One outcome per trace field, in `LINKS` order.
        integrity: Stored files that do not match the manifest, or an
            invalid signature; empty when the manifest is intact.
        signed: Whether a signature verified.
    """

    bundle: str
    run: str
    data: str
    checks: tuple[Check, ...]
    integrity: tuple[str, ...]
    signed: bool

    @property
    def traced(self) -> bool:
        """Whether the manifest is intact and every field verifies."""
        return not self.integrity and all(c.verified for c in self.checks)

    def problems(self) -> list[str]:
        """Every reason the bundle does not trace.

        Returns:
            The integrity problems, then ``field: absent`` for a field the
            manifest lacks and ``field: problem`` for each failed
            verification; empty when the bundle traces.
        """
        found = list(self.integrity)
        for check in self.checks:
            if not check.present:
                found.append(f"{check.name}: absent")
            found.extend(
                f"{check.name}: {problem}" for problem in check.problems
            )
        return found

    def dump(self) -> dict[str, Any]:
        """A JSON-ready record.

        Returns:
            The attributes, each check with its `present` and `verified`,
            and `traced`.
        """
        return {
            "bundle": self.bundle,
            "run": self.run,
            "data": self.data,
            "signed": self.signed,
            "integrity": list(self.integrity),
            "checks": {
                check.name: {
                    "value": check.value,
                    "present": check.present,
                    "verified": check.verified,
                    "problems": list(check.problems),
                }
                for check in self.checks
            },
            "traced": self.traced,
        }


def inspect(
    bundle: Path, manifest: Mapping[str, Any], key: Any
) -> tuple[list[str], bool]:
    """Checks the stored files against the manifest and its signature.

    Args:
        bundle: The bundle directory.
        manifest: Its parsed manifest.
        key: The signing key, or None to use the environment's.

    Returns:
        The integrity problems and whether a signature verified.
    """
    problems = []
    for name, digest in sorted(dict(manifest.get("files", {})).items()):
        path = bundle / name
        if not path.is_file():
            problems.append(f"{name} is missing from the bundle")
        elif checksum(path) != digest:
            problems.append(f"{name} does not match its manifest digest")
    try:
        signed = seal.verify(bundle, key)
    except errors.ArtifactError as error:
        problems.append(str(error))
        signed = False
    return problems, signed


def load(bundle: Path) -> dict[str, Any]:
    """Reads a bundle's manifest.

    Args:
        bundle: The bundle directory.

    Returns:
        The parsed manifest.

    Raises:
        ArtifactError: If the directory holds no readable manifest.
    """
    path = bundle / layout.MANIFEST_FILE
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise errors.ArtifactError(
            f"{bundle} holds no readable {layout.MANIFEST_FILE}: {error}"
        ) from error
    if not isinstance(found, dict):
        raise errors.ArtifactError(f"{path} is not a JSON object")
    return found


def source() -> Path:
    """The repository holding the running scmle source.

    Returns:
        The parent of the `scmle` package directory.
    """
    return Path(scmle.__file__).resolve().parent.parent


def trace(
    bundle: str | Path,
    *,
    run: str | Path | None = None,
    repository: str | Path | None = None,
    data: str | Path | None = None,
    key: str | bytes | None = None,
    links: Sequence[Link] = LINKS,
) -> Report:
    """Checks that a bundle traces to data, configuration, commit, environment.

    Args:
        bundle: The bundle directory (``model/`` of a run, or a saved copy).
        run: The run directory that built it; defaults to the bundle's
            parent when that holds a ``config.json``.
        repository: The git repository the commit is looked up in; defaults
            to the one holding the running scmle source.
        data: A data file to rehash in place of the run's recorded source
            (for a run that read an in-memory frame, or a moved file).
        key: The signing key; defaults to ``$SCMLE_SIGNING_KEY``.
        links: The fields to check.

    Returns:
        The report.

    Raises:
        ArtifactError: If the bundle has no readable manifest.
        InputError: If `run` or `data` is given but does not exist.
    """
    folder = Path(bundle)
    manifest = load(folder)
    if run is None and (folder.parent / CONFIG).is_file():
        run = folder.parent
    for given in (run, data):
        if given is not None and not Path(given).exists():
            raise errors.InputError(f"{given} does not exist")
    records = Records(
        folder,
        manifest,
        Path(run) if run is not None else None,
        Path(repository) if repository is not None else source(),
        Path(data) if data is not None else None,
    )
    integrity, signed = inspect(folder, manifest, key)
    checks = tuple(
        Check(
            link.name,
            link.value(records),
            tuple(link.check(records)) if link.value(records) else (),
        )
        for link in links
    )
    located = records.locate()
    return Report(
        str(folder),
        str(records.run or ""),
        str(located or ""),
        checks,
        tuple(integrity),
        signed,
    )
