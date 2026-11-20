"""Finds what is stored about a dataset and plans erasing it.

A folder is recognised by the files scmle writes into it:

* a run: `config.json` and `dataset_manifest.json`;
* a model bundle: `model_manifest.json`, whose `dataset` is the training
  data's digest; inside a registry (`registry.json` two levels up) it is a
  registered version;
* a sealed trail: `trail.json`, with the run's records under
  `retained/run/`;
* a prediction log: `<output>.manifest.json` from `scmle batch`, whose
  `artifact_digest` names the bundle that scored it.

A dataset is matched by any of its digests: the row-content hash, the
snapshot identity, the dataset identifier or the raw file's SHA-256.
"""

from collections.abc import Iterable
import dataclasses
import json
import os
import pathlib

from scmle.api.erasure import steps
from scmle.artifacts.bundle import layout
from scmle.assurance.custody import journal
from scmle.contracts import errors
from scmle.contracts import records
from scmle.contracts.data import provenance
import scmle.deploy.provenance
from scmle.governance import registry
import scmle.trail.records

DATASET = "dataset_manifest.json"
CONFIG = "config.json"
MODEL = "model"
TRACKING = "experiments.sqlite"
LOG = ".manifest.json"
# The run records an erasure keeps. Each holds digests, ids, signatures and
# the run's own decisions, never row values, and together they show what was
# trained on which data, what was released and what was later erased: the
# provenance and attestations, the dataset's identity, the event and custody
# logs, and a charter run's hypothesis ledger, charter and confirmed claims.
CUSTODY = frozenset(
    {
        "provenance.json",
        "release_attestation.json",
        "evidence_graph.json",
        DATASET,
        "events.jsonl",
        "complete.json",
        "analysis_charter.json",
        "hypothesis_ledger.jsonl",
        "confirmed_claims.json",
        "custody_log.jsonl",
    }
)


def read(path: pathlib.Path) -> records.Layout:
    """Reads a JSON object scmle wrote.

    Args:
        path: The file.

    Returns:
        The object.

    Raises:
        ArtifactError: If the file cannot be read or is not an object; an
            erasure fails closed rather than skip what it cannot read.
    """
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise errors.ArtifactError(f"cannot read {path}: {error}") from error
    if not isinstance(found, dict):
        raise errors.ArtifactError(f"{path} is not a JSON object")
    return found


def identities(path: pathlib.Path) -> frozenset[str]:
    """Returns every digest a dataset manifest names.

    Args:
        path: A `dataset_manifest.json`, in any stored version.

    Returns:
        The row-content hash, identifier, snapshot and lineage digests.

    Raises:
        ArtifactError: If the manifest cannot be read.
    """
    try:
        manifest = provenance.Manifest.from_dict(read(path))
    except (KeyError, TypeError, errors.InputError) as error:
        raise errors.ArtifactError(f"cannot read {path}: {error}") from error
    found = {
        manifest.identifier,
        manifest.snapshot,
        manifest.lineage.snapshot,
        manifest.digests.get("data", ""),
        *manifest.lineage.digests.values(),
    }
    return frozenset(found - {""})


def registration(bundle: pathlib.Path) -> tuple[str, str, int]:
    """Finds the registry version a bundle directory is, if any.

    Args:
        bundle: The bundle directory.

    Returns:
        The registry root, model name and version; empty and 0 when the
        bundle is not a registered version.
    """
    root = bundle.resolve().parent.parent
    index = root / registry.INDEX_FILE
    if not index.is_file():
        return "", "", 0
    for name, entry in read(index).get("models", {}).items():
        for version, record in entry.get("versions", {}).items():
            if (root / record.get("path", "")).resolve() == bundle.resolve():
                return os.fspath(root), name, int(version)
    return "", "", 0


def model(bundle: pathlib.Path) -> list[steps.Step]:
    """Plans retiring a bundle and removing what it learned from the data.

    Args:
        bundle: The bundle directory.

    Returns:
        A `Mark` first, so the model is refused before anything is removed,
        then one `Removal` per member outside `steps.KEPT`.
    """
    root, name, version = registration(bundle)
    content = registry.identify(bundle)
    planned: list[steps.Step] = [
        steps.Mark(
            os.fspath(bundle.resolve()),
            {"content": content},
            root,
            name,
            version,
        )
    ]
    planned += [
        steps.Removal.of(member)
        for member in sorted(bundle.iterdir())
        if member.name not in steps.KEPT
    ]
    return planned


def dossier(run: pathlib.Path) -> list[steps.Step]:
    """Plans erasing a run directory, keeping its custody records.

    Args:
        run: The run directory.

    Returns:
        The model's steps, a `Custody` per kept record (the run's own and its
        custody journal beside the run), a `Removal` per other entry, and an
        `Unlisting` from the tracking database beside the run.
    """
    bundle = run / MODEL
    registered = (bundle / layout.MANIFEST_FILE).is_file()
    planned = model(bundle) if registered else []
    for entry in sorted(run.iterdir()):
        if entry.name in CUSTODY:
            planned.append(steps.Custody.of(entry))
        elif entry != bundle or not registered:
            planned.append(steps.Removal.of(entry))
    log = run.parent / journal.FOLDER / f"{run.name}.jsonl"
    if log.is_file():
        planned.append(steps.Custody.of(log))
    tracking = run.parent / TRACKING
    if tracking.is_file():
        planned.append(
            steps.Unlisting(os.fspath(tracking.resolve()), {}, run.name)
        )
    return planned


@dataclasses.dataclass
class Survey:
    """What a scan found about a set of digests.

    Attributes:
        runs: Run directories trained on the data.
        plan: The steps erasing what was found.
        bundles: Content digests of the bundles retired, which name the
            prediction logs they wrote.
        logs: Every prediction log manifest seen, matched after the scan.
    """

    runs: list[pathlib.Path] = dataclasses.field(default_factory=list)
    plan: list[steps.Step] = dataclasses.field(default_factory=list)
    bundles: set[str] = dataclasses.field(default_factory=set)
    logs: list[pathlib.Path] = dataclasses.field(default_factory=list)

    def visit(
        self, folder: pathlib.Path, names: list[str], wanted: frozenset[str]
    ) -> bool:
        """Plans what one folder holds about the data.

        Args:
            folder: The folder.
            names: The files directly in it.
            wanted: The dataset's digests.

        Returns:
            Whether the folder was recognised, so its contents need no scan.
        """
        self.logs += [folder / n for n in names if n.endswith(LOG)]
        if scmle.trail.records.TRAIL_FILE in names:
            retained = folder / scmle.trail.records.RETAINED / "run" / DATASET
            if retained.is_file() and identities(retained) & wanted:
                self.plan.append(steps.Custody.of(folder))
            return True
        if DATASET in names and CONFIG in names:
            if identities(folder / DATASET) & wanted:
                self.runs.append(folder)
                self.plan += dossier(folder)
                self.retired(folder / MODEL)
            return True
        if layout.MANIFEST_FILE in names:
            stored = read(folder / layout.MANIFEST_FILE).get("dataset", "")
            if stored and stored in wanted:
                self.plan += model(folder)
                self.retired(folder)
            return True
        return False

    def retired(self, bundle: pathlib.Path) -> None:
        """Remembers a retired bundle's digest to find its prediction logs.

        Args:
            bundle: The bundle directory.
        """
        if (bundle / layout.MANIFEST_FILE).is_file():
            self.bundles.add(scmle.deploy.provenance.digest(bundle))

    def predictions(self) -> None:
        """Plans removing every prediction log a retired bundle wrote."""
        for manifest in sorted(set(self.logs)):
            if read(manifest).get("artifact_digest") in self.bundles:
                output = manifest.with_name(manifest.name.removesuffix(LOG))
                self.plan += [
                    steps.Removal.of(output),
                    steps.Removal.of(manifest),
                ]


def scan(
    roots: Iterable[pathlib.Path],
    wanted: frozenset[str],
    excluded: pathlib.Path,
) -> Survey:
    """Finds every run, model, trail and prediction log of a dataset.

    Args:
        roots: The folders to search, with their sub-folders.
        wanted: The dataset's digests.
        excluded: A folder never searched (the erasure's own receipts).

    Returns:
        The runs found and the plan.

    Raises:
        InputError: If a root is not a folder.
    """
    survey = Survey()
    for root in roots:
        if not root.is_dir():
            raise errors.InputError(f"no such folder: {root}")
        for folder, names, files in os.walk(root):
            here = pathlib.Path(folder)
            if here.resolve() == excluded.resolve() or survey.visit(
                here, files, wanted
            ):
                names.clear()
            names.sort()
    survey.predictions()
    return survey
