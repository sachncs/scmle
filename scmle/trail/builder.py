"""Builds a trail folder from the records of one deployed model."""

from collections.abc import Mapping
from pathlib import Path
import shutil

from scmle.trail import chain
from scmle.trail import records


def copy_present(source: Path, target: Path, names: tuple[str, ...]) -> None:
    """Copies the named files that exist, keeping their relative paths.

    Args:
        source: Directory the files are read from.
        target: Directory the files are copied into; subdirectories are created
            as needed.
        names: Paths relative to `source`; those that are not files are skipped.
    """
    for name in names:
        found = source / name
        if found.is_file():
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(found, destination)


def build(
    output: str | Path,
    *,
    run_dir: str | Path,
    framing_dir: str | Path,
    registry_dir: str | Path,
    deployment_dir: str | Path,
    inputs_file: str | Path,
    keys: Mapping[str, str | bytes | None],
) -> chain.Report:
    """Copies the retained records and `inputs.json`, and seals the trail.

    Only the files named in `records` that exist are copied; missing ones are
    skipped, and the trial files are taken from `run_dir/trials`.

    Args:
        output: Trail folder to write (created if absent).
        run_dir: The charter run directory.
        framing_dir: The ``scmle frame`` output directory.
        registry_dir: The local registry root.
        deployment_dir: The loop's deployment directory, which also holds
            ``observed_outcomes.json``.
        inputs_file: The signed ``inputs.json``; it must exist.
        keys: Keys by role (``analyst`` signs the trail; the others are used
            to check the signed sections of `inputs.json`).

    Returns:
        The verification of the trail just written.
    """
    folder = Path(output)
    base = folder / records.RETAINED
    copy_present(Path(run_dir), base / "run", records.RUN_FILES)
    for trial in sorted((Path(run_dir) / "trials").glob("trial_*.json")):
        copy_present(Path(run_dir), base / "run", (f"trials/{trial.name}",))
    copy_present(Path(framing_dir), base / "framing", records.FRAMING_FILES)
    copy_present(Path(registry_dir), base / "registry", records.REGISTRY_FILES)
    copy_present(
        Path(deployment_dir), base / "deployment", records.DEPLOYMENT_FILES
    )
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(inputs_file, folder / records.INPUTS)
    return chain.seal(folder, keys)
