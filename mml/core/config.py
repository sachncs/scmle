"""Reading and writing project configuration files."""

import json
import pathlib
from typing import Any

from mml.contracts import errors
from mml.contracts import tasks

SECTIONS = {
    "search": tasks.SearchSpec,
    "strategies": tasks.StrategySpec,
    "ensemble": tasks.EnsembleSpec,
    "feature_graph": tasks.FeatureGraphSpec,
    "selection": tasks.SelectionSpec,
    "decision": tasks.DecisionSpec,
}


def load_sections(path: str | pathlib.Path | None) -> dict[str, Any]:
    """Reads the optional JSON file of nested specification sections.

    Args:
        path: JSON file whose keys are section names.

    Returns:
        Section name to specification object; empty without a file.

    Raises:
        ConfigError: For unknown sections.
    """
    if path is None:
        return {}
    record = json.loads(pathlib.Path(path).read_text())
    unknown = set(record) - set(SECTIONS)
    if unknown:
        raise errors.ConfigError(
            f"unknown config sections {sorted(unknown)}; allowed: "
            f"{sorted(SECTIONS)}"
        )
    return {name: SECTIONS[name](**body) for name, body in record.items()}


def load_spec(path: str | pathlib.Path) -> tasks.ProjectSpec:
    """Reads a complete project specification (for example `config.json`).

    Args:
        path: JSON file written by `save_spec` or found in a run directory.

    Returns:
        The validated specification.

    Raises:
        ConfigError: If the file is not a valid specification.
    """
    try:
        return tasks.ProjectSpec.model_validate_json(
            pathlib.Path(path).read_text()
        )
    except ValueError as error:
        raise errors.ConfigError(
            f"invalid project specification: {error}"
        ) from error


def save_spec(spec: tasks.ProjectSpec, path: str | pathlib.Path) -> None:
    """Writes a specification as canonical, key-sorted JSON.

    Args:
        spec: The specification.
        path: Destination file.
    """
    pathlib.Path(path).write_text(spec.canonical_json())
