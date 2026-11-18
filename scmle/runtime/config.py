"""Reading project configuration files."""

import json
import pathlib
from typing import Any

from scmle.contracts import errors
from scmle.contracts import tasks

SECTIONS = {
    "search": tasks.Search,
    "strategies": tasks.Strategies,
    "models": tasks.Candidates,
    "ensemble": tasks.Stacking,
    "feature_graph": tasks.Derivation,
    "derivation": tasks.Derivation,
    "selection": tasks.Selection,
    "decision": tasks.Threshold,
}
KEYWORDS = {"derivation": "feature_graph"}
"""Stored section names whose `Project` keyword is still the field name."""


def overlay(path: str | pathlib.Path | None) -> dict[str, Any]:
    """Reads the optional JSON file of nested specification sections.

    Args:
        path: JSON file whose keys are section names, in either spelling
            (`derivation` or the older `feature_graph`).

    Returns:
        `Project` keyword to specification object; empty without a file.

    Raises:
        ConfigError: For unknown sections.
    """
    if path is None:
        return {}
    record = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    unknown = set(record) - set(SECTIONS)
    if unknown:
        raise errors.ConfigError(
            f"unknown config sections {sorted(unknown)}; allowed: "
            f"{sorted(SECTIONS)}"
        )
    return {
        KEYWORDS.get(name, name): SECTIONS[name](**body)
        for name, body in record.items()
    }
