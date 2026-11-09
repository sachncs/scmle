"""Layout of a trail folder and the small readers its modules share.

A trail folder holds ``trail.json`` (the chained steps), ``reports/`` (the
derived reports) and ``retained/`` (verbatim copies of the records the trail
was built from). Everything a reviewer needs is inside it; nothing is read
from the machine that produced the model.
"""

from collections.abc import Mapping
import hashlib
import json
import operator
from pathlib import Path
from typing import Any

from scmle.contracts import signing

FORMAT = 1
TRAIL_FILE = "trail.json"
RETAINED = "retained"
REPORTS = "reports"
INPUTS = "inputs.json"
EXPLORATORY = "reports/exploratory_report.json"
EXPERIMENTS = "reports/experiment_comparison.json"
OUTCOME_TRACKING = "reports/outcome_tracking.json"
ROLES = ("analyst", "stakeholder", "reviewer")
RUN_FILES = (
    "analysis_charter.json",
    "hypothesis_ledger.jsonl",
    "confirmed_claims.json",
    "negative_control.json",
    "baseline_report.json",
    "causal_claims.json",
    "custody_log.jsonl",
    "error_budget.json",
    "split_manifest.json",
    "evaluation.json",
    "curation_report.json",
    "dataset_manifest.json",
    "feasibility.json",
    "framing_link.json",
    "model/model_manifest.json",
)
FRAMING_FILES = (
    "framing_draft.json",
    "framing_questions.json",
    "decision_cost_model.json",
)
REGISTRY_FILES = ("registry.json", "audit.jsonl", "release_state.json")
DEPLOYMENT_FILES = (
    "deployment_ledger.jsonl",
    "loop_state.json",
    "observed_outcomes.json",
)
OPERATORS = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
}


def sha256_file(path: Path) -> str:
    """Digests a file's bytes.

    Args:
        path: The file.

    Returns:
        The hex SHA-256 digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: Any) -> str:
    """Digests the canonical JSON of a JSON-native value.

    Args:
        value: A JSON-native value.

    Returns:
        The hex SHA-256 digest.
    """
    return hashlib.sha256(signing.canonical_payload(value)).hexdigest()


def read_json(path: Path) -> Any:
    """Reads a JSON file.

    Args:
        path: The file.

    Returns:
        The parsed JSON value. A missing file raises `FileNotFoundError` and
        text that is not JSON raises `ValueError`.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path: Path) -> list[dict[str, Any]]:
    """Reads a JSON-lines file.

    Args:
        path: The file.

    Returns:
        One parsed object per non-blank line.
    """
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    """Writes sorted, indented JSON with a trailing newline.

    Args:
        path: The file; its directory is created.
        value: A JSON-serializable value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def lookup(document: Any, dotted: str) -> Any:
    """Follows ``a.b.0.c`` through mappings and lists.

    A list index that is not an integer raises `ValueError` and one out of
    range raises `IndexError`.

    Args:
        document: A parsed JSON value.
        dotted: Path components separated by dots; numbers index lists.

    Returns:
        The value at the path.

    Raises:
        KeyError: If a mapping lacks a component, or the path continues into
            a value that is neither a mapping nor a list.
    """
    value = document
    for part in dotted.split("."):
        if isinstance(value, Mapping):
            value = value[part]
        elif isinstance(value, list):
            value = value[int(part)]
        else:
            raise KeyError(dotted)
    return value


def compare(value: Any, op: str, bound: Any) -> bool:
    """Applies one of the comparison operators in ``OPERATORS``.

    Args:
        value: The observed value.
        op: One of `>=`, `<=`, `>`, `<`, `==`.
        bound: The value compared against.

    Returns:
        The result of `value op bound`.

    Raises:
        ValueError: For an unknown operator.
    """
    if op not in OPERATORS:
        raise ValueError(f"unknown operator {op!r}")
    return bool(OPERATORS[op](value, bound))


def resolve(directory: Path, spec: str, derived: Mapping[str, Any]) -> Any:
    """Reads ``file#dotted.path``.

    The file is relative to ``retained/``; a name that is a key of `derived`
    (the tracer uses ``exploratory``, ``experiments`` and ``outcomes``) names
    that derived report instead.

    Args:
        directory: The trail folder.
        spec: The file name, optionally followed by `#` and a dotted path.
        derived: The derived reports by pseudo-file name.

    Returns:
        The whole document, or the value at the dotted path. A path missing in
        the document raises `KeyError` (see `lookup` for the list-index
        errors) and a missing file raises `FileNotFoundError`.
    """
    name, ignored, dotted = spec.partition("#")
    if name in derived:
        document = derived[name]
    else:
        document = read_json(directory / RETAINED / name)
    return lookup(document, dotted) if dotted else document
