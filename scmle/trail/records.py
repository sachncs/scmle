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
    """SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: Any) -> str:
    """SHA-256 of the canonical JSON of a JSON-native value."""
    return hashlib.sha256(signing.canonical_payload(value)).hexdigest()


def read_json(path: Path) -> Any:
    """Reads a JSON file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If it is not JSON.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path: Path) -> list[dict[str, Any]]:
    """Reads a JSON-lines file."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    """Writes sorted, indented JSON with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def lookup(document: Any, dotted: str) -> Any:
    """Follows ``a.b.0.c`` through mappings and lists.

    Raises:
        KeyError: If a component is missing.
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

    Raises:
        ValueError: For an unknown operator.
    """
    if op not in OPERATORS:
        raise ValueError(f"unknown operator {op!r}")
    return bool(OPERATORS[op](value, bound))


def resolve(directory: Path, spec: str, derived: Mapping[str, Any]) -> Any:
    """Reads ``file#dotted.path``.

    The file is relative to ``retained/``; the pseudo-files ``outcomes``
    and ``exploratory`` name the derived reports.

    Raises:
        KeyError: If the path is missing in the document.
        FileNotFoundError: If the file is missing.
    """
    name, ignored, dotted = spec.partition("#")
    if name in derived:
        document = derived[name]
    else:
        document = read_json(directory / RETAINED / name)
    return lookup(document, dotted) if dotted else document
