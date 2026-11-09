"""Benchmark definitions for the breadth slices of Phase 18 (F15 item 15).

Each file in this folder defines, before the slice is built, how its
robustness, drift, explanation and monitoring will be measured: the damage or
signal, the ground truth, the metric and the rule that passes. None has been
run, no parameter is fixed (the deployment owner fixes them first), and the
presence of a file says nothing about support. Status of every slice:
``docs/BREADTH.md``.
"""

import json
from pathlib import Path
from typing import Any

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]
SLICES = (
    "accelerated_training",
    "association",
    "audio",
    "clustering",
    "dimensionality_reduction",
    "document_nlp",
    "forecasting",
    "geospatial",
    "graph",
    "image",
    "pretrained_multimodal",
    "ranking",
    "streaming",
    "video",
)
SECTIONS = ("robustness", "drift", "explanation", "monitoring")
SECTION_FIELDS = (
    "question",
    "perturbation_or_signal",
    "truth",
    "metric",
    "acceptance_rule",
    "parameters",
    "status",
)
BUILT = ("none", "library_only", "tabular_beta_path", "experimental_lifecycle")
LIFECYCLE = (
    "ingestion",
    "evaluation",
    "packaging",
    "deployment",
    "monitoring",
    "recovery",
)
TOP_FIELDS = (
    "schema_version",
    "slice",
    "status",
    "task_family",
    "modality",
    "built",
    "evidence_in_repo",
    "deployment_requirement",
    "external_referent",
    "benchmarks",
    "lifecycle_required",
    "parameter_rule",
    "not_claimed",
)


def load(name: str) -> dict[str, Any]:
    """Reads one slice definition."""
    found: dict[str, Any] = json.loads(
        (FOLDER / f"{name}.json").read_text(encoding="utf-8")
    )
    return found


def load_all() -> dict[str, dict[str, Any]]:
    """Reads every slice definition, by slice name."""
    return {name: load(name) for name in SLICES}


def section_problems(name: str, kind: str, section: Any) -> list[str]:
    """What is wrong with one benchmark section."""
    where = f"{name}.{kind}"
    if not isinstance(section, dict):
        return [f"{where} is not an object"]
    found = [
        f"{where} lacks {field}"
        for field in SECTION_FIELDS
        if field not in section
    ]
    for field in SECTION_FIELDS[:5]:
        value = section.get(field)
        if field in section and not (isinstance(value, str) and value.strip()):
            found.append(f"{where}.{field} is empty")
    if section.get("status") != "not_run":
        found.append(f"{where} claims a status other than not_run")
    parameters = section.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        found.append(f"{where} names no parameter")
    else:
        rule = str(section.get("acceptance_rule", ""))
        for key, value in parameters.items():
            if value is not None:
                found.append(f"{where}.{key} is fixed before the slice exists")
            if key not in rule:
                found.append(f"{where}.{key} is not used by the rule")
    return found


def problems(name: str, definition: dict[str, Any]) -> list[str]:
    """What is wrong with one slice definition (empty when well formed)."""
    found = [
        f"{name} lacks {field}"
        for field in TOP_FIELDS
        if field not in definition
    ]
    if found:
        return found
    if definition["slice"] != name:
        found.append(f"{name} names slice {definition['slice']}")
    if definition["status"] != "definition_only":
        found.append(f"{name} has status {definition['status']}")
    if definition["built"] not in BUILT:
        found.append(f"{name} has unknown built value")
    if tuple(definition["lifecycle_required"]) != LIFECYCLE:
        found.append(f"{name} drops a lifecycle stage")
    if definition["deployment_requirement"].get("declared") is not False:
        found.append(f"{name} claims a declared deployment requirement")
    if tuple(definition["benchmarks"]) != SECTIONS:
        found.append(f"{name} must define exactly {', '.join(SECTIONS)}")
    for kind in SECTIONS:
        found += section_problems(
            name, kind, definition["benchmarks"].get(kind)
        )
    for path in definition["evidence_in_repo"]:
        if not (ROOT / path).exists():
            found.append(f"{name} cites {path}, which does not exist")
    claims = " ".join(definition["not_claimed"])
    if "production-supported" not in claims:
        found.append(f"{name} does not disclaim production support")
    return found
