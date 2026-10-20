#!/usr/bin/env python3
"""Build and validate the runtime dependency license inventory."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from importlib import metadata
import json
from pathlib import Path
import re
import tomllib

FORBIDDEN_LICENSES = re.compile(
    r"(?:^|[^A-Z])(AGPL|BUSL|GPL|LGPL|SSPL|ELASTIC)(?:[^A-Z]|$)",
    re.IGNORECASE,
)
LICENSE_CLASSIFIERS = {
    "Apache Software License": "Apache-2.0",
    "BSD License": "BSD",
    "MIT License": "MIT",
    "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
}
LICENSE_FILE_NAMES = frozenset(
    {
        "copying",
        "copying.md",
        "copying.txt",
        "license",
        "license.md",
        "license.txt",
    }
)
LICENSE_TEXT_PATTERNS = {
    "Apache-2.0": (
        "licensed under the apache license, version 2.0",
        "apache license, version 2.0",
    ),
    "BSD": (
        "redistribution and use in source and binary forms",
        "redistribution and use in source forms",
    ),
    "GPL": (
        "gnu general public license",
        "gnu lesser general public license",
    ),
    "LGPL": ("gnu lesser general public license",),
    "MIT": (
        "permission is hereby granted, free of charge",
        "mit license",
    ),
    "MPL-2.0": ("mozilla public license version 2.0",),
    "SSPL": ("server side public license",),
}


def metadata_value(
    package_metadata: metadata.PackageMetadata, field: str
) -> str:
    """Returns the first value for a metadata field, if present."""
    values = package_metadata.get_all(field) or []
    return values[0] if values else ""


def distribution_map() -> dict[str, metadata.Distribution]:
    """Returns installed distributions keyed by normalized name."""
    return {
        name.lower().replace("_", "-"): distribution
        for distribution in metadata.distributions()
        if (name := metadata_value(distribution.metadata, "Name"))
    }


def dependency_names(project_file: Path, include_all_extras: bool) -> set[str]:
    """Returns the project's declared runtime dependency roots."""
    project = tomllib.loads(project_file.read_text(encoding="utf-8"))["project"]
    requirements = list(project.get("dependencies", []))
    if include_all_extras:
        for extra in project.get("optional-dependencies", {}).values():
            requirements.extend(extra)
    from packaging.requirements import Requirement

    return {
        parsed.name.lower().replace("_", "-")
        for item in requirements
        if (parsed := Requirement(item)).marker is None
        or parsed.marker.evaluate()
    }


def applicable_requirements(
    distribution: metadata.Distribution,
) -> Iterable[str]:
    """Yields dependency names whose installed markers apply."""
    for requirement in distribution.requires or []:
        try:
            from packaging.requirements import Requirement

            parsed = Requirement(requirement)
            if parsed.marker is None or parsed.marker.evaluate():
                yield parsed.name.lower().replace("_", "-")
        except Exception as error:  # pragma: no cover - malformed metadata
            raise RuntimeError(
                f"cannot parse dependency metadata: {requirement!r}"
            ) from error


def license_evidence(
    distribution: metadata.Distribution,
) -> tuple[str, str]:
    """Returns a normalized license label and its metadata source."""
    expression = metadata_value(
        distribution.metadata, "License-Expression"
    ).strip()
    if expression:
        return expression, "license-expression"

    classifiers = distribution.metadata.get_all("Classifier") or []
    classifier_labels = {
        label
        for classifier in classifiers
        if classifier.startswith("License ::")
        for label in LICENSE_CLASSIFIERS
        if classifier.endswith(label)
    }
    if classifier_labels:
        normalized = sorted(
            LICENSE_CLASSIFIERS[label] for label in classifier_labels
        )
        return " OR ".join(normalized), "classifier"

    license_text = metadata_value(distribution.metadata, "License").strip()
    first_line = license_text.splitlines()[0].strip() if license_text else ""
    if first_line:
        return first_line[:200], "license-field"

    file_licenses = license_file_evidence(distribution)
    if file_licenses:
        return file_licenses, "license-file"
    return "UNKNOWN", "missing-metadata"


def license_file_evidence(distribution: metadata.Distribution) -> str:
    """Returns recognized license names from a distribution license file."""
    if not distribution.files:
        return ""
    recognized: set[str] = set()
    for file_path in distribution.files:
        file_name = Path(str(file_path)).name.lower()
        if file_name not in LICENSE_FILE_NAMES:
            continue
        license_path = distribution.locate_file(file_path)
        try:
            content = Path(str(license_path)).read_bytes()[:131_072].decode(
                "utf-8", errors="replace"
            ).lower()
        except OSError:
            continue
        recognized.update(
            license_name
            for license_name, patterns in LICENSE_TEXT_PATTERNS.items()
            if any(pattern in content for pattern in patterns)
        )
    return " OR ".join(sorted(recognized))


def collect_inventory(
    project_file: Path, include_all_extras: bool
) -> tuple[list[dict[str, str]], list[str]]:
    """Collects the transitive runtime closure and missing roots."""
    installed = distribution_map()
    pending = sorted(dependency_names(project_file, include_all_extras))
    missing = [name for name in pending if name not in installed]
    visited: set[str] = set()
    inventory: list[dict[str, str]] = []
    while pending:
        name = pending.pop(0)
        if name in visited or name not in installed:
            continue
        visited.add(name)
        distribution = installed[name]
        license_name, source = license_evidence(distribution)
        inventory.append(
            {
                "name": distribution.metadata["Name"],
                "version": distribution.version,
                "license": license_name,
                "evidence": source,
            }
        )
        pending.extend(
            dependency
            for dependency in applicable_requirements(distribution)
            if dependency not in visited and dependency in installed
        )
    inventory.sort(key=lambda item: (item["name"].lower(), item["version"]))
    return inventory, missing


def validate_inventory(
    inventory: list[dict[str, str]], missing: list[str]
) -> None:
    """Fails when the runtime closure is incomplete or policy-violating."""
    failures = [f"missing installed dependency: {name}" for name in missing]
    for package in inventory:
        if package["license"] == "UNKNOWN":
            failures.append(f"missing license evidence: {package['name']}")
        elif FORBIDDEN_LICENSES.search(package["license"]):
            failures.append(
                f"forbidden license {package['license']!r}: {package['name']}"
            )
    if failures:
        raise SystemExit("\n".join(failures))


def parse_arguments() -> argparse.Namespace:
    """Parses command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-file", type=Path, default=Path("pyproject.toml")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--all-extras",
        action="store_true",
        help="include every optional extra in the runtime roots",
    )
    return parser.parse_args()


def main() -> None:
    """Writes the deterministic inventory and enforces the license policy."""
    arguments = parse_arguments()
    inventory, missing = collect_inventory(
        arguments.project_file, arguments.all_extras
    )
    validate_inventory(inventory, missing)
    record = {
        "schema_version": 1,
        "project": "super-charge-ml",
        "scope": (
            "all-optional-extras" if arguments.all_extras else "base-runtime"
        ),
        "packages": inventory,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"license inventory: {len(inventory)} packages")


if __name__ == "__main__":
    main()
