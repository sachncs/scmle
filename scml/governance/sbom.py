"""Deterministic CycloneDX SBOM generation from an environment lock."""

from collections.abc import Mapping
import datetime
import hashlib
import json
import re
from typing import Any
import uuid

COMPONENT_NAME = "super-charge-ml"
CYCLONEDX_VERSION = "1.5"
SBOM_SCHEMA = "https://cyclonedx.org/schema/bom-1.5.schema.json"
PACKAGE_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def normalize_name(name: str) -> str:
    """Return a canonical PyPI package name for an SBOM component."""
    value = name.strip().lower().replace("_", "-")
    if not value or PACKAGE_NAME.fullmatch(value) is None:
        raise ValueError(f"invalid distribution name: {name!r}")
    return value


def component_ref(name: str, version: str) -> str:
    """Return a stable package URL used as the component reference."""
    normalized = normalize_name(name)
    version_value = version.strip()
    if not version_value:
        raise ValueError(f"version for {name!r} must not be empty")
    return f"pkg:pypi/{normalized}@{version_value}"


def build(
    distributions: Mapping[str, str],
    *,
    run_id: str,
    software_version: str,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Build a CycloneDX SBOM from a pinned distribution mapping.

    The function records package identities and versions only. License and
    vulnerability claims require authoritative package metadata and scanners,
    so this artifact does not invent either field.
    """
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if not software_version.strip():
        raise ValueError("software_version must not be empty")
    if not distributions:
        raise ValueError("distributions must not be empty")
    components = []
    ordered = sorted(distributions.items(), key=lambda item: item[0])
    for name, version in ordered:
        normalized = normalize_name(name)
        reference = component_ref(normalized, str(version))
        components.append(
            {
                "type": "library",
                "bom-ref": reference,
                "name": normalized,
                "version": str(version),
                "purl": reference,
            }
        )
    serial_seed = json.dumps(
        {
            "run_id": run_id,
            "software_version": software_version,
            "components": components,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    serial = uuid.uuid5(
        uuid.NAMESPACE_URL,
        hashlib.sha256(serial_seed.encode("utf-8")).hexdigest(),
    )
    timestamp = observed_at or datetime.datetime.now(datetime.UTC).isoformat()
    return {
        "bomFormat": "CycloneDX",
        "specVersion": CYCLONEDX_VERSION,
        "serialNumber": f"urn:uuid:{serial}",
        "version": 1,
        "$schema": SBOM_SCHEMA,
        "metadata": {
            "timestamp": timestamp,
            "component": {
                "type": "application",
                "name": COMPONENT_NAME,
                "version": software_version,
            },
            "properties": [{"name": "scml.run_id", "value": run_id}],
        },
        "components": components,
    }
