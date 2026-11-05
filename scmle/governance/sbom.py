"""Deterministic CycloneDX SBOM generation from an environment lock."""

from collections.abc import Mapping
import datetime
import hashlib
import json
import os
import re
from typing import Any
import uuid

COMPONENT_NAME = "scmle"
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
            "properties": [{"name": "scmle.run_id", "value": run_id}],
        },
        "components": components,
    }


def parse_pins(text: str) -> dict[str, str]:
    """Reads `name==version` lines of a lock file; comments are skipped."""
    pins = {}
    for line in text.splitlines():
        entry = line.split("#", 1)[0].strip()
        if not entry or "==" not in entry:
            continue
        parts = entry.partition("==")
        pins[normalize_name(parts[0])] = parts[2].strip().split(" ")[0]
    return pins


def source_timestamp() -> str:
    """The build time: `SOURCE_DATE_EPOCH` if set, else the current time."""
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch is not None and epoch.isdigit():
        moment = datetime.datetime.fromtimestamp(int(epoch), datetime.UTC)
        return moment.isoformat()
    return datetime.datetime.now(datetime.UTC).isoformat()


def build_service(
    distributions: Mapping[str, str],
    *,
    model_name: str,
    model_version: str,
    bundle_digest: str,
    base_image: str,
    software_version: str,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Build the CycloneDX SBOM of a packaged service.

    On top of the Python distributions it lists the model bundle (by the
    SHA-256 of its manifest) and the base image (by reference; the digest is
    recorded in the build's attestation once the base image is resolved), and
    a dependency edge from the service to each of them.
    """
    if not bundle_digest.startswith("sha256:"):
        raise ValueError("bundle_digest must be a sha256: digest")
    document = build(
        distributions,
        run_id=bundle_digest,
        software_version=software_version,
        observed_at=observed_at or source_timestamp(),
    )
    model_ref = f"model:{model_name}@{bundle_digest}"
    image_ref = f"image:{base_image}"
    document["components"] = [
        *document["components"],
        {
            "type": "machine-learning-model",
            "bom-ref": model_ref,
            "name": model_name,
            "version": model_version,
            "hashes": [
                {
                    "alg": "SHA-256",
                    "content": bundle_digest.removeprefix("sha256:"),
                }
            ],
        },
        {
            "type": "container",
            "bom-ref": image_ref,
            "name": base_image.split(":", 1)[0],
            "version": base_image.split(":", 1)[-1],
        },
    ]
    document["dependencies"] = [
        {
            "ref": COMPONENT_NAME,
            "dependsOn": sorted(
                [item["bom-ref"] for item in document["components"]]
            ),
        }
    ]
    return document
