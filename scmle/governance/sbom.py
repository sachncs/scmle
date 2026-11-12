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


def build(
    distributions: Mapping[str, str],
    *,
    run_id: str,
    software_version: str,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Builds a CycloneDX SBOM from a pinned distribution mapping.

    The function records package identities and versions only. License and
    vulnerability claims require authoritative package metadata and scanners,
    so this artifact does not invent either field.

    Args:
        distributions: Mapping from distribution name to pinned version.
        run_id: Identifier of the run, recorded as a property and used to
            derive the serial number.
        software_version: Version of the software the SBOM describes.
        observed_at: ISO 8601 timestamp for the metadata; the current time
            if None or empty.

    Returns:
        The SBOM document. Its components follow the order of the given
        distribution names sorted as written, before normalization.

    Raises:
        ValueError: If the run id, the software version or the
            distributions are empty, or a name or version is invalid.
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
        normalized = name.strip().lower().replace("_", "-")
        if not normalized or PACKAGE_NAME.fullmatch(normalized) is None:
            raise ValueError(f"invalid distribution name: {name!r}")
        pinned = str(version).strip()
        if not pinned:
            raise ValueError(f"version for {normalized!r} must not be empty")
        reference = f"pkg:pypi/{normalized}@{pinned}"
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
                "bom-ref": COMPONENT_NAME,
                "name": COMPONENT_NAME,
                "version": software_version,
            },
            "properties": [{"name": "scmle.run_id", "value": run_id}],
        },
        "components": components,
    }


def source_timestamp() -> str:
    """Returns the build time as `SOURCE_DATE_EPOCH` or the current time.

    The environment variable is used only when it holds digits; any other
    value falls back to the current time.

    Returns:
        An ISO 8601 UTC timestamp.
    """
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
    """Builds the CycloneDX SBOM of a packaged service.

    On top of the Python distributions it lists the model bundle (by the
    SHA-256 of its manifest) and the base image (by reference only, with no
    digest), and a single dependency entry whose `ref` is `scmle` and whose
    `dependsOn` lists every component. The bundle digest also serves as the
    run id of the underlying SBOM.

    Args:
        distributions: Mapping from distribution name to pinned version.
        model_name: Name of the packaged model.
        model_version: Version of the packaged model.
        bundle_digest: `sha256:`-prefixed digest of the bundle manifest.
        base_image: Reference of the base container image.
        software_version: Version of the software the SBOM describes.
        observed_at: ISO 8601 timestamp; the build time if None.

    Returns:
        The SBOM document with the model and the image appended as extra
        components and a `dependencies` entry listing every component
        reference, sorted.

    Raises:
        ValueError: If `bundle_digest` is not a `sha256:` digest, or the
            underlying SBOM inputs are invalid.
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
