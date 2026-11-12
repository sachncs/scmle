"""Digests that identify an exact model bundle, the same way everywhere.

The bundle digest is the SHA-256 of the canonical JSON of the bundle manifest.
The manifest lists the checksum of every stored file, so the digest names the
whole bundle. A running service, a deployment spec and an attestation all quote
this one value.
"""

from collections.abc import Mapping
import hashlib
import json
import pathlib
from typing import Any

from scmle.contracts import errors

MANIFEST_FILE = "model_manifest.json"


def sha256_digest(data: bytes) -> str:
    """Digests bytes.

    Args:
        data: The bytes to digest.

    Returns:
        `sha256:` followed by the hex SHA-256 of `data`.
    """
    return "sha256:" + hashlib.sha256(data).hexdigest()


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Digests a parsed bundle manifest.

    Args:
        manifest: The manifest as parsed from JSON.

    Returns:
        The bundle digest, the SHA-256 digest of its canonical JSON.
    """
    return sha256_digest(
        json.dumps(
            manifest, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    )


def bundle_digest(directory: str | pathlib.Path) -> str:
    """Digests a bundle directory by its manifest.

    Args:
        directory: Bundle directory containing `model_manifest.json`.

    Returns:
        The bundle digest of the manifest.

    Raises:
        ArtifactError: If the directory has no readable manifest.
    """
    path = pathlib.Path(directory) / MANIFEST_FILE
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise errors.ArtifactError(
            f"cannot read the bundle manifest at {path}"
        ) from error
    if not isinstance(manifest, dict):
        raise errors.ArtifactError("the bundle manifest is not an object")
    return manifest_digest(manifest)


def file_digest(path: str | pathlib.Path) -> str:
    """Digests a file, reading it in 1 MiB chunks.

    Args:
        path: File to digest.

    Returns:
        `sha256:` followed by the hex SHA-256 of the file's bytes.
    """
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()
