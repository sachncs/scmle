"""Deterministic signing primitives shared by serializable contracts."""

from collections.abc import Mapping
import hashlib
import hmac
import json
import os
from typing import Any

SIGNING_KEY_ENV = "SCMLE_SIGNING_KEY"
SIGNING_KEY_FILE_ENV = "SCMLE_SIGNING_KEY_FILE"
SIGNATURE_ALGORITHM = "hmac-sha256"


def key_from_environment(environ: Mapping[str, str]) -> str | None:
    """The configured key: `SCMLE_SIGNING_KEY`, else the file it names.

    The file form lets a secret manager mount the key (a Kubernetes secret, a
    Secrets Manager or Secret Manager volume) so it never sits in an
    environment variable or an image.

    Args:
        environ: The environment variables.

    Returns:
        The key from the variable, else the trimmed content of the file the
        file variable names, or None if neither yields a key.
    """
    direct = environ.get(SIGNING_KEY_ENV)
    if direct:
        return direct
    path = environ.get(SIGNING_KEY_FILE_ENV)
    if path:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip() or None
    return None


def resolve_key(signing_key: str | bytes | None) -> bytes | None:
    """Return the explicit signing key or the configured environment key.

    Args:
        signing_key: The explicit key, or None to use the configured one.

    Returns:
        The key as bytes, or None if there is none or it is empty.
    """
    key = (
        signing_key
        if signing_key is not None
        else key_from_environment(os.environ)
    )
    if key is None or key == "":
        return None
    return key.encode() if isinstance(key, str) else key


def canonical_payload(payload: Mapping[str, Any]) -> bytes:
    """Return the canonical JSON bytes covered by a signature.

    Args:
        payload: The payload to sign.

    Returns:
        The compact, key-sorted ASCII JSON of the payload as bytes.
    """
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sign_payload(payload: Mapping[str, Any], key: str | bytes) -> str:
    """Return an HMAC signature for a canonical contract payload.

    Args:
        payload: The payload to sign.
        key: The signing key.

    Returns:
        The hex HMAC-SHA256 of the canonical payload.

    Raises:
        ValueError: If the key is empty.
    """
    resolved = resolve_key(key)
    if resolved is None:
        raise ValueError("signing key must not be empty")
    return hmac.new(
        resolved,
        canonical_payload(payload),
        hashlib.sha256,
    ).hexdigest()
