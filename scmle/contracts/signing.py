"""Deterministic signing primitives shared by serializable contracts."""

from collections.abc import Mapping
import hashlib
import hmac
import json
import os
import pathlib
from typing import Any

from scmle.contracts import errors

SIGNING_KEY_ENV = "SCMLE_SIGNING_KEY"
SIGNING_KEY_FILE_ENV = "SCMLE_SIGNING_KEY_FILE"
SIGNATURE_ALGORITHM = "hmac-sha256"


def key_from_environment(environ: Mapping[str, str]) -> str | None:
    """Returns the configured key: `SCMLE_SIGNING_KEY`, else the file it names.

    The file form lets a secret manager mount the key (a Kubernetes secret, a
    Secrets Manager or Secret Manager volume) so it never sits in an
    environment variable or an image.

    Args:
        environ: The environment variables.

    Returns:
        The key from the variable, else the stripped content of the file
        named by the file variable, or None when neither variable is set.

    Raises:
        ConfigError: If a variable is set but empty, the file cannot be
            read, or the file is empty. Each of these used to mean "no key"
            and left records unsigned without a word (ERR-3).
    """
    direct = environ.get(SIGNING_KEY_ENV)
    if direct is not None:
        if not direct:
            raise errors.ConfigError(f"{SIGNING_KEY_ENV} is set but empty")
        return direct
    path = environ.get(SIGNING_KEY_FILE_ENV)
    if path is None:
        return None
    if not path:
        raise errors.ConfigError(f"{SIGNING_KEY_FILE_ENV} is set but empty")
    try:
        text = pathlib.Path(path).read_text(encoding="utf-8").strip()
    except OSError as error:
        raise errors.ConfigError(
            f"cannot read the signing key file named by "
            f"{SIGNING_KEY_FILE_ENV}: {error}"
        ) from error
    if not text:
        raise errors.ConfigError(f"the signing key file {path} is empty")
    return text


def resolve_key(signing_key: str | bytes | None) -> bytes | None:
    """Returns the explicit signing key or the configured environment key.

    Args:
        signing_key: The explicit key, or None to use the configured one.

    Returns:
        The key as bytes (a string is UTF-8 encoded), or None if no key is
        given or configured.

    Raises:
        ValueError: If the key is empty; an empty HMAC key would sign with a
            publicly known secret, and treating it as "no key" left records
            unsigned silently (ERR-3).
        ConfigError: If the configured key is unusable (see
            `key_from_environment`).
    """
    key = (
        signing_key
        if signing_key is not None
        else key_from_environment(os.environ)
    )
    if key is None:
        return None
    resolved = key.encode() if isinstance(key, str) else key
    if not resolved:
        raise ValueError("signing key must not be empty")
    return resolved


def canonicalise(payload: Mapping[str, Any]) -> bytes:
    """Returns the canonical JSON bytes covered by a signature.

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
    """Returns an HMAC signature for a canonical contract payload.

    Args:
        payload: The payload to sign.
        key: The signing key.

    Returns:
        The hex HMAC-SHA256 of the canonical payload.

    Raises:
        ValueError: If the key is empty, as text or as bytes.
    """
    resolved = resolve_key(key)
    if resolved is None:
        raise ValueError("signing key must not be empty")
    return hmac.new(
        resolved,
        canonicalise(payload),
        hashlib.sha256,
    ).hexdigest()
