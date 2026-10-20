"""Deterministic signing primitives shared by serializable contracts."""

from collections.abc import Mapping
import hashlib
import hmac
import json
import os
from typing import Any

SIGNING_KEY_ENV = "SCML_SIGNING_KEY"
SIGNATURE_ALGORITHM = "hmac-sha256"


def resolve_key(signing_key: str | bytes | None) -> bytes | None:
    """Return the explicit signing key or the configured environment key."""
    key = (
        signing_key
        if signing_key is not None
        else os.environ.get(SIGNING_KEY_ENV)
    )
    if key is None or key == "":
        return None
    return key.encode() if isinstance(key, str) else key


def canonical_payload(payload: Mapping[str, Any]) -> bytes:
    """Return the canonical JSON bytes covered by a signature."""
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sign_payload(payload: Mapping[str, Any], key: str | bytes) -> str:
    """Return an HMAC signature for a canonical contract payload."""
    resolved = resolve_key(key)
    if resolved is None:
        raise ValueError("signing key must not be empty")
    return hmac.new(
        resolved,
        canonical_payload(payload),
        hashlib.sha256,
    ).hexdigest()
