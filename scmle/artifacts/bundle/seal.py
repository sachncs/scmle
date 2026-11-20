"""The HMAC signature of a bundle's manifest."""

import hashlib
import hmac
import json
import pathlib

from scmle.artifacts.bundle import layout
from scmle.contracts import errors
from scmle.contracts import signing

SIGNATURE_FILE = "signature.json"
SIGNATURE_ALGORITHM = "hmac-sha256"


def sign(manifest_bytes: bytes, key: bytes) -> str:
    """Computes the HMAC-SHA256 of the manifest.

    The manifest carries the checksum of every stored file.

    Args:
        manifest_bytes: The manifest text as bytes.
        key: The HMAC key.

    Returns:
        The hex digest.
    """
    return hmac.new(key, manifest_bytes, hashlib.sha256).hexdigest()


def verify(
    root: pathlib.Path,
    signing_key: str | bytes | None,
    text: bytes | None = None,
) -> bool:
    """Checks the manifest signature.

    A missing manifest surfaces as the `OSError` of reading it, not as an
    `ArtifactError`. A caller that goes on to trust the manifest passes the
    bytes it parsed as `text`, so the signature covers exactly those bytes
    and not a file that may have been replaced since.

    Args:
        root: Bundle directory.
        signing_key: HMAC key (defaults to the environment).
        text: The manifest bytes to check; None reads the manifest file.

    Returns:
        True when a signature exists and verifies; False when the bundle is
        unsigned or no key is available.

    Raises:
        ArtifactError: If a signature exists but is unreadable, uses an
            unsupported algorithm or does not verify.
    """
    signature_path = root / SIGNATURE_FILE
    key = signing.keying(signing_key)
    if not signature_path.exists() or key is None:
        return False
    try:
        record = json.loads(signature_path.read_text(encoding="utf-8"))
        algorithm = record.get("algorithm")
        recorded = record.get("signature", "")
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        AttributeError,
    ) as error:
        raise errors.ArtifactError(
            "invalid bundle signature metadata"
        ) from error
    if algorithm != SIGNATURE_ALGORITHM:
        raise errors.ArtifactError(
            f"unsupported bundle signature algorithm {algorithm!r}"
        )
    if text is None:
        text = (root / layout.MANIFEST_FILE).read_bytes()
    expected = sign(text, key)
    if not hmac.compare_digest(recorded, expected):
        raise errors.ArtifactError(
            "bundle signature is invalid: the bundle was modified or "
            "signed with a different key"
        )
    return True
