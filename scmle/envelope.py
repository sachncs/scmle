"""Authenticated envelopes for pickled model payloads (SEC-1, SEC-2).

Unpickling executes code, so a payload is unpickled only after it is
authenticated. An envelope is a magic line, one JSON header line and the raw
payload bytes. The header holds the caller's metadata, the payload's SHA-256
and an HMAC over both, made with the signing key (`signing.keying`).
`read` parses the header without executing anything, checks the digest and
the HMAC, and returns the payload bytes for the caller to unpickle. An
envelope without an HMAC, or one read with no key, is refused unless the
caller vouches for its source with `trusted=True`, the same rule bundles
follow.
"""

from collections.abc import Mapping
import hashlib
import hmac
import json
import pathlib
from typing import Any

from scmle import durability
from scmle.contracts import errors
from scmle.contracts import signing

MAGIC = b"SCMLE-ENVELOPE 1\n"
LIMIT = 1 << 20


def mac(header: Mapping[str, Any], key: bytes) -> str:
    """HMAC-SHA256 of the canonical JSON of the header without its MAC."""
    body = {k: v for k, v in header.items() if k != "mac"}
    return hmac.new(key, signing.canonicalise(body), hashlib.sha256).hexdigest()


def seal(
    path: str | pathlib.Path,
    meta: Mapping[str, Any],
    payload: bytes,
    signing_key: str | bytes | None = None,
) -> pathlib.Path:
    """Writes an envelope atomically, signed when a key is available.

    Args:
        path: Destination file; missing parent folders are created.
        meta: JSON-ready metadata stored in the header.
        payload: The pickled bytes.
        signing_key: HMAC key; the configured key is used when None. With no
            key at all the envelope is unsigned and can be read only with
            `trusted=True`.

    Returns:
        The path written.
    """
    key = signing.keying(signing_key)
    header: dict[str, Mapping[str, Any] | str | None] = {
        "meta": dict(meta),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    header["mac"] = None if key is None else mac(header, key)
    line = json.dumps(header, sort_keys=True).encode("utf-8") + b"\n"
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    durability.write_atomically(destination, MAGIC + line + payload)
    return destination


def read(
    path: str | pathlib.Path,
    signing_key: str | bytes | None = None,
    trusted: bool = False,
) -> tuple[Mapping[str, Any], bytes]:
    """Reads and authenticates an envelope; nothing is unpickled here.

    Args:
        path: The envelope file.
        signing_key: HMAC key; the configured key is used when None.
        trusted: Whether the caller vouches for an unsigned envelope.

    Returns:
        The metadata and the payload bytes, safe for the caller to unpickle.

    Raises:
        ValueError: If the file is not an envelope, its header is malformed
            or the payload does not match its digest.
        ArtifactError: If the envelope is unsigned, or no key is available,
            and `trusted` is false, or its HMAC does not verify.
    """
    data = pathlib.Path(path).read_bytes()
    if not data.startswith(MAGIC):
        raise errors.InputError(f"{path} is not an scmle envelope")
    end = data.find(b"\n", len(MAGIC), len(MAGIC) + LIMIT)
    if end < 0:
        raise errors.InputError(f"{path} has no envelope header")
    try:
        header = json.loads(data[len(MAGIC) : end])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise errors.InputError(
            f"{path} has a malformed envelope header"
        ) from error
    if not isinstance(header, dict) or not isinstance(header.get("meta"), dict):
        raise errors.InputError(f"{path} has a malformed envelope header")
    payload = data[end + 1 :]
    digest = str(header.get("sha256", ""))
    if not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), digest):
        raise errors.InputError(f"{path} does not match its digest")
    key = signing.keying(signing_key)
    signed = isinstance(header.get("mac"), str)
    if signed and key is not None:
        if not hmac.compare_digest(str(header["mac"]), mac(header, key)):
            raise errors.ArtifactError(
                f"{path} is not authentic: its signature does not verify"
            )
    elif not trusted:
        raise errors.ArtifactError(
            f"refusing to load {path}: it is pickle-based and loading executes "
            "code, and it is "
            + ("signed but no key is set" if signed else "not signed")
            + f". Set ${signing.SIGNING_KEY_ENV}, or pass trusted=True if you "
            "trust its source."
        )
    return dict(header["meta"]), payload


def sign(
    manifest: Mapping[str, Any], signing_key: str | bytes | None = None
) -> Mapping[str, Any]:
    """Signs a package manifest, which records its payload's digest.

    Args:
        manifest: JSON-ready manifest; any `signature` it has is replaced.
        signing_key: HMAC key; the configured key is used when None.

    Returns:
        The manifest with `signature` set to the HMAC of the rest, or None
        when no key is available (the package then loads only when
        trusted).
    """
    key = signing.keying(signing_key)
    body = {k: v for k, v in manifest.items() if k != "signature"}
    return {**body, "signature": None if key is None else mac(body, key)}


def check(
    manifest: Mapping[str, Any],
    where: str | pathlib.Path,
    signing_key: str | bytes | None = None,
    trusted: bool = False,
) -> None:
    """Refuses to go on to unpickle unless the manifest is authentic.

    Call before loading the payload, after checking that the payload
    matches the digest the manifest records.

    Args:
        manifest: The package manifest, as read.
        where: The package, for messages.
        signing_key: HMAC key; the configured key is used when None.
        trusted: Whether the caller vouches for an unsigned package.

    Raises:
        ArtifactError: If the manifest's signature does not verify, or it is
            unsigned (or no key is set) and `trusted` is false.
    """
    key = signing.keying(signing_key)
    signature = manifest.get("signature")
    signed = isinstance(signature, str)
    if signed and key is not None:
        body = {k: v for k, v in manifest.items() if k != "signature"}
        if not hmac.compare_digest(str(signature), mac(body, key)):
            raise errors.ArtifactError(
                f"{where} is not authentic: its signature does not verify"
            )
        return
    if not trusted:
        raise errors.ArtifactError(
            f"refusing to load {where}: it is pickle-based and loading "
            "executes code, and it is "
            + ("signed but no key is set" if signed else "not signed")
            + f". Set ${signing.SIGNING_KEY_ENV}, or pass trusted=True if you "
            "trust its source."
        )
