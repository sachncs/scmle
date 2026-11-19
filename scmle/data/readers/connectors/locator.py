"""Canonical source URIs, source identities and resumable bookmarks."""

import base64
import dataclasses
import hashlib
import pathlib
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

from scmle.contracts import errors

SHA256_HEX_LENGTH = 64


def canonicalize(source_uri: str | pathlib.Path) -> str:
    """Returns a deterministic URI without changing source semantics.

    Bare paths are treated as local files. URI schemes are normalized to
    lower-case, while query strings and fragments remain part of identity.

    Args:
        source_uri: A URI or a local path (a single-letter drive prefix such
            as ``C:`` is treated as a path, not a scheme).

    Returns:
        The URI with a lower-cased scheme, or, for a path, the absolute
        ``file://`` URI after expanding ``~`` and resolving the path.

    Raises:
        ValueError: If `source_uri` is blank.
    """
    value = str(source_uri).strip()
    if not value:
        raise errors.InputError("source_uri must not be empty")
    parsed = urlsplit(value)
    if parsed.scheme and not (len(parsed.scheme) == 1 and value[1:2] == ":"):
        return urlunsplit(
            (
                parsed.scheme.lower(),
                parsed.netloc,
                parsed.path,
                parsed.query,
                parsed.fragment,
            )
        )
    return pathlib.Path(value).expanduser().resolve(strict=False).as_uri()


def identify(source_uri: str | pathlib.Path) -> str:
    """Returns a stable identity for a logical source URI.

    Args:
        source_uri: A URI or local path; an empty value is rejected by
            `canonicalize`.

    Returns:
        ``src_`` followed by the SHA-256 hex digest of the canonical URI.
    """
    canonical = canonicalize(source_uri)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"src_{digest}"


def bookmark(source_identity_value: str, offset: int) -> str:
    """Creates a deterministic opaque cursor for a source offset.

    Args:
        source_identity_value: Identity of the source the offset belongs to.
        offset: Position in the source, in the connector's unit (rows or
            bytes).

    Returns:
        A URL-safe, unpadded base64 string encoding the identity and
        offset.

    Raises:
        ValueError: If the identity is empty or the offset is negative.
    """
    if not source_identity_value:
        raise errors.InputError("source_identity_value must not be empty")
    if offset < 0:
        raise errors.InputError("offset must be non-negative")
    payload = f"{source_identity_value}:{offset}".encode()
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def unwrap(cursor: str) -> tuple[str, int]:
    """Decodes a cursor into its source identity and offset.

    Args:
        cursor: A string produced by `bookmark`.

    Returns:
        The source identity and the non-negative offset.

    Raises:
        ValueError: If the cursor is empty, cannot be decoded, or encodes
            an empty identity or a negative offset.
    """
    if not cursor:
        raise errors.InputError("cursor must not be empty")
    padding = "=" * (-len(cursor) % 4)
    try:
        payload = base64.urlsafe_b64decode(cursor + padding).decode("utf-8")
        identity, offset_value = payload.rsplit(":", maxsplit=1)
        offset = int(offset_value)
    except (ValueError, UnicodeDecodeError) as error:
        raise errors.InputError(
            "cursor is not a valid connector cursor"
        ) from error
    if not identity or offset < 0:
        raise errors.InputError("cursor contains an invalid source or offset")
    return identity, offset


@dataclasses.dataclass(frozen=True)
class Bookmark:
    """A portable source-position checkpoint bound to a source identity.

    A row-oriented connector uses row offsets; a range-oriented connector may
    use byte offsets. The unit is defined by the connector contract. A
    checkpoint can be supplied by explicit ``offset`` or by the opaque
    ``cursor`` returned in a previous batch. When both are present they must
    agree. A checkpoint returned by a connector always contains both values.

    Attributes:
        offset: Position in the source, in the connector's unit. Zero is
            treated as unset when a cursor is also given.
        cursor: Opaque cursor encoding the source identity and offset, or
            None if not yet bound.
        identity: Identity of the source the checkpoint belongs to,
            or an empty string if not yet bound.
        digest: SHA-256 hex digest of the source content (or of the
            rows before the offset) the checkpoint was taken against, or
            None if unknown.
    """

    offset: int = 0
    cursor: str | None = None
    identity: str = ""
    digest: str | None = None

    def __post_init__(self) -> None:
        """Validates and normalizes cursor-derived checkpoint fields.

        An empty `identity` or a zero `offset` is filled in from the
        cursor when one is given.

        Raises:
            ValueError: If `offset` is negative, `digest` is not a
                64-character lower-case hex digest, the cursor is invalid,
                or the cursor's identity or (non-zero) offset disagrees
                with the checkpoint.
        """
        if self.offset < 0:
            raise errors.InputError("offset must be non-negative")
        if self.digest is not None and (
            len(self.digest) != SHA256_HEX_LENGTH
            or any(
                character not in "0123456789abcdef" for character in self.digest
            )
        ):
            raise errors.InputError(
                "content_sha256 must be a SHA-256 hex digest"
            )
        if self.cursor is None:
            return
        cursor_identity, cursor_offset = unwrap(self.cursor)
        if self.identity and self.identity != cursor_identity:
            raise errors.InputError(
                "cursor source identity does not match checkpoint"
            )
        if self.offset and self.offset != cursor_offset:
            raise errors.InputError(
                "cursor offset does not match checkpoint offset"
            )
        if not self.identity:
            object.__setattr__(self, "identity", cursor_identity)
        if not self.offset:
            object.__setattr__(self, "offset", cursor_offset)

    def bind(self, identity: str) -> "Bookmark":
        """Binds an unbound checkpoint to ``identity`` safely.

        Args:
            identity: Identity of the source to bind to.

        Returns:
            A checkpoint with the same offset and content digest, bound to
            `identity`, with a cursor generated if none was present.

        Raises:
            ValueError: If the checkpoint is already bound to a different
                source.
        """
        if self.identity and self.identity != identity:
            raise errors.InputError("checkpoint belongs to a different source")
        return Bookmark(
            offset=self.offset,
            cursor=self.cursor or bookmark(identity, self.offset),
            identity=identity,
            digest=self.digest,
        )

    def seal(self, content_sha256: str) -> "Bookmark":
        """Binds a checkpoint to an immutable source-content digest.

        Args:
            content_sha256: SHA-256 hex digest of the observed content.

        Returns:
            A copy of the checkpoint carrying `digest`.

        Raises:
            ValueError: If the checkpoint already carries a different
                digest.
        """
        if self.digest and self.digest != content_sha256:
            raise errors.InputError(
                "checkpoint content digest does not match source"
            )
        return Bookmark(
            offset=self.offset,
            cursor=self.cursor,
            identity=self.identity,
            digest=content_sha256,
        )

    def advance(self, rows: int) -> "Bookmark":
        """Returns the checkpoint after consuming ``rows`` rows.

        Args:
            rows: Amount consumed, in the connector's unit (rows or bytes).

        Returns:
            A checkpoint with the offset increased by `rows` and a matching
            cursor.

        Raises:
            ValueError: If `rows` is negative or the checkpoint is not
                bound to a source.
        """
        if rows < 0:
            raise errors.InputError("rows must be non-negative")
        identity = self.identity
        if not identity:
            raise errors.InputError("checkpoint must be bound before advancing")
        next_offset = self.offset + rows
        return Bookmark(
            offset=next_offset,
            cursor=bookmark(identity, next_offset),
            identity=identity,
            digest=self.digest,
        )
