"""Validated image samples and their transformation provenance."""

from collections.abc import Iterable
from collections.abc import Mapping
import dataclasses
import hashlib
import io
import json
import pathlib
from types import MappingProxyType
from typing import Any
from typing import ClassVar
from typing import Self

from scmle.contracts import errors
from scmle.contracts import records
from scmle.data.modalities import image
from scmle.data.modalities.image.failures import ViolationError
from scmle.data.modalities.image.pixels import Pixels

SHA256_HEX_LENGTH = 64

DEFAULT_MAX_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_PIXELS = 25_000_000
DEFAULT_PROFILE_SIZE = (16, 16)

SENSITIVE_METADATA_TAGS = {
    271: "camera_make",
    272: "camera_model",
    305: "software",
    315: "artist",
    33432: "copyright",
    34853: "gps_info",
    36867: "date_time_original",
}


def bounds(
    width: int, height: int, max_pixels: int | None, source: str
) -> None:
    """Checks that decoded image dimensions are positive and within limits.

    Args:
        width: Decoded width in pixels.
        height: Decoded height in pixels.
        max_pixels: Maximum decoded pixel count, or ``None`` for no limit.
        source: Stable source label, recorded in the error.

    Raises:
        ViolationError: If a dimension is not positive or the pixel
            count exceeds ``max_pixels``.
    """
    if width <= 0 or height <= 0:
        raise ViolationError(
            "image dimensions must be positive",
            code="empty_dimensions",
            source=source,
        )
    pixel_count = width * height
    if max_pixels is not None and pixel_count > max_pixels:
        raise ViolationError(
            f"image has {pixel_count} pixels; limit is {max_pixels}",
            code="pixel_count_limit",
            source=source,
            details={"pixel_count": pixel_count, "max_pixels": max_pixels},
        )


def identify(source: str | pathlib.Path) -> str:
    """Returns a stable source identifier for a path or URI-like value.

    Args:
        source: A filesystem path, a value containing ``://``, or empty.

    Returns:
        ``memory://`` for an empty value, the stripped value itself when it
        contains ``://``, otherwise the absolute resolved path as a file URI.
    """
    value = str(source).strip()
    if not value:
        return "memory://"
    if "://" in value:
        return value
    return pathlib.Path(value).expanduser().resolve(strict=False).as_uri()


def imprint(
    image: Any,
    width: int,
    height: int,
) -> str:
    """Returns a format-independent hash of decoded RGBA content.

    Args:
        image: A decoded Pillow image.
        width: Width in pixels, included in the hashed payload.
        height: Height in pixels, included in the hashed payload.

    Returns:
        The SHA-256 hex digest of a version tag, the dimensions and the RGBA
        pixel bytes.
    """
    canonical = image.convert("RGBA")
    payload = b"scmle-image-content-v1\0"
    payload += f"{width}x{height}\0".encode("ascii")
    payload += canonical.tobytes()
    return hashlib.sha256(payload).hexdigest()


def examine(
    raw: bytes, max_pixels: int | None, source: str
) -> tuple[int, int, str, int, str, str, tuple[str, ...], tuple[str, ...]]:
    """Verifies and decodes an encoded image and reads its metadata.

    Args:
        raw: The encoded image payload.
        max_pixels: Maximum decoded pixel count, or ``None`` for no limit.
        source: Stable source label, recorded in errors.

    Returns:
        Width, height, Pillow mode, channel count, format name (``unknown``
        when Pillow names none), content hash, sorted EXIF tag numbers as
        text and sorted names of the sensitive EXIF fields present.

    Raises:
        ViolationError: If the payload is corrupt, cannot be decoded,
            or exceeds ``max_pixels``.
    """
    try:
        with image.PillowModule.open(io.BytesIO(raw)) as opened:
            opened.verify()
        with image.PillowModule.open(io.BytesIO(raw)) as opened:
            width, height = opened.size
            bounds(width, height, max_pixels, source)
            mode = opened.mode
            channels = len(opened.getbands())
            format_name = opened.format or "unknown"
            opened.load()
            content_hash = imprint(opened, width, height)
            exif_tags = tuple(sorted(str(tag) for tag in opened.getexif()))
            sensitive_metadata = tuple(
                sorted(
                    SENSITIVE_METADATA_TAGS[tag]
                    for tag in opened.getexif()
                    if tag in SENSITIVE_METADATA_TAGS
                )
            )
    except ViolationError:
        raise
    except Exception as error:
        raise ViolationError(
            f"could not decode image: {error}",
            code="corrupt_payload",
            source=source,
            details={"exception": type(error).__name__},
        ) from error
    return (
        width,
        height,
        mode,
        channels,
        format_name,
        content_hash,
        exif_tags,
        sensitive_metadata,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class Augmentation:
    """Caller-declared provenance for a transformed image sample.

    Attributes:
        source: Source of the image the sample was derived from,
            normalised with `identify`.
        operation: Non-empty name of the transformation.
        parameters: JSON-native transformation parameters, read-only.
        tool: Name of the tool that applied the transformation; may be
            empty.
    """

    source: str
    operation: str
    parameters: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    tool: str = ""

    KEYS: ClassVar[records.Migration] = records.Migration(
        renames={"parent_source": "parent"}
    )
    """Version-1 keys to the written ones; `downgrade` restores version 1."""

    def __post_init__(self) -> None:
        """Validates and normalizes transformation provenance.

        Raises:
            ValueError: If the operation is blank, a parameter key is not a
                non-empty string, or the parameters are not JSON-native.
        """
        parent = identify(self.source)
        operation = self.operation.strip()
        tool = self.tool.strip()
        if not operation:
            raise errors.InputError("augmentation operation must not be empty")
        keys: Iterable[object] = self.parameters
        if any(not isinstance(key, str) or not key.strip() for key in keys):
            raise errors.InputError(
                "augmentation parameter keys must be non-empty text"
            )
        try:
            normalized = json.loads(
                json.dumps(
                    dict(self.parameters), allow_nan=False, sort_keys=True
                )
            )
        except (TypeError, ValueError) as error:
            raise errors.InputError(
                "augmentation parameters must be JSON-native"
            ) from error
        object.__setattr__(self, "source", parent)
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "parameters", MappingProxyType(normalized))
        object.__setattr__(self, "tool", tool)

    @property
    def identity(self) -> str:
        """A deterministic identity string for the transformation record.

        It is the compact, key-sorted JSON of `to_dict` at version 1
        (`parent_source`), so renaming the stored key keeps every dataset
        hash that includes it.
        """
        return json.dumps(
            self.KEYS.downgrade(self.to_dict()),
            sort_keys=True,
            separators=(",", ":"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native transformation provenance.

        Returns:
            A dict with `parent`, `operation`, `parameters` and `tool`.
        """
        return self.KEYS.upgrade(
            {
                "parent_source": self.source,
                "operation": self.operation,
                "parameters": dict(self.parameters),
                "tool": self.tool,
            }
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Sample(Pixels):
    """An immutable, validated image snapshot.

    The encoded bytes are retained so a sample remains tied to the identities
    and metadata that were validated at ingestion time. ``file_hash`` hashes
    the encoded source bytes; ``content_hash`` hashes canonical decoded RGBA
    pixels and therefore supports content-based deduplication across formats.

    Attributes:
        source: Non-empty source label or URI.
        encoded_bytes: The encoded image payload.
        file_hash: SHA-256 hex digest of the encoded bytes.
        content_hash: SHA-256 hex digest of the canonical decoded pixels.
        width: Width in pixels; positive.
        height: Height in pixels; positive.
        channels: Number of colour bands; positive.
        mode: Pillow mode of the stored image.
        format: Encoded format name, or ``unknown``.
        size: Length of ``encoded``.
        label: Optional task label.
        group: Optional identity used to keep related samples in one
            split.
        tags: Sorted EXIF tag numbers present, as text.
        metadata: Sorted names of sensitive EXIF fields present.
        augmentation: Provenance if the sample is a transformed image.
    """

    source: str
    encoded: bytes = dataclasses.field(repr=False)
    file_hash: str
    content_hash: str
    width: int
    height: int
    channels: int
    mode: str
    format: str
    size: int
    label: str | int | float | bool | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()
    metadata: tuple[str, ...] = ()
    augmentation: Augmentation | None = None

    KEYS: ClassVar[records.Migration] = records.nest(
        {
            "byte_size": "size",
            "group_id": "group",
            "exif_tags": "exif",
            "sensitive_metadata": "sensitive",
        },
        groups={"digests": {"file_hash": "file", "content_hash": "content"}},
    )
    """Version-1 keys to the written ones; `downgrade` restores version 1."""

    def __post_init__(self) -> None:
        """Validates immutable identity and geometry metadata.

        Raises:
            ValueError: If source is blank, group_id is blank, byte_size
                differs from the payload length, a dimension or the channel
                count is not positive, or a hash is not 64 characters.
            TypeError: If augmentation is neither None nor an
                Augmentation.
        """
        object.__setattr__(self, "encoded", bytes(self.encoded))
        if not self.source.strip():
            raise errors.InputError("source must not be empty")
        if self.group is not None and not self.group.strip():
            raise errors.InputError("group_id must not be empty when provided")
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "metadata", tuple(self.metadata))
        if self.augmentation is not None and not isinstance(
            self.augmentation, Augmentation
        ):
            raise errors.KindError("augmentation must be Augmentation or None")
        if len(self.encoded) != self.size:
            raise errors.InputError("byte_size does not match encoded_bytes")
        if self.width <= 0 or self.height <= 0:
            raise errors.InputError("image dimensions must be positive")
        if self.channels <= 0:
            raise errors.InputError("channels must be positive")
        if (
            len(self.file_hash) != SHA256_HEX_LENGTH
            or len(self.content_hash) != SHA256_HEX_LENGTH
        ):
            raise errors.InputError("image hashes must be SHA-256 hex digests")

    @classmethod
    def parse(
        cls,
        encoded_bytes: bytes,
        source: str = "memory://",
        max_bytes: int | None = DEFAULT_MAX_BYTES,
        max_pixels: int | None = DEFAULT_MAX_PIXELS,
        label: str | float | bool | None = None,
        group_id: str | None = None,
        augmentation: Augmentation | None = None,
    ) -> Self:
        """Builds a validated image sample from encoded bytes.

        Args:
            encoded_bytes: Complete encoded image payload.
            source: Stable source label or URI for lineage.
            max_bytes: Maximum encoded payload size, or ``None`` for no limit.
            max_pixels: Maximum decoded pixel count, or ``None`` for no limit.
            label: Optional task label retained for curation evidence.
            group_id: Optional identity used to prevent split leakage.
            augmentation: Optional caller-declared transformation provenance.

        Returns:
            A validated immutable image snapshot.

        Raises:
            ViolationError: If the payload is empty, corrupt, or too
                large. An DependencyError (Pillow missing) or a
                ValueError (invalid limit) from the helpers it calls
                propagates unchanged.
        """
        image.pillow("image ingestion")
        raw = bytes(encoded_bytes)
        stable_source = identify(source)
        cls.validate(max_bytes, max_pixels)
        if not raw:
            raise ViolationError(
                "image payload is empty",
                code="empty_payload",
                source=stable_source,
            )
        if max_bytes is not None and len(raw) > max_bytes:
            raise ViolationError(
                f"image payload is {len(raw)} bytes; limit is {max_bytes}",
                code="encoded_size_limit",
                source=stable_source,
                details={"byte_size": len(raw), "max_bytes": max_bytes},
            )
        (
            width,
            height,
            mode,
            channels,
            format_name,
            content_hash,
            exif_tags,
            sensitive_metadata,
        ) = examine(raw, max_pixels, stable_source)
        return cls(
            source=stable_source,
            encoded=raw,
            file_hash=hashlib.sha256(raw).hexdigest(),
            content_hash=content_hash,
            width=width,
            height=height,
            channels=channels,
            mode=mode,
            format=format_name,
            size=len(raw),
            label=label,
            group=group_id,
            tags=exif_tags,
            metadata=sensitive_metadata,
            augmentation=augmentation,
        )

    @classmethod
    def load(
        cls,
        path: str | pathlib.Path,
        max_bytes: int | None = DEFAULT_MAX_BYTES,
        max_pixels: int | None = DEFAULT_MAX_PIXELS,
        label: str | float | bool | None = None,
        group_id: str | None = None,
        augmentation: Augmentation | None = None,
    ) -> Self:
        """Builds a validated image sample from a local file.

        Args:
            path: Path of the image file; ``~`` is expanded.
            max_bytes: Maximum file size, or ``None`` for no limit.
            max_pixels: Maximum decoded pixel count, or ``None`` for no
                limit.
            label: Optional task label retained for curation evidence.
            group_id: Optional identity used to prevent split leakage.
            augmentation: Optional caller-declared transformation provenance.

        Returns:
            A validated immutable image snapshot.

        Raises:
            ViolationError: If the file is missing, not a regular
                file, larger than ``max_bytes``, or unreadable. Failures of
                `parse`, including DependencyError, propagate.
        """
        file_path = pathlib.Path(path).expanduser()
        source = identify(file_path)
        if not file_path.exists():
            raise ViolationError(
                "image file was not found",
                code="missing_file",
                source=source,
            )
        if not file_path.is_file():
            raise ViolationError(
                "image source is not a regular file",
                code="not_a_file",
                source=source,
            )
        cls.validate(max_bytes, max_pixels)
        file_size = file_path.stat().st_size
        if max_bytes is not None and file_size > max_bytes:
            raise ViolationError(
                f"image file is {file_size} bytes; limit is {max_bytes}",
                code="encoded_size_limit",
                source=source,
                details={"byte_size": file_size, "max_bytes": max_bytes},
            )
        try:
            payload = file_path.read_bytes()
        except OSError as error:
            raise ViolationError(
                f"could not read image file: {error}",
                code="read_failure",
                source=source,
            ) from error
        return cls.parse(
            payload,
            source,
            max_bytes,
            max_pixels,
            label,
            group_id,
            augmentation,
        )

    @staticmethod
    def validate(
        max_bytes: int | None,
        max_pixels: int | None,
    ) -> None:
        """Validates optional resource limits before reading or decoding.

        Args:
            max_bytes: Encoded size limit, or None for no limit.
            max_pixels: Decoded pixel-count limit, or None for no limit.

        Raises:
            ValueError: If a limit is neither None nor a positive integer.
        """
        for name, value in (
            ("max_bytes", max_bytes),
            ("max_pixels", max_pixels),
        ):
            candidate: object = value
            if candidate is not None and (
                not isinstance(candidate, int)
                or isinstance(candidate, bool)
                or candidate <= 0
            ):
                raise errors.InputError(
                    f"{name} must be a positive integer or None"
                )

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible metadata without embedding image bytes.

        Returns:
            A dict of every field except ``encoded`` in single-word
            keys, the hashes under `digests.{file, content}`; the tag tuples
            become lists and the augmentation becomes its own dict or None.
        """
        return self.KEYS.upgrade(
            {
                "source": self.source,
                "file_hash": self.file_hash,
                "content_hash": self.content_hash,
                "width": self.width,
                "height": self.height,
                "channels": self.channels,
                "mode": self.mode,
                "format": self.format,
                "byte_size": self.size,
                "label": self.label,
                "group_id": self.group,
                "exif_tags": list(self.tags),
                "sensitive_metadata": list(self.metadata),
                "augmentation": (
                    self.augmentation.to_dict()
                    if self.augmentation is not None
                    else None
                ),
            }
        )
