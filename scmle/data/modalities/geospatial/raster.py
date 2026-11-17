"""Raster tiles: identity, metadata and bounded decoding."""

from collections.abc import Callable
import dataclasses
import hashlib
import math
from numbers import Real

import numpy

from scmle.contracts import errors
from scmle.data.modalities.geospatial.geometry import GeometryError

AFFINE_TRANSFORM_LENGTH = 6

RasterDecoder = Callable[[bytes], numpy.ndarray]


def projector(
    transform: tuple[float, ...],
) -> Callable[[int, int], tuple[float, float]]:
    """Builds the map from a pixel boundary to source-CRS coordinates.

    Args:
        transform: A six-value affine transform in the order ``a, b, c, d,
            e, f``, or a nine-value row-major homogeneous matrix.

    Returns:
        A function from (column, row) to (x, y). Under a homogeneous
        transform it raises ``GeometryError`` where the transform is
        singular.
    """
    if len(transform) == AFFINE_TRANSFORM_LENGTH:
        first, second, third, fourth, fifth, sixth = transform

        def project(column: int, row: int) -> tuple[float, float]:
            return (
                first * column + second * row + third,
                fourth * column + fifth * row + sixth,
            )

        return project
    (
        first,
        second,
        third,
        fourth,
        fifth,
        sixth,
        seventh,
        eighth,
        ninth,
    ) = transform

    def homogeneous(column: int, row: int) -> tuple[float, float]:
        denominator = seventh * column + eighth * row + ninth
        if denominator == 0:
            raise GeometryError(
                "raster transform is singular at a pixel boundary",
                {"code": "singular_transform"},
            )
        return (
            (first * column + second * row + third) / denominator,
            (fourth * column + fifth * row + sixth) / denominator,
        )

    return homogeneous


def conform(decoded: numpy.ndarray, tile: "Tile", max_pixels: int) -> None:
    """Checks decoded pixels against a tile's shape and the pixel limit.

    Args:
        decoded: The decoded pixel array.
        tile: The tile whose metadata the pixels must match.
        max_pixels: Maximum decoded array elements allowed.

    Raises:
        GeometryError: If the array is not 2-D or 3-D, is empty or larger
            than ``max_pixels``, or does not match the declared dimensions.
    """
    if decoded.ndim not in {2, 3}:
        raise GeometryError(
            "decoded raster pixels must be grayscale or channel-last",
            {
                "code": "invalid_pixel_shape",
                "source": tile.source,
                "shape": list(decoded.shape),
            },
        )
    if decoded.size < 1 or decoded.size > max_pixels:
        raise GeometryError(
            "decoded raster exceeds the configured pixel limit",
            {
                "code": "pixel_limit",
                "source": tile.source,
                "pixel_count": int(decoded.size),
                "max_pixels": max_pixels,
            },
        )
    expected_shape = (tile.height, tile.width)
    if tile.bands == 1 and decoded.shape == expected_shape:
        return
    if decoded.shape != (tile.height, tile.width, tile.bands):
        raise GeometryError(
            "decoded raster dimensions do not match metadata",
            {
                "code": "pixel_dimension_mismatch",
                "source": tile.source,
                "expected": [tile.height, tile.width, tile.bands],
                "observed": list(decoded.shape),
            },
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Tile:
    """Raster identity and metadata without assuming a raster codec.

    Attributes:
        source: Non-empty source identifier.
        encoded_bytes: Encoded raster payload, which must not be empty.
        file_hash: SHA-256 hex digest of ``encoded_bytes``.
        width: Raster width in pixels.
        height: Raster height in pixels.
        bands: Number of bands.
        dtype: Non-empty pixel data type name.
        crs: Non-empty CRS identifier.
        transform: Empty, a six-value affine transform, or a nine-value
            row-major homogeneous matrix.
        nodata_value: Declared nodata sentinel, stored as a float, or None.
        nodata_fraction: Measured fraction of nodata pixels in ``[0, 1]``,
            or None.
        byte_size: Length of ``encoded_bytes``.
    """

    source: str
    encoded_bytes: bytes = dataclasses.field(repr=False)
    file_hash: str
    width: int
    height: int
    bands: int
    dtype: str
    crs: str
    transform: tuple[float, ...]
    nodata_value: float | int | None = None
    nodata_fraction: float | None = None
    byte_size: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Validates raster shape, CRS provenance, and source identity.

        Raises:
            ValueError: If the source, CRS, or dtype is empty, dimensions or
                bands are not positive, the transform length is not 0, 6, or
                9, nodata values are non-numeric, non-finite, or out of
                range, or ``file_hash`` does not match the payload.
            GeometryError: If the payload is empty.
        """
        raw = bytes(self.encoded_bytes)
        object.__setattr__(self, "encoded_bytes", raw)
        if (
            not self.source.strip()
            or not self.crs.strip()
            or not self.dtype.strip()
        ):
            raise errors.InputError("source, crs, and dtype must not be empty")
        if min(self.width, self.height, self.bands) < 1:
            raise errors.InputError(
                "raster dimensions and bands must be positive"
            )
        if len(self.transform) not in {0, 6, 9}:
            raise errors.InputError(
                "transform must be empty or an affine transform"
            )
        self.normalize()
        if not raw:
            raise GeometryError(
                "raster payload is empty", {"code": "empty_payload"}
            )
        if hashlib.sha256(raw).hexdigest() != self.file_hash:
            raise errors.InputError("file_hash does not match encoded_bytes")
        object.__setattr__(self, "byte_size", len(raw))

    def normalize(self) -> None:
        """Validates the nodata value and fraction, normalising the value.

        Raises:
            ValueError: If the nodata value is non-numeric or non-finite, or
                the nodata fraction is non-finite or outside [0, 1].
        """
        if self.nodata_value is not None:
            nodata: object = self.nodata_value
            if isinstance(nodata, bool) or not isinstance(nodata, Real):
                raise errors.InputError("nodata_value must be numeric")
            if not math.isfinite(float(nodata)):
                raise errors.InputError("nodata_value must be finite")
            object.__setattr__(self, "nodata_value", float(nodata))
        if (
            self.nodata_fraction is not None
            and not 0.0 <= float(self.nodata_fraction) <= 1.0
        ):
            raise errors.InputError("nodata_fraction must be in [0, 1]")
        if self.nodata_fraction is not None and not math.isfinite(
            float(self.nodata_fraction)
        ):
            raise errors.InputError("nodata_fraction must be finite")

    @property
    def nodata_status(self) -> str:
        """``"measured"``, ``"declared"``, or ``"unknown"`` nodata state."""
        if self.nodata_fraction is not None:
            return "measured"
        if self.nodata_value is not None:
            return "declared"
        return "unknown"

    def extent(
        self,
        row_start: int,
        row_end: int,
        column_start: int,
        column_end: int,
    ) -> tuple[float, float, float, float] | None:
        """Returns the CRS bounds of a validated pixel window.

        Pixel boundaries are transformed rather than pixel centers so the
        returned envelope describes the full window. Empty transforms have no
        spatial reference and therefore return ``None``. Six-value transforms
        use the conventional affine order ``a, b, c, d, e, f``; nine-value
        transforms use a row-major homogeneous matrix.

        Args:
            row_start: Inclusive pixel-row boundary.
            row_end: Exclusive pixel-row boundary.
            column_start: Inclusive pixel-column boundary.
            column_end: Exclusive pixel-column boundary.

        Returns:
            Minimum x, minimum y, maximum x, and maximum y in the source CRS,
            or ``None`` when the sample has no transform. A homogeneous
            transform that is singular at a window boundary raises
            ``GeometryError``.

        Raises:
            ValueError: If the window is outside the raster or empty.
        """
        if not (
            0 <= row_start < row_end <= self.height
            and 0 <= column_start < column_end <= self.width
        ):
            raise errors.InputError(
                "pixel window must be non-empty and in bounds"
            )
        if not self.transform:
            return None
        corners = (
            (column_start, row_start),
            (column_start, row_end),
            (column_end, row_start),
            (column_end, row_end),
        )
        project = projector(self.transform)
        projected = tuple(project(column, row) for column, row in corners)
        x_values = tuple(point[0] for point in projected)
        y_values = tuple(point[1] for point in projected)
        return (min(x_values), min(y_values), max(x_values), max(y_values))

    def decode(
        self,
        decoder: RasterDecoder,
        max_pixels: int = 4_000_000,
    ) -> numpy.ndarray:
        """Decodes bounded numeric pixels through a caller-owned decoder.

        The core package does not select a raster codec. The caller owns the
        decoder and must return a grayscale ``(height, width)`` array or a
        channel-last ``(height, width, bands)`` array. The decoded values are
        copied into a read-only float matrix after shape, size, type, and
        finiteness checks.

        Args:
            decoder: Caller-owned function that decodes the encoded payload.
            max_pixels: Maximum decoded array elements allowed per sample.

        Returns:
            A read-only, finite, channel-last-compatible pixel matrix.

        Raises:
            TypeError: If the decoder is not callable.
            ValueError: If ``max_pixels`` is not positive.
            GeometryError: If the decoder raises, the decoded
                array is not 2-D or 3-D, is empty or larger than
                ``max_pixels``, does not match the declared dimensions, is
                not real numeric, or contains non-finite values.
        """
        if not callable(decoder):
            raise errors.KindError("decoder must be callable")
        if max_pixels < 1:
            raise errors.InputError("max_pixels must be positive")
        try:
            decoded = numpy.asarray(decoder(self.encoded_bytes))
        except Exception as error:
            raise GeometryError(
                "raster decoding failed",
                {"code": "decode_failed", "source": self.source},
            ) from error
        conform(decoded, self, max_pixels)
        if not numpy.issubdtype(
            decoded.dtype, numpy.number
        ) or numpy.issubdtype(decoded.dtype, numpy.complexfloating):
            raise GeometryError(
                "decoded raster pixels must be real numeric values",
                {"code": "invalid_pixel_type", "source": self.source},
            )
        values = numpy.asarray(decoded, dtype=numpy.float64).copy()
        if not numpy.isfinite(values).all():
            raise GeometryError(
                "decoded raster pixels must be finite",
                {"code": "non_finite_pixels", "source": self.source},
            )
        values.setflags(write=False)
        return values

    def measured_nodata_fraction(
        self,
        decoder: RasterDecoder,
        max_pixels: int = 4_000_000,
    ) -> float | None:
        """Measures the declared nodata sentinel in decoded pixels.

        Args:
            decoder: Caller-owned function that decodes the encoded payload.
            max_pixels: Maximum decoded array elements allowed per sample.

        Returns:
            The fraction of decoded values equal to the declared nodata
            value, or None when no nodata value is declared. Decoding errors
            are those of ``decode``.
        """
        if self.nodata_value is None:
            return None
        pixels = self.decode(decoder, max_pixels=max_pixels)
        return float(numpy.equal(pixels, self.nodata_value).mean())
