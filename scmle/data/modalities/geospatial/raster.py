"""Raster tiles: identity, metadata, bounded decoding and description.

`describe` reads a tile file once, in strips of rows bounded by `STRIP`
bytes, so the memory it holds does not grow with the tile (register row
MM-6). Pixels equal to the declared nodata value, and pixels that are not
finite (NaN nodata, MM-16), are left out of every statistic. It returns the
tile's per-band features, the digest of its decoded pixels (exact
duplicates), a standardised thumbnail whose distance to another tile's
measures how alike the two look (near-duplicates), and the tile's bounds.

`.npy` arrays (height, width[, bands]) are read with numpy; GeoTIFF is read
through the optional `rasterio` library, and without it a GeoTIFF raises
`GeometryError` naming the library.
"""

from collections.abc import Callable
from collections.abc import Iterator
import dataclasses
import hashlib
import importlib
import math
from numbers import Real
import pathlib
from typing import Any

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
        encoded: Encoded raster payload, which must not be empty.
        digest: SHA-256 hex digest of ``encoded``.
        width: Raster width in pixels.
        height: Raster height in pixels.
        bands: Number of bands.
        dtype: Non-empty pixel data type name.
        crs: Non-empty CRS identifier.
        transform: Empty, a six-value affine transform, or a nine-value
            row-major homogeneous matrix.
        sentinel: Declared nodata sentinel, stored as a float, or None.
        nodata: Measured fraction of nodata pixels in ``[0, 1]``,
            or None.
        size: Length of ``encoded``.
    """

    source: str
    encoded: bytes = dataclasses.field(repr=False)
    digest: str
    width: int
    height: int
    bands: int
    dtype: str
    crs: str
    transform: tuple[float, ...]
    sentinel: float | int | None = None
    nodata: float | None = None
    size: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Validates raster shape, CRS provenance, and source identity.

        Raises:
            ValueError: If the source, CRS, or dtype is empty, dimensions or
                bands are not positive, the transform length is not 0, 6, or
                9, nodata values are non-numeric, non-finite, or out of
                range, or ``digest`` does not match the payload.
            GeometryError: If the payload is empty.
        """
        raw = bytes(self.encoded)
        object.__setattr__(self, "encoded", raw)
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
        if hashlib.sha256(raw).hexdigest() != self.digest:
            raise errors.InputError("file_hash does not match encoded_bytes")
        object.__setattr__(self, "size", len(raw))

    def normalize(self) -> None:
        """Validates the nodata value and fraction, normalising the value.

        Raises:
            ValueError: If the nodata value is non-numeric or non-finite, or
                the nodata fraction is non-finite or outside [0, 1].
        """
        if self.sentinel is not None:
            nodata: object = self.sentinel
            if isinstance(nodata, bool) or not isinstance(nodata, Real):
                raise errors.InputError("nodata_value must be numeric")
            if not math.isfinite(float(nodata)):
                raise errors.InputError("nodata_value must be finite")
            object.__setattr__(self, "sentinel", float(nodata))
        if self.nodata is not None and not 0.0 <= float(self.nodata) <= 1.0:
            raise errors.InputError("nodata_fraction must be in [0, 1]")
        if self.nodata is not None and not math.isfinite(float(self.nodata)):
            raise errors.InputError("nodata_fraction must be finite")

    @property
    def status(self) -> str:
        """``"measured"``, ``"declared"``, or ``"unknown"`` nodata state."""
        if self.nodata is not None:
            return "measured"
        if self.sentinel is not None:
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
            decoded = numpy.asarray(decoder(self.encoded))
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

    def measure(
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
        if self.sentinel is None:
            return None
        pixels = self.decode(decoder, max_pixels=max_pixels)
        return float(numpy.equal(pixels, self.sentinel).mean())


STRIP = 1024 * 1024
"""Bytes of float64 pixels decoded per strip. A strip's statistics make
about a dozen strip-sized temporaries, so 1 MiB keeps the working set of
one tile near 16 MiB whatever its size."""

GRID = 8
"""Thumbnail cells per side: 8 x 8 = 64 cells, as many as an audio
envelope."""

NOISE = 20.0
"""Signal-to-noise ratio in dB down to which a noisy copy of a tile is the
same scene: noise at 20 dB has a tenth of the signal's amplitude."""

RADIUS = 10.0 ** (-NOISE / 20.0)
"""Largest root-mean-square thumbnail difference of two near-copies. A
thumbnail is standardised, so noise `NOISE` dB below the signal moves a
cell by at most a tenth of a standard deviation before the cell's average
shrinks it further."""


@dataclasses.dataclass(frozen=True)
class Shape:
    """What a tile file says about its pixels.

    Attributes:
        height: Rows of pixels.
        width: Columns of pixels.
        bands: Bands per pixel.
        nodata: The declared nodata value, or None.
        bounds: West, south, east and north in the CRS, or None.
        crs: The CRS identifier; empty when unknown.
    """

    height: int
    width: int
    bands: int
    nodata: float | None
    bounds: tuple[float, float, float, float] | None
    crs: str


class Array:
    """A `.npy` tile, height by width or height by width by bands.

    Attributes:
        path: The tile's file.
        values: The memory-mapped array.
    """

    def __init__(self, path: pathlib.Path) -> None:
        """Maps the file without reading it.

        Args:
            path: The tile's file.

        Raises:
            GeometryError: If the file is not a numeric 2-D or 3-D array.
        """
        self.path = path
        try:
            self.values = numpy.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, EOFError) as error:
            raise GeometryError(
                f"'{path}' is not a readable .npy array",
                {"code": "invalid_raster", "source": str(path)},
            ) from error
        if (
            self.values.ndim not in {2, 3}
            or not self.values.size
            or not numpy.issubdtype(self.values.dtype, numpy.number)
            or numpy.issubdtype(self.values.dtype, numpy.complexfloating)
        ):
            raise GeometryError(
                f"'{path}' must hold real numbers, height by width[ by bands]",
                {
                    "code": "invalid_pixel_shape",
                    "source": str(path),
                    "shape": list(self.values.shape),
                },
            )

    def shape(self) -> Shape:
        """The array's dimensions; it declares no nodata, bounds or CRS.

        Returns:
            The shape.
        """
        dims = (*self.values.shape, 1)
        return Shape(dims[0], dims[1], dims[2], None, None, "")

    def strips(self, rows: int) -> Iterator[numpy.ndarray]:
        """Reads `rows` rows at a time.

        Args:
            rows: Rows per strip.

        Yields:
            Strips of rows by columns by bands, as float64.
        """
        for start in range(0, self.values.shape[0], rows):
            block = numpy.asarray(
                self.values[start : start + rows], dtype=numpy.float64
            )
            yield block.reshape(block.shape[0], block.shape[1], -1)


class Geotiff:
    """A GeoTIFF tile read through the optional `rasterio` library.

    Attributes:
        path: The tile's file.
        library: The imported `rasterio` module.
    """

    def __init__(self, path: pathlib.Path) -> None:
        """Binds the file and imports the library.

        Args:
            path: The tile's file.

        Raises:
            GeometryError: If `rasterio` is not installed.
        """
        self.path = path
        try:
            self.library: Any = importlib.import_module("rasterio")
        except ImportError as error:
            raise GeometryError(
                f"'{path.name}' is a GeoTIFF; reading it needs the optional "
                "rasterio library (pip install rasterio), or save the tile "
                "as .npy",
                {"code": "decoder_missing", "source": str(path)},
            ) from error

    def shape(self) -> Shape:
        """The tile's dimensions, nodata value, bounds and CRS.

        Returns:
            The shape.
        """
        with self.library.open(self.path) as opened:
            bounds = tuple(float(value) for value in opened.bounds)
            return Shape(
                opened.height,
                opened.width,
                opened.count,
                None if opened.nodata is None else float(opened.nodata),
                (bounds[0], bounds[1], bounds[2], bounds[3])
                if opened.transform is not None
                else None,
                str(opened.crs or ""),
            )

    def strips(self, rows: int) -> Iterator[numpy.ndarray]:
        """Reads `rows` rows at a time.

        Args:
            rows: Rows per strip.

        Yields:
            Strips of rows by columns by bands, as float64.
        """
        windows = self.library.windows
        with self.library.open(self.path) as opened:
            for start in range(0, opened.height, rows):
                window = windows.Window(
                    0, start, opened.width, min(rows, opened.height - start)
                )
                block = opened.read(window=window).astype(numpy.float64)
                yield numpy.moveaxis(block, 0, -1)


def opener(path: pathlib.Path) -> Array | Geotiff:
    """The reader of a tile's format.

    Args:
        path: The tile's file.

    Returns:
        `Array` for `.npy`, `Geotiff` for `.tif` and `.tiff`.

    Raises:
        GeometryError: For any other suffix.
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return Array(path)
    if suffix in {".tif", ".tiff"}:
        return Geotiff(path)
    raise GeometryError(
        f"'{path.name}' is neither .npy nor GeoTIFF",
        {"code": "unsupported_raster", "source": str(path)},
    )


class Scan:
    """Running sums over a tile's strips.

    Attributes:
        shape: The tile's shape.
        count: Valid pixels per band.
        total: Sum per band.
        energy: Sum of squares per band.
        low: Smallest value per band.
        high: Largest value per band.
        rough: Sum of absolute neighbour differences per band.
        pairs: Valid neighbour pairs per band.
        last: The previous strip's last row, or None.
        row: Rows already read.
        cells: Thumbnail sums of the band mean.
        weights: Valid pixels per thumbnail cell.
        hasher: The digest of the decoded pixels.
    """

    def __init__(self, shape: Shape) -> None:
        """Starts empty sums.

        Args:
            shape: The tile's shape.
        """
        self.shape = shape
        bands = shape.bands
        self.count = numpy.zeros(bands)
        self.total = numpy.zeros(bands)
        self.energy = numpy.zeros(bands)
        self.low = numpy.full(bands, numpy.inf)
        self.high = numpy.full(bands, -numpy.inf)
        self.rough = numpy.zeros(bands)
        self.pairs = numpy.zeros(bands)
        self.last: numpy.ndarray | None = None
        self.row = 0
        self.cells = numpy.zeros((GRID, GRID))
        self.weights = numpy.zeros((GRID, GRID))
        self.hasher = hashlib.sha256(
            f"{shape.height}:{shape.width}:{shape.bands}".encode()
        )

    def add(self, strip: numpy.ndarray) -> None:
        """Adds one strip.

        Args:
            strip: Rows by columns by bands; nodata already NaN.
        """
        self.hasher.update(strip.tobytes())
        valid = numpy.isfinite(strip)
        zeroed = numpy.where(valid, strip, 0.0)
        self.count += valid.sum(axis=(0, 1))
        self.total += zeroed.sum(axis=(0, 1))
        self.energy += numpy.square(zeroed).sum(axis=(0, 1))
        self.low = numpy.fmin(
            self.low, numpy.nanmin(strip, axis=(0, 1), initial=numpy.inf)
        )
        self.high = numpy.fmax(
            self.high, numpy.nanmax(strip, axis=(0, 1), initial=-numpy.inf)
        )
        stacked = (
            strip
            if self.last is None
            else numpy.concatenate([self.last[None], strip])
        )
        for steps in (numpy.diff(strip, axis=1), numpy.diff(stacked, axis=0)):
            finite = numpy.isfinite(steps)
            self.rough += numpy.where(finite, numpy.abs(steps), 0.0).sum(
                axis=(0, 1)
            )
            self.pairs += finite.sum(axis=(0, 1))
        self.last = strip[-1]
        counted = valid.sum(axis=2)
        mean = zeroed.sum(axis=2) / numpy.maximum(counted, 1)
        rows = (
            (self.row + numpy.arange(strip.shape[0]))
            * GRID
            // self.shape.height
        )
        columns = numpy.arange(strip.shape[1]) * GRID // self.shape.width
        where = (rows[:, None], columns[None, :])
        present = counted > 0
        numpy.add.at(self.cells, where, numpy.where(present, mean, 0.0))
        numpy.add.at(self.weights, where, present)
        self.row += strip.shape[0]

    def finish(self) -> "Picture":
        """Closes the sums.

        Returns:
            The tile's features, digest, thumbnail and bounds.

        Raises:
            GeometryError: If the tile has no valid pixel.
        """
        if not self.count.sum():
            raise GeometryError(
                "the tile holds no valid pixel (all nodata)",
                {"code": "all_nodata"},
            )
        count = numpy.maximum(self.count, 1)
        mean = self.total / count
        spread = numpy.sqrt(
            numpy.maximum(self.energy / count - mean * mean, 0.0)
        )
        size = float(self.shape.height * self.shape.width)
        features: dict[str, float] = {
            "raster_height": float(self.shape.height),
            "raster_width": float(self.shape.width),
            "raster_bands": float(self.shape.bands),
        }
        for band in range(self.shape.bands):
            empty = not self.count[band]
            features |= {
                f"raster_band{band}_mean": float(mean[band]),
                f"raster_band{band}_std": float(spread[band]),
                f"raster_band{band}_min": 0.0
                if empty
                else float(self.low[band]),
                f"raster_band{band}_max": 0.0
                if empty
                else float(self.high[band]),
                f"raster_band{band}_valid": float(self.count[band] / size),
                f"raster_band{band}_rough": float(
                    self.rough[band] / max(self.pairs[band], 1.0)
                ),
            }
        return Picture(
            features,
            self.hasher.hexdigest(),
            thumbnail(self),
            self.shape.bounds,
            self.shape.crs,
        )


def thumbnail(scan: Scan) -> tuple[float, ...]:
    """The standardised `GRID` x `GRID` thumbnail of the band mean.

    Standardising (mean 0, standard deviation 1 over the cells) makes the
    thumbnail blind to gain and offset, so a brighter or rescaled copy of a
    tile keeps it. A flat thumbnail is all zeros.

    Args:
        scan: The closed sums of a tile.

    Returns:
        `GRID` x `GRID` cells, row by row.
    """
    cells = scan.cells / numpy.maximum(scan.weights, 1.0)
    cells = numpy.where(
        scan.weights > 0, cells, float(numpy.mean(cells[scan.weights > 0]))
    )
    spread = float(cells.std())
    if spread <= 0.0:
        return (0.0,) * cells.size
    return tuple(
        float(value) for value in ((cells - cells.mean()) / spread).flat
    )


@dataclasses.dataclass(frozen=True)
class Picture:
    """What one tile yields.

    Attributes:
        features: Feature name to finite value.
        digest: SHA-256 of the decoded pixels and their shape.
        envelope: The standardised thumbnail, for near-duplicates.
        bounds: West, south, east and north in the CRS, or None.
        crs: The CRS identifier; empty when unknown.
    """

    features: dict[str, float]
    digest: str
    envelope: tuple[float, ...]
    bounds: tuple[float, float, float, float] | None
    crs: str


def describe(
    path: pathlib.Path,
    nodata: float | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    crs: str = "",
) -> Picture:
    """Reads a whole tile in bounded strips and describes it.

    Args:
        path: The tile's file.
        nodata: The nodata value, overriding one the file declares.
        bounds: West, south, east and north, overriding the file's.
        crs: The CRS identifier, overriding the file's.

    Returns:
        Its features, digest, thumbnail, bounds and CRS.

    Raises:
        GeometryError: If the file is unreadable, its format unsupported or
            missing its optional library, or it holds no valid pixel.
    """
    source = opener(path)
    found = source.shape()
    shape = dataclasses.replace(
        found,
        nodata=found.nodata if nodata is None else float(nodata),
        bounds=found.bounds if bounds is None else bounds,
        crs=crs or found.crs,
    )
    scan = Scan(shape)
    rows = max(1, STRIP // (8 * shape.width * shape.bands))
    for strip in source.strips(rows):
        scan.add(
            strip
            if shape.nodata is None
            else numpy.where(strip == shape.nodata, numpy.nan, strip)
        )
    try:
        return scan.finish()
    except GeometryError as error:
        error.details = {**error.details, "source": str(path)}
        raise
