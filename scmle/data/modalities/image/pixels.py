"""Deterministic pixel decoding and the feature configuration it accepts."""

import io
from typing import TYPE_CHECKING

import numpy

from scmle.contracts import errors
from scmle.data.modalities import image
from scmle.data.modalities.image.failures import ViolationError

if TYPE_CHECKING:
    from collections.abc import Iterable

IMAGE_SIZE_DIMENSIONS = 2

MIN_HASH_SIZE = 2
MAX_HASH_SIZE = 32
MATRIX_DIMENSIONS = 2

SUPPORTED_FEATURE_MODES = ("L", "RGB", "RGBA")


def configure(target_size: tuple[int, int], mode: str) -> tuple[int, int]:
    """Validates the deterministic baseline feature configuration.

    Args:
        target_size: ``(width, height)`` of the resized image.
        mode: Pillow mode; one of ``SUPPORTED_FEATURE_MODES``.

    Returns:
        ``target_size`` unchanged.

    Raises:
        ValueError: If ``target_size`` is not two positive integers, or
            ``mode`` is not a supported mode.
    """
    sizes: Iterable[object] = target_size
    if len(target_size) != IMAGE_SIZE_DIMENSIONS or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in sizes
    ):
        raise errors.InputError(
            "target_size must contain two positive integers"
        )
    if mode not in SUPPORTED_FEATURE_MODES:
        raise errors.InputError(
            f"mode must be one of {SUPPORTED_FEATURE_MODES}, got {mode!r}"
        )
    return target_size


class Pixels:
    """Pixel decoding of a validated image sample.

    Attributes:
        source: Source label or URI of the sample.
        encoded: The encoded image payload.
        mode: Pillow mode of the stored image.
    """

    __slots__ = ()

    source: str
    encoded: bytes
    mode: str

    def fingerprint(self, hash_size: int = 8) -> str:
        """Returns a bounded luminance signature for near-duplicate review.

        Args:
            hash_size: Side length of the square luminance thumbnail.

        Returns:
            A hex string of ``hash_size ** 2`` digits, one per pixel of the
            grayscale thumbnail in row-major order, each the pixel value
            divided by 16.

        Raises:
            ValueError: If ``hash_size`` is not an integer from 2 to 32.
        """
        candidate: object = hash_size
        if (
            not isinstance(candidate, int)
            or isinstance(candidate, bool)
            or not MIN_HASH_SIZE <= candidate <= MAX_HASH_SIZE
        ):
            raise errors.InputError(
                "hash_size must be an integer between 2 and 32"
            )
        pixels = self.decode("L", (hash_size, hash_size))[..., 0]
        quantized = numpy.asarray(pixels, dtype=numpy.uint8) // 16
        return "".join(f"{int(value):x}" for value in quantized.flat)

    def decode(
        self,
        mode: str | None = None,
        target_size: tuple[int, int] | None = None,
    ) -> numpy.ndarray:
        """Decodes pixels into a deterministic ``uint8`` NumPy array.

        Args:
            mode: Optional Pillow mode, limited to ``L``, ``RGB`` or ``RGBA``.
            target_size: Optional ``(width, height)`` resize target.

        Returns:
            Array shaped ``(height, width, channels)``.

        Raises:
            ValueError: If the mode is unsupported or ``target_size`` is not
                two positive integers.
            ViolationError: If the stored bytes cannot be decoded.
        """
        image_library = image.pillow("image decoding")
        selected_mode = mode or self.mode
        if selected_mode not in SUPPORTED_FEATURE_MODES:
            raise errors.InputError(
                f"mode must be one of {SUPPORTED_FEATURE_MODES}, got "
                f"{selected_mode!r}"
            )
        if target_size is not None:
            configure(target_size, selected_mode)
        try:
            with image_library.open(io.BytesIO(self.encoded)) as opened:
                converted = opened.convert(selected_mode)
                if target_size is not None:
                    resampling = image_library.Resampling.BILINEAR
                    converted = converted.resize(target_size, resampling)
                pixels = numpy.asarray(converted, dtype=numpy.uint8)
        except Exception as error:
            raise ViolationError(
                f"could not decode validated image: {error}",
                code="decode_failure",
                source=self.source,
                details={"exception": type(error).__name__},
            ) from error
        if pixels.ndim == MATRIX_DIMENSIONS:
            pixels = pixels[..., numpy.newaxis]
        return numpy.ascontiguousarray(pixels)
