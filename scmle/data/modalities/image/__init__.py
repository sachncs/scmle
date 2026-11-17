"""Immutable image samples, bounded profiling, and baseline features.

Pillow is intentionally optional. Importing this module remains safe without
it, while every operation that needs image decoding raises
``DependencyError`` with an actionable, structured error envelope.

The Pillow guard (`PillowModule`, `PILLOW_AVAILABLE` and `pillow`) lives in
this package module rather than a submodule, so that replacing either
attribute here governs every decoding path, as it did when the package was
one module.
"""

from typing import Any

from scmle.data.modalities.image.dataset import Dataset
from scmle.data.modalities.image.duplicates import islands
from scmle.data.modalities.image.evidence import Profile
from scmle.data.modalities.image.evidence import Split
from scmle.data.modalities.image.failures import DependencyError
from scmle.data.modalities.image.failures import ImageError
from scmle.data.modalities.image.failures import ImageValidationError
from scmle.data.modalities.image.pixels import configure
from scmle.data.modalities.image.pixels import IMAGE_SIZE_DIMENSIONS
from scmle.data.modalities.image.pixels import MATRIX_DIMENSIONS
from scmle.data.modalities.image.pixels import MAX_HASH_SIZE
from scmle.data.modalities.image.pixels import MIN_HASH_SIZE
from scmle.data.modalities.image.pixels import SUPPORTED_FEATURE_MODES
from scmle.data.modalities.image.sample import Augmentation
from scmle.data.modalities.image.sample import bounds
from scmle.data.modalities.image.sample import canonical_source
from scmle.data.modalities.image.sample import DEFAULT_MAX_BYTES
from scmle.data.modalities.image.sample import DEFAULT_MAX_PIXELS
from scmle.data.modalities.image.sample import DEFAULT_PROFILE_SIZE
from scmle.data.modalities.image.sample import imprint
from scmle.data.modalities.image.sample import Sample
from scmle.data.modalities.image.sample import SENSITIVE_METADATA_TAGS
from scmle.data.modalities.image.sample import SHA256_HEX_LENGTH

PillowModule: Any = None
try:
    from PIL import Image  # pyright: ignore[reportMissingImports]  # optional extra, imported lazily and guarded; not installed in the type-check environment
except ImportError:
    # handled: optional Pillow absent; PILLOW_AVAILABLE flag reports it
    pass
else:
    PillowModule = Image


PILLOW_AVAILABLE = PillowModule is not None


def pillow(operation: str = "image data") -> Any:
    """Returns the optional Pillow module or raises a structured error.

    Args:
        operation: Operation that requires image decoding.

    Returns:
        The imported Pillow image module.

    Raises:
        DependencyError: If Pillow is not installed.
    """
    if not PILLOW_AVAILABLE or PillowModule is None:
        raise DependencyError(operation)
    return PillowModule


__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_PIXELS",
    "DEFAULT_PROFILE_SIZE",
    "IMAGE_SIZE_DIMENSIONS",
    "MATRIX_DIMENSIONS",
    "MAX_HASH_SIZE",
    "MIN_HASH_SIZE",
    "PILLOW_AVAILABLE",
    "SENSITIVE_METADATA_TAGS",
    "SHA256_HEX_LENGTH",
    "SUPPORTED_FEATURE_MODES",
    "Augmentation",
    "Dataset",
    "DependencyError",
    "ImageError",
    "ImageValidationError",
    "PillowModule",
    "Profile",
    "Sample",
    "Split",
    "bounds",
    "canonical_source",
    "configure",
    "imprint",
    "islands",
    "pillow",
]
