"""Cancellation, feature batches, partitions and package identity."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import json
import threading
from typing import Any
from typing import ClassVar

import numpy

from scmle.contracts import errors
from scmle.contracts import records
from scmle.pipeline.modality_lifecycle import constants


class AbortedError(errors.CancelledError):
    """Raised when a modality lifecycle receives a cancellation request."""

    def __init__(self, message: str) -> None:
        """Creates a cancellation error with an actionable message.

        Args:
            message: Explanation of why and where the run was cancelled.
        """
        super().__init__(message, {"scope": "modality_lifecycle"})


class Cancellation:
    """Thread-safe cooperative cancellation signal for one lifecycle run."""

    def __init__(self) -> None:
        """Creates a cancellation signal in the not-cancelled state."""
        self.signal = threading.Event()

    def cancel(self) -> None:
        """Requests cancellation at the next lifecycle stage boundary."""
        self.signal.set()

    def is_cancelled(self) -> bool:
        """Reports whether cancellation has been requested.

        Returns:
            True once `cancel` has been called, otherwise False.
        """
        return self.signal.is_set()


@dataclasses.dataclass(frozen=True, slots=True)
class Batch:
    """A finite, bounded feature matrix tied to stable sample identities.

    Attributes:
        sample_ids: Unique identifier of each row, in row order.
        feature_names: Non-empty name of each column, in column order.
        values: Read-only float matrix with one row per sample ID and one column
            per feature name; every value is finite.
    """

    sample_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: numpy.ndarray = dataclasses.field(repr=False)

    def __post_init__(self) -> None:
        """Validates shape, finiteness, and immutable identity metadata.

        Raises:
            ValueError: If the values are not a two-dimensional matrix, the row
                or column counts do not match the IDs or names, a value is not
                finite, sample IDs repeat, or a feature name is blank.
        """
        matrix = numpy.asarray(self.values, dtype=numpy.float64)
        if matrix.ndim != constants.MATRIX_RANK:
            raise errors.InputError(
                "feature values must be a two-dimensional matrix"
            )
        if matrix.shape[0] != len(self.sample_ids):
            raise errors.InputError("feature rows must match sample IDs")
        if matrix.shape[1] != len(self.feature_names):
            raise errors.InputError("feature columns must match feature names")
        if not numpy.isfinite(matrix).all():
            raise errors.InputError("feature values must be finite")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise errors.InputError(
                "sample IDs must be unique within a feature batch"
            )
        if any(not name.strip() for name in self.feature_names):
            raise errors.InputError("feature names must not be empty")
        frozen = matrix.copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "sample_ids", tuple(self.sample_ids))
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "values", frozen)


@dataclasses.dataclass(frozen=True, slots=True)
class Partitions:
    """Leakage-aware train, validation, and test partitions.

    Attributes:
        train: Samples used to fit the model.
        validation: Samples held out for validation.
        test: Samples held out for the final evaluation.
        groups: Mapping of partition name to the sorted group identifiers it
            contains.
    """

    train: tuple[Any, ...]
    validation: tuple[Any, ...]
    test: tuple[Any, ...]
    groups: dict[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        """Normalizes partitions to tuples and sorts each group list."""
        object.__setattr__(self, "train", tuple(self.train))
        object.__setattr__(self, "validation", tuple(self.validation))
        object.__setattr__(self, "test", tuple(self.test))
        normalized = {
            name: tuple(sorted(values)) for name, values in self.groups.items()
        }
        object.__setattr__(self, "groups", normalized)


@dataclasses.dataclass(frozen=True, slots=True)
class Package:
    """Digest and signature identity of an atomically written package.

    Attributes:
        path: Package directory.
        model_sha256: SHA-256 digest of the serialized model.
        signature: Signature value of the package manifest.
        schema_version: Version of the package layout.
        KEYS: Version-1 keys to the written ones (`digests.model`);
            `downgrade` restores version 1, the layout a lifecycle report's
            signature covers.
    """

    KEYS: ClassVar[records.Migration] = records.nest(
        {}, groups={"digests": {"model_sha256": "model"}}
    )

    path: str
    model_sha256: str
    signature: str
    schema_version: int = constants.PACKAGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validates package identity before it enters lifecycle evidence.

        Raises:
            ValueError: If the path or signature is blank, the model digest is
                not a SHA-256 digest, or the schema version is unsupported.
        """
        path: object = self.path
        model_sha256: object = self.model_sha256
        signature: object = self.signature
        if not isinstance(path, str) or not path.strip():
            raise errors.InputError("package path must be non-empty")
        if not isinstance(model_sha256, str) or not hashlike(model_sha256):
            raise errors.InputError(
                "package model_sha256 must be a SHA-256 digest"
            )
        if not isinstance(signature, str) or not signature.strip():
            raise errors.InputError("package signature must be non-empty")
        if self.schema_version != constants.PACKAGE_SCHEMA_VERSION:
            raise errors.InputError(
                "unsupported modality package schema version"
            )

    def to_dict(self) -> dict[str, object]:
        """Converts the package identity to JSON-native values.

        Returns:
            The path, signature and schema version, and the model digest
            under `digests.model`.
        """
        return self.KEYS.upgrade(dataclasses.asdict(self))


def hashlike(value: str) -> bool:
    """Checks for a canonical lowercase SHA-256 digest.

    Args:
        value: Text to check.

    Returns:
        True if `value` is 64 lowercase hexadecimal characters, else False.
    """
    return len(value) == constants.SHA256_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


def normalise(value: Mapping[str, object], name: str) -> dict[str, object]:
    """Normalizes a finite JSON object or rejects the evidence boundary.

    Args:
        value: Mapping to normalize.
        name: Field name used in error messages.

    Returns:
        A new dictionary of JSON-native values with sorted keys.

    Raises:
        TypeError: If `value` is not a mapping.
        ValueError: If `value` is not JSON-native, has non-finite numbers, or
            does not serialize to an object.
    """
    if not isinstance(value, Mapping):
        raise errors.KindError(f"{name} must be a mapping")
    try:
        normalized = json.loads(
            json.dumps(dict(value), allow_nan=False, sort_keys=True)
        )
    except (TypeError, ValueError) as error:
        raise errors.InputError(f"{name} must be JSON-native") from error
    if not isinstance(normalized, dict):
        raise errors.InputError(f"{name} must serialize to an object")
    return normalized
