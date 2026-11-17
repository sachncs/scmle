"""The schema, missingness, duplicate and label evidence of a catalog."""

from collections.abc import Iterable
from collections.abc import Mapping
import dataclasses
import math
from typing import Any
from typing import Self

from scmle.contracts import errors


@dataclasses.dataclass(frozen=True, slots=True)
class Field:
    """One ordered field in an observed dataset schema."""

    __module__ = "scmle.data.catalog"

    name: str
    data_type: str

    def __post_init__(self) -> None:
        """Validates schema field names and types, and strips their spaces.

        Raises:
            ValueError: If ``name`` or ``data_type`` is not a string or is
                empty after stripping whitespace.
        """
        name: object = self.name
        data_type: object = self.data_type
        if not isinstance(name, str) or not name.strip():
            raise errors.InputError("field name must not be empty")
        if not isinstance(data_type, str) or not data_type.strip():
            raise errors.InputError("field data_type must not be empty")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "data_type", self.data_type.strip())

    def to_dict(self) -> dict[str, str]:
        """Returns JSON-native field evidence.

        Returns:
            A dictionary with ``name`` and ``data_type``.
        """
        return {"name": self.name, "data_type": self.data_type}

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a schema field from JSON evidence.

        Args:
            record: A mapping with ``name`` and ``data_type`` entries.

        Returns:
            The reconstructed field.
        """
        return cls(str(record["name"]), str(record["data_type"]))


@dataclasses.dataclass(frozen=True, slots=True)
class Missingness:
    """Observed missing-value state, including fields with missingness."""

    __module__ = "scmle.data.catalog"

    status: str
    total_values: int
    missing_values: int
    rows_with_missing: int
    by_field: Mapping[str, int]

    def __post_init__(self) -> None:
        """Validates missingness accounting.

        Raises:
            TypeError: If a count is not an integer (booleans are rejected).
            ValueError: If ``status`` is empty, a count is negative,
                ``missing_values`` exceeds ``total_values``, or ``by_field``
                has a blank name or a negative or non-integer count.
        """
        if not self.status.strip():
            raise errors.InputError("missingness status must not be empty")
        for field_name, field_value in (
            ("total_values", self.total_values),
            ("missing_values", self.missing_values),
            ("rows_with_missing", self.rows_with_missing),
        ):
            if isinstance(field_value, bool) or not isinstance(
                field_value, int
            ):
                raise errors.KindError(f"{field_name} must be an integer")
            if field_value < 0:
                raise errors.InputError(f"{field_name} must not be negative")
        if self.missing_values > self.total_values:
            raise errors.InputError("missing_values cannot exceed total_values")
        by_field_items: Iterable[tuple[object, object]] = self.by_field.items()
        if any(
            not isinstance(name, str)
            or not name.strip()
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for name, count in by_field_items
        ):
            raise errors.InputError(
                "by_field must contain non-negative named counts"
            )
        object.__setattr__(
            self, "by_field", dict(sorted(self.by_field.items()))
        )

    @property
    def fraction(self) -> float:
        """The fraction of observed values that are missing.

        This is ``missing_values / total_values``, or 0.0 when there are no
        values.
        """
        if self.total_values == 0:
            return 0.0
        return self.missing_values / self.total_values

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native missingness evidence.

        Returns:
            A dictionary of the counts, the per-field counts, and the computed
            ``missing_fraction``.
        """
        return {
            "status": self.status,
            "total_values": self.total_values,
            "missing_values": self.missing_values,
            "missing_fraction": self.fraction,
            "rows_with_missing": self.rows_with_missing,
            "by_field": dict(self.by_field),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds missingness evidence and recomputes its fraction.

        Args:
            record: A mapping as produced by ``to_dict``.  A recorded
                ``missing_fraction`` is optional and is checked when present.

        Returns:
            The reconstructed summary.

        Raises:
            ValueError: If a recorded ``missing_fraction`` does not match the
                fraction computed from the counts.
        """
        result = cls(
            status=str(record["status"]),
            total_values=int(record["total_values"]),
            missing_values=int(record["missing_values"]),
            rows_with_missing=int(record["rows_with_missing"]),
            by_field={
                str(name): int(count)
                for name, count in dict(record.get("by_field", {})).items()
            },
        )
        observed_fraction = record.get("missing_fraction")
        if observed_fraction is not None and not math.isclose(
            float(observed_fraction), result.fraction
        ):
            raise errors.InputError(
                "missing_fraction does not match missing values"
            )
        return result


@dataclasses.dataclass(frozen=True, slots=True)
class Duplication:
    """Observed duplicate-row state and the method used to measure it."""

    __module__ = "scmle.data.catalog"

    status: str
    duplicate_rows: int
    duplicate_fraction: float
    method: str

    def __post_init__(self) -> None:
        """Validates duplicate accounting.

        Raises:
            ValueError: If ``status`` or ``method`` is empty, ``duplicate_rows``
                is negative, or ``duplicate_fraction`` is not a finite number
                in [0, 1].
            TypeError: If ``duplicate_rows`` is not an integer (booleans are
                rejected).
        """
        if not self.status.strip() or not self.method.strip():
            raise errors.InputError(
                "duplicate status and method must not be empty"
            )
        if isinstance(self.duplicate_rows, bool) or not isinstance(
            self.duplicate_rows, int
        ):
            raise errors.KindError("duplicate_rows must be an integer")
        if self.duplicate_rows < 0:
            raise errors.InputError("duplicate_rows must not be negative")
        if (
            not math.isfinite(self.duplicate_fraction)
            or not 0 <= (self.duplicate_fraction) <= 1
        ):
            raise errors.InputError(
                "duplicate_fraction must be between zero and one"
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native duplicate evidence.

        Returns:
            A dictionary of status, duplicate row count, duplicate fraction,
            and method.
        """
        return {
            "status": self.status,
            "duplicate_rows": self.duplicate_rows,
            "duplicate_fraction": self.duplicate_fraction,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds duplicate evidence.

        Args:
            record: A mapping as produced by ``to_dict``.

        Returns:
            The reconstructed summary.
        """
        return cls(
            status=str(record["status"]),
            duplicate_rows=int(record["duplicate_rows"]),
            duplicate_fraction=float(record["duplicate_fraction"]),
            method=str(record["method"]),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Labeling:
    """Explicit label state without inferring a learning task."""

    __module__ = "scmle.data.catalog"

    status: str
    field: str | None
    observed_count: int
    missing_count: int
    distinct_count: int
    distribution: Mapping[str, int]

    def __post_init__(self) -> None:
        """Validates label accounting.

        Raises:
            ValueError: If ``status`` is empty, ``field`` is blank, a count is
                negative, ``distinct_count`` exceeds ``observed_count``,
                ``distribution`` has an empty key or a negative or non-integer
                count, or its counts do not sum to ``observed_count``.
            TypeError: If a count is not an integer (booleans are rejected).
        """
        if not self.status.strip():
            raise errors.InputError("label status must not be empty")
        if self.field is not None and not self.field.strip():
            raise errors.InputError("label field must not be empty")
        for field_name, field_value in (
            ("observed_count", self.observed_count),
            ("missing_count", self.missing_count),
            ("distinct_count", self.distinct_count),
        ):
            if isinstance(field_value, bool) or not isinstance(
                field_value, int
            ):
                raise errors.KindError(f"{field_name} must be an integer")
            if field_value < 0:
                raise errors.InputError(f"{field_name} must not be negative")
        if self.distinct_count > self.observed_count:
            raise errors.InputError(
                "distinct_count cannot exceed observed_count"
            )
        distribution_items: Iterable[tuple[object, object]] = (
            self.distribution.items()
        )
        if any(
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for key, value in distribution_items
        ):
            raise errors.InputError(
                "distribution must contain non-negative counts"
            )
        if sum(self.distribution.values()) != self.observed_count:
            raise errors.InputError(
                "distribution does not match observed_count"
            )
        object.__setattr__(
            self, "distribution", dict(sorted(self.distribution.items()))
        )

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native label evidence.

        Returns:
            A dictionary of status, field, counts, and the label distribution.
        """
        return {
            "status": self.status,
            "field": self.field,
            "observed_count": self.observed_count,
            "missing_count": self.missing_count,
            "distinct_count": self.distinct_count,
            "distribution": dict(self.distribution),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds label evidence.

        Args:
            record: A mapping as produced by ``to_dict``.  ``distribution`` is
                optional and defaults to empty.

        Returns:
            The reconstructed summary.
        """
        return cls(
            status=str(record["status"]),
            field=record.get("field"),
            observed_count=int(record["observed_count"]),
            missing_count=int(record["missing_count"]),
            distinct_count=int(record["distinct_count"]),
            distribution={
                str(label): int(count)
                for label, count in dict(record.get("distribution", {})).items()
            },
        )
