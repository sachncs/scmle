"""The schema, missingness, duplicate and label evidence of a catalog.

These parts carry no version of their own: the catalog that holds them is
versioned, and each part lists the renames of its stored keys in
`migrations`, which the catalog's own migration applies to it.
"""

from collections.abc import Iterable
from collections.abc import Mapping
import dataclasses
import math
from typing import Any
from typing import ClassVar
from typing import Self

from scmle.contracts import errors
from scmle.contracts import records


@dataclasses.dataclass(frozen=True, slots=True)
class Field:
    """One ordered field in an observed dataset schema.

    Attributes:
        name: The field's name.
        kind: The field's data type name.
    """

    __module__ = "scmle.data.catalog"

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(renames={"data_type": "kind"}),
    )

    name: str
    kind: str

    def __post_init__(self) -> None:
        """Validates schema field names and types, and strips their spaces.

        Raises:
            ValueError: If ``name`` or ``kind`` is not a string or is empty
                after stripping whitespace.
        """
        name: object = self.name
        kind: object = self.kind
        if not isinstance(name, str) or not name.strip():
            raise errors.InputError("field name must not be empty")
        if not isinstance(kind, str) or not kind.strip():
            raise errors.InputError("field kind must not be empty")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "kind", self.kind.strip())

    def to_dict(self) -> dict[str, str]:
        """Returns JSON-native field evidence.

        Returns:
            A dictionary with ``name`` and ``kind``.
        """
        return {"name": self.name, "kind": self.kind}

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a schema field from JSON evidence.

        Args:
            record: A mapping with ``name`` and ``kind`` entries.

        Returns:
            The reconstructed field.
        """
        return cls(str(record["name"]), str(record["kind"]))


@dataclasses.dataclass(frozen=True, slots=True)
class Missingness:
    """Observed missing-value state, including fields with missingness.

    Attributes:
        status: Whether missingness was observed.
        total: Number of values observed.
        missing: Number of those values that are missing.
        incomplete: Number of rows with at least one missing value.
        fields: Field name to its number of missing values.
    """

    __module__ = "scmle.data.catalog"

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={
                "total_values": "total",
                "missing_values": "missing",
                "rows_with_missing": "incomplete",
                "by_field": "fields",
            }
        ),
    )

    status: str
    total: int
    missing: int
    incomplete: int
    fields: Mapping[str, int]

    def __post_init__(self) -> None:
        """Validates missingness accounting.

        Raises:
            TypeError: If a count is not an integer (booleans are rejected).
            ValueError: If ``status`` is empty, a count is negative,
                ``missing`` exceeds ``total``, or ``fields`` has a blank name
                or a negative or non-integer count.
        """
        if not self.status.strip():
            raise errors.InputError("missingness status must not be empty")
        for field_name, field_value in (
            ("total", self.total),
            ("missing", self.missing),
            ("incomplete", self.incomplete),
        ):
            if isinstance(field_value, bool) or not isinstance(
                field_value, int
            ):
                raise errors.KindError(f"{field_name} must be an integer")
            if field_value < 0:
                raise errors.InputError(f"{field_name} must not be negative")
        if self.missing > self.total:
            raise errors.InputError("missing cannot exceed total")
        field_items: Iterable[tuple[object, object]] = self.fields.items()
        if any(
            not isinstance(name, str)
            or not name.strip()
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for name, count in field_items
        ):
            raise errors.InputError(
                "fields must contain non-negative named counts"
            )
        object.__setattr__(self, "fields", dict(sorted(self.fields.items())))

    @property
    def fraction(self) -> float:
        """The fraction of observed values that are missing.

        This is ``missing / total``, or 0.0 when there are no values.
        """
        if self.total == 0:
            return 0.0
        return self.missing / self.total

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native missingness evidence.

        Returns:
            A dictionary of the counts, the per-field counts, and the computed
            ``missing_fraction``.
        """
        return {
            "status": self.status,
            "total": self.total,
            "missing": self.missing,
            "missing_fraction": self.fraction,
            "incomplete": self.incomplete,
            "fields": dict(self.fields),
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
            total=int(record["total"]),
            missing=int(record["missing"]),
            incomplete=int(record["incomplete"]),
            fields={
                str(name): int(count)
                for name, count in dict(record.get("fields", {})).items()
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
    """Observed duplicate-row state and the method used to measure it.

    Attributes:
        status: Whether duplicates were observed.
        rows: Number of duplicate rows.
        fraction: Fraction of rows that are duplicates, in [0, 1].
        method: How duplicates were measured.
    """

    __module__ = "scmle.data.catalog"

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={"duplicate_rows": "rows", "duplicate_fraction": "fraction"}
        ),
    )

    status: str
    rows: int
    fraction: float
    method: str

    def __post_init__(self) -> None:
        """Validates duplicate accounting.

        Raises:
            ValueError: If ``status`` or ``method`` is empty, ``rows`` is
                negative, or ``fraction`` is not a finite number in [0, 1].
            TypeError: If ``rows`` is not an integer (booleans are rejected).
        """
        if not self.status.strip() or not self.method.strip():
            raise errors.InputError(
                "duplicate status and method must not be empty"
            )
        if isinstance(self.rows, bool) or not isinstance(self.rows, int):
            raise errors.KindError("rows must be an integer")
        if self.rows < 0:
            raise errors.InputError("rows must not be negative")
        if not math.isfinite(self.fraction) or not 0 <= self.fraction <= 1:
            raise errors.InputError("fraction must be between zero and one")

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native duplicate evidence.

        Returns:
            A dictionary of status, duplicate row count, duplicate fraction,
            and method.
        """
        return {
            "status": self.status,
            "rows": self.rows,
            "fraction": self.fraction,
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
            rows=int(record["rows"]),
            fraction=float(record["fraction"]),
            method=str(record["method"]),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Labeling:
    """Explicit label state without inferring a learning task.

    Attributes:
        status: Whether a label was declared and observed.
        field: The label column, or None when none was declared.
        observed: Number of observed (non-missing) labels.
        missing: Number of missing labels.
        distinct: Number of distinct observed labels.
        distribution: Label to its count; the counts sum to ``observed``.
    """

    __module__ = "scmle.data.catalog"

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={
                "observed_count": "observed",
                "missing_count": "missing",
                "distinct_count": "distinct",
            }
        ),
    )

    status: str
    field: str | None
    observed: int
    missing: int
    distinct: int
    distribution: Mapping[str, int]

    def __post_init__(self) -> None:
        """Validates label accounting.

        Raises:
            ValueError: If ``status`` is empty, ``field`` is blank, a count is
                negative, ``distinct`` exceeds ``observed``, ``distribution``
                has an empty key or a negative or non-integer count, or its
                counts do not sum to ``observed``.
            TypeError: If a count is not an integer (booleans are rejected).
        """
        if not self.status.strip():
            raise errors.InputError("label status must not be empty")
        if self.field is not None and not self.field.strip():
            raise errors.InputError("label field must not be empty")
        for field_name, field_value in (
            ("observed", self.observed),
            ("missing", self.missing),
            ("distinct", self.distinct),
        ):
            if isinstance(field_value, bool) or not isinstance(
                field_value, int
            ):
                raise errors.KindError(f"{field_name} must be an integer")
            if field_value < 0:
                raise errors.InputError(f"{field_name} must not be negative")
        if self.distinct > self.observed:
            raise errors.InputError("distinct cannot exceed observed")
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
        if sum(self.distribution.values()) != self.observed:
            raise errors.InputError("distribution does not match observed")
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
            "observed": self.observed,
            "missing": self.missing,
            "distinct": self.distinct,
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
            observed=int(record["observed"]),
            missing=int(record["missing"]),
            distinct=int(record["distinct"]),
            distribution={
                str(label): int(count)
                for label, count in dict(record.get("distribution", {})).items()
            },
        )
