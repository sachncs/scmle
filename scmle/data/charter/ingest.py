"""The ingest block of the charter (feature F9, formats 7 and 8).

Everything here is optional: a charter without an `IngestSpec` serialises,
hashes and loads exactly as a format 1 to 6 charter did, and an `IngestSpec`
without the format 8 sub-blocks (`entity`) serialises exactly as under
format 7. The block fixes, before
any data is read, how much the source may be asked (queries, rows scanned,
seconds), how many features the grammar may propose (`B_feat`), the windows
and the lag of the as-of cutoff, the largest recording delay the reconciliation
accepts, and the tolerances of key and foreign-key inference. The tables and
columns are not here; they are in the plan file the data spec names.
"""

from collections.abc import Mapping
import dataclasses
import re
from typing import Any
from typing import Self

from scmle.contracts import evidence
from scmle.contracts import qualification

HEX64 = re.compile(r"^[0-9a-f]{64}$")
PASS_KINDS = ("exact", "prefix", "soundex")
DEFAULT_WINDOWS = (7.0, 30.0, 90.0)
DEFAULT_FEATURE_BUDGET = 200


def non_negative(value: Any, name: str) -> float:
    """A finite number that is not negative."""
    number = qualification.finite(value, name)
    if number < 0:
        raise ValueError(f"{name} must not be negative")
    return number


@dataclasses.dataclass(frozen=True, slots=True)
class EntityField:
    """A column of the run table compared to find duplicates of an entity.

    Attributes:
        name: Column name.
        fuzzy: Three-level string comparison instead of exact agreement.
    """

    name: str
    fuzzy: bool = True

    def __post_init__(self) -> None:
        """Validates the record."""
        evidence.require_text(self.name, "name")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the record, rejecting unknown or missing fields."""
        qualification.require_fields(cls, record)
        return cls(**record)


@dataclasses.dataclass(frozen=True, slots=True)
class EntityPass:
    """One blocking pass of the entity resolution.

    Attributes:
        field: Column the block key is built from.
        kind: `exact`, `prefix` or `soundex`.
        size: Prefix length for `prefix`.
    """

    field: str
    kind: str = "exact"
    size: int = 3

    def __post_init__(self) -> None:
        """Validates the record."""
        evidence.require_text(self.field, "field")
        if self.kind not in PASS_KINDS:
            raise ValueError(f"kind must be one of {PASS_KINDS}")
        qualification.count(self.size, "size", minimum=1)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the record, rejecting unknown or missing fields."""
        qualification.require_fields(cls, record)
        return cls(**record)


@dataclasses.dataclass(frozen=True, slots=True)
class EntitySpec:
    """Entity resolution whose result is the split's group column (format 8).

    Attributes:
        fields: Columns compared, fixed before any data is read.
        passes: Blocking passes.
        column: Name of the group column added to the run table; the split
            groups by it, so one real-world entity never straddles
            development and test.
    """

    fields: tuple[EntityField, ...]
    passes: tuple[EntityPass, ...]
    column: str = "entity_group"

    def __post_init__(self) -> None:
        """Validates the block."""
        qualification.require_tuple(self.fields, "fields")
        qualification.require_tuple(self.passes, "passes")
        if not self.fields or not self.passes:
            raise ValueError("an entity block needs fields and passes")
        names = [f.name for f in self.fields]
        if len(set(names)) != len(names):
            raise ValueError("duplicate entity field")
        evidence.require_text(self.column, "column")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return {
            "fields": [f.to_dict() for f in self.fields],
            "passes": [p.to_dict() for p in self.passes],
            "column": self.column,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the block, rejecting unknown or missing fields."""
        qualification.require_fields(cls, record)
        return cls(
            tuple(EntityField.from_dict(f) for f in record["fields"]),
            tuple(EntityPass.from_dict(p) for p in record["passes"]),
            record["column"],
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ColumnTotal:
    """The total of one column that the source system owns.

    Attributes:
        column: Column of the extracted view.
        total: Its sum in the source system.
    """

    column: str
    total: float

    def __post_init__(self) -> None:
        """Validates the record."""
        evidence.require_text(self.column, "column")
        qualification.finite(self.total, "total")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the record, rejecting unknown or missing fields."""
        qualification.require_fields(cls, record)
        return cls(**record)


@dataclasses.dataclass(frozen=True, slots=True)
class TableTotals:
    """What the source system says one extracted view must contain.

    Attributes:
        view: Name of a view (or nest) of the ingest plan.
        rows: Row count in the source, or None.
        sums: Column totals in the source.
        checksum_column: Column whose values are checksummed.
        checksum: SHA-256 of the sorted values of that column as text, joined
            by newlines (`scmle.data.ingest.declared.key_checksum`).
    """

    view: str
    rows: int | None = None
    sums: tuple[ColumnTotal, ...] = ()
    checksum_column: str | None = None
    checksum: str | None = None

    def __post_init__(self) -> None:
        """Validates the record."""
        evidence.require_text(self.view, "view")
        qualification.require_tuple(self.sums, "sums")
        if self.rows is not None:
            qualification.count(self.rows, "rows")
        if (self.checksum_column is None) != (self.checksum is None):
            raise ValueError("checksum and checksum_column go together")
        if self.checksum is not None and not HEX64.match(self.checksum):
            raise ValueError("checksum must be 64 lower-case hex digits")
        if self.rows is None and not self.sums and self.checksum is None:
            raise ValueError(f"nothing is declared for view {self.view!r}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return {
            "view": self.view,
            "rows": self.rows,
            "sums": [s.to_dict() for s in self.sums],
            "checksum_column": self.checksum_column,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the record, rejecting unknown or missing fields."""
        qualification.require_fields(cls, record)
        return cls(
            record["view"],
            record["rows"],
            tuple(ColumnTotal.from_dict(s) for s in record["sums"]),
            record["checksum_column"],
            record["checksum"],
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ReconcileSpec:
    """Totals the extract must reproduce, or the run stops (format 8).

    Attributes:
        tables: One record per view; the relative gap allowed in a sum is the
            ingest block's `delta_rec`, counts and checksums are exact.
    """

    tables: tuple[TableTotals, ...]

    def __post_init__(self) -> None:
        """Validates the block."""
        qualification.require_tuple(self.tables, "tables")
        if not self.tables:
            raise ValueError("a reconcile block needs at least one table")
        names = [t.view for t in self.tables]
        if len(set(names)) != len(names):
            raise ValueError("duplicate view in the reconcile block")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return {"tables": [t.to_dict() for t in self.tables]}

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the block, rejecting unknown or missing fields."""
        qualification.require_fields(cls, record)
        return cls(tuple(TableTotals.from_dict(t) for t in record["tables"]))


@dataclasses.dataclass(frozen=True, slots=True)
class IngestSpec:
    """Budget, windows and tolerances of an ingestion.

    Attributes:
        max_queries: Source queries allowed, or None for no limit.
        max_rows_scanned: Rows the source may be estimated to scan, or None.
        max_seconds: Wall time allowed, or None.
        feature_budget: `B_feat`, the cap on synthesized features.
        windows: Declared windows in days.
        lag: Days between the cutoff and the prediction moment.
        delta_rec: Relative gap in the extract reconciliation that needs no
            explanation.
        eps_key: Largest `g3` accepted as a key.
        eps_null: Largest null rate accepted in a key.
        eps_fk: Largest share of child values without a parent for a link.
        strict: Drop features whose weakest source is `unverified`.
        entity: Entity resolution on the run table whose groups become the
            split's group column (format 8), or None.
        reconcile: Source totals the extracted views must reproduce
            (format 8), or None.
    """

    max_queries: int | None = None
    max_rows_scanned: int | None = None
    max_seconds: float | None = None
    feature_budget: int = DEFAULT_FEATURE_BUDGET
    windows: tuple[float, ...] = DEFAULT_WINDOWS
    lag: float = 0.0
    delta_rec: float = 0.0
    eps_key: float = 0.0
    eps_null: float = 0.0
    eps_fk: float = 0.0
    strict: bool = False
    entity: EntitySpec | None = None
    reconcile: ReconcileSpec | None = None

    def __post_init__(self) -> None:
        """Validates the block."""
        for name in ("max_queries", "max_rows_scanned"):
            value = getattr(self, name)
            if value is not None:
                qualification.count(value, name)
        if self.max_seconds is not None:
            non_negative(self.max_seconds, "max_seconds")
        qualification.count(self.feature_budget, "feature_budget", minimum=1)
        qualification.require_tuple(self.windows, "windows")
        if not self.windows or len(set(self.windows)) != len(self.windows):
            raise ValueError("windows must be non-empty and distinct")
        for window in self.windows:
            if qualification.finite(window, "window") <= 0:
                raise ValueError("windows must be positive")
        for name in ("lag", "delta_rec", "eps_key", "eps_null", "eps_fk"):
            non_negative(getattr(self, name), name)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        record = dataclasses.asdict(self)
        record["windows"] = list(self.windows)
        del record["entity"], record["reconcile"]
        if self.entity is not None:
            record["entity"] = self.entity.to_dict()
        if self.reconcile is not None:
            record["reconcile"] = self.reconcile.to_dict()
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the block, rejecting unknown or missing fields."""
        values = {"entity": None, "reconcile": None, **record}
        qualification.require_fields(cls, values)
        values["windows"] = tuple(values["windows"])
        block = values["entity"]
        values["entity"] = (
            None if block is None else EntitySpec.from_dict(block)
        )
        totals = values["reconcile"]
        values["reconcile"] = (
            None if totals is None else ReconcileSpec.from_dict(totals)
        )
        return cls(**values)
