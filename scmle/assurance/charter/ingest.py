"""The ingest block of the charter (formats 7 and 8).

Everything here is optional: a charter without an `Ingestion` serialises,
hashes and loads exactly as a format 1 to 6 charter did, and an `Ingestion`
without the format 8 sub-blocks (`entity` and `reconcile`) serialises
exactly as under format 7. The block fixes, before any data is read, how
much the source may be asked (queries, rows scanned, seconds), how many
features the grammar may propose (`B_feat`), the windows and the lag of the
as-of cutoff, the largest recording delay the reconciliation accepts, and
the tolerances of key and foreign-key inference. The tables and columns are
not here; they are in the ingest-plan file that the data spec names.
"""

from collections.abc import Mapping
import dataclasses
import re
from typing import Any
from typing import ClassVar
from typing import Self

from scmle.contracts import errors
from scmle.contracts import evidence
from scmle.contracts import qualification
from scmle.contracts import records

HEX64 = re.compile(r"^[0-9a-f]{64}$")
PASS_KINDS = ("exact", "prefix", "soundex")
DEFAULT_WINDOWS = (7.0, 30.0, 90.0)
DEFAULT_FEATURE_BUDGET = 200
EPSILONS = {"eps_key": "key", "eps_null": "null", "eps_fk": "link"}


def nonnegative(value: Any, name: str) -> float:
    """Checks that a number is finite and not negative.

    Args:
        value: The candidate number.
        name: Name used in the error message.

    Returns:
        `value` as a float.

    Raises:
        ValueError: If `value` is not a finite int or float (a bool is
            rejected) or is negative.
    """
    number = qualification.finite(value, name)
    if number < 0:
        raise errors.InputError(f"{name} must not be negative")
    return number


@dataclasses.dataclass(frozen=True, slots=True)
class Attribute:
    """A column of the run table compared to find duplicates of an entity.

    Attributes:
        name: Column name.
        fuzzy: Three-level string comparison instead of exact agreement.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = ()

    name: str
    fuzzy: bool = True

    def __post_init__(self) -> None:
        """Validates the record.

        A blank `name` raises `ValueError` (from `evidence.require_text`).
        """
        evidence.require_text(self.name, "name")

    def to_dict(self) -> dict[str, Any]:
        """Serializes the record.

        Returns:
            A JSON-ready dict of the record's fields.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the record from a dict.

        Keys other than the fields, and missing fields, are rejected with
        the `ValueError` of `qualification.require`.

        Args:
            record: Field names to values, as produced by `to_dict`.

        Returns:
            The rebuilt record.
        """
        return cls(**qualification.require(cls, record))


@dataclasses.dataclass(frozen=True, slots=True)
class Pass:
    """One blocking pass of the entity resolution.

    Attributes:
        field: Column the block key is built from.
        kind: `exact`, `prefix` or `soundex`.
        size: Prefix length for `prefix`.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = ()

    field: str
    kind: str = "exact"
    size: int = 3

    def __post_init__(self) -> None:
        """Validates the record.

        Raises:
            ValueError: If `field` is blank, `kind` is not in `PASS_KINDS`,
                or `size` is not an integer of at least 1.
        """
        evidence.require_text(self.field, "field")
        if self.kind not in PASS_KINDS:
            raise errors.InputError(f"kind must be one of {PASS_KINDS}")
        qualification.count(self.size, "size", minimum=1)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the record.

        Returns:
            A JSON-ready dict of the record's fields.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the record from a dict.

        Keys other than the fields, and missing fields, are rejected with
        the `ValueError` of `qualification.require`.

        Args:
            record: Field names to values, as produced by `to_dict`.

        Returns:
            The rebuilt record.
        """
        return cls(**qualification.require(cls, record))


@dataclasses.dataclass(frozen=True, slots=True)
class Entity:
    """Entity resolution whose result is the split's group column (format 8).

    Attributes:
        fields: Columns compared, fixed before any data is read.
        passes: Blocking passes.
        column: Name of the group column added to the run table; the split
            groups by it, so one real-world entity never straddles
            development and test.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = ()

    fields: tuple[Attribute, ...]
    passes: tuple[Pass, ...]
    column: str = "entity_group"

    def __post_init__(self) -> None:
        """Validates the block.

        Raises:
            ValueError: If `fields` or `passes` is not a tuple or is empty,
                two fields share a name, or `column` is blank.
        """
        qualification.tupled(self.fields, "fields")
        qualification.tupled(self.passes, "passes")
        if not self.fields or not self.passes:
            raise errors.InputError("an entity block needs fields and passes")
        names = [f.name for f in self.fields]
        if len(set(names)) != len(names):
            raise errors.InputError("duplicate entity field")
        evidence.require_text(self.column, "column")

    def to_dict(self) -> dict[str, Any]:
        """Serializes the block.

        Returns:
            A JSON-ready dict with `fields` and `passes` as lists of dicts
            and `column`.
        """
        return {
            "fields": [f.to_dict() for f in self.fields],
            "passes": [p.to_dict() for p in self.passes],
            "column": self.column,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the block from a dict.

        Keys other than the fields, and missing fields, are rejected with
        the `ValueError` of `qualification.require`.

        Args:
            record: Field names to values, as produced by `to_dict`.

        Returns:
            The rebuilt block.
        """
        record = qualification.require(cls, record)
        return cls(
            tuple(Attribute.from_dict(f) for f in record["fields"]),
            tuple(Pass.from_dict(p) for p in record["passes"]),
            record["column"],
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Total:
    """The total of one column that the source system owns.

    Attributes:
        column: Column of the extracted view.
        total: Its sum in the source system.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = ()

    column: str
    total: float

    def __post_init__(self) -> None:
        """Validates the record.

        A blank `column` or a `total` that is not finite raises `ValueError`
        (from `evidence.require_text` and `qualification.finite`).
        """
        evidence.require_text(self.column, "column")
        qualification.finite(self.total, "total")

    def to_dict(self) -> dict[str, Any]:
        """Serializes the record.

        Returns:
            A JSON-ready dict of the record's fields.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the record from a dict.

        Keys other than the fields, and missing fields, are rejected with
        the `ValueError` of `qualification.require`.

        Args:
            record: Field names to values, as produced by `to_dict`.

        Returns:
            The rebuilt record.
        """
        return cls(**qualification.require(cls, record))


@dataclasses.dataclass(frozen=True, slots=True)
class TableTotals:
    """What the source system says one extracted view must contain.

    Attributes:
        view: Name of a view (or nest) of the ingest plan.
        rows: Row count in the source, or None.
        sums: Column totals in the source.
        column: Column whose values are checksummed.
        checksum: SHA-256 of the sorted values of that column as text, joined
            by newlines (`scmle.assurance.ingest.declared.sha`).
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(renames={"checksum_column": "column"}),
    )

    view: str
    rows: int | None = None
    sums: tuple[Total, ...] = ()
    column: str | None = None
    checksum: str | None = None

    def __post_init__(self) -> None:
        """Validates the record.

        Raises:
            ValueError: If `view` is blank, `sums` is not a tuple, `rows` is
                set but not a non-negative integer, only one of `checksum`
                and `column` is set, `checksum` is not 64
                lower-case hex digits, or none of `rows`, `sums` and
                `checksum` is declared.
        """
        evidence.require_text(self.view, "view")
        qualification.tupled(self.sums, "sums")
        if self.rows is not None:
            qualification.count(self.rows, "rows")
        if (self.column is None) != (self.checksum is None):
            raise errors.InputError("checksum and column go together")
        if self.checksum is not None and not HEX64.match(self.checksum):
            raise errors.InputError("checksum must be 64 lower-case hex digits")
        if self.rows is None and not self.sums and self.checksum is None:
            raise errors.InputError(
                f"nothing is declared for view {self.view!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serializes the record.

        Returns:
            A JSON-ready dict with `view`, `rows`, `sums` (a list of dicts),
            `column` and `checksum`.
        """
        return {
            "view": self.view,
            "rows": self.rows,
            "sums": [s.to_dict() for s in self.sums],
            "column": self.column,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the record from a dict.

        Keys other than the fields, and missing fields, are rejected with
        the `ValueError` of `qualification.require`.

        Args:
            record: Field names to values, as produced by `to_dict`.

        Returns:
            The rebuilt record.
        """
        record = qualification.require(cls, record)
        return cls(
            record["view"],
            record["rows"],
            tuple(Total.from_dict(s) for s in record["sums"]),
            record["column"],
            record["checksum"],
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Expectation:
    """Totals the extract must reproduce, or the run stops (format 8).

    Attributes:
        tables: One record per view; the relative gap allowed in a sum is the
            ingest block's `reconciliation`, counts and checksums are exact.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(renames={}, parts={"tables": TableTotals.migrations[0]}),
    )

    tables: tuple[TableTotals, ...]

    def __post_init__(self) -> None:
        """Validates the block.

        Raises:
            ValueError: If `tables` is not a tuple or is empty, or two
                records name the same view.
        """
        qualification.tupled(self.tables, "tables")
        if not self.tables:
            raise errors.InputError(
                "a reconcile block needs at least one table"
            )
        names = [t.view for t in self.tables]
        if len(set(names)) != len(names):
            raise errors.InputError("duplicate view in the reconcile block")

    def to_dict(self) -> dict[str, Any]:
        """Serializes the block.

        Returns:
            A JSON-ready dict with `tables` as a list of dicts.
        """
        return {"tables": [t.to_dict() for t in self.tables]}

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the block from a dict.

        Keys other than the fields, and missing fields, are rejected with
        the `ValueError` of `qualification.require`.

        Args:
            record: Field names to values, as produced by `to_dict`.

        Returns:
            The rebuilt block.
        """
        record = qualification.require(cls, record)
        return cls(tuple(TableTotals.from_dict(t) for t in record["tables"]))


def fold(layout: records.Layout) -> records.Layout:
    """Moves the three `eps_*` keys of a version 1 block into `tolerances`.

    Args:
        layout: The ingest block; a block that already holds `tolerances`
            and no `eps_*` key is returned unchanged.

    Returns:
        The block with its key tolerances nested.
    """
    if not EPSILONS.keys() & layout.keys():
        return layout
    return records.gather(layout, {"tolerances": EPSILONS})


def unfold(layout: records.Layout) -> records.Layout:
    """Undoes `fold`.

    Args:
        layout: The ingest block at version 2.

    Returns:
        The block with `eps_key`, `eps_null` and `eps_fk` at the top.
    """
    if "tolerances" not in layout:
        return layout
    return records.scatter(layout, {"tolerances": EPSILONS})


@dataclasses.dataclass(frozen=True, slots=True)
class Tolerances:
    """Tolerances of key and foreign-key inference.

    Attributes:
        key: Largest `g3` accepted as a key (`eps_key`).
        null: Largest null rate accepted in a key (`eps_null`).
        link: Largest share of child values without a parent for a link
            (`eps_fk`).
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = ()

    key: float = 0.0
    null: float = 0.0
    link: float = 0.0

    def __post_init__(self) -> None:
        """Validates the tolerances.

        A value that is not a finite non-negative number raises `ValueError`
        (from `nonnegative`).
        """
        for name in ("key", "null", "link"):
            nonnegative(getattr(self, name), name)


@dataclasses.dataclass(frozen=True, slots=True)
class Ingestion:
    """Budget, windows and tolerances of an ingestion.

    Attributes:
        queries: Source queries allowed, or None for no limit.
        scanned: Rows the source may be estimated to scan, or None.
        seconds: Wall time allowed, in seconds, or None.
        features: `B_feat`, the cap on synthesized features.
        windows: Declared windows in days.
        lag: Days between the cutoff and the prediction moment.
        reconciliation: Serves two purposes (`delta_rec`): the relative gap
            a declared sum may have in the extract reconciliation, and the
            recording delay in days that applies to a view's time rule when
            the rule states none.
        tolerances: Tolerances of key and foreign-key inference.
        strict: Drop features whose weakest source is `unverified`.
        entity: Entity resolution on the run table whose groups become the
            split's group column (format 8), or None.
        reconcile: Source totals the extracted views must reproduce
            (format 8), or None.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={
                "max_queries": "queries",
                "max_rows_scanned": "scanned",
                "max_seconds": "seconds",
                "feature_budget": "features",
                "delta_rec": "reconciliation",
            },
            change=lambda layout: records.nest(
                {}, parts={"reconcile": Expectation.migrations[0]}
            ).upgrade(fold(layout)),
            revert=lambda layout: unfold(
                records.nest(
                    {}, parts={"reconcile": Expectation.migrations[0]}
                ).downgrade(layout)
            ),
        ),
    )

    queries: int | None = None
    scanned: int | None = None
    seconds: float | None = None
    features: int = DEFAULT_FEATURE_BUDGET
    windows: tuple[float, ...] = DEFAULT_WINDOWS
    lag: float = 0.0
    reconciliation: float = 0.0
    tolerances: Tolerances = dataclasses.field(default_factory=Tolerances)
    strict: bool = False
    entity: Entity | None = None
    reconcile: Expectation | None = None

    def __post_init__(self) -> None:
        """Validates the block.

        Raises:
            ValueError: If `queries` or `scanned` is set but is not a
                non-negative integer, `seconds`, `lag` or `reconciliation`
                is not a finite non-negative number (`seconds` only when
                set), `features` is not an integer of at least 1, or
                `windows` is not a tuple, is empty, has a repeat, or holds a
                value that is not finite and positive.
        """
        for name in ("queries", "scanned"):
            value = getattr(self, name)
            if value is not None:
                qualification.count(value, name)
        if self.seconds is not None:
            nonnegative(self.seconds, "seconds")
        qualification.count(self.features, "features", minimum=1)
        qualification.tupled(self.windows, "windows")
        if not self.windows or len(set(self.windows)) != len(self.windows):
            raise errors.InputError("windows must be non-empty and distinct")
        for window in self.windows:
            if qualification.finite(window, "window") <= 0:
                raise errors.InputError("windows must be positive")
        for name in ("lag", "reconciliation"):
            nonnegative(getattr(self, name), name)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the block.

        Returns:
            A JSON-ready dict of the fields, with `windows` as a list and
            `tolerances` as a nested dict. `entity` and `reconcile` are
            included, as nested dicts, only when set.
        """
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
        """Rebuilds the block from a dict.

        `entity` and `reconcile` may be absent and then default to None.
        Other missing fields, and keys that are not fields, are rejected
        with the `ValueError` of `qualification.require`.

        Args:
            record: Field names to values, as produced by `to_dict`.

        Returns:
            The rebuilt block.
        """
        values: dict[str, Any] = qualification.require(
            cls,
            {
                "entity": None,
                "reconcile": None,
                **records.load(cls.__name__, cls.migrations, record),
            },
        )
        values["windows"] = tuple(values["windows"])
        values["tolerances"] = Tolerances(
            **qualification.require(Tolerances, values["tolerances"])
        )
        block = values["entity"]
        values["entity"] = None if block is None else Entity.from_dict(block)
        totals = values["reconcile"]
        values["reconcile"] = (
            None if totals is None else Expectation.from_dict(totals)
        )
        return cls(**values)
