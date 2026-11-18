"""Versioned records: the one pydantic layer for every persisted record.

A record kind derives from `Record`: a frozen pydantic model that rejects
unknown keys and carries an integer `schema_version`. Loading is
`Kind.model_validate(raw)` and writing is `record.model_dump(mode="json")`;
both are pydantic's own methods. A part nested inside a record derives from
`Schema` and carries no version of its own: it is versioned by the record
that holds it.

A kind lists its version changes as a tuple of `Migration` steps, its
migration registry; the current version is one more than the number of
steps. Loading detects the stored version (the `schema_version` key, the
kind's `legacy` spelling, or 1 when neither is present, because records
written before this layer carried no version), applies the steps from that
version up, and refuses a record newer than the build. Every step can be
undone, so a record can always be restored to an older layout.

`schema_version` is the one snake_case key kept on purpose: eight record
kinds have a domain field named `version`, so one word would be ambiguous.
"""

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
from typing import Any
from typing import ClassVar

import pydantic

from scmle.contracts import errors

KEY = "schema_version"
ORIGIN = 1

Layout = dict[str, Any]


class Schema(pydantic.BaseModel):
    """Immutable pydantic base that rejects unknown fields."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)


@dataclasses.dataclass(frozen=True)
class Migration:
    """One version step of a record kind, applicable in both directions.

    Attributes:
        renames: Old key to new key; distinct old keys map to distinct new
            keys, so a rename can be undone.
        change: A transformation applied after the renames, or None.
        revert: The inverse of `change`; required exactly when `change` is.
    """

    renames: Mapping[str, str] = dataclasses.field(default_factory=dict)
    change: Callable[[Layout], Layout] | None = None
    revert: Callable[[Layout], Layout] | None = None

    def __post_init__(self) -> None:
        """Rejects a step that could not be undone.

        Raises:
            InputError: If two keys are renamed to the same key, or a change
                comes without its revert (or a revert without its change).
        """
        if len(set(self.renames.values())) != len(self.renames):
            raise errors.InputError("a migration's renames must be injective")
        if (self.change is None) != (self.revert is None):
            raise errors.InputError("a migration's change needs its revert")

    def upgrade(self, layout: Layout) -> Layout:
        """Returns a stored layout moved one version up.

        Args:
            layout: The record at the step's older version.

        Returns:
            The record at the step's newer version.

        Raises:
            InputError: If a renamed key would overwrite a key that is
                already present.
        """
        return self.apply(layout, self.renames, None, self.change)

    def downgrade(self, layout: Layout) -> Layout:
        """Returns a stored layout moved one version down.

        Args:
            layout: The record at the step's newer version.

        Returns:
            The record at the step's older version.

        Raises:
            InputError: If a renamed key would overwrite a key that is
                already present.
        """
        inverse = {new: old for old, new in self.renames.items()}
        return self.apply(layout, inverse, self.revert, None)

    @staticmethod
    def apply(
        layout: Layout,
        renames: Mapping[str, str],
        before: Callable[[Layout], Layout] | None,
        after: Callable[[Layout], Layout] | None,
    ) -> Layout:
        """Renames keys between two optional transformations.

        Args:
            layout: The record to move.
            renames: Old key to new key.
            before: Applied before the renames, or None.
            after: Applied after the renames, or None.

        Returns:
            The moved record.

        Raises:
            InputError: If a renamed key would overwrite a key that is
                already present.
        """
        source = layout if before is None else before(dict(layout))
        clashes = {
            new
            for old, new in renames.items()
            if old in source and new in source
        }
        if clashes:
            raise errors.InputError(
                f"renaming would overwrite stored keys: {sorted(clashes)}"
            )
        moved = {renames.get(key, key): value for key, value in source.items()}
        return moved if after is None else after(moved)


class Record(Schema):
    """A persisted record kind with a schema version and its migrations.

    Attributes:
        schema_version: The layout version the record is stored in.
        migrations: The kind's version steps, oldest first.
        legacy: The kind's old spelling of the version key, or None. A key
            named `version` is never read as a schema version unless the kind
            declares it here.
    """

    schema_version: int = pydantic.Field(default=ORIGIN, ge=ORIGIN)
    migrations: ClassVar[tuple[Migration, ...]] = ()
    legacy: ClassVar[str | None] = None

    def __init__(self, /, **data: Any) -> None:
        """Builds a record in code at the kind's current version.

        Args:
            **data: The field values; `schema_version` defaults to the
                kind's current version.
        """
        data.setdefault(KEY, type(self).latest())
        super().__init__(**data)

    # pydantic calls a custom `__init__` from `model_validate` too, which
    # would stamp every stored record as current; this marks it as the base
    # initialiser, so loading reads the stored version instead.
    __init__.__pydantic_base_init__ = True  # type: ignore[attr-defined]

    @classmethod
    def latest(cls) -> int:
        """Returns the kind's current version: one more than its steps."""
        return ORIGIN + len(cls.migrations)

    @pydantic.model_validator(mode="before")
    @classmethod
    def migrate(cls, raw: Any) -> Any:
        """Moves a stored record to the current version before validation.

        Args:
            raw: The stored record, or a value pydantic will reject.

        Returns:
            The record at the current version, or `raw` unchanged when it is
            not a mapping.

        Raises:
            InputError: If the stored version is not a positive integer or is
                newer than this build.
        """
        if not isinstance(raw, Mapping):
            return raw
        return load(cls.__name__, cls.migrations, raw, cls.legacy)

    @classmethod
    def detect(cls, layout: Layout) -> int:
        """Removes and returns the stored version of a record.

        Args:
            layout: The stored record; its version key is removed.

        Returns:
            The stored version; 1 when the record carries none.

        Raises:
            InputError: If the version is not an integer of at least 1.
        """
        return detect(cls.__name__, layout, cls.legacy)


class Spelled(Schema):
    """A part or record whose stored keys differ from its field names.

    Its `spelling` step moves a layout keyed by field name to the stored
    layout (one word per key, shared prefixes as objects) and back, so code
    keeps the field names while files carry the approved vocabulary. Dumping
    writes the stored layout; loading accepts either layout. A kind that is
    also a `Record` (listed before it in the bases) migrates the stored
    version first.

    Attributes:
        spelling: Field-name layout to stored layout, applied at the top
            level only; nested parts carry their own.
    """

    spelling: ClassVar[Migration] = Migration()

    @pydantic.model_validator(mode="before")
    @classmethod
    def migrate(cls, raw: Any) -> Any:
        """Moves a stored layout to the field names before validation.

        Args:
            raw: The stored record, keyword arguments, or a value pydantic
                will reject.

        Returns:
            The layout keyed by field name, or `raw` unchanged when it is not
            a mapping.

        Raises:
            InputError: If the stored version is not a positive integer or is
                newer than this build.
        """
        if not isinstance(raw, Mapping):
            return raw
        layout = dict(raw)
        if issubclass(cls, Record):
            layout = load(cls.__name__, cls.migrations, layout, cls.legacy)
        if set(layout) - set(cls.model_fields):
            layout = cls.spelling.downgrade(layout)
        return layout

    @pydantic.model_serializer(mode="wrap")
    def spell(self, handler: pydantic.SerializerFunctionWrapHandler) -> Any:
        """Writes the stored layout.

        Args:
            handler: pydantic's serializer for the fields.

        Returns:
            The dumped record with its stored keys.
        """
        dumped = handler(self)
        if not isinstance(dumped, dict):
            return dumped
        return type(self).spelling.upgrade(dumped)


def anchor(record: Record) -> Layout:
    """Returns a record's dump at version 1, the layout its digest names.

    Identity digests are computed on this layout, so renaming a stored key
    never changes the digest of stored content.

    Args:
        record: The record.

    Returns:
        The record's JSON dump moved down to version 1, with
        `schema_version` set to 1.
    """
    stored = record.model_dump(mode="json")
    stored.pop(KEY, None)
    return {KEY: ORIGIN, **downgrade(type(record).migrations, stored)}


def detect(name: str, layout: Layout, legacy: str | None = None) -> int:
    """Removes and returns the stored version of a record.

    Args:
        name: The record kind, for the error message.
        layout: The stored record; its version key is removed.
        legacy: The kind's old spelling of the version key, or None.

    Returns:
        The stored version; 1 when the record carries none.

    Raises:
        InputError: If the version is not an integer of at least 1.
    """
    if KEY in layout:
        value = layout.pop(KEY)
    elif legacy is not None and legacy in layout:
        value = layout.pop(legacy)
    else:
        return ORIGIN
    if isinstance(value, bool) or not isinstance(value, int):
        raise errors.InputError(f"{name} schema version must be an integer")
    if value < ORIGIN:
        raise errors.InputError(
            f"{name} schema version must be at least {ORIGIN}"
        )
    return value


def load(
    name: str,
    migrations: tuple[Migration, ...],
    raw: Mapping[str, Any],
    legacy: str | None = None,
) -> Layout:
    """Moves a stored record of any kind to its current version.

    This is the loading path of `Record`, also used by the record kinds that
    are still dataclasses with their own `from_dict`.

    Args:
        name: The record kind, for error messages.
        migrations: The kind's version steps, oldest first.
        raw: The stored record.
        legacy: The kind's old spelling of the version key, or None.

    Returns:
        The record at the current version, with `schema_version` set to it.

    Raises:
        InputError: If the stored version is not a positive integer or is
            newer than this build.
    """
    layout = dict(raw)
    stored = detect(name, layout, legacy)
    current = ORIGIN + len(migrations)
    if stored > current:
        raise errors.InputError(
            f"{name} record has schema version {stored}, newer than this "
            f"build's {current}"
        )
    layout = upgrade(migrations, layout, stored)
    layout[KEY] = current
    return layout


def upgrade(
    migrations: tuple[Migration, ...], layout: Layout, version: int
) -> Layout:
    """Moves a layout without a version key from `version` to the current one.

    Args:
        migrations: The kind's version steps, oldest first.
        layout: The record (or a part of one) at `version`.
        version: The version `layout` is in.

    Returns:
        The layout at the kind's current version.
    """
    for step in migrations[version - ORIGIN :]:
        layout = step.upgrade(layout)
    return layout


def downgrade(
    migrations: tuple[Migration, ...], layout: Layout, version: int = ORIGIN
) -> Layout:
    """Moves a layout without a version key from the current version down.

    Identity digests are computed on the layout at the kind's anchor version
    (1), so renaming a key never changes the identity of stored content.

    Args:
        migrations: The kind's version steps, oldest first.
        layout: The record (or a part of one) at the current version.
        version: The version to move to.

    Returns:
        The layout at `version`.
    """
    for step in reversed(migrations[version - ORIGIN :]):
        layout = step.downgrade(layout)
    return layout


def nest(
    renames: Mapping[str, str],
    parts: Mapping[str, Migration] | None = None,
    groups: Mapping[str, Mapping[str, str]] | None = None,
) -> Migration:
    """Builds a step that also moves nested parts and gathers keys.

    A part (a mapping, or a list of mappings) is moved by its own step, and
    a group collects top-level keys into one object: `{"digests":
    {"data_hash": "data"}}` moves `data_hash` to `digests.data`. Both are
    undone by the step's revert.

    Args:
        renames: Old key to new key at the top level.
        parts: New key of a nested part to the part's step.
        groups: Name of a new object to its old keys and their new names.

    Returns:
        The step.
    """
    moves = dict(parts or {})
    folds = {name: dict(keys) for name, keys in (groups or {}).items()}

    def change(layout: Layout) -> Layout:
        for key, step in moves.items():
            if key in layout:
                layout[key] = shift(layout[key], step.upgrade)
        return gather(layout, folds)

    def revert(layout: Layout) -> Layout:
        layout = scatter(layout, folds)
        for key, step in moves.items():
            if key in layout:
                layout[key] = shift(layout[key], step.downgrade)
        return layout

    return Migration(renames=renames, change=change, revert=revert)


def shift(value: Any, move: Callable[[Layout], Layout]) -> Any:
    """Moves a nested part, or each part of a list, one version.

    Args:
        value: A mapping, a list of mappings, or anything else (unchanged).
        move: The part's step in one direction.

    Returns:
        The moved value.
    """
    if isinstance(value, Mapping):
        return move(dict(value))
    if isinstance(value, list | tuple):
        return [
            move(dict(item)) if isinstance(item, Mapping) else item
            for item in value
        ]
    return value


def gather(layout: Layout, folds: Mapping[str, Mapping[str, str]]) -> Layout:
    """Collects top-level keys into named objects.

    Args:
        layout: The record; changed in place.
        folds: Name of each object to its old keys and their new names.

    Returns:
        The record.

    Raises:
        InputError: If an object's name is already a stored key.
    """
    for name, keys in folds.items():
        if name in layout:
            raise errors.InputError(
                f"gathering would overwrite the stored key {name!r}"
            )
        layout[name] = {
            new: layout.pop(old) for old, new in keys.items() if old in layout
        }
    return layout


def scatter(layout: Layout, folds: Mapping[str, Mapping[str, str]]) -> Layout:
    """Undoes `gather`: spreads named objects back to top-level keys.

    Args:
        layout: The record; changed in place.
        folds: Name of each object to its old keys and their new names.

    Returns:
        The record.

    Raises:
        InputError: If an object holds a key the older layout has no
            place for.
    """
    for name, keys in folds.items():
        group = dict(layout.pop(name, {}))
        for old, new in keys.items():
            if new in group:
                layout[old] = group.pop(new)
        if group:
            raise errors.InputError(
                f"{name} holds keys an older layout lacks: {sorted(group)}"
            )
    return layout
