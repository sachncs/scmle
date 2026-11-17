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
        layout = dict(raw)
        stored = cls.detect(layout)
        current = cls.latest()
        if stored > current:
            raise errors.InputError(
                f"{cls.__name__} record has schema version {stored}, newer "
                f"than this build's {current}"
            )
        for step in cls.migrations[stored - ORIGIN :]:
            layout = step.upgrade(layout)
        layout[KEY] = current
        return layout

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
        if KEY in layout:
            value = layout.pop(KEY)
        elif cls.legacy is not None and cls.legacy in layout:
            value = layout.pop(cls.legacy)
        else:
            return ORIGIN
        if isinstance(value, bool) or not isinstance(value, int):
            raise errors.InputError(
                f"{cls.__name__} schema version must be an integer"
            )
        if value < ORIGIN:
            raise errors.InputError(
                f"{cls.__name__} schema version must be at least {ORIGIN}"
            )
        return value
