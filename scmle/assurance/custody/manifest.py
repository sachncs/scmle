"""Split manifest v2: a full-length hash over the whole three-way split.

The v1 `Manifest.fingerprint` is `sha256(repr(...))` cut to 16 hex
characters. That is too short to be a commitment and depends on Python's
`repr`. Version 2 hashes canonical JSON of every id, the strategy, seed,
embargo, the charter hash and the data hash, and keeps the full 64 hex
digits. Version 1 manifests remain readable and are still verified against
their own fingerprint.
"""

from collections.abc import Mapping
import dataclasses
import hashlib
from typing import Any
from typing import ClassVar
from typing import Self

import pandas

from scmle.assurance.custody import merkle
from scmle.contracts import errors
from scmle.contracts import evidence
from scmle.contracts import qualification
from scmle.contracts import records
from scmle.contracts import signing
from scmle.data.splitting import splitters

MANIFEST_VERSION = 2
SCOPES = ("confirmatory", "exploratory_unconfirmed")
HASH_FIELD = "sha256"
Fold = tuple[tuple[int, ...], tuple[int, ...]]


def ids(value: Any, name: str) -> tuple[int, ...]:
    """Requires a tuple of integers.

    Args:
        value: The value to check.
        name: Field name used in error messages.

    Returns:
        `value`, unchanged.

    Raises:
        ValueError: If a member is not an integer (a bool does not count).
            `qualification.tupled` raises first when `value` is not
            a tuple.
    """
    qualification.tupled(value, name)
    if not all(isinstance(i, int) and not isinstance(i, bool) for i in value):
        raise errors.InputError(f"{name} must hold integers")
    members: tuple[int, ...] = value
    return members


@dataclasses.dataclass(frozen=True, slots=True)
class Digests:
    """Hashes a split manifest binds.

    Attributes:
        charter: Hash of the governing charter.
        data: Hash of the dataset the ids index into.
    """

    charter: str
    data: str


@dataclasses.dataclass(frozen=True, slots=True)
class Sets:
    """Row ids of each part of the split.

    Attributes:
        exploration: Rows of `X`, the only rows exploration may read.
        confirmation: Rows of `C`, held sealed until the one unlock.
        embargo: Rows dropped from both sets.
        final: Rows of the final-evaluation set (`F` for temporal data,
            role `final_test` otherwise).
    """

    exploration: tuple[int, ...]
    confirmation: tuple[int, ...]
    embargo: tuple[int, ...]
    final: tuple[int, ...]


SETS = {
    "exploration_ids": "exploration",
    "confirmation_ids": "confirmation",
    "embargo_ids": "embargo",
    "final_test_ids": "final",
}


@dataclasses.dataclass(frozen=True, slots=True)
class Manifest:
    """Immutable record of the three-way split.

    The manifest is stored at schema version 2; a version 1 record (flat
    `charter_hash`, `data_hash` and `*_ids` keys) loads through
    `migrations`. Its seal (`sha256`) is computed on the version 1 layout
    (`anchor`), so a manifest sealed before the rename keeps its hash, and
    the custody log and the sealed view that bind that hash still match.

    Attributes:
        strategy: Strategy of the underlying development/final split.
        seed: Seed of the carve.
        embargo: Rows dropped around temporal confirmation blocks.
        digests: Hashes of the governing charter and of the dataset.
        sets: Row ids of the exploration, confirmation, embargo and final
            sets.
        folds: Cross-validation folds as positions within
            `sets.exploration`.
        isolation: Isolation level per sealed set, as `(set, level)` pairs.
        scope: `confirmatory` or `exploratory_unconfirmed`.
        notes: Adjustments made while splitting.
        roots: Merkle root over the row content of each set, as
            `(set, root)` pairs. Empty for manifests written before roots
            existed; their hash is unchanged.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            renames={"embargo_rows": "embargo", "row_roots": "roots"},
            groups={
                "digests": {"charter_hash": "charter", "data_hash": "data"},
                "sets": SETS,
            },
        ),
    )

    strategy: str
    seed: int
    embargo: int
    digests: Digests
    sets: Sets
    folds: tuple[Fold, ...]
    isolation: tuple[tuple[str, str], ...]
    scope: str
    notes: tuple[str, ...] = ()
    roots: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        """Validates the manifest.

        Raises:
            ValueError: If `scope` is not a known scope, or a set repeats a
                row, or two sets share a row. The field checks of
                `evidence.require_text`, `qualification.count`,
                `qualification.tupled` and `ids` raise on a malformed
                field.
        """
        evidence.require_text(self.strategy, "strategy")
        evidence.require_text(self.digests.charter, "digests.charter")
        evidence.require_text(self.digests.data, "digests.data")
        qualification.count(self.seed, "seed")
        qualification.count(self.embargo, "embargo")
        if self.scope not in SCOPES:
            raise errors.InputError(f"scope must be one of {SCOPES}")
        groups = {
            name: ids(getattr(self.sets, name), f"sets.{name}")
            for name in SETS.values()
        }
        seen: set[int] = set()
        for name, members in groups.items():
            if len(set(members)) != len(members):
                raise errors.InputError(f"sets.{name} repeats a row")
            if seen & set(members):
                raise errors.InputError(f"sets.{name} overlap another set")
            seen |= set(members)
        qualification.tupled(self.folds, "folds")
        qualification.tupled(self.isolation, "isolation")
        qualification.tupled(self.notes, "notes")
        qualification.tupled(self.roots, "roots")

    def body(self) -> dict[str, Any]:
        """Builds the content at the current schema version, without hash.

        Returns:
            A JSON-ready dict including the manifest version and the
            `schema_version`. `roots` is present only when it is not empty.
        """
        body: dict[str, Any] = {
            records.KEY: records.ORIGIN + len(self.migrations),
            "manifest_version": MANIFEST_VERSION,
            "strategy": self.strategy,
            "seed": self.seed,
            "embargo": self.embargo,
            "digests": {
                "charter": self.digests.charter,
                "data": self.digests.data,
            },
            "sets": {
                name: list(getattr(self.sets, name)) for name in SETS.values()
            },
            "folds": [
                {"fit": list(fit), "val": list(val)} for fit, val in self.folds
            ],
            "isolation": [list(pair) for pair in self.isolation],
            "scope": self.scope,
            "notes": list(self.notes),
        }
        if self.roots:
            body["roots"] = [list(pair) for pair in self.roots]
        return body

    def anchor(self) -> records.Layout:
        """Builds the hashed content in its version 1 layout.

        Returns:
            The `body` as version 1 stored it, without `schema_version`.
        """
        current = self.body()
        del current[records.KEY]
        return records.downgrade(self.migrations, current)

    @property
    def sha256(self) -> str:
        """SHA-256 of the canonical JSON of the version 1 manifest."""
        return hashlib.sha256(signing.canonicalise(self.anchor())).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Builds the JSON-ready record including the hash.

        Returns:
            The `body` with the `sha256` field added.
        """
        return {**self.body(), HASH_FIELD: self.sha256}

    def rowids(self) -> dict[str, tuple[int, ...]]:
        """Lists the row ids of each set that carries a content root.

        Returns:
            A dict from set name (`exploration`, `confirmation`,
            `final_test`) to its row ids.
        """
        return {
            "exploration": self.sets.exploration,
            "confirmation": self.sets.confirmation,
            "final_test": self.sets.final,
        }

    def verify_rows(self, frame: pandas.DataFrame) -> None:
        """Checks the content of the rows against the recorded roots.

        Args:
            frame: The full dataset, indexed by position as the ids are.

        Raises:
            ArtifactError: If the manifest holds no roots or a root differs,
                which means a value inside a row changed.
        """
        if not self.roots:
            raise errors.ArtifactError("manifest holds no row content roots")
        ids_of = self.rowids()
        for name, root in self.roots:
            if merkle.merkleize(frame, ids_of[name]) != root:
                raise errors.ArtifactError(
                    f"rows of {name} differ from the manifest: a value changed"
                )

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the manifest and verifies its hash.

        A version 1 record is verified on its stored bytes before it is
        migrated; every record is verified again, after it is rebuilt,
        against the hash of its version 1 layout. Unknown or missing fields
        raise ValueError from `qualification.require`, as does any invalid
        value from the constructor; only `notes` and `roots` may be absent.
        The version is checked before the fields.

        Args:
            record: A parsed manifest record as written by `to_dict`, at any
                schema version.

        Returns:
            The manifest.

        Raises:
            ArtifactError: If the manifest version is not 2 or the hash does
                not match the content.
        """
        values = dict(record)
        claimed = values.pop(HASH_FIELD, None)
        if values.get("manifest_version") != MANIFEST_VERSION:
            raise errors.ArtifactError("unsupported split manifest version")
        if (
            records.KEY not in values
            and claimed
            != hashlib.sha256(signing.canonicalise(values)).hexdigest()
        ):
            raise errors.ArtifactError(
                "split manifest hash mismatch: the manifest was edited"
            )
        del values["manifest_version"]
        values = qualification.require(
            cls,
            {
                "notes": (),
                "roots": (),
                **records.load(cls.__name__, cls.migrations, values),
            },
        )
        sets = values["sets"]
        manifest = cls(
            strategy=values["strategy"],
            seed=values["seed"],
            embargo=values["embargo"],
            digests=Digests(
                charter=values["digests"]["charter"],
                data=values["digests"]["data"],
            ),
            sets=Sets(**{name: tuple(sets[name]) for name in SETS.values()}),
            folds=tuple(
                (tuple(f["fit"]), tuple(f["val"])) for f in values["folds"]
            ),
            isolation=tuple(tuple(pair) for pair in values["isolation"]),
            scope=values["scope"],
            notes=tuple(values["notes"]),
            roots=tuple(tuple(p) for p in values["roots"]),
        )
        if claimed != manifest.sha256:
            raise errors.ArtifactError(
                "split manifest hash mismatch: the manifest was edited"
            )
        return manifest


def restore(
    record: Mapping[str, Any],
) -> Manifest | splitters.Manifest:
    """Reads a manifest of either version and verifies it.

    Args:
        record: A parsed manifest file.

    Returns:
        A `Manifest` (v2) or a `Manifest` (v1).

    Raises:
        ArtifactError: On an unknown version or a failed hash check.
    """
    version = record.get("manifest_version")
    if version == MANIFEST_VERSION:
        return Manifest.from_dict(record)
    if version is None and ("training" in record or "train_ids" in record):
        ids = splitters.Manifest.KEYS.downgrade(dict(record))
        v1 = splitters.Manifest(
            strategy=record["strategy"],
            seed=record["seed"],
            train_ids=tuple(ids["train_ids"]),
            test_ids=tuple(ids["test_ids"]),
            folds=tuple(
                (tuple(f["train"]), tuple(f["val"])) for f in record["folds"]
            ),
            notes=tuple(record.get("notes", ())),
        )
        if record.get("fingerprint") != v1.fingerprint:
            raise errors.ArtifactError(
                "split manifest fingerprint mismatch: the manifest was edited"
            )
        return v1
    raise errors.ArtifactError(
        f"unsupported split manifest version: {version!r}"
    )
