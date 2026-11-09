"""Split manifest v2: a full-length hash over the whole three-way split.

The v1 `SplitManifest.fingerprint` is `sha256(repr(...))` cut to 16 hex
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
from typing import Self

import pandas as pd

from scmle.contracts import errors
from scmle.contracts import evidence
from scmle.contracts import qualification
from scmle.contracts import signing
from scmle.data.custody import merkle
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
            `qualification.require_tuple` raises first when `value` is not
            a tuple.
    """
    qualification.require_tuple(value, name)
    if not all(isinstance(i, int) and not isinstance(i, bool) for i in value):
        raise ValueError(f"{name} must hold integers")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class CustodyManifest:
    """Immutable record of the three-way split.

    Attributes:
        strategy: Strategy of the underlying development/final split.
        seed: Seed of the carve.
        embargo_rows: Rows dropped around temporal confirmation blocks.
        charter_hash: Hash of the governing charter.
        data_hash: Hash of the dataset the ids index into.
        exploration_ids: Rows of `X`, the only rows exploration may read.
        confirmation_ids: Rows of `C`, held sealed until the one unlock.
        embargo_ids: Rows dropped from both sets.
        final_test_ids: Rows of the final-evaluation set (`F` for temporal
            data, role `final_test` otherwise).
        folds: Cross-validation folds as positions within `exploration_ids`.
        isolation: Isolation level per sealed set, as `(set, level)` pairs.
        scope: `confirmatory` or `exploratory_unconfirmed`.
        notes: Adjustments made while splitting.
        row_roots: Merkle root over the row content of each set, as
            `(set, root)` pairs. Empty for manifests written before roots
            existed; their hash is unchanged.
    """

    strategy: str
    seed: int
    embargo_rows: int
    charter_hash: str
    data_hash: str
    exploration_ids: tuple[int, ...]
    confirmation_ids: tuple[int, ...]
    embargo_ids: tuple[int, ...]
    final_test_ids: tuple[int, ...]
    folds: tuple[Fold, ...]
    isolation: tuple[tuple[str, str], ...]
    scope: str
    notes: tuple[str, ...] = ()
    row_roots: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        """Validates the manifest.

        Raises:
            ValueError: If `scope` is not a known scope, or a set repeats a
                row, or two sets share a row. The field checks of
                `evidence.require_text`, `qualification.count`,
                `qualification.require_tuple` and `ids` raise on a malformed
                field.
        """
        evidence.require_text(self.strategy, "strategy")
        evidence.require_text(self.charter_hash, "charter_hash")
        evidence.require_text(self.data_hash, "data_hash")
        qualification.count(self.seed, "seed")
        qualification.count(self.embargo_rows, "embargo_rows")
        if self.scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}")
        groups = {
            "exploration_ids": ids(self.exploration_ids, "exploration_ids"),
            "confirmation_ids": ids(self.confirmation_ids, "confirmation_ids"),
            "embargo_ids": ids(self.embargo_ids, "embargo_ids"),
            "final_test_ids": ids(self.final_test_ids, "final_test_ids"),
        }
        seen: set[int] = set()
        for name, members in groups.items():
            if len(set(members)) != len(members):
                raise ValueError(f"{name} repeats a row")
            if seen & set(members):
                raise ValueError(f"{name} overlap another set")
            seen |= set(members)
        qualification.require_tuple(self.folds, "folds")
        qualification.require_tuple(self.isolation, "isolation")
        qualification.require_tuple(self.notes, "notes")
        qualification.require_tuple(self.row_roots, "row_roots")

    def body(self) -> dict[str, Any]:
        """Builds the hashed content: everything except the hash itself.

        Returns:
            A JSON-ready dict including the manifest version. `row_roots`
            is present only when it is not empty.
        """
        body: dict[str, Any] = {
            "manifest_version": MANIFEST_VERSION,
            "strategy": self.strategy,
            "seed": self.seed,
            "embargo_rows": self.embargo_rows,
            "charter_hash": self.charter_hash,
            "data_hash": self.data_hash,
            "exploration_ids": list(self.exploration_ids),
            "confirmation_ids": list(self.confirmation_ids),
            "embargo_ids": list(self.embargo_ids),
            "final_test_ids": list(self.final_test_ids),
            "folds": [
                {"fit": list(fit), "val": list(val)} for fit, val in self.folds
            ],
            "isolation": [list(pair) for pair in self.isolation],
            "scope": self.scope,
            "notes": list(self.notes),
        }
        if self.row_roots:
            body["row_roots"] = [list(pair) for pair in self.row_roots]
        return body

    @property
    def sha256(self) -> str:
        """SHA-256 of the canonical JSON of the manifest (64 hex digits)."""
        return hashlib.sha256(
            signing.canonical_payload(self.body())
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Builds the JSON-ready record including the hash.

        Returns:
            The `body` with the `sha256` field added.
        """
        return {**self.body(), HASH_FIELD: self.sha256}

    def set_ids(self) -> dict[str, tuple[int, ...]]:
        """Lists the row ids of each set that carries a content root.

        Returns:
            A dict from set name (`exploration`, `confirmation`,
            `final_test`) to its row ids.
        """
        return {
            "exploration": self.exploration_ids,
            "confirmation": self.confirmation_ids,
            "final_test": self.final_test_ids,
        }

    def verify_rows(self, frame: pd.DataFrame) -> None:
        """Checks the content of the rows against the recorded roots.

        Args:
            frame: The full dataset, indexed by position as the ids are.

        Raises:
            ArtifactError: If the manifest holds no roots or a root differs,
                which means a value inside a row changed.
        """
        if not self.row_roots:
            raise errors.ArtifactError("manifest holds no row content roots")
        ids_of = self.set_ids()
        for name, root in self.row_roots:
            if merkle.frame_root(frame, ids_of[name]) != root:
                raise errors.ArtifactError(
                    f"rows of {name} differ from the manifest: a value changed"
                )

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the manifest and verifies its hash.

        Unknown or missing fields raise ValueError from
        `qualification.require_fields`, as does any invalid value from the
        constructor; only `notes` and `row_roots` may be absent. The version
        is checked before the fields.

        Args:
            record: A parsed manifest record as written by `to_dict`.

        Returns:
            The manifest.

        Raises:
            ArtifactError: If the manifest version is not 2 or the hash does
                not match the content.
        """
        values = dict(record)
        claimed = values.pop(HASH_FIELD, None)
        if values.pop("manifest_version", None) != MANIFEST_VERSION:
            raise errors.ArtifactError("unsupported split manifest version")
        qualification.require_fields(
            cls,
            {
                **values,
                "notes": values.get("notes", ()),
                "row_roots": values.get("row_roots", ()),
            },
        )
        manifest = cls(
            strategy=values["strategy"],
            seed=values["seed"],
            embargo_rows=values["embargo_rows"],
            charter_hash=values["charter_hash"],
            data_hash=values["data_hash"],
            exploration_ids=tuple(values["exploration_ids"]),
            confirmation_ids=tuple(values["confirmation_ids"]),
            embargo_ids=tuple(values["embargo_ids"]),
            final_test_ids=tuple(values["final_test_ids"]),
            folds=tuple(
                (tuple(f["fit"]), tuple(f["val"])) for f in values["folds"]
            ),
            isolation=tuple(tuple(pair) for pair in values["isolation"]),
            scope=values["scope"],
            notes=tuple(values["notes"]),
            row_roots=tuple(tuple(p) for p in values.get("row_roots", ())),
        )
        if claimed != manifest.sha256:
            raise errors.ArtifactError(
                "split manifest hash mismatch: the manifest was edited"
            )
        return manifest


def read_v1(record: Mapping[str, Any]) -> splitters.SplitManifest:
    """Rebuilds a v1 manifest and checks its 16-hex fingerprint.

    Args:
        record: A parsed v1 manifest record.

    Returns:
        The v1 manifest.

    Raises:
        ArtifactError: If the recorded fingerprint differs from the one
            computed from the content.
    """
    manifest = splitters.SplitManifest(
        strategy=record["strategy"],
        seed=record["seed"],
        train_ids=tuple(record["train_ids"]),
        test_ids=tuple(record["test_ids"]),
        folds=tuple(
            (tuple(f["train"]), tuple(f["val"])) for f in record["folds"]
        ),
        notes=tuple(record.get("notes", ())),
    )
    if record.get("fingerprint") != manifest.fingerprint:
        raise errors.ArtifactError(
            "split manifest fingerprint mismatch: the manifest was edited"
        )
    return manifest


def load_manifest(
    record: Mapping[str, Any],
) -> CustodyManifest | splitters.SplitManifest:
    """Reads a manifest of either version and verifies it.

    Args:
        record: A parsed manifest file.

    Returns:
        A `CustodyManifest` (v2) or a `SplitManifest` (v1).

    Raises:
        ArtifactError: On an unknown version or a failed hash check.
    """
    version = record.get("manifest_version")
    if version == MANIFEST_VERSION:
        return CustodyManifest.from_dict(record)
    if version is None and "train_ids" in record:
        return read_v1(record)
    raise errors.ArtifactError(
        f"unsupported split manifest version: {version!r}"
    )
