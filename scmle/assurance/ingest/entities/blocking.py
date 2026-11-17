"""Compared fields and blocking passes that choose the pairs to compare."""

from collections.abc import Sequence
import dataclasses
import itertools
from typing import Any

import pandas

from scmle.assurance.ingest.entities.similarity import normalize
from scmle.assurance.ingest.entities.similarity import soundex
from scmle.contracts import errors

MAX_BLOCK = 200


Pair = tuple[int, int]


@dataclasses.dataclass(frozen=True, slots=True)
class Field:
    """A field to compare.

    Attributes:
        name: Column name.
        fuzzy: Use a three-level string comparison (disagree, similar,
            agree) instead of exact agreement.
    """

    name: str
    fuzzy: bool = True


@dataclasses.dataclass(frozen=True, slots=True)
class Pass:
    """One blocking pass.

    Attributes:
        field: Column the key is built from.
        kind: `exact`, `prefix` (first `size` characters) or `soundex`.
        size: Prefix length.
    """

    field: str
    kind: str = "exact"
    size: int = 3

    def key(self, value: Any) -> str:
        """Builds the block key of one value.

        Args:
            value: The field value of a record.

        Returns:
            The key for this pass; an empty string means the record is not
            blocked in this pass.

        Raises:
            ValueError: If `kind` is not `exact`, `prefix` or `soundex`.
        """
        text = normalize(value)
        if not text:
            return ""
        if self.kind == "exact":
            return text
        if self.kind == "prefix":
            return text[: self.size]
        if self.kind == "soundex":
            return soundex(text)
        raise errors.InputError(f"unknown blocking kind: {self.kind}")


def pairing(
    records: pandas.DataFrame,
    passes: Sequence[Pass],
    max_block: int = MAX_BLOCK,
) -> tuple[list[Pair], int]:
    """Finds pairs of records sharing a block key in any pass.

    A block larger than `max_block` is skipped whole and counted, because its
    pair count would dominate the work; the loss shows up in pair
    completeness, never silently.

    Args:
        records: One row per record.
        passes: Blocking passes.
        max_block: Largest block that is compared.

    Returns:
        The sorted, distinct record-index pairs, and the number of blocks
        skipped as too big.
    """
    pairs: set[Pair] = set()
    skipped = 0
    for rule in passes:
        blocks: dict[str, list[int]] = {}
        for index, value in enumerate(records[rule.field].tolist()):
            key = rule.key(value)
            if key:
                blocks.setdefault(key, []).append(index)
        for members in blocks.values():
            if len(members) > max_block:
                skipped += 1
                continue
            pairs.update(itertools.combinations(members, 2))
    return sorted(pairs), skipped
