"""The one group split of the modalities (MM-22).

Every modality splits its samples by group, so related samples (chunks of
one document, frames of one video, near duplicates) never straddle
partitions. `divide` is that split: groups are allocated whole by
`curation.allocate`, stratified by each group's most frequent label
(`consensus`, MM-8) when labels are given.
"""

from collections.abc import Sequence
from typing import Any
from typing import TypeVar

from scmle.contracts import errors
from scmle.data.modalities import curation

Item = TypeVar("Item")
"""The sample type `divide` splits."""


def bucket(label: Any) -> str:
    """Names the stable bucket of one label.

    Args:
        label: A label of any type; None is unlabeled.

    Returns:
        `unlabeled` for None, otherwise `<type name>:<label>`.
    """
    return "unlabeled" if label is None else f"{type(label).__name__}:{label}"


def consensus(keys: Sequence[str], labels: Sequence[Any]) -> dict[str, str]:
    """The stratum of each group: its most frequent label bucket.

    Args:
        keys: The group of each item.
        labels: The label of each item; None is unlabeled.

    Returns:
        For each group, its most frequent bucket (`bucket`); a tie goes to
        the first bucket by name.
    """
    tallies: dict[str, dict[str, int]] = {}
    for key, label in zip(keys, labels, strict=True):
        tally = tallies.setdefault(key, {})
        tally[bucket(label)] = tally.get(bucket(label), 0) + 1
    return {
        key: min(tally, key=lambda name: (-tally[name], name))
        for key, tally in tallies.items()
    }


def divide(
    items: Sequence[Item],
    keys: Sequence[str],
    labels: Sequence[Any] | None,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[tuple[Item, ...], tuple[Item, ...], tuple[Item, ...]]:
    """Splits items so no group spans two partitions.

    The one group split of the modalities (MM-22). With labels, a group's
    stratum is its most frequent label bucket (ties go to the first bucket
    by name) and the groups are allocated with `curation.allocate` under those
    strata (MM-8); without, they are allocated unstratified.

    Args:
        items: The samples, in order.
        keys: The group of each item.
        labels: The label of each item (None is unlabeled), or None to
            allocate without strata.
        validation_fraction: Share of groups for validation, in [0, 1).
        test_fraction: Share of groups for testing, in [0, 1).
        seed: Seed of the allocation.

    Returns:
        The training, validation and test items, each in input order.

    Raises:
        InputError: If a fraction is outside [0, 1), the fractions sum to
            1 or more, or the sequences differ in length.
    """
    curation.proportions(validation_fraction, test_fraction)
    if len(items) != len(keys) or (
        labels is not None and len(labels) != len(keys)
    ):
        raise errors.InputError("items, keys and labels must align")
    strata = None if labels is None else consensus(keys, labels)
    validation, test = curation.allocate(
        set(keys), validation_fraction, test_fraction, seed, strata
    )
    sides = [
        tuple(
            item
            for item, key in zip(items, keys, strict=True)
            if (key in test, key in validation) == flags
        )
        for flags in ((False, False), (False, True), (True, False))
    ]
    return sides[0], sides[1], sides[2]
