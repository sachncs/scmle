"""The result of resolving a record set, and its grade against the truth."""

from collections.abc import Sequence
import dataclasses
import itertools
import math
from typing import Any
from typing import ClassVar

import numpy

from scmle.assurance.detectors import certificates
from scmle.assurance.ingest.entities.blocking import Pair
from scmle.assurance.ingest.entities.fitting import LEVELS
from scmle.assurance.ingest.entities.fitting import Model
from scmle.assurance.ingest.entities.sampling import Estimate
from scmle.contracts import qualification
from scmle.contracts import records


@dataclasses.dataclass(frozen=True, slots=True)
class Resolution:
    """Result of resolving a record set.

    Attributes:
        groups: Group id per record (link-then-group); use as the group
            column for splitting.
        links: Pairs at or above the grouping threshold.
        group: Weight threshold used for grouping.
        t_match: Weight above which a pair is called a match.
        model: Fitted parameters.
        compared: Pairs compared after blocking.
        reduction: Share of all `N(N-1)/2` pairs avoided.
        blocks: Blocks too large to compare.
        estimate: Clerical-sample estimate, or None when unlabelled.
        tier: `attested` with a named reviewer's sample, else `unverified`.
        migrations: The written layout's version steps, oldest first;
            version 2 writes `thresholds.{group, linkage}`, `reduction`
            and `skipped`.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            {"reduction_ratio": "reduction", "skipped_blocks": "skipped"},
            groups={"thresholds": {"t_group": "group", "t_match": "linkage"}},
        ),
    )

    groups: list[int]
    links: tuple[Pair, ...]
    group: float
    t_match: float
    model: Model
    compared: int
    reduction: float
    blocks: int
    estimate: Estimate | None
    tier: qualification.Tier

    def to_dict(self) -> dict[str, Any]:
        """Serializes the resolution, for `entity_resolution.json`.

        Returns:
            A JSON-ready mapping with the tier, thresholds, counts, model
            and estimate.
        """
        return {
            "tier": self.tier.value,
            "thresholds": {"group": self.group, "linkage": self.t_match},
            "compared_pairs": self.compared,
            "reduction": self.reduction,
            "skipped": self.blocks,
            "links": len(self.links),
            "groups": len(set(self.groups)),
            "records": len(self.groups),
            "model": {
                "prior": self.model.prior,
                "m": self.model.m.tolist(),
                "u": self.model.u.tolist(),
                "iterations": self.model.iterations,
            },
            "estimate": None
            if self.estimate is None
            else self.estimate.to_dict(),
        }


def isolate(
    count: int, width: int, reduction: float, skipped: int
) -> Resolution:
    """Resolves a record set in which blocking left no pair to compare.

    Args:
        count: Number of records; each becomes its own group.
        width: Number of fields compared.
        reduction: Share of all pairs avoided.
        skipped: Blocks too large to compare.

    Returns:
        An unverified resolution with no links, infinite thresholds and a
        flat model.
    """
    model = Model(
        0.0,
        numpy.ones((width, LEVELS)),
        numpy.ones((width, LEVELS)),
        0,
    )
    return Resolution(
        list(range(count)),
        (),
        math.inf,
        math.inf,
        model,
        0,
        reduction,
        skipped,
        None,
        qualification.Tier.UNVERIFIED,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class Quality:
    """Measures against known truth (benchmarks only).

    Attributes:
        completeness: Share of true-match pairs the blocking compared.
        recall: Share of true-match pairs placed in one group.
        interval: Wilson interval of `recall`.
        matches: True-match pairs.
        pairs: True-match pairs found in one group.
        spurious: Non-match pairs placed in one group.
    """

    completeness: float
    recall: float
    interval: tuple[float, float]
    matches: int
    pairs: int
    spurious: int


def grade(
    entity: Sequence[Any], resolution: Resolution, pairs: Sequence[Pair]
) -> Quality:
    """Compares groups with the true entity of each record.

    Args:
        entity: True entity of each record, by record index.
        resolution: The resolution to judge.
        pairs: The compared pairs, to give pair completeness.

    Returns:
        The measures against truth; pair completeness and link recall are
        1.0 when there are no true-match pairs.
    """
    by_entity: dict[Any, list[int]] = {}
    for index, key in enumerate(entity):
        by_entity.setdefault(key, []).append(index)
    truth = {
        pair
        for members in by_entity.values()
        for pair in itertools.combinations(members, 2)
    }
    grouped = sum(
        resolution.groups[i] == resolution.groups[j] for i, j in truth
    )
    compared = set(pairs)
    completeness = len(truth & compared) / len(truth) if truth else 1.0
    by_group: dict[int, list[int]] = {}
    for index, group in enumerate(resolution.groups):
        by_group.setdefault(group, []).append(index)
    false_pairs = sum(
        entity[i] != entity[j]
        for members in by_group.values()
        for i, j in itertools.combinations(members, 2)
    )
    recall = grouped / len(truth) if truth else 1.0
    interval = certificates.bound(grouped, len(truth)) if truth else (0.0, 1.0)
    return Quality(
        completeness, recall, interval, len(truth), grouped, false_pairs
    )
