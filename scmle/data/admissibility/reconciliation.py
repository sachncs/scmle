"""Reconciliation of the extract against independent records.

Row counts, key cardinalities and control totals of the extract are compared
with independent records (source counts, finance totals, log counts). For one
quantity the gap is `|extract - source| / |source|`. A gap above the
tolerance blocks unless named mechanisms account for it: each explanation
names one mechanism (a filter, late arrivals, de-duplication), describes it,
and states the signed amount of `extract - source` it accounts for. What
remains, `(extract - source) - sum(amounts)`, must itself be within the
tolerance. An explanation without an amount, or with an unnamed mechanism, is
not accepted.

The arithmetic is exact, with a closed-form referent: the tests compare it
with a brute-force evaluation, and the certificate plants gaps of known size.
The independent record is the weak point: its tier is `attested` at best,
since a person supplied the source number and its name.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import enum
from typing import Any
from typing import Self

from scmle.contracts import evidence
from scmle.contracts import qualification
from scmle.data.detectors import certificates

DETECTOR_ID = "f4-reconciliation"
DEFECT_TYPE = "unexplained_gap"
FORMAT_VERSION = 1


class Mechanism(enum.StrEnum):
    """Named mechanisms that may explain a gap."""

    FILTER = "filter"
    LATE_ARRIVAL = "late_arrival"
    DEDUPE = "dedupe"


class Status(enum.StrEnum):
    """Result for one quantity."""

    MATCHED = "matched"
    EXPLAINED = "explained"
    UNEXPLAINED = "unexplained"


@dataclasses.dataclass(frozen=True, slots=True)
class Explanation:
    """A named mechanism and the part of the gap it accounts for.

    Attributes:
        mechanism: Which mechanism.
        detail: What it is, concretely (the rule, the cut-off, the job).
        amount: Signed part of `extract - source` it accounts for.
        attested_by: Who vouches for it, or None.
    """

    mechanism: Mechanism
    detail: str
    amount: float
    attested_by: str | None = None

    def __post_init__(self) -> None:
        """Validate the record.

        Raises:
            ValueError: If `mechanism` is not a `Mechanism`, `detail` or
                `attested_by` (when given) is empty, or `amount` is not
                finite.
        """
        if not isinstance(self.mechanism, Mechanism):
            raise ValueError("mechanism must be a Mechanism")
        evidence.require_text(self.detail, "detail")
        qualification.finite(self.amount, "amount")
        if self.attested_by is not None:
            evidence.require_text(self.attested_by, "attested_by")

    def to_dict(self) -> dict[str, Any]:
        """Convert the explanation to a JSON-ready record.

        Returns:
            A dict with the mechanism (its string value), detail, amount
            and attester.
        """
        return {
            "mechanism": self.mechanism.value,
            "detail": self.detail,
            "amount": self.amount,
            "attested_by": self.attested_by,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild the record, rejecting unknown or missing fields.

        Args:
            record: A mapping in the form `to_dict` returns.

        Returns:
            The explanation. Unexpected or missing fields, an unknown
            mechanism and invalid values are rejected with a `ValueError`.
        """
        qualification.require_fields(cls, record)
        return cls(
            Mechanism(record["mechanism"]),
            record["detail"],
            record["amount"],
            record["attested_by"],
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Item:
    """One quantity to reconcile.

    Attributes:
        name: What is counted (for example "orders.row_count").
        extract: Value in the extract.
        source: Value in the independent record; must be non-zero.
        source_name: The independent record (system, report, ticket).
        explanations: Mechanisms offered for a gap.
    """

    name: str
    extract: float
    source: float
    source_name: str
    explanations: tuple[Explanation, ...] = ()

    def __post_init__(self) -> None:
        """Validate the record.

        Raises:
            ValueError: If `name` or `source_name` is empty, `extract` or
                `source` is not finite, `source` is zero, or
                `explanations` is not a tuple.
        """
        evidence.require_text(self.name, "name")
        evidence.require_text(self.source_name, "source_name")
        qualification.finite(self.extract, "extract")
        if qualification.finite(self.source, "source") == 0.0:
            raise ValueError("source must be non-zero to form a gap ratio")
        qualification.require_tuple(self.explanations, "explanations")

    @property
    def gap(self) -> float:
        """`|extract - source| / |source|`."""
        return abs(self.extract - self.source) / abs(self.source)


@dataclasses.dataclass(frozen=True, slots=True)
class ItemResult:
    """Verdict for one quantity.

    Attributes:
        item: The quantity.
        gap: Relative gap before explanations.
        residual_gap: Relative gap left after the explanations.
        status: `matched`, `explained` or `unexplained`.
        tier: `attested` when the source is named and, for an explained gap,
            every explanation is attested; otherwise `unverified`.
        check: Ledger record.
    """

    item: Item
    gap: float
    residual_gap: float
    status: Status
    tier: qualification.Tier
    check: qualification.Check

    def to_dict(self) -> dict[str, Any]:
        """Convert the verdict to a JSON-ready record.

        Returns:
            A dict with the quantity, both values, gaps, status, tier and
            the explanations offered.
        """
        return {
            "name": self.item.name,
            "extract": self.item.extract,
            "source": self.item.source,
            "source_name": self.item.source_name,
            "gap": self.gap,
            "residual_gap": self.residual_gap,
            "status": self.status.value,
            "tier": self.tier.value,
            "explanations": [e.to_dict() for e in self.item.explanations],
        }


def reconcile_item(
    item: Item,
    tolerance: float = 0.0,
    gate: certificates.GateStatus = certificates.GateStatus.DESCRIPTIVE,
    set_id: qualification.SetId = qualification.SetId.ALL,
    seed_path: qualification.SeedPath = (0, DETECTOR_ID, 0),
) -> ItemResult:
    """Reconcile one quantity.

    A gap equal to the tolerance matches. The check blocks when the gap is
    unexplained and `gate` is `GATING`; otherwise it warns.

    Args:
        item: The quantity.
        tolerance: Largest relative gap that needs no explanation, in [0, 1].
        gate: Whether the reconciliation detector has a current certificate.
        set_id: Set the extract belongs to. Reconciliation reads no target.
        seed_path: Recorded on the check; no randomness is drawn.

    Returns:
        The verdict and its ledger record. A `tolerance` outside [0, 1] is
        rejected with a `ValueError`.
    """
    level = qualification.probability(tolerance, "tolerance")
    gap = item.gap
    accounted = sum(e.amount for e in item.explanations)
    residual = abs((item.extract - item.source) - accounted) / abs(item.source)
    if gap <= level:
        status = Status.MATCHED
    elif item.explanations and residual <= level:
        status = Status.EXPLAINED
    else:
        status = Status.UNEXPLAINED
    attested = status is Status.MATCHED or (
        status is Status.EXPLAINED
        and all(e.attested_by is not None for e in item.explanations)
    )
    tier = (
        qualification.Tier.ATTESTED
        if attested
        else qualification.Tier.UNVERIFIED
    )
    decision = (
        qualification.Decision.BLOCK
        if status is Status.UNEXPLAINED
        else qualification.Decision.PASS
    )
    check = qualification.Check(
        check_id=f"reconciliation.{item.name}",
        null_hypothesis=(
            f"the extract equals {item.source_name} within {level:g}, or a "
            "named mechanism explains the gap"
        ),
        statistic=gap,
        p_value=None,
        interval=(min(gap, residual), max(gap, residual)),
        effect=gap,
        n=1,
        decision=certificates.demote(decision, gate),
        set_id=set_id,
        seed_path=seed_path,
        tier=tier,
    )
    return ItemResult(item, gap, residual, status, tier, check)


@dataclasses.dataclass(frozen=True, slots=True)
class Reconciliation:
    """All quantities of an extract, as `reconciliation.json`.

    Attributes:
        results: One result per quantity.
        tolerance: Relative gap that needs no explanation.
    """

    results: tuple[ItemResult, ...]
    tolerance: float

    @property
    def passed(self) -> bool:
        """Whether no quantity has an unexplained gap."""
        return all(r.status is not Status.UNEXPLAINED for r in self.results)

    @property
    def checks(self) -> tuple[qualification.Check, ...]:
        """Ledger records, in order."""
        return tuple(r.check for r in self.results)

    def to_dict(self) -> dict[str, Any]:
        """Convert the reconciliation to a JSON-ready document.

        Returns:
            A dict with the format version, tolerance, whether it passed
            and the per-quantity results.
        """
        return {
            "format_version": FORMAT_VERSION,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "results": [r.to_dict() for r in self.results],
        }


def reconcile(
    items: Sequence[Item],
    tolerance: float = 0.0,
    gate: certificates.GateStatus = certificates.GateStatus.DESCRIPTIVE,
    set_id: qualification.SetId = qualification.SetId.ALL,
) -> Reconciliation:
    """Reconcile every quantity.

    Args:
        items: Quantities, with unique names.
        tolerance: Relative gap that needs no explanation, in [0, 1].
        gate: Whether the detector has a current certificate.
        set_id: Set the extract belongs to.

    Returns:
        The reconciliation; `passed` is False if any gap is unexplained.

    Raises:
        ValueError: If `items` is empty, names repeat, or `tolerance` is
            outside [0, 1].
    """
    if not items:
        raise ValueError("at least one item is required")
    names = [i.name for i in items]
    if len(set(names)) != len(names):
        raise ValueError("item names must be unique")
    results = tuple(
        reconcile_item(i, tolerance, gate, set_id, (0, DETECTOR_ID, k))
        for k, i in enumerate(items)
    )
    return Reconciliation(results, tolerance)
