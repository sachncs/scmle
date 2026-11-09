"""Where a drafted field came from, and the tier its claims may reach.

An LLM-drafted field has no error rate the instrument can derive. A claim
that depends on one stays `unverified` while the charter is unsigned. Costs
read from data are `unverified` as well. Fields written by a
person or filled by a deterministic rule do not cap the tier.
"""

from collections.abc import Iterable
import enum

from scmle.contracts import qualification


class FieldSource(enum.StrEnum):
    """How a charter field was filled."""

    TEMPLATE = "template"
    LLM = "llm"
    DATA = "data"
    HUMAN = "human"


def claim_tier(
    ceiling: qualification.Tier,
    sources: Iterable[FieldSource],
    *,
    signed: bool,
    cost_from_data: bool = False,
) -> qualification.Tier:
    """Highest tier a claim may carry given the fields it depends on.

    Args:
        ceiling: The tier the test itself supports.
        sources: Source of each field the claim depends on.
        signed: Whether the charter carries a verified signature.
        cost_from_data: Whether a cost entry was read from data.

    Returns:
        `unverified` if an unsigned LLM field or a data-derived cost is
        involved, otherwise `ceiling`.
    """
    if cost_from_data:
        return qualification.Tier.UNVERIFIED
    if not signed and FieldSource.LLM in set(sources):
        return qualification.Tier.UNVERIFIED
    return ceiling
