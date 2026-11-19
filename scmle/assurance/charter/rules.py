"""The pre-registered rules of a charter run (charter format 3).

A charter may carry the ledger rule, the confirmation rule, the failure
policy and the stability thresholds as `Procedure`. The charter hash and the
signature cover them, so altering any one of them changes `charter_hash`.

Versioning. A charter without rules serialises exactly as before (the field is
omitted), so every charter signed under format 1 or 2 keeps its hash, its
signature and loads unchanged. Such a charter runs with the defaults and the
run reports the rules as `not chartered`: they are tiered `unverified`, not
`attested` by the signature.
"""

from collections.abc import Mapping
import dataclasses
from typing import Any
from typing import ClassVar
from typing import Self

from scmle.assurance.ledger import budget
from scmle.assurance.ledger import confirm
from scmle.assurance.ledger import online_fdr
import scmle.assurance.ledger.online_fdr
from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.contracts import records

DEFAULT_S_MIN = 0.8
DEFAULT_BAND = 0.5
DEFAULT_Q_MIN = 0.8


@dataclasses.dataclass(frozen=True, slots=True)
class Procedure:
    """Choices that decide which findings are claimed, fixed before data.

    Attributes:
        ledger: Online rule of the exploration (`e-lond`, `lord++` or
            `saffron`).
        confirmation: Multiplicity rule of the single confirmation (`bh`,
            `by`, `holm` or `bonferroni`).
        failure: Failure policy: what a crashed confirmation does, one of
            `confirm.POLICIES`: `unconfirmed` (claims stay unverified) or
            `thresholdout` (the run also answers adaptive queries on the
            single opening of `C` through a reuse session).
        stability: Stability filter: whether survivors with a `fragile`
            stability verdict are withdrawn before `C` is opened. It acts
            only when the stability rule has a current passing certificate
            for these exact thresholds; otherwise the verdict stays a
            warning and the report says so.
        agreement: Stability rule, smallest sign-agreement share (`s_min`).
        band: Stability rule, largest relative distance from the headline.
        exclusion: Stability rule, smallest share of intervals excluding
            zero (`q_min`).
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={
                "ledger_rule": "ledger",
                "confirmation_rule": "confirmation",
                "failure_policy": "failure",
                "stability_filter": "stability",
                "s_min": "agreement",
                "q_min": "exclusion",
            }
        ),
    )

    ledger: str = scmle.assurance.ledger.online_fdr.ELOND
    confirmation: str = budget.Correction.BY.value
    failure: str = confirm.UNCONFIRMED
    stability: bool = False
    agreement: float = DEFAULT_S_MIN
    band: float = DEFAULT_BAND
    exclusion: float = DEFAULT_Q_MIN

    def __post_init__(self) -> None:
        """Validates every field.

        Raises:
            ValueError: If `ledger` is not in `online_fdr.RULE_NAMES`,
                `confirmation` is not a `Correction` value, `failure` is not
                in `confirm.POLICIES`, `stability` is not a bool,
                `agreement` or `exclusion` is not finite or outside (0, 1],
                or `band` is not finite or not positive.
        """
        if self.ledger not in online_fdr.RULE_NAMES:
            raise errors.InputError(
                f"ledger must be one of {online_fdr.RULE_NAMES}"
            )
        allowed = tuple(r.value for r in budget.Correction)
        if self.confirmation not in allowed:
            raise errors.InputError(f"confirmation must be one of {allowed}")
        if self.failure not in confirm.POLICIES:
            raise errors.InputError(
                f"failure must be one of {confirm.POLICIES}"
            )
        if not isinstance(self.stability, bool):
            raise errors.InputError("stability must be a boolean")
        for name in ("agreement", "exclusion"):
            share = qualification.finite(getattr(self, name), name)
            if not 0.0 < share <= 1.0:
                raise errors.InputError(f"{name} must be within (0, 1]")
        if qualification.finite(self.band, "band") <= 0.0:
            raise errors.InputError("band must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Serializes the record.

        Returns:
            A JSON-ready dict of the record's fields.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the rules from a dict.

        Keys other than the fields, and missing fields, are rejected with
        the `ValueError` of `qualification.require`.

        Args:
            record: Field names to values, as produced by `to_dict`.

        Returns:
            The rebuilt rules.
        """
        return cls(**qualification.require(cls, record))
