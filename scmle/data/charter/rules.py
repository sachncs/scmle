"""The pre-registered rules of a charter run (charter format 3).

Before this module the ledger rule, the confirmation rule, the failure policy
and the stability thresholds were options of the run specification, outside
the charter hash and the signature, so they could change after signing. A
charter may now carry them as `RunRules`; the hash covers them, and altering
any one of them changes `charter_hash`.

Versioning. A charter without rules serialises exactly as before (the field is
omitted), so every charter signed under format 1 or 2 keeps its hash, its
signature and loads unchanged. Such a charter runs with the defaults and the
run reports the rules as `not chartered`: they are tiered `unverified`, not
`attested` by the signature.
"""

from collections.abc import Mapping
import dataclasses
from typing import Any
from typing import Self

from scmle.contracts import qualification
from scmle.data.ledger import budget as budget_module
from scmle.data.ledger import confirm
from scmle.data.ledger import online_fdr

DEFAULT_S_MIN = 0.8
DEFAULT_BAND = 0.5
DEFAULT_Q_MIN = 0.8


@dataclasses.dataclass(frozen=True, slots=True)
class RunRules:
    """Choices that decide which findings are claimed, fixed before data.

    Attributes:
        ledger_rule: Online rule of the exploration (`e-lond`, `lord++` or
            `saffron`).
        confirmation_rule: Multiplicity rule of the single confirmation
            (`bh`, `by`, `holm` or `bonferroni`).
        failure_policy: What a crashed confirmation does (`unconfirmed`).
        stability_filter: Whether survivors with a `fragile` stability
            verdict are withdrawn before `C` is opened. It acts only when the
            stability rule has a current passing certificate for these exact
            thresholds; otherwise the verdict stays a warning and the report
            says so.
        s_min: Stability rule, smallest sign-agreement share.
        band: Stability rule, largest relative distance from the headline.
        q_min: Stability rule, smallest share of intervals excluding zero.
    """

    ledger_rule: str = online_fdr.DEFAULT_RULE
    confirmation_rule: str = budget_module.MultiplicityRule.BY.value
    failure_policy: str = confirm.UNCONFIRMED
    stability_filter: bool = False
    s_min: float = DEFAULT_S_MIN
    band: float = DEFAULT_BAND
    q_min: float = DEFAULT_Q_MIN

    def __post_init__(self) -> None:
        """Validates every field."""
        if self.ledger_rule not in online_fdr.RULE_NAMES:
            raise ValueError(
                f"ledger_rule must be one of {online_fdr.RULE_NAMES}"
            )
        allowed = tuple(r.value for r in budget_module.MultiplicityRule)
        if self.confirmation_rule not in allowed:
            raise ValueError(f"confirmation_rule must be one of {allowed}")
        if self.failure_policy != confirm.UNCONFIRMED:
            raise ValueError(
                f"failure_policy must be '{confirm.UNCONFIRMED}': no other "
                "policy is implemented"
            )
        if not isinstance(self.stability_filter, bool):
            raise ValueError("stability_filter must be a boolean")
        for name in ("s_min", "q_min"):
            share = qualification.finite(getattr(self, name), name)
            if not 0.0 < share <= 1.0:
                raise ValueError(f"{name} must be within (0, 1]")
        if qualification.finite(self.band, "band") <= 0.0:
            raise ValueError("band must be positive")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the rules, rejecting unknown or missing fields."""
        qualification.require_fields(cls, record)
        return cls(**record)
