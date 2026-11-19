"""The results of every leakage test on one feature and on a table."""

import dataclasses
import math
from typing import Any

from scmle.assurance.admissibility.leakage.conditional import Conditional
from scmle.assurance.admissibility.leakage.missingness import Missingness
from scmle.assurance.admissibility.leakage.skill import Skill
from scmle.assurance.detectors import certificates
from scmle.contracts import qualification


@dataclasses.dataclass(frozen=True, slots=True)
class Assessment:
    """All leakage tests for one feature.

    Attributes:
        feature: Feature name.
        skill: Skill test.
        conditional: Conditional test, run for suspects.
        missingness: Missingness test, when the feature has gaps.
        suspect: Either the skill or the missingness test called it one.
        escalated: A suspect that also keeps its signal given the others.
    """

    feature: str
    skill: Skill
    conditional: Conditional | None
    missingness: Missingness | None
    suspect: bool
    escalated: bool

    @property
    def checks(self) -> tuple[qualification.Check, ...]:
        """Every ledger record of this feature."""
        parts = (self.skill, self.conditional, self.missingness)
        return tuple(p.check for p in parts if p is not None)


@dataclasses.dataclass(frozen=True, slots=True)
class Assessments:
    """Leakage tests for many features, reported as suspicions only.

    Attributes:
        assessments: One per feature, in the order given.
        gate: Whether decisions were allowed to quarantine.
    """

    assessments: tuple[Assessment, ...]
    gate: certificates.Authority

    @property
    def suspects(self) -> dict[str, bool]:
        """Suspicion per feature."""
        return {a.feature: a.suspect for a in self.assessments}

    @property
    def checks(self) -> tuple[qualification.Check, ...]:
        """Every ledger record."""
        return tuple(c for a in self.assessments for c in a.checks)

    def to_dict(self) -> dict[str, Any]:
        """Converts the report to a JSON-ready document.

        A suspicion is not a finding, and the document says so.

        Returns:
            A dict with a note, the gate value and, per feature, the
            suspect, escalated and low-power flags, the skill, its lower
            bound (None when not finite) and p-value, the conditional and
            missingness p-values and the missingness Cramer's V (each None
            when that test did not run).
        """
        return {
            "note": "statistical suspicion only; not proof of leakage",
            "gate": self.gate.value,
            "features": [
                {
                    "feature": a.feature,
                    "suspect": a.suspect,
                    "escalated": a.escalated,
                    "skill": a.skill.skill,
                    "skill_lower_bound": (
                        a.skill.floor if math.isfinite(a.skill.floor) else None
                    ),
                    "skill_p_value": a.skill.p_value,
                    "low_power": a.skill.power,
                    "conditional_p_value": None
                    if a.conditional is None
                    else a.conditional.p_value,
                    "missingness_p_value": None
                    if a.missingness is None
                    else a.missingness.p_value,
                    "missingness_cramers_v": None
                    if a.missingness is None
                    else a.missingness.association,
                }
                for a in self.assessments
            ],
        }
