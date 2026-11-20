"""The tier of a causal claim as a pure function of its reports.

The three tiers read as follows for a causal claim:

* `confirmed` needs all five conditions: (1) the estimate and interval were
  computed on `C`, opened once; (2) every design diagnostic of the strategy
  passed at a certified threshold; (3) the pre-registered falsification
  battery ran and passed, with at least one negative control declared;
  (4) the estimate replicated in at least two strata with one sign and no
  heterogeneity; (5) the sensitivity gate holds at the declared strength.
  Each gate and the strategy's interval must also have a current passing
  certificate; an uncertified gate is descriptive and cannot count.
* `attested` means every assumption the strategy needs is declared with a
  named attestor and the design diagnostics passed, but a condition for
  `confirmed` is missing or cannot be met (no negative control, one treated
  unit, a weak instrument, strata disagree, sensitivity fails or was not
  declared, an uncertified gate).
* `unverified` means an assumption lacks a named attestor, a design
  diagnostic failed, or a falsification test rejected: a rejected control is
  evidence against the attested assumptions, so the claim does not keep
  `attested`.

`confirmed` says: under the attested assumptions the interval has its
nominal coverage (measured at finite `n` in the certificate), and the
battery would have flagged a violation of at least the certified `delta_min`
and did not. It does not say the assumptions are true. A violation the
negative controls do not share passes undetected, and the held-out set alone
never yields `confirmed`.
"""

from collections.abc import Collection
from collections.abc import Mapping
import dataclasses
from typing import Any

from scmle.contracts import qualification

NOT_IDENTIFIED = "not_identified"


@dataclasses.dataclass(frozen=True, slots=True)
class Inputs:
    """Facts about a causal run, copied from its report files.

    Attributes:
        identified: The selector found an eligible strategy.
        needed: Assumptions the strategy needs.
        attestors: Assumption to the named attestor (empty text is none).
        confirmatory: The estimate used `C`, opened once.
        sound: Every design diagnostic passed its threshold.
        gates: Certificate gates the design diagnostics rest on.
        interval: Certificate gate of the strategy's interval.
        ran: At least one falsification test ran.
        passed: No test was rejected at `alpha_falsify`.
        negative: The charter declared a negative control.
        gate: Certificate gate of the battery.
        replicated: Two or more strata with one sign, no
            heterogeneity; None when not run.
        holds: The sensitivity gate holds; None when it was not
            declared or could not run.
        sensitivity: Certificate gate of the sensitivity analysis, or
            None when the strategy needs none.
        unit: One treated unit (nothing to hold out).
        instrument: The first stage failed its threshold.
        certified: Gates with a current passing certificate.
        checked: The back-door adjustment set was verified against
            a declared DAG (True for strategies that have no such set).
    """

    identified: bool
    needed: tuple[str, ...]
    attestors: Mapping[str, str]
    confirmatory: bool
    sound: bool
    gates: tuple[str, ...]
    interval: str
    ran: bool
    passed: bool
    negative: bool
    gate: str
    replicated: bool | None
    holds: bool | None
    sensitivity: str | None
    unit: bool
    instrument: bool
    certified: Collection[str]
    checked: bool = True


@dataclasses.dataclass(frozen=True)
class Assignment:
    """The tier, the five conditions and the reasons for any shortfall.

    Attributes:
        tier: A `Tier`, or None when the claim is `not_identified`.
        status: `identified` or `not_identified`.
        conditions: The five conditions for `confirmed`.
        reasons: Why the tier is not higher.
        action: The decision-map action implied (`refuse_claim`,
            `downgrade_tier` or an empty string).
    """

    tier: qualification.Tier | None
    status: str
    conditions: dict[str, bool]
    reasons: tuple[str, ...]
    action: str

    def to_dict(self) -> dict[str, Any]:
        """Builds the JSON-ready record.

        Returns:
            A mapping with the tier value (None when not identified), the
            status, the conditions, the reasons and the action.
        """
        return {
            "tier": None if self.tier is None else self.tier.value,
            "status": self.status,
            "conditions": dict(self.conditions),
            "reasons": list(self.reasons),
            "action": self.action,
        }


def assess(facts: Inputs, certified: Collection[str]) -> dict[str, bool]:
    """Evaluates the five conditions of the tier rule.

    Args:
        facts: The facts about the run.
        certified: The gates that carry a passing certificate.

    Returns:
        Whether each of the five conditions holds.
    """
    gates_ok = all(g in certified for g in facts.gates)
    return {
        "estimated_on_c": facts.confirmatory and not facts.unit,
        "design_diagnostics": facts.sound
        and facts.checked
        and gates_ok
        and facts.interval in certified
        and not facts.instrument,
        "falsification_battery": facts.ran
        and facts.passed
        and facts.negative
        and facts.gate in certified,
        "replication": bool(facts.replicated),
        "sensitivity": bool(facts.holds)
        and (facts.sensitivity is None or facts.sensitivity in certified),
    }


def shortfalls(
    facts: Inputs, conditions: Mapping[str, bool], certified: Collection[str]
) -> list[str]:
    """Lists why the design conditions fall short of `confirmed`.

    Args:
        facts: The facts about the run.
        conditions: Whether each of the five conditions holds.
        certified: The gates that carry a passing certificate.

    Returns:
        The reasons for the use of `C`, a weak instrument, an unchecked
        adjustment set and a design gate without a certificate, in that
        order.
    """
    reasons: list[str] = []
    if not conditions["estimated_on_c"]:
        reasons.append(
            "one treated unit: nothing to hold out"
            if facts.unit
            else "the estimate did not use C"
        )
    if facts.instrument:
        reasons.append("weak instrument: only the robust interval is valid")
    if not facts.checked:
        reasons.append(
            "the adjustment set is an unchecked attestation (no declared "
            "DAG confirms it)"
        )
    gates_ok = all(g in certified for g in facts.gates)
    if not gates_ok or facts.interval not in certified:
        reasons.append("a design gate has no passing certificate")
    return reasons


def diagnose(
    facts: Inputs, conditions: Mapping[str, bool], certified: Collection[str]
) -> list[str]:
    """Lists why the falsification, replication and sensitivity fall short.

    Args:
        facts: The facts about the run.
        conditions: Whether each of the five conditions holds.
        certified: The gates that carry a passing certificate.

    Returns:
        The reasons for the negative control and battery, the replication
        and the sensitivity analysis, in that order.
    """
    reasons: list[str] = []
    if not facts.ran or not facts.negative:
        reasons.append("no negative control ran")
    elif facts.gate not in certified:
        reasons.append("the battery has no passing certificate")
    if facts.replicated is None:
        reasons.append("replication did not run (fewer than two strata)")
    elif not facts.replicated:
        reasons.append("strata disagree or are heterogeneous")
    if facts.holds is None:
        reasons.append("no sensitivity analysis was declared or possible")
    elif not facts.holds:
        reasons.append("the sensitivity gate failed at the declared strength")
    elif conditions["sensitivity"] is False:
        reasons.append("the sensitivity analysis has no passing certificate")
    return reasons


def decide(facts: Inputs) -> Assignment:
    """Applies the tier rule.

    Args:
        facts: The facts about the run, copied from its report files.

    Returns:
        The tier decision: `not_identified` with no tier when no strategy
        was eligible; `unverified` when an assumption lacks a named
        attestor, a design diagnostic failed or a falsification test
        rejected; `confirmed` when all five conditions hold; otherwise
        `attested`. For `attested` the reasons list every shortfall; for
        `unverified` they list only the failures that caused it; they are
        empty for `confirmed`.
    """
    if not facts.identified:
        return Assignment(
            None,
            NOT_IDENTIFIED,
            dict.fromkeys(CONDITIONS, False),
            ("no eligible identification strategy",),
            "refuse_claim",
        )
    certified = set(facts.certified)
    unnamed = [
        a for a in facts.needed if not facts.attestors.get(a, "").strip()
    ]
    conditions = assess(facts, certified)
    reasons: list[str] = []
    if unnamed:
        reasons.append("no named attestor for: " + ", ".join(unnamed))
    if not facts.sound:
        reasons.append("a design diagnostic failed")
    if facts.ran and not facts.passed:
        reasons.append("a falsification test rejected")
    if reasons:
        return Assignment(
            qualification.Tier.UNVERIFIED,
            "identified",
            conditions,
            tuple(reasons),
            "downgrade_tier",
        )
    reasons.extend(shortfalls(facts, conditions, certified))
    reasons.extend(diagnose(facts, conditions, certified))
    if all(conditions.values()):
        return Assignment(
            qualification.Tier.CONFIRMED, "identified", conditions, (), ""
        )
    return Assignment(
        qualification.Tier.ATTESTED,
        "identified",
        conditions,
        tuple(reasons),
        "downgrade_tier",
    )


CONDITIONS = (
    "estimated_on_c",
    "design_diagnostics",
    "falsification_battery",
    "replication",
    "sensitivity",
)
