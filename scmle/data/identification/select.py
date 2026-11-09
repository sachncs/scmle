"""Choosing an identification strategy from the charter alone.

A strategy is eligible only if the charter's estimand matches it, every
assumption it needs is attested and the data design it needs is declared.
The function reads the causal block and (optionally) a declared DAG; it never
sees outcome values and imports no estimator, so choosing is not a
specification search. When nothing is eligible the result is a refusal:
`status="not_identified"`, the missing pieces per strategy, and no estimate.
"""

from collections.abc import Mapping
import dataclasses
from typing import Any

import scmle.data.admissibility.dag
from scmle.data.charter import causal

IDENTIFIED = "identified"
NOT_IDENTIFIED = "not_identified"
ASSOCIATIONAL = (
    "any summary printed next to this report is associational, not causal"
)


@dataclasses.dataclass(frozen=True, slots=True)
class Method:
    """What a strategy needs before it may run.

    Attributes:
        name: Registry name.
        estimands: Charter estimands it can answer.
        assumptions: Required attested assumptions (the word
            `unconfoundedness_given` stands for the prefixed form).
        design: Causal-block fields that must be declared.
        family: Short description of the family.
    """

    name: str
    estimands: tuple[str, ...]
    assumptions: tuple[str, ...]
    design: tuple[str, ...]
    family: str


STRATEGIES: dict[str, Method] = {
    s.name: s
    for s in (
        Method(
            "randomized_diff",
            ("ATE",),
            ("randomized_assignment", "no_interference"),
            (),
            "difference in means with a recorded assignment mechanism",
        ),
        Method(
            "backdoor_aipw",
            ("ATE", "ATT"),
            ("unconfoundedness_given", "positivity", "no_interference"),
            (),
            "cross-fitted augmented IPW under back-door adjustment",
        ),
        Method(
            "did",
            ("ATT_DID",),
            ("parallel_trends", "no_anticipation", "no_interference"),
            ("unit", "time", "adoption"),
            "group-time difference in differences",
        ),
        Method(
            "iv",
            ("LATE",),
            (
                "instrument_relevance",
                "exclusion",
                "independence",
                "monotonicity",
            ),
            ("instrument",),
            "just-identified two-stage least squares",
        ),
        Method(
            "rd",
            ("LATE_RD",),
            ("continuity_at_cutoff", "no_manipulation"),
            ("running_variable", "cutoff"),
            "local linear regression discontinuity",
        ),
        Method(
            "synth",
            ("ATT_SC",),
            ("synthetic_hull", "no_interference"),
            ("unit", "time"),
            "synthetic control with placebo-in-space inference",
        ),
    )
}


@dataclasses.dataclass(frozen=True, slots=True)
class Verdict:
    """Why a strategy is or is not eligible.

    Attributes:
        strategy: Registry name.
        eligible: Whether every requirement holds.
        missing_assumptions: Required assumptions not attested.
        missing_design: Required design fields not declared.
        reasons: Other reasons (estimand mismatch, DAG failure, ...).
        checked: For back-door adjustment, whether a DAG verified the
            adjustment set; False caps the tier at `attested`.
    """

    strategy: str
    eligible: bool
    missing_assumptions: tuple[str, ...] = ()
    missing_design: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    checked: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Builds the JSON-ready record.

        Returns:
            A mapping of the strategy, eligibility, the missing pieces, the
            reasons and whether a DAG checked the adjustment set.
        """
        return {
            "strategy": self.strategy,
            "eligible": self.eligible,
            "missing_assumptions": list(self.missing_assumptions),
            "missing_design": list(self.missing_design),
            "reasons": list(self.reasons),
            "adjustment_checked_by_dag": self.checked,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Plan:
    """The selector's output (`identification_plan.json`).

    Attributes:
        status: `identified` or `not_identified`.
        primary: The first eligible strategy in priority order, or None.
        multiverse: Other eligible strategies, in order.
        verdicts: One verdict per considered strategy.
        order: The priority order that was used.
        chartered_order: Whether the order came from the charter.
    """

    status: str
    primary: str | None
    multiverse: tuple[str, ...]
    verdicts: tuple[Verdict, ...]
    order: tuple[str, ...]
    chartered_order: bool

    def verdict(self, name: str) -> Verdict:
        """Looks up the verdict of one strategy.

        Args:
            name: Registry name of the strategy.

        Returns:
            The verdict for that strategy.

        Raises:
            KeyError: If the strategy was not considered.
        """
        for item in self.verdicts:
            if item.strategy == name:
                return item
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        """Builds the JSON-ready plan.

        Returns:
            A mapping of the status, primary strategy, multiverse, order and
            per-strategy verdicts. A plan that is not identified adds a null
            estimate, the action `refuse_claim` and a note that any printed
            summary is associational.
        """
        record: dict[str, Any] = {
            "status": self.status,
            "primary": self.primary,
            "multiverse": list(self.multiverse),
            "order": list(self.order),
            "order_chartered": self.chartered_order,
            "strategies": [v.to_dict() for v in self.verdicts],
        }
        if self.status == NOT_IDENTIFIED:
            record["estimate"] = None
            record["action"] = "refuse_claim"
            record["note"] = ASSOCIATIONAL
        return record


def judge(
    strategy: Method,
    spec: causal.Question,
    dag: scmle.data.admissibility.dag.Dag | None,
) -> Verdict:
    """Decides whether one strategy may run on this charter.

    Args:
        strategy: The strategy to judge.
        spec: The charter's causal block.
        dag: The declared DAG, or None.

    Returns:
        The verdict: eligible only when the estimand matches, every needed
        assumption is attested, every needed design field is declared and,
        for back-door adjustment, no DAG failure is found: a supplied DAG
        must accept the adjustment set, and a DAG the charter declares must
        have been supplied. The back-door check runs only when nothing else
        is missing.
    """
    reasons: list[str] = []
    if spec.estimand not in strategy.estimands:
        reasons.append(
            f"estimand {spec.estimand} is not answered by {strategy.name} "
            f"(answers {', '.join(strategy.estimands)})"
        )
    attested = set(spec.assumptions)
    missing = []
    for word in strategy.assumptions:
        if word == "unconfoundedness_given":
            if causal.covariates_of(spec.assumptions) is None:
                missing.append(causal.UNCONFOUNDEDNESS_PREFIX + "<columns>")
        elif word not in attested:
            missing.append(word)
    absent = tuple(n for n in strategy.design if getattr(spec, n) is None)
    checked = False
    if strategy.name == "backdoor_aipw" and not missing and not reasons:
        reasons, checked = check_adjustment(spec, dag)
    return Verdict(
        strategy.name,
        not (missing or absent or reasons),
        tuple(missing),
        absent,
        tuple(reasons),
        checked,
    )


def check_adjustment(
    spec: causal.Question, dag: scmle.data.admissibility.dag.Dag | None
) -> tuple[list[str], bool]:
    """Verifies the declared adjustment set against a declared DAG.

    Args:
        spec: The charter's causal block.
        dag: The declared DAG, or None.

    Returns:
        The reasons the set fails (empty when none) and whether a DAG
        verified it. Without a DAG the set is unchecked: no reason is given
        unless the charter declares a DAG that was not supplied.
    """
    names = causal.covariates_of(spec.assumptions) or ()
    if dag is None:
        if spec.dag:
            return (["a DAG is declared but was not supplied"], False)
        return ([], False)
    try:
        valid = dag.backdoor_valid(spec.treatment, spec.outcome, names)
    except ValueError as error:
        return ([f"the declared DAG cannot check the set: {error}"], False)
    if not valid:
        return (
            [
                (
                    "the declared adjustment set is not a valid back-door set "
                    "for the declared DAG"
                )
            ],
            False,
        )
    return ([], True)


def select(
    spec: causal.Question, dag: scmle.data.admissibility.dag.Dag | None = None
) -> Plan:
    """Chooses the primary strategy and the declared multiverse.

    Args:
        spec: The charter's causal block.
        dag: The declared DAG hypothesis, if the charter declares one.

    Returns:
        The plan; `status` is `not_identified` when nothing is eligible.
    """
    chartered = bool(spec.strategies)
    order = spec.strategies if chartered else tuple(STRATEGIES)
    verdicts = []
    for name in order:
        strategy = STRATEGIES.get(name)
        if strategy is None:
            verdicts.append(
                Verdict(
                    name, eligible=False, reasons=(f"unknown strategy {name}",)
                )
            )
        else:
            verdicts.append(judge(strategy, spec, dag))
    eligible = tuple(v.strategy for v in verdicts if v.eligible)
    return Plan(
        status=IDENTIFIED if eligible else NOT_IDENTIFIED,
        primary=eligible[0] if eligible else None,
        multiverse=eligible[1:],
        verdicts=tuple(verdicts),
        order=tuple(order),
        chartered_order=chartered,
    )


def refusal(plan: Plan, spec: causal.Question) -> Mapping[str, Any]:
    """Builds the `identification_report.json` of a run that cannot identify.

    Args:
        plan: The selector's plan, with its verdicts.
        spec: The charter's causal block.

    Returns:
        A mapping with status `not_identified`, the estimand, treatment and
        outcome, a null estimate, the action `refuse_claim`, what each
        strategy is missing and a note that any printed summary is
        associational.
    """
    missing = {
        v.strategy: {
            "missing_assumptions": list(v.missing_assumptions),
            "missing_design": list(v.missing_design),
            "reasons": list(v.reasons),
        }
        for v in plan.verdicts
    }
    return {
        "status": NOT_IDENTIFIED,
        "estimand": spec.estimand,
        "treatment": spec.treatment,
        "outcome": spec.outcome,
        "estimate": None,
        "action": "refuse_claim",
        "missing_per_strategy": missing,
        "note": ASSOCIATIONAL,
    }
