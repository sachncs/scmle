"""Selects, estimates, falsifies and tiers one chartered causal claim."""

from collections.abc import Mapping
import dataclasses
from typing import Any

import numpy
import pandas

import scmle.assurance.admissibility.dag
from scmle.assurance.charter import causal
from scmle.assurance.ledger import budget
from scmle.causal.identification import battery
from scmle.causal.identification import select
from scmle.causal.identification import tier
import scmle.causal.identification.certified
from scmle.causal.identification.run.adjustment import Backdoor
from scmle.causal.identification.run.adjustment import RandomizedEstimator
from scmle.causal.identification.run.documents import arbitrate
from scmle.causal.identification.run.documents import assemble
from scmle.causal.identification.run.documents import Outcome
from scmle.causal.identification.run.documents import prerequisites
from scmle.causal.identification.run.documents import refuse
from scmle.causal.identification.run.estimator import Est
from scmle.causal.identification.run.estimator import Estimator
from scmle.causal.identification.run.exogeneity import Instrument
from scmle.causal.identification.run.exogeneity import RdEstimator
from scmle.causal.identification.run.panel import DiD
from scmle.causal.identification.run.panel import Synth
from scmle.causal.identification.run.strata import replication
from scmle.causal.identification.run.strata import subgroup_results
from scmle.contracts import qualification

INTERVAL_GATES = {
    "backdoor_aipw": "coverage_aipw",
    "randomized_diff": "coverage_randomized",
    "did": "coverage_did",
    "iv": "coverage_iv",
    "rd": "coverage_rd",
    "synth": "size_synth",
}


DESIGN_GATES = {
    "backdoor_aipw": ("overlap",),
    "iv": ("iv_strength",),
    "rd": ("density_rd",),
}


BATTERY_GATES = {
    "backdoor_aipw": "battery_nco",
    "randomized_diff": "battery_nco",
    "did": "battery_pretrend",
    "iv": "battery_iv",
    "rd": "battery_rd",
    "synth": "battery_synth",
}


SENSITIVITY_GATES = {
    "backdoor_aipw": "sensitivity_bound",
    "did": "sensitivity_did",
}


ADAPTERS: dict[str, type[Estimator]] = {
    "backdoor_aipw": Backdoor,
    "randomized_diff": RandomizedEstimator,
    "did": DiD,
    "iv": Instrument,
    "rd": RdEstimator,
    "synth": Synth,
}


def alphas(error_budget: budget.Alpha) -> tuple[float, float]:
    """Splits the confirmation level into the primary and falsification levels.

    Args:
        error_budget: The charter's error budget.

    Returns:
        ``(alpha_primary, alpha_falsify)``, the confirmation level divided in
        the ratio 4 to 1.
    """
    parts = budget.apportion(error_budget, battery.FALSIFY_WEIGHTS)
    return (
        float(parts["estimate"].confirmation.alpha_confirm),
        float(parts["falsify"].confirmation.alpha_confirm),
    )


def fragility(
    adapter: Estimator, c: pandas.DataFrame, level: float
) -> dict[str, Any]:
    """Assesses sensitivity to unmeasured confounding, if it can be done.

    Args:
        adapter: The strategy's estimator.
        c: The confirmation rows.
        level: Confidence level.

    Returns:
        The sensitivity document of the strategy, or a note that a single
        treated unit has only the placebo permutation.
    """
    if adapter.single_unit:
        return {
            "holds": None,
            "note": "single treated unit: inference is the placebo permutation",
        }
    return adapter.sensitivity(c, level)


def caveats(
    spec: causal.Question, strategy: str, *, checked: bool
) -> list[str]:
    """Lists the notes about what the run could not do or check.

    Args:
        spec: The charter's causal block.
        strategy: Registry name of the strategy.
        checked: Whether a declared DAG verified the adjustment set.

    Returns:
        A note for each negative-control exposure that the strategy does
        not support, then a note when the adjustment set is unchecked.
    """
    notes = [
        f"negative-control exposure {control.variable} is only "
        "supported for backdoor_aipw and was not run"
        for control in spec.controls
        if control.kind == "exposure" and strategy != "backdoor_aipw"
    ]
    if not checked:
        notes.append(
            "the adjustment set is an unchecked attestation (no DAG): the "
            "tier cannot exceed attested"
        )
    return notes


@dataclasses.dataclass(frozen=True)
class Trial:
    """The primary estimate of the chosen strategy and the levels it used.

    Attributes:
        plan: The selector's plan.
        spec: The charter's causal block.
        strategy: Registry name of the chosen strategy.
        adapter: The strategy's estimator, prepared on `X`.
        primary: The primary estimate on `C`.
        level: Confidence level of the primary interval.
        alpha_primary: The primary level, `1 - level`.
        alpha_falsify: Familywise level of the falsification battery.
    """

    plan: select.Plan
    spec: causal.Question
    strategy: str
    adapter: Estimator
    primary: Est
    level: float
    alpha_primary: float
    alpha_falsify: float


def design(
    plan: select.Plan,
    spec: causal.Question,
    strategy: str,
    x: pandas.DataFrame,
    seed: int,
    learner: str,
    alpha_falsify: float,
    dag_checks: Mapping[str, Any] | None,
) -> Estimator | Outcome:
    """Builds the strategy's estimator and runs its design check on `X`.

    Args:
        plan: The selector's plan.
        spec: The charter's causal block.
        strategy: Registry name of the chosen strategy.
        x: Exploration rows.
        seed: Seed of the learners.
        learner: `linear` or `boosted` nuisance learners.
        alpha_falsify: Familywise level of the falsification battery.
        dag_checks: What was found about the declared DAG, or None.

    Returns:
        The prepared estimator, or the refusal documents when the design
        check raises `ValueError` or refuses the claim.
    """
    adapter = ADAPTERS[strategy](spec, x, seed, learner, alpha_falsify)
    try:
        adapter.prepare()
    except ValueError as error:
        # handled: returned as failed finding/problem/refusal with error text
        return refuse(
            plan, spec, f"design could not be assessed: {error}", dag_checks
        )
    if adapter.refusal:
        return refuse(plan, spec, adapter.refusal, dag_checks)
    return adapter


def falsify(
    trial: Trial, c: pandas.DataFrame
) -> tuple[battery.Battery, Mapping[str, Any], list[str]]:
    """Runs the falsification battery and the replication on `C`.

    Args:
        trial: The primary estimate and its levels.
        c: The confirmation rows.

    Returns:
        The battery report, the replication result (no columns and `passed`
        None when no strata are declared or there is a single treated unit)
        and the notes: one when the controls raised `ValueError`, in which
        case the battery ran on no tests.
    """
    adapter = trial.adapter
    tests: list[battery.Test] = []
    notes: list[str] = []
    try:
        tests = adapter.controls(c, trial.level)
    except ValueError as error:
        # handled: returned as failed finding/problem/refusal with error text
        notes.append(f"controls could not run: {error}")
    report_battery = battery.run_battery(tests, trial.alpha_falsify)
    rep: dict[str, Any]
    if trial.spec.strata and not adapter.single_unit:
        rep = replication(
            adapter, c, trial.level, trial.primary.estimate, trial.alpha_falsify
        )
    else:
        rep = {"by_column": {}, "passed": None}
    return report_battery, rep, notes


def grade(
    trial: Trial,
    report_battery: battery.Battery,
    rep: Mapping[str, Any],
    sens: Mapping[str, Any],
    attestors: Mapping[str, str] | None,
    certified: tuple[str, ...],
) -> tuple[tier.Assignment, tuple[str, ...], bool]:
    """Tiers the claim from what the run found.

    Args:
        trial: The primary estimate and its levels.
        report_battery: The falsification battery.
        rep: The replication result.
        sens: The sensitivity document.
        attestors: Assumption to the named attestor.
        certified: Gates with a current passing certificate.

    Returns:
        The tier decision, the assumptions the strategy relies on, and
        whether the adjustment set is checked (always for a strategy other
        than backdoor_aipw; by a declared DAG for backdoor_aipw).
    """
    plan, spec, strategy = trial.plan, trial.spec, trial.strategy
    adapter = trial.adapter
    needed = prerequisites(strategy, spec)
    checked = strategy != "backdoor_aipw" or plan.verdict(strategy).checked
    facts = tier.Inputs(
        identified=True,
        needed=needed,
        attestors=dict(attestors or {}),
        estimated_on_c=not adapter.single_unit,
        design_passed=True,
        design_gates=DESIGN_GATES.get(strategy, ()),
        interval_gate=INTERVAL_GATES[strategy],
        battery_ran=report_battery.ran,
        battery_passed=report_battery.passed,
        negative_control_declared=bool(spec.controls),
        battery_gate=BATTERY_GATES[strategy],
        replication_passed=rep["passed"],
        sensitivity_holds=sens.get("holds"),
        sensitivity_gate=SENSITIVITY_GATES.get(strategy),
        single_unit=adapter.single_unit,
        weak_instrument=adapter.weak,
        certified=certified,
        adjustment_checked=checked,
    )
    return tier.decide(facts), needed, checked


def chronicle(
    trial: Trial,
    notes: list[str],
    dag_checks: Mapping[str, Any] | None,
    subgroup_found: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Builds the identification report of an estimated claim.

    Args:
        trial: The primary estimate and its levels.
        notes: Remarks for the report.
        dag_checks: What was found about the declared DAG, or None.
        subgroup_found: The labelled subgroup report, or None.

    Returns:
        The report: status, strategy, levels, the primary estimate, the
        design diagnostics, notes and the stated verdict, plus the DAG
        checks and subgroups when given.
    """
    primary = trial.primary
    verdict = arbitrate(trial.spec)
    verdict["interval_excludes_zero"] = (
        primary.conclusive() if numpy.isfinite(primary.low) else None
    )
    report = {
        "status": select.IDENTIFIED,
        "strategy": trial.strategy,
        "estimand": trial.spec.estimand,
        "level": trial.level,
        "alpha_primary": trial.alpha_primary,
        "alpha_falsify": trial.alpha_falsify,
        "estimate": {
            "point": primary.estimate,
            "se": primary.se,
            "low": primary.low,
            "high": primary.high,
            "n": primary.n,
            "detail": dict(primary.detail),
        },
        "design": trial.adapter.design,
        "multiverse_not_run": list(trial.plan.multiverse),
        "notes": notes,
        "verdict": verdict,
    }
    if dag_checks is not None:
        report["dag"] = dict(dag_checks)
    if subgroup_found is not None:
        report["subgroups"] = subgroup_found
    return report


def confirm(
    trial: Trial,
    c: pandas.DataFrame,
    attestors: Mapping[str, str] | None,
    certified: tuple[str, ...],
    named: tuple[str, ...] | None,
    dag_checks: Mapping[str, Any] | None,
    attestation_report: Mapping[str, Any] | None,
) -> Outcome:
    """Falsifies, assesses and tiers a claim whose primary estimate exists.

    Args:
        trial: The primary estimate and its levels.
        c: The confirmation rows.
        attestors: Assumption to the named attestor.
        certified: Gates with a current passing certificate.
        named: The certified gates the caller named, or None (see
            `strata.certify`).
        dag_checks: What was found about the declared DAG, or None.
        attestation_report: What was decided about the signed attestations,
            or None.

    Returns:
        The five documents, their hashes and the tier decision.
    """
    adapter, spec, level = trial.adapter, trial.spec, trial.level
    report_battery, rep, notes = falsify(trial, c)
    sens = fragility(adapter, c, level)
    subgroup_found = subgroup_results(adapter, spec, c, level, named)
    decision, needed, checked = grade(
        trial, report_battery, rep, sens, attestors, certified
    )
    notes.extend(caveats(spec, trial.strategy, checked=checked))
    report = chronicle(trial, notes, dag_checks, subgroup_found)
    falsification = {
        **report_battery.to_dict(),
        "replication": rep,
        "gate": BATTERY_GATES[trial.strategy],
    }
    return assemble(
        trial.plan,
        spec,
        report,
        falsification,
        sens,
        decision,
        trial.strategy,
        needed,
        attestors,
        certified,
        subgroup_found,
        attestation_report,
    )


def identify(
    spec: causal.Question,
    x: pandas.DataFrame,
    c: pandas.DataFrame,
    error_budget: budget.Alpha,
    attestors: Mapping[str, str] | None = None,
    dag: scmle.assurance.admissibility.dag.Dag | None = None,
    seed: int = 0,
    learner: str = "linear",
    certified: tuple[str, ...] | None = None,
    dag_checks: Mapping[str, Any] | None = None,
    attestation_report: Mapping[str, Any] | None = None,
) -> Outcome:
    """Selects, estimates, falsifies, assesses and tiers a causal claim.

    Args:
        spec: The charter's causal block.
        x: Exploration rows (design, nuisances, bandwidth).
        c: Confirmation rows; used for the estimate, controls, replication,
            sensitivity and subgroups, never to fit nuisances or a bandwidth.
        error_budget: The charter's error budget; its confirmation level is
            split into the primary level and `alpha_falsify` (weights 4:1).
        attestors: Assumption to the named attestor.
        dag: The declared DAG, if any.
        seed: Seed of the learners.
        learner: `linear` or `boosted` nuisance learners.
        certified: Gates with a current passing certificate; None loads the
            shipped certificate file, and an empty tuple certifies nothing.
        dag_checks: What the admissibility stage found about the declared DAG
            on `X` (its hash, the conditional-independence falsification, the
            adjustment members the graph rules refused); copied into the
            identification report.
        attestation_report: What `attest.apply` decided about the signed
            attestations (`Applied.to_dict()`), copied into the claim file.

    Returns:
        The five documents, their hashes and the tier decision. A
        `ValueError` from the adapter's design check or primary estimate is
        not raised: it gives the refusal documents with the error text as the
        reason, as does a failed overlap or density gate. A `ValueError` from
        the controls is recorded as a note instead.
    """
    plan = select.select(spec, dag)
    if plan.primary is None:
        return refuse(plan, spec, dag_checks=dag_checks)
    named = certified
    if certified is None:
        certified = tuple(
            sorted(scmle.causal.identification.certified.certificated())
        )
    primary_alpha, alpha_falsify = alphas(error_budget)
    level = 1.0 - primary_alpha
    strategy = plan.primary
    adapter = design(
        plan, spec, strategy, x, seed, learner, alpha_falsify, dag_checks
    )
    if isinstance(adapter, Outcome):
        return adapter
    try:
        primary = adapter.estimate(c, level)
    except ValueError as error:
        # handled: returned as failed finding/problem/refusal with error text
        return refuse(plan, spec, str(error), dag_checks)
    trial = Trial(
        plan,
        spec,
        strategy,
        adapter,
        primary,
        level,
        primary_alpha,
        alpha_falsify,
    )
    return confirm(
        trial, c, attestors, certified, named, dag_checks, attestation_report
    )


NOT_ESTIMATED = (
    "no estimate was computed on the confirmation set: it was not available "
    "to the run, or the claim could not be registered, so the claim is "
    "unverified. `identify` computes the estimate on the single opening of C "
    "when the ledger hands it the released rows (`scmle.causal.claim`)."
)


def design_only(
    spec: causal.Question,
    x: pandas.DataFrame,
    error_budget: budget.Alpha,
    dag: scmle.assurance.admissibility.dag.Dag | None = None,
    seed: int = 0,
    learner: str = "linear",
    dag_checks: Mapping[str, Any] | None = None,
) -> Outcome:
    """Selects and diagnoses the design on `X`, with no estimate.

    Used by the pipeline stage, which never hands `C` to this module: the
    plan, the refusal if there is one, the design diagnostics and a claim
    file at tier `unverified`.

    Args:
        spec: The charter's causal block.
        x: Exploration rows.
        error_budget: The charter's error budget.
        dag: The declared DAG, if any.
        seed: Seed of the learners.
        learner: `linear` or `boosted` nuisance learners.
        dag_checks: What was found about the declared DAG, copied into the
            report.

    Returns:
        The five documents, their hashes and a tier decision at `unverified`
        (or the refusal documents when the claim is not identified).
    """
    plan = select.select(spec, dag)
    if plan.primary is None:
        return refuse(plan, spec, dag_checks=dag_checks)
    alpha_falsify = alphas(error_budget)[1]
    strategy = plan.primary
    adapter = design(
        plan, spec, strategy, x, seed, learner, alpha_falsify, dag_checks
    )
    if isinstance(adapter, Outcome):
        return adapter
    report = {
        "status": select.IDENTIFIED,
        "strategy": strategy,
        "estimand": spec.estimand,
        "estimate": None,
        "design": adapter.design,
        "multiverse_not_run": list(plan.multiverse),
        "notes": [NOT_ESTIMATED],
    }
    if dag_checks is not None:
        report["dag"] = dict(dag_checks)
    decision = tier.Assignment(
        qualification.Tier.UNVERIFIED,
        "identified",
        dict.fromkeys(tier.CONDITIONS, False),
        (NOT_ESTIMATED,),
        "downgrade_tier",
    )
    return assemble(
        plan,
        spec,
        report,
        {"ran": False, "note": NOT_ESTIMATED},
        {"holds": None, "note": NOT_ESTIMATED},
        decision,
        strategy,
        prerequisites(strategy, spec),
    )
