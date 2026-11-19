"""The result of the single-feature skill test and its ledger check."""

import dataclasses
import math

import numpy

from scmle.assurance.admissibility.leakage.settings import Config
from scmle.assurance.detectors import certificates
from scmle.contracts import qualification
from scmle.runtime import randomness


@dataclasses.dataclass(frozen=True, slots=True)
class Skill:
    """Single-feature skill against a permutation null.

    Attributes:
        feature: Feature name.
        skill: Observed out-of-fold skill.
        null: Mean skill over the permuted targets.
        floor: One-sided lower bound on the skill.
        p_value: Permutation p-value.
        n: Rows.
        power: Too few rows; the feature is never a suspect.
        suspect: There are enough rows, p <= alpha and the lower bound is
            above the ceiling.
        check: Ledger record.
    """

    feature: str
    skill: float
    null: float
    floor: float
    p_value: float
    n: int
    power: bool
    suspect: bool
    check: qualification.Check


def decide(
    suspect: bool, low_power: bool, gate: certificates.Authority
) -> qualification.Decision:
    """Chooses the decision of a leakage test, demoted when not certified.

    Args:
        suspect: Whether the test called the feature a suspect.
        low_power: Whether there were too few rows.
        gate: Whether the detector may gate.

    Returns:
        `quarantine` for a suspect, `warn` for low power, `pass` otherwise;
        a quarantine is lowered to a warning unless `gate` is gating.
    """
    if suspect:
        raw = qualification.Decision.QUARANTINE
    elif low_power:
        raw = qualification.Decision.WARN
    else:
        raw = qualification.Decision.PASS
    return certificates.demote(raw, gate)


def summarize(
    path: str,
    cfg: Config,
    result: tuple[float, list[float], float, float],
    n: int,
    decision: qualification.Decision,
    set_id: qualification.Subset,
) -> qualification.Check:
    """Writes the ledger check of one skill test.

    Args:
        path: Seed path of the test, also its check id.
        cfg: Settings (`run_seed`).
        result: The observed skill, the permuted skills, the lower bound
            and the p-value.
        n: Rows tested.
        decision: The test's decision.
        set_id: Set the rows belong to.

    Returns:
        The check; its interval is None when the lower bound is -inf.
    """
    skill, null, bound, p_value = result
    return qualification.Check(
        identifier=path,
        null=(
            "the feature has no predictive skill beyond a target permuted "
            "within its blocks"
        ),
        statistic=skill,
        p_value=p_value,
        interval=(bound, max(skill, bound)) if math.isfinite(bound) else None,
        effect=skill - float(numpy.mean(null)),
        n=n,
        decision=decision,
        subset=set_id,
        seeds=randomness.seed_path(cfg.seed, path, 1),
        tier=qualification.Tier.UNVERIFIED,
    )
