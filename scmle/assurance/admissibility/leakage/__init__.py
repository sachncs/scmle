"""Statistical leakage tests: suspicion, never proof.

Statistics cannot separate a leak from a strong legitimate predictor, so every
result here is a suspicion for a person (or a certified gate) to weigh against
provenance. All tests read the exploration set `X` only. Each test
returns a `Check` for the hypothesis ledger.

Tests
-----
1. Single-feature skill against a permutation null. A small tree
   (`max_leaf_nodes` leaves) is fitted on one feature, with cross-fitted
   (out-of-fold) predictions. Skill is the out-of-fold ROC AUC (macro
   one-vs-rest for more than two classes) or R^2 for a continuous target. The
   null is the same procedure on `B` permutations of the target, permuted
   within groups or consecutive time blocks when given, and the p-value is
   `(1 + #{null >= observed}) / (B + 1)`, the exact permutation p-value for
   randomly drawn permutations (Phipson and Smyth, 2010, Stat. Appl. Genet.
   Mol. Biol. 9(1): the formula was read in the arXiv version, 1603.05766).
   A one-sided lower confidence bound on the skill comes from a bootstrap of
   the out-of-fold predictions (clustered by group when groups are given).
   Folds share training rows, so the interval is approximate; its coverage is
   not claimed, only the measured size and recall of the decision rule.
   A feature is a *suspect* when the p-value is at most `alpha` AND the
   lower bound exceeds the plausibility ceiling `skill_ceiling`, and only
   when there are enough rows (otherwise `low_power`, never a suspect).
2. Conditional permutation test. The feature is permuted within strata of the
   other admitted features, then the same single-feature skill is compared
   with its permuted distribution. The strata are the leaves of a small tree
   that predicts the feature from the others, a
   crude estimate of its conditional distribution. This follows the idea of
   Berrett, Wang, Barber and Samworth (2020, JRSS-B 82(1): 175-197, the
   conditional permutation test: permute the feature in a way that respects
   its dependence on the other covariates). It is a simplification: strata
   instead of their MCMC sampler of permutations, and it has not been
   validated against the paper's sampler. That the paper's abstract
   describes the method was verified; the equivalence of this
   approximation was not.
3. Missingness as leakage. When values are recorded only for some outcomes,
   the missing-value indicator is tested against the target (binned to
   quartiles when continuous) with Fisher's exact test for a 2x2 table with
   an expected count below 5, otherwise Pearson's chi-square without
   continuity correction, plus Cramer's V.

Guarantees
----------
Under exchangeability within the permutation blocks, the permutation
p-value is valid, so the rate of calling a signal-free feature a suspect is
at most `alpha`, and in practice far lower because the lower bound must also
clear the ceiling. Nothing is claimed about power against leaks weaker than
the ceiling. The measured operating characteristics are in the certificate
(`benchmarks.qualification.f4_certify`); until a current certificate passes,
`adjudication` is descriptive and decisions are demoted to warnings.
"""

from collections.abc import Hashable
from collections.abc import Sequence
import dataclasses
import sys

import numpy
import pandas

from scmle.assurance.admissibility.leakage.assessment import Assessment
from scmle.assurance.admissibility.leakage.assessment import Assessments
from scmle.assurance.admissibility.leakage.conditional import Conditional
from scmle.assurance.admissibility.leakage.conditional import scramble
from scmle.assurance.admissibility.leakage.conditional import strata_labels
from scmle.assurance.admissibility.leakage.fitting import auc
from scmle.assurance.admissibility.leakage.fitting import carve
from scmle.assurance.admissibility.leakage.fitting import crossfit
from scmle.assurance.admissibility.leakage.fitting import design
from scmle.assurance.admissibility.leakage.fitting import lower_bound
from scmle.assurance.admissibility.leakage.fitting import measure
from scmle.assurance.admissibility.leakage.fitting import out_of_fold
from scmle.assurance.admissibility.leakage.fitting import permute_within
from scmle.assurance.admissibility.leakage.fitting import predictiveness
from scmle.assurance.admissibility.leakage.fitting import prepare
from scmle.assurance.admissibility.leakage.fitting import rescore
from scmle.assurance.admissibility.leakage.fitting import significance
from scmle.assurance.admissibility.leakage.fitting import sufficient
from scmle.assurance.admissibility.leakage.fitting import vectorize
from scmle.assurance.admissibility.leakage.missingness import bucket
from scmle.assurance.admissibility.leakage.missingness import cramers_v
from scmle.assurance.admissibility.leakage.missingness import Missingness
from scmle.assurance.admissibility.leakage.missingness import missingness_test
from scmle.assurance.admissibility.leakage.settings import BINARY_CLASSES
from scmle.assurance.admissibility.leakage.settings import CERTIFIED
from scmle.assurance.admissibility.leakage.settings import CONDITIONAL
from scmle.assurance.admissibility.leakage.settings import Config
from scmle.assurance.admissibility.leakage.settings import DETECTOR_ID
from scmle.assurance.admissibility.leakage.settings import EXPECTED_FISHER
from scmle.assurance.admissibility.leakage.settings import LEAK_DEFECTS
from scmle.assurance.admissibility.leakage.settings import MAX_LEVELS
from scmle.assurance.admissibility.leakage.settings import MIN_CLASSES
from scmle.assurance.admissibility.leakage.settings import MIN_STRATUM_ROWS
from scmle.assurance.admissibility.leakage.settings import MISSING
from scmle.assurance.admissibility.leakage.settings import SKILL
from scmle.assurance.admissibility.leakage.settings import Splits
from scmle.assurance.admissibility.leakage.settings import TARGET_BINS
from scmle.assurance.admissibility.leakage.settings import TASKS
from scmle.assurance.admissibility.leakage.skill import decide
from scmle.assurance.admissibility.leakage.skill import Skill
from scmle.assurance.admissibility.leakage.skill import summarize
from scmle.assurance.detectors import certificates
from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.runtime import randomness

__all__ = [
    "BINARY_CLASSES",
    "CERTIFIED",
    "CONDITIONAL",
    "DETECTOR_ID",
    "EXPECTED_FISHER",
    "LEAK_DEFECTS",
    "MAX_LEVELS",
    "MIN_CLASSES",
    "MIN_STRATUM_ROWS",
    "MISSING",
    "SKILL",
    "TARGET_BINS",
    "TASKS",
    "Assessment",
    "Assessments",
    "Conditional",
    "Config",
    "Missingness",
    "Skill",
    "Splits",
    "auc",
    "bucket",
    "carve",
    "cramers_v",
    "crossfit",
    "decide",
    "design",
    "lower_bound",
    "measure",
    "missingness_test",
    "out_of_fold",
    "permute_within",
    "predictiveness",
    "prepare",
    "rescore",
    "scramble",
    "significance",
    "strata_labels",
    "sufficient",
    "summarize",
    "vectorize",
]


def probe(
    feature: pandas.Series,
    target: pandas.Series,
    task: str,
    cfg: Config,
    groups: pandas.Series | None = None,
    block_size: int | None = None,
    gate: certificates.Authority = certificates.Authority.DESCRIPTIVE,
    set_id: qualification.Subset = qualification.Subset.EXPLORATION,
) -> Skill:
    """Tests whether one feature predicts the target beyond chance.

    Args:
        feature: The feature, aligned to `target`.
        target: The outcome.
        task: `classification` or `regression`.
        cfg: Settings.
        groups: Group label per row; folds and permutations respect it.
        block_size: Consecutive rows per time block (permute within blocks
            and use contiguous folds); exclusive with `groups`.
        gate: Whether the detector has a current certificate.
        set_id: Set the rows belong to (the exploration set for anything
            touching the target).

    Returns:
        The test, with a ledger `Check`. The feature is a suspect only when
        there are enough rows (see `sufficient`); with too few it is
        `low_power` and never a suspect. The check's interval is None when
        the lower bound is -inf.

    Raises:
        ValueError: If both `groups` and `block_size` are given, `feature`
            and `target` differ in length, or the target is invalid (see
            `vectorize`).
    """
    if len(feature) != len(target):
        raise errors.InputError("feature and target must have the same length")
    name, y, group_values, x = prepare(
        feature, target, task, groups, block_size
    )
    path = f"{SKILL}.{name}"
    skill, pred, null = measure(
        x,
        y,
        task,
        cfg,
        group_values,
        block_size,
        path,
        lambda rng: (x, y[permute_within(y, rng, group_values, block_size)]),
    )
    p_value = significance(skill, null)
    rng = randomness.component_rng(cfg.run_seed, path, 2)
    bound = lower_bound(pred, y, task, cfg, group_values, rng)
    powered = sufficient(y, task, cfg)
    suspect = powered and p_value <= cfg.alpha and bound > cfg.skill_ceiling
    decision = decide(suspect, not powered, gate)
    result = (skill, null, bound, p_value)
    check = summarize(path, cfg, result, len(y), decision, set_id)
    return Skill(
        name,
        skill,
        float(numpy.mean(null)),
        bound,
        p_value,
        len(y),
        not powered,
        suspect,
        check,
    )


def condition(
    feature: pandas.Series,
    others: pandas.DataFrame,
    target: pandas.Series,
    task: str,
    cfg: Config,
    groups: pandas.Series | None = None,
    block_size: int | None = None,
    set_id: qualification.Subset = qualification.Subset.EXPLORATION,
) -> Conditional:
    """Tests whether the feature keeps predictive value given the others.

    Args:
        feature: The feature under test.
        others: The other admitted features (the conditioning set).
        target: The outcome.
        task: `classification` or `regression`.
        cfg: Settings.
        groups: Group label per row, for the cross-fitting folds.
        block_size: Consecutive rows per time block, for the folds.
        set_id: Set the rows belong to.

    Returns:
        The test. It never gates: it only supports an escalation.

    Raises:
        ValueError: If both `groups` and `block_size` are given, or the
            target is invalid (see `vectorize`).
    """
    name, y, group_values, x = prepare(
        feature, target, task, groups, block_size
    )
    path = f"{CONDITIONAL}.{name}"
    labels = strata_labels(feature, others, cfg)
    members = [numpy.flatnonzero(labels == v) for v in numpy.unique(labels)]
    skill, ignored, null = measure(
        x,
        y,
        task,
        cfg,
        group_values,
        block_size,
        path,
        lambda rng: (x[scramble(len(y), members, rng)], y),
    )
    p_value = significance(skill, null)
    keeps = p_value <= cfg.alpha
    check = qualification.Check(
        check_id=path,
        null_hypothesis=(
            "given the other features, the feature has no predictive skill "
            "beyond permutation within strata"
        ),
        statistic=skill,
        p_value=p_value,
        interval=None,
        effect=skill - float(numpy.mean(null)),
        n=len(y),
        decision=(
            qualification.Decision.WARN
            if keeps
            else qualification.Decision.PASS
        ),
        set_id=set_id,
        seed_path=randomness.seed_path(cfg.run_seed, path, 1),
        tier=qualification.Tier.UNVERIFIED,
    )
    return Conditional(
        name,
        skill,
        float(numpy.mean(null)),
        p_value,
        len(members),
        keeps,
        check,
    )


def assess(
    feature: pandas.Series,
    others: pandas.DataFrame,
    target: pandas.Series,
    task: str,
    cfg: Config,
    groups: pandas.Series | None = None,
    block_size: int | None = None,
    gate: certificates.Authority = certificates.Authority.DESCRIPTIVE,
    set_id: qualification.Subset = qualification.Subset.EXPLORATION,
) -> Assessment:
    """Runs the skill, missingness and (for suspects) conditional tests.

    Args:
        feature: The feature.
        others: The other admitted features.
        target: The outcome.
        task: `classification` or `regression`.
        cfg: Settings.
        groups: Group label per row.
        block_size: Consecutive rows per time block.
        gate: Whether the detector has a current certificate.
        set_id: Set the rows belong to.

    Returns:
        The assessment. A feature is a suspect when its skill test or its
        missingness test says so, and is escalated when the conditional
        test, run for suspects only, finds it keeps its signal.
    """
    skill = probe(feature, target, task, cfg, groups, block_size, gate, set_id)
    gaps = missingness_test(feature, target, task, cfg, gate, set_id)
    suspect = skill.suspect or (gaps is not None and gaps.suspect)
    conditional = (
        condition(
            feature, others, target, task, cfg, groups, block_size, set_id
        )
        if suspect
        else None
    )
    escalated = conditional is not None and conditional.keeps_signal and suspect
    return Assessment(
        str(feature.name), skill, conditional, gaps, suspect, escalated
    )


def scan(
    frame: pandas.DataFrame,
    target: str,
    task: str,
    cfg: Config,
    features: Sequence[Hashable] | None = None,
    groups: pandas.Series | None = None,
    block_size: int | None = None,
    gate: certificates.Authority = certificates.Authority.DESCRIPTIVE,
    set_id: qualification.Subset = qualification.Subset.EXPLORATION,
) -> Assessments:
    """Tests every feature of the exploration set for leakage suspicion.

    Args:
        frame: Exploration-set data including the target.
        target: Name of the target column.
        task: `classification` or `regression`.
        cfg: Settings.
        features: Columns to test; defaults to all but the target.
        groups: Group label per row.
        block_size: Consecutive rows per time block.
        gate: Whether the detector has a current certificate.
        set_id: Set the rows belong to.

    Returns:
        The report, with one assessment per tested feature in the order
        given, recording `gate` as passed (see `adjudication` for deriving
        it from the certificates).

    Raises:
        ValueError: If `features` names a column that is missing or is the
            target, or a test rejects its inputs.
    """
    names = [c for c in frame.columns if c != target]
    chosen = names if features is None else list(features)
    unknown = [c for c in chosen if c not in names]
    if unknown:
        raise errors.InputError(f"not feature columns: {unknown}")
    outcomes = frame[target]
    assessments = tuple(
        assess(
            frame[name],
            frame[[c for c in chosen if c != name]],
            outcomes,
            task,
            cfg,
            groups,
            block_size,
            gate,
            set_id,
        )
        for name in chosen
    )
    return Assessments(assessments, gate)


def adjudication(
    store: certificates.Store,
    cfg: Config,
    task: str = "classification",
    n_classes: int = 2,
) -> certificates.Authority:
    """Decides whether the skill test may quarantine, from the certificates.

    Gating needs a current certificate whose every required leak type
    passed, and settings equal to the certified ones (`CERTIFIED`, any run
    seed). Only a binary classification target was certified; any other
    target is descriptive.

    Args:
        store: Certificates on file.
        cfg: The settings the tests will run with.
        task: `classification` or `regression`.
        n_classes: Classes of a classification target.

    Returns:
        `GATING` or `DESCRIPTIVE`.
    """
    if task != "classification" or n_classes != BINARY_CLASSES:
        return certificates.Authority.DESCRIPTIVE
    if dataclasses.replace(cfg, run_seed=CERTIFIED.run_seed) != CERTIFIED:
        return certificates.Authority.DESCRIPTIVE
    digest = code_digest()
    for defect in LEAK_DEFECTS:
        if store.status(DETECTOR_ID, digest, defect) is not (
            certificates.Authority.GATING
        ):
            return certificates.Authority.DESCRIPTIVE
    return certificates.Authority.GATING


def code_digest() -> str:
    """Hashes this package's source, the key of its certificate.

    Returns:
        The code hash of the package and each of its modules, in name
        order, as computed by `certificates.code_hash`.
    """
    return certificates.code_hash(
        *(
            sys.modules[name]
            for name in sorted(sys.modules)
            if name == __name__ or name.startswith(f"{__name__}.")
        )
    )
