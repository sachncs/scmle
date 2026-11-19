"""Runs the stability and shift checks on `X` for a charter run.

This is the wiring between the robustness package and a charter run. It reads
the exploration view and, when the data spec names one, a deployment sample;
it never touches `C`, and it does not open `F` (the final-test set is opened
by the final evaluation only, so the forward check is reported as not run
here; `forward.check` is the function that opens `F`, once).

Certificates are shipped with the package
(`scmle/certificates/f6_certificates.json`). When the file is missing,
unreadable or stale for the current code, the checks are descriptive: a
`fragile` finding is a warning, not a quarantine.

The stability verdicts are reported. They remove findings from the
confirmation family only through the pre-registered second stage
(`prefilter.Prefilter`), which runs this module between exploration and
the opening of `C` when the charter's rules ask for it.
"""

from collections.abc import Sequence
import dataclasses
from pathlib import Path

import numpy
import pandas

from scmle.assurance.admissibility import qualify
from scmle.assurance.charter import contracts
from scmle.assurance.detectors import certificates
from scmle.assurance.robustness import coverage
from scmle.assurance.robustness import forward
from scmle.assurance.robustness import multiverse
from scmle.assurance.robustness import periods
from scmle.assurance.robustness import shift
from scmle.assurance.robustness import stability

F6_STORE = "f6_certificates.json"
BUDGET_SECONDS = 60.0
MAX_FINDINGS = 50
MAX_BOOTSTRAP = 100
MAX_CURVE_PERMUTATIONS = 200
MIN_PERMUTATIONS = 99
MAX_PERMUTATIONS = 499
MAX_REFERENCE_ROWS = 20000
TEMPORAL_SKIP = (
    "F is the final test set, which only the final evaluation opens; "
    "the forward check was not run in this stage"
)
STATIC_SKIP = "the data are not temporal: no forward check"


@dataclasses.dataclass(frozen=True)
class Robustness:
    """What the robustness checks found on `X`.

    Attributes:
        grid: The specification grid, or None when stability was not run.
        applicable: Declared choices the estimator could vary.
        findings: One stability verdict per surviving finding.
        stability_note: Why stability was not run, or an empty string.
        notes: Choices or options that were declared and not run.
        shift: The comparison with the deployment sample.
        coverage: Subgroup coverage rows.
        forward: The forward check (skipped in a run).
        gate: Whether a current certificate lets stability act.
        classifier: Gate status of the classifier test.
        kernel: Gate status of the kernel test.
        store_note: Why no certificate store was used, or an empty string.
        rule: The survival rule the verdicts used.
        note: Why the rule is descriptive, or an empty string.
        drift: Each time period of `X` against the first.
    """

    grid: multiverse.Grid | None
    applicable: tuple[contracts.Choice, ...]
    findings: tuple[stability.Stability, ...]
    stability_note: str
    notes: tuple[str, ...]
    shift: shift.Shift
    coverage: tuple[coverage.Cell, ...]
    forward: forward.Forward
    gate: certificates.Authority
    classifier: certificates.Authority
    kernel: certificates.Authority
    store_note: str
    rule: stability.Thresholds = stability.DEFAULT_RULE
    note: str = ""
    drift: periods.Periods = periods.UNTIMED

    @property
    def fragile(self) -> tuple[str, ...]:
        """The names of the findings whose verdict is `fragile`."""
        return tuple(
            f.name for f in self.findings if f.status == stability.FRAGILE
        )


def authorize(
    store: certificates.Store | None, rule: stability.Thresholds
) -> tuple[certificates.Authority, str]:
    """Decides whether the stability verdicts may gate.

    Args:
        store: The certificate store, or None.
        rule: Survival thresholds. The certificate was measured for
            `stability.DEFAULT_RULE` only, so any other rule is descriptive:
            its verdicts are reported and never gate.

    Returns:
        The stability gate and why the rule is descriptive (or an empty
        string).
    """
    if rule == stability.DEFAULT_RULE:
        return stability.adjudication(store), ""
    return certificates.Authority.DESCRIPTIVE, (
        "the chartered stability thresholds differ from the certified "
        "rule: the verdicts are descriptive"
    )


def triage(
    multiclass: bool,
    survivors: Sequence[str],
    grid: multiverse.Grid,
    notes: tuple[str, ...],
) -> tuple[str, list[str], tuple[str, ...]]:
    """Chooses the findings whose stability is assessed.

    Args:
        multiclass: Whether the target has more than two classes; the
            stability estimator is a rank correlation, so such findings are
            not evaluated.
        survivors: Columns of the findings that survived exploration.
        grid: The specification grid.
        notes: Notes so far.

    Returns:
        Why stability is not run (an empty string when it is), the findings
        to assess (the first `MAX_FINDINGS` survivors; none when stability
        is not run) and the notes, with one more when survivors were cut.
    """
    reason = ""
    if multiclass:
        reason = (
            "the target has more than two classes: findings use Kruskal-Wallis "
            "and the rank-correlation stability estimator does not apply"
        )
    elif not survivors:
        reason = "no finding survived exploration"
    elif len(grid.specs) < stability.MIN_SPECS:
        reason = (
            f"the declared choices give {len(grid.specs)} specification(s) "
            f"this estimator can vary; at least {stability.MIN_SPECS} are "
            "needed, so no finding can be called stable"
        )
    chosen = list(survivors)[:MAX_FINDINGS]
    if len(survivors) > MAX_FINDINGS:
        notes = (
            *notes,
            (
                f"only the first {MAX_FINDINGS} of {len(survivors)} findings "
                "were assessed"
            ),
        )
    return reason, [] if reason else chosen, notes


def examine(
    features: pandas.DataFrame,
    codes: pandas.Series,
    chosen: Sequence[str],
    grid: multiverse.Grid,
    seed: int,
    strata: numpy.ndarray | None,
    permutations: int,
    budget_seconds: float,
    gate: certificates.Authority,
    rule: stability.Thresholds,
) -> tuple[stability.Stability, ...]:
    """Assesses the stability of each chosen finding.

    Args:
        features: Exploration features.
        codes: Encoded exploration target.
        chosen: Columns of the findings to assess.
        grid: The specification grid.
        seed: Charter seed.
        strata: Group or block labels for resampling and permuting.
        permutations: Charter permutation count; the specification-curve
            test uses at most `MAX_CURVE_PERMUTATIONS` of them.
        budget_seconds: Time allowed for the whole multiverse, divided
            equally among the chosen findings.
        gate: Whether a current certificate lets stability act.
        rule: Survival thresholds.

    Returns:
        One verdict per chosen finding, in order.
    """
    share = budget_seconds / max(len(chosen), 1)
    curve = min(permutations, MAX_CURVE_PERMUTATIONS)
    return tuple(
        stability.assess(
            name,
            features[name].to_numpy(dtype=float),
            codes.to_numpy(dtype=float),
            grid,
            seed,
            bootstrap=MAX_BOOTSTRAP,
            strata=strata,
            curve_permutations=curve,
            budget_seconds=share,
            gate=gate,
            rule=rule,
        )
        for name in chosen
    )


def compare(
    features: pandas.DataFrame,
    candidates: Sequence[str],
    reference: pandas.DataFrame | None,
    seed: int,
    permutations: int,
    store: certificates.Store | None,
) -> tuple[shift.Shift, certificates.Authority, certificates.Authority]:
    """Compares the exploration rows with the deployment sample.

    Args:
        features: Exploration features.
        candidates: Feature columns compared with the reference; those not
            in `features` are skipped.
        reference: Deployment sample, or None.
        seed: Charter seed.
        permutations: Charter permutation count, clamped to
            `MIN_PERMUTATIONS`..`MAX_PERMUTATIONS`.
        store: The certificate store, or None.

    Returns:
        The comparison on at most `MAX_REFERENCE_ROWS` rows of each side,
        and the gates of the classifier and the kernel test.
    """
    c2st_gate, mmd_gate = (
        certificates.Authority.DESCRIPTIVE
        if store is None
        else store.status(detector, shift.fingerprint(), shift.DEFECT_TYPE)
        for detector in (shift.C2ST_ID, shift.MMD_ID)
    )
    sample = features[[c for c in candidates if c in features.columns]]
    found = shift.assess(
        sample.head(MAX_REFERENCE_ROWS),
        None if reference is None else reference.head(MAX_REFERENCE_ROWS),
        seed,
        permutations=min(max(permutations, MIN_PERMUTATIONS), MAX_PERMUTATIONS),
        c2st_gate=c2st_gate,
        mmd_gate=mmd_gate,
    )
    return found, c2st_gate, mmd_gate


def represent(
    features: pandas.DataFrame,
    reference: pandas.DataFrame | None,
    sensitive: Sequence[str],
) -> tuple[coverage.Cell, ...]:
    """Reports the subgroup coverage of the sensitive columns.

    Args:
        features: Exploration features.
        reference: Deployment sample, or None.
        sensitive: Columns whose subgroup coverage is reported; those not
            in `features` are skipped.

    Returns:
        The coverage rows, against the deployment shares when the sample
        holds the column.
    """
    rows: list[coverage.Cell] = []
    for name in sensitive:
        if name not in features.columns:
            continue
        deployment = (
            coverage.shares(reference[name])
            if reference is not None and name in reference.columns
            else None
        )
        rows.extend(coverage.represent(features[name], deployment))
    return tuple(rows)


def run(
    features: pandas.DataFrame,
    codes: pandas.Series,
    *,
    survivors: Sequence[str],
    multiclass: bool,
    choices: Sequence[contracts.Choice],
    seed: int,
    permutations: int,
    strata: numpy.ndarray | None = None,
    reference: pandas.DataFrame | None = None,
    candidates: Sequence[str] = (),
    sensitive: Sequence[str] = (),
    temporal: bool = False,
    directory: Path | None = None,
    budget_seconds: float = BUDGET_SECONDS,
    rule: stability.Thresholds = stability.DEFAULT_RULE,
) -> Robustness:
    """Assesses survivor stability and the shift to a deployment sample.

    Args:
        features: Exploration features.
        codes: Encoded exploration target.
        survivors: Columns of the findings that survived exploration.
        multiclass: Whether the target has more than two classes.
        choices: The charter's multiverse.
        seed: Charter seed.
        permutations: Charter permutation count, capped per test.
        strata: Group or block labels for resampling and permuting.
        reference: Deployment sample, or None.
        candidates: Feature columns compared with the reference.
        sensitive: Columns whose subgroup coverage is reported.
        temporal: Whether the data have a time column; only the wording of
            the skipped forward check depends on it.
        directory: Folder of the certificate store, for tests.
        budget_seconds: Time allowed for the whole multiverse.
        rule: Survival thresholds; any but the certified default is
            descriptive (see `authorize`).

    Returns:
        The stability verdicts of the findings `triage` chooses (none, and
        `stability_note` says why, when it chooses none), the shift test,
        the subgroup coverage, a skipped forward check and the gates.
    """
    loaded = qualify.load(F6_STORE, directory)
    gate, rule_note = authorize(loaded.store, rule)
    applicable, notes = stability.actionable(choices)
    grid = multiverse.universe(applicable, seed)
    note, chosen, notes = triage(multiclass, survivors, grid, notes)
    findings = examine(
        features,
        codes,
        chosen,
        grid,
        seed,
        strata,
        permutations,
        budget_seconds,
        gate,
        rule,
    )
    result, c2st_gate, mmd_gate = compare(
        features, candidates, reference, seed, permutations, loaded.store
    )
    return Robustness(
        grid=grid,
        applicable=applicable,
        findings=findings,
        stability_note=note,
        notes=tuple(notes),
        shift=result,
        coverage=represent(features, reference, sensitive),
        forward=forward.skipped(TEMPORAL_SKIP if temporal else STATIC_SKIP),
        gate=gate,
        classifier=c2st_gate,
        kernel=mmd_gate,
        store_note=loaded.note,
        rule=rule,
        note=rule_note,
    )
