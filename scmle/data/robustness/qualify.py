"""Runs the stability and shift checks on `X` for a charter run.

This is the wiring between the robustness package and a charter run. It reads
the exploration view and, when the data spec names one, a deployment sample;
it never touches `C`, and it does not open `F` (the final-test set is opened
by the final evaluation only, so the forward check is reported as not run
here; `forward.forward_check` is the function that opens `F`, once).

Certificates live beside the qualification harness
(`benchmarks/qualification/f6_certificates.json`), which the installed
package does not ship. When the file is missing, unreadable or stale for the
current code, the checks are descriptive: a `fragile` finding is a warning,
not a quarantine.

The stability verdicts are reported. They remove findings from the
confirmation family only through the pre-registered second stage
(`prefilter.StabilityStage`), which runs this module between exploration and
the opening of `C` when the charter's rules ask for it.
"""

from collections.abc import Sequence
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from scmle.data.admissibility import qualify as admissibility
from scmle.data.charter import contracts as charter_contracts
from scmle.data.detectors import certificates
from scmle.data.robustness import coverage
from scmle.data.robustness import forward
from scmle.data.robustness import multiverse
from scmle.data.robustness import shift
from scmle.data.robustness import stability

F6_STORE = "f6_certificates.json"
BUDGET_SECONDS = 60.0
MAX_FINDINGS = 50
MAX_BOOTSTRAP = 100
MAX_CURVE_PERMUTATIONS = 200
MIN_PERMUTATIONS = 99
MAX_PERMUTATIONS = 499
MAX_REFERENCE_ROWS = 20000


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
        stability_gate: Whether a current certificate lets stability act.
        c2st_gate: Gate status of the classifier test.
        mmd_gate: Gate status of the kernel test.
        store_note: Why no certificate store was used, or an empty string.
        rule: The survival rule the verdicts used.
        rule_note: Why the rule is descriptive, or an empty string.
    """

    grid: multiverse.Grid | None
    applicable: tuple[charter_contracts.Choice, ...]
    findings: tuple[stability.Stability, ...]
    stability_note: str
    notes: tuple[str, ...]
    shift: shift.ShiftResult
    coverage: tuple[coverage.LevelCoverage, ...]
    forward: forward.ForwardResult
    stability_gate: certificates.GateStatus
    c2st_gate: certificates.GateStatus
    mmd_gate: certificates.GateStatus
    store_note: str
    rule: stability.Rule = stability.DEFAULT_RULE
    rule_note: str = ""

    @property
    def fragile(self) -> tuple[str, ...]:
        """The names of the findings whose verdict is `fragile`."""
        return tuple(
            f.name for f in self.findings if f.status == stability.FRAGILE
        )


def run(
    features: pd.DataFrame,
    codes: pd.Series,
    *,
    survivors: Sequence[str],
    multiclass: bool,
    choices: Sequence[charter_contracts.Choice],
    seed: int,
    permutations: int,
    strata: np.ndarray | None = None,
    reference: pd.DataFrame | None = None,
    candidates: Sequence[str] = (),
    sensitive: Sequence[str] = (),
    temporal: bool = False,
    directory: Path | None = None,
    budget_seconds: float = BUDGET_SECONDS,
    rule: stability.Rule = stability.DEFAULT_RULE,
) -> Robustness:
    """Assesses survivor stability and the shift to a deployment sample.

    Args:
        features: Exploration features.
        codes: Encoded exploration target.
        survivors: Columns of the findings that survived exploration.
        multiclass: Whether the target has more than two classes; the
            stability estimator is a rank correlation, so such findings are
            not evaluated.
        choices: The charter's multiverse.
        seed: Charter seed.
        permutations: Charter permutation count. The specification-curve
            test uses at most `MAX_CURVE_PERMUTATIONS` of them; the shift
            tests use it clamped to `MIN_PERMUTATIONS`..`MAX_PERMUTATIONS`.
        strata: Group or block labels for resampling and permuting.
        reference: Deployment sample, or None.
        candidates: Feature columns compared with the reference.
        sensitive: Columns whose subgroup coverage is reported.
        temporal: Whether the data have a time column; only the wording of
            the skipped forward check depends on it.
        directory: Folder of the certificate store, for tests.
        budget_seconds: Time allowed for the whole multiverse, divided
            equally among the assessed findings.
        rule: Survival thresholds. The certificate was measured for
            `stability.DEFAULT_RULE` only, so any other rule is descriptive:
            its verdicts are reported and never gate.

    Returns:
        The stability verdicts (at most `MAX_FINDINGS` findings, in the
        order of `survivors`), the shift comparison, the subgroup coverage
        rows, a skipped forward check and the certificate gates. Stability
        is not run, and `stability_note` says why, for a multiclass target,
        no survivor or too few specifications.
    """
    loaded = admissibility.load_store(F6_STORE, directory)
    store = loaded.store
    stability_gate = stability.gate_status(store)
    rule_note = ""
    if rule != stability.DEFAULT_RULE:
        stability_gate = certificates.GateStatus.DESCRIPTIVE
        rule_note = (
            "the chartered stability thresholds differ from the certified "
            "rule: the verdicts are descriptive"
        )
    c2st_gate, mmd_gate = (
        certificates.GateStatus.DESCRIPTIVE
        if store is None
        else store.status(detector, shift.code_digest(), shift.DEFECT_TYPE)
        for detector in (shift.C2ST_ID, shift.MMD_ID)
    )
    applicable, notes = stability.applicable_choices(choices)
    grid = multiverse.enumerate_grid(applicable, seed)
    findings: list[stability.Stability] = []
    stability_note = ""
    if multiclass:
        stability_note = (
            "the target has more than two classes: findings use Kruskal-Wallis "
            "and the rank-correlation stability estimator does not apply"
        )
    elif not survivors:
        stability_note = "no finding survived exploration"
    elif len(grid.specs) < stability.MIN_SPECS:
        stability_note = (
            f"the declared choices give {len(grid.specs)} specification(s) "
            f"this estimator can vary; at least {stability.MIN_SPECS} are "
            "needed, so no finding can be called stable"
        )
    chosen = list(survivors)[:MAX_FINDINGS]
    if len(survivors) > MAX_FINDINGS:
        notes = (
            *notes,
            f"only the first {MAX_FINDINGS} of {len(survivors)} findings "
            "were assessed",
        )
    if not stability_note:
        share = budget_seconds / max(len(chosen), 1)
        curve = min(permutations, MAX_CURVE_PERMUTATIONS)
        for name in chosen:
            findings.append(
                stability.assess_finding(
                    name,
                    features[name].to_numpy(dtype=float),
                    codes.to_numpy(dtype=float),
                    grid,
                    seed,
                    bootstrap=MAX_BOOTSTRAP,
                    strata=strata,
                    curve_permutations=curve,
                    budget_seconds=share,
                    gate=stability_gate,
                    rule=rule,
                )
            )
    sample = features[[c for c in candidates if c in features.columns]]
    reference_frame = (
        None if reference is None else reference.head(MAX_REFERENCE_ROWS)
    )
    result = shift.assess(
        sample.head(MAX_REFERENCE_ROWS),
        reference_frame,
        seed,
        permutations=min(max(permutations, MIN_PERMUTATIONS), MAX_PERMUTATIONS),
        c2st_gate=c2st_gate,
        mmd_gate=mmd_gate,
    )
    rows: list[coverage.LevelCoverage] = []
    for name in sensitive:
        if name not in features.columns:
            continue
        deployment = (
            coverage.shares(reference[name])
            if reference is not None and name in reference.columns
            else None
        )
        rows.extend(coverage.subgroup_coverage(features[name], deployment))
    skipped_forward = (
        forward.skipped(
            "F is the final test set, which only the final evaluation opens; "
            "the forward check was not run in this stage"
        )
        if temporal
        else forward.skipped("the data are not temporal: no forward check")
    )
    return Robustness(
        grid=grid,
        applicable=applicable,
        findings=tuple(findings),
        stability_note=stability_note,
        notes=tuple(notes),
        shift=result,
        coverage=tuple(rows),
        forward=skipped_forward,
        stability_gate=stability_gate,
        c2st_gate=c2st_gate,
        mmd_gate=mmd_gate,
        store_note=loaded.note,
        rule=rule,
        rule_note=rule_note,
    )
