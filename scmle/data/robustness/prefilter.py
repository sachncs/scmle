"""The stability filter as the pre-registered second stage of a charter run.

`governed.run_governed` calls `StabilityStage` after exploration and the
negative control and before `C` is opened. The stage re-runs the survivors
over the declared specification grid on the exploration set and, when the
charter's `Procedure.stability_filter` is true and the verdicts are allowed to
gate, withdraws the `fragile` survivors from the family tested on `C`.

Error control (see `confirm.STAGE_TWO_STATEMENT` and the "Second stage"
section of `scmle.data.ledger.confirm`). The stage reads the exploration
features and target and the charter, and nothing else; it is handed no `C`
and no `F`. Its verdict is therefore a function of `X`, the charter and the
seed. Under the assumption already made for confirmation, that the rows of `C`
are independent of those of `X`, withdrawing hypotheses on that basis leaves
the confirmation rule's guarantee over the remaining family at
``alpha_confirm``, spends no alpha, and cannot create a false discovery. It
can lose power: a true effect that is `fragile` is not tested. A `fragile`
verdict is a statement about the declared grid, not about truth.

When the verdicts may gate. The stability rule's operating characteristics
were measured for `stability.DEFAULT_RULE` only
(`benchmarks/qualification/f6_certificates.json`). The filter acts only if a
current passing certificate exists and the chartered thresholds equal that
rule; otherwise the request is recorded as not applied, with the reason, and
the verdicts stay warnings. Findings the estimator cannot assess
(multiclass targets, fewer than `MIN_SPECS` specifications, columns beyond
`MAX_FINDINGS`) are never withdrawn.

Diagnostics. Whether or not the filter is requested, the stage returns the
stability checks, the two overall shift checks and the per-column shift tests
as wealth-free ledger diagnostics. They describe `X` against a reference
sample and are uncertified for the per-column tests (descriptive).
"""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses

from scmle.contracts import qualification
from scmle.core import randomness
from scmle.data.detectors import certificates
from scmle.data.ledger import confirm
from scmle.data.ledger import ledger
from scmle.data.robustness import qualify
from scmle.data.robustness import shift
from scmle.data.robustness import stability

FILTER_ID = "f6-stability-filter"
FEATURE_SHIFT_ID = "f6-shift-feature"
MAX_FEATURE_ROWS = 200


def feature_checks(
    result: shift.Shift, rows: int, seed: int
) -> tuple[qualification.Check, ...]:
    """Per-column shift tests as checks (descriptive, never gating).

    Args:
        result: The shift result.
        rows: Exploration rows the tests compared.
        seed: Run seed, for the check's seed path.

    Returns:
        One check per column, at most `MAX_FEATURE_ROWS`, rejected ones first.
    """
    ordered = sorted(
        enumerate(result.features), key=lambda item: not item[1].rejected
    )
    checks = []
    for index, row in ordered[:MAX_FEATURE_ROWS]:
        checks.append(
            qualification.Check(
                check_id=f"{FEATURE_SHIFT_ID}:{row.column}",
                null_hypothesis=f"'{row.column}' has the same distribution "
                "in the exploration rows and the reference sample",
                statistic=row.statistic,
                p_value=row.p_value,
                interval=None,
                effect=row.effect,
                n=rows,
                decision=(
                    qualification.Decision.WARN
                    if row.rejected
                    else qualification.Decision.PASS
                ),
                set_id=qualification.SetId.EXPLORATION,
                seed_path=randomness.seed_path(seed, FEATURE_SHIFT_ID, index),
                tier=qualification.Tier.UNVERIFIED,
            )
        )
    return tuple(checks)


def diagnostic_rows(
    found: qualify.Robustness, rows: int, seed: int
) -> tuple[tuple[str, qualification.Check], ...]:
    """Collects stability and shift results as ledger diagnostics.

    Only checks that read the exploration set are returned; the ledger
    refuses any other.

    Args:
        found: The robustness results.
        rows: Exploration rows the tests compared.
        seed: Run seed, for the per-column checks' seed paths.

    Returns:
        (source, check) pairs: the per-finding stability checks, the
        classifier and kernel shift checks that exist, then the per-column
        shift checks, restricted to those whose set is the exploration set.
    """
    out: list[tuple[str, qualification.Check]] = []
    for finding in found.findings:
        if finding.check is not None:
            out.append((stability.DETECTOR_ID, finding.check))
    for name, check in (
        (shift.C2ST_ID, found.shift.c2st),
        (shift.MMD_ID, found.shift.mmd),
    ):
        if check is not None:
            out.append((name, check))
    out.extend(
        (FEATURE_SHIFT_ID, check)
        for check in feature_checks(found.shift, rows, seed)
    )
    return tuple(
        (source, check)
        for source, check in out
        if check.set_id is qualification.SetId.EXPLORATION
    )


@dataclasses.dataclass
class StabilityStage:
    """The second stage handed to `run_governed`.

    Attributes:
        compute: Runs the robustness stage for the given survivor columns on
            the exploration set (a closure over `X`; it must not read `C`).
        requested: The charter's `stability_filter`.
        rows: Number of exploration rows, for the diagnostics.
        seed: Charter seed.
        found: The robustness results, set when the stage has run.
        applied: Whether the filter was requested and allowed to gate.
        note: What the stage did, for the report.
    """

    compute: Callable[[tuple[str, ...]], qualify.Robustness]
    requested: bool
    rows: int
    seed: int
    found: qualify.Robustness | None = None
    applied: bool = False
    note: str = ""

    def __call__(
        self, survivors: Sequence[ledger.Hypothesis]
    ) -> confirm.Withdrawal:
        """Assesses the survivors on `X` and withdraws the fragile ones.

        Sets `found`, `applied` and `note` as a side effect.

        Args:
            survivors: The frozen survivors; those without a feature column
                are not assessed.

        Returns:
            The withdrawal. Its reasons name each fragile survivor (by
            hypothesis id) only when the filter is applied, and are empty
            otherwise. It also carries the rule record and the diagnostics,
            and is accounted only when the filter was requested.
        """
        columns = tuple(h.spec.column for h in survivors if h.spec.column)
        found = self.compute(columns)
        self.found = found
        gating = found.stability_gate is certificates.GateStatus.GATING
        self.applied = self.requested and gating
        fragile = set(found.fragile)
        reasons: dict[str, str] = {}
        if self.applied:
            by_name = {f.name: f for f in found.findings}
            for hypothesis in survivors:
                name = hypothesis.spec.column
                if name in fragile and name is not None:
                    reasons[hypothesis.hypothesis_id] = (
                        f"fragile across the declared grid: "
                        f"{by_name[name].reason}"
                    )
        if not self.requested:
            self.note = (
                "not applied: the charter does not ask for fragile survivors "
                "to be withdrawn"
            )
        elif not gating:
            self.note = (
                "requested but not applied: the stability rule has no "
                "current passing certificate for these thresholds, so "
                "fragile verdicts are warnings"
                + (
                    f" ({found.rule_note or found.store_note})"
                    if (found.rule_note or found.store_note)
                    else ""
                )
            )
        else:
            self.note = (
                f"applied before C was opened: {len(reasons)} of "
                f"{len(survivors)} survivors withdrawn as fragile; the "
                "verdicts read the exploration set only"
            )
        record = {
            "name": FILTER_ID,
            "requested": self.requested,
            "applied": self.applied,
            "gate": found.stability_gate.value,
            "thresholds": found.rule.to_dict(),
            "assessed": [f.name for f in found.findings],
            "fragile": sorted(fragile),
            "note": self.note,
        }
        return confirm.Withdrawal(
            reasons,
            record,
            diagnostic_rows(found, self.rows, self.seed),
            accounted=self.requested,
        )
