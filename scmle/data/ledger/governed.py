"""The charter-governed exploration and confirmation of one run.

This is the single entry point the pipeline calls. It takes exploration data
and, separately, the sealed confirmation set as an argument:

1. open the ledger (a new file or the run's existing one);
2. explore: one registered hypothesis per usable feature, tested under the
   online rule on `X`;
3. negative control: re-run the whole exploration `B` times with the target
   permuted within groups or time blocks, record the discoveries, and gate on
   the 95th percentile;
4. optionally, a second stage: a rule computed from `X` alone withdraws some
   survivors from the family (see "Second stage" in `confirm`);
5. confirm: freeze the survivors and test the family once on `C`.

The module imports nothing from custody; the caller hands `C` over as
`source`, so the code that selects survivors cannot reach it.
"""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
from fractions import Fraction
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scmle.contracts import qualification
from scmle.data.ledger import budget as budget_module
from scmle.data.ledger import confirm
from scmle.data.ledger import controls
from scmle.data.ledger import explore
from scmle.data.ledger import ledger as ledger_module
from scmle.data.ledger import multiplicity
from scmle.data.ledger import online_fdr

CONTROL_FILE = "negative_control.json"
DEFAULT_BLOCKS = 20


@dataclasses.dataclass(frozen=True, slots=True)
class GovernedRun:
    """What a governed exploration and confirmation produced.

    Attributes:
        ledger: The ledger, sealed by the confirmation.
        plan: The exploration plan.
        control: The negative control.
        confirmation: The confirmation outcome.
        scope: `strata` or `global`, how the control permuted the target.
    """

    ledger: ledger_module.Ledger
    plan: explore.Plan
    control: controls.ControlResult
    confirmation: confirm.Confirmation
    scope: str

    def summary(self) -> dict[str, Any]:
        """Counts and hashes for the run event."""
        results = self.ledger.results.values()
        return {
            "rule": self.ledger.header["rule"]["rule"],
            "hypotheses": len(self.ledger.hypotheses),
            "tested": sum(1 for r in results if r.check is not None),
            "rejected_on_x": sum(1 for r in results if r.rejected),
            "skipped_features": len(self.plan.skipped),
            "confirmed": len(self.confirmation.confirmed),
            "outcome": self.confirmation.outcome,
            "control_exceeds": self.control.exceeds,
            "control_observed": self.control.observed,
            "control_p95": self.control.p95,
            "control_scope": self.scope,
            "ledger_head": self.ledger.head,
            "survivors_sha256": self.confirmation.survivors_sha256,
            "withdrawn": (
                None
                if self.confirmation.stage_two is None
                else len(self.confirmation.stage_two["withdrawn"])
            ),
        }


def strata_labels(
    features: pd.DataFrame,
    group_column: str | None,
    time_column: str | None,
    blocks: int = DEFAULT_BLOCKS,
) -> np.ndarray | None:
    """Labels within which the target may be permuted.

    Groups win over time: rows of one entity stay together. With only a time
    column the rows are cut into `blocks` contiguous blocks in time order.
    Blocks must be short against the timescale of the drift: on random-walk
    data 10 blocks let the control be exceeded in about 7.7% of null worlds
    and 20 blocks in about 3.7% (see `benchmarks.qualification.ledger_checks`).
    With neither, the rows are taken as exchangeable and None is returned
    (a global permutation).

    Args:
        features: Exploration features, holding the structure columns.
        group_column: Entity column, if any.
        time_column: Time column, if any.
        blocks: Number of contiguous time blocks.

    Returns:
        One label per row, or None for exchangeable rows.
    """
    if group_column and group_column in features.columns:
        return features[group_column].astype(str).to_numpy()
    if time_column and time_column in features.columns:
        order = pd.to_numeric(features[time_column], errors="coerce")
        size = max(2, len(features) // max(blocks, 1))
        return controls.block_labels(order.to_numpy(dtype=float), size)
    return None


def run_governed(
    *,
    charter: Any,
    features: pd.DataFrame,
    target: pd.Series,
    classification: bool,
    source: confirm.ConfirmationSource | None,
    run_dir: str | Path | None,
    exclude: tuple[str, ...] = (),
    strata: np.ndarray | None = None,
    rule: str = online_fdr.DEFAULT_RULE,
    confirmation_rule: budget_module.MultiplicityRule = (
        multiplicity.DEFAULT_RULE
    ),
    policy: str = confirm.UNCONFIRMED,
    diagnostics: Sequence[tuple[str, qualification.Check]] = (),
    second_stage: Callable[
        [Sequence[ledger_module.Hypothesis]], confirm.Withdrawal
    ]
    | None = None,
) -> GovernedRun:
    """Explores `X` under the ledger, runs the control and confirms on `C`.

    Args:
        charter: The governing `AnalysisCharter`.
        features: Exploration features (never confirmation rows).
        target: Exploration target.
        classification: Whether the target holds classes.
        source: The sealed confirmation set, or None if none was carved.
        run_dir: Directory for the artifacts, or None to keep them in memory.
        exclude: Structure columns (group, time) that are not candidates.
        strata: Permutation labels for the control (see `strata_labels`).
        rule: Online rule for exploration.
        confirmation_rule: Multiplicity rule for the confirmation.
        policy: Failure policy of the confirmation.
        diagnostics: (source, check) pairs from detectors run on `X`; they
            are logged once, before exploration, and spend no wealth.
        second_stage: Pre-registered withdrawal rule. It receives the
            survivors (rejected hypotheses, in order) after the negative
            control and must compute its answer from `X` alone; the closure
            the caller builds holds no `C`. Its diagnostics are logged
            without wealth before the survivors are frozen, and the
            withdrawn survivors are not tested on `C`.

    Returns:
        The governed run; artifacts are written when `run_dir` is given.
    """
    confirm.check_policy(policy)
    kind = explore.target_kind(target, classification)
    codes = explore.encode_target(target, classification)
    classes = (
        tuple(sorted(target.dropna().unique(), key=str))
        if classification
        else None
    )
    plan = explore.plan_exploration(features, kind, exclude)
    if run_dir is None:
        ledger = ledger_module.Ledger.create(
            None,
            charter_hash=charter.charter_hash,
            config=online_fdr.RuleConfig(
                rule, charter.error_budget.exploration.initial
            ),
            alpha_confirm=charter.error_budget.confirmation.alpha_confirm,
            seed=charter.seed,
        )
    else:
        ledger = ledger_module.Ledger.open(run_dir, charter, rule=rule)
    config = ledger.config
    if config is None:  # pragma: no cover - open() always sets it
        raise ledger_module.LedgerError("ledger has no rule")
    if diagnostics and not ledger.diagnostics:
        for source_name, found in diagnostics:
            ledger.record_diagnostic(found, source_name)
    results = explore.run_exploration(ledger, features, codes, plan)
    observed = sum(1 for r in results if r.rejected)

    def discoveries(frame: pd.DataFrame, shuffled: pd.Series) -> int:
        return explore.count_discoveries(config, plan, frame, shuffled)

    control = controls.permuted_target_control(
        discoveries,
        features,
        codes,
        observed=observed,
        b=charter.error_budget.permutations_b,
        seed=charter.seed,
        strata=strata,
    )
    ledger.record_control(control.to_dict())
    withdrawal = None
    if second_stage is not None:
        withdrawal = second_stage(ledger.survivors())
        for source_name, found in withdrawal.diagnostics:
            ledger.record_diagnostic(found, source_name)
    alpha = Fraction(charter.error_budget.confirmation.alpha_confirm)
    confirmation = confirm.confirm(
        ledger,
        source,
        alpha_confirm=alpha,
        rule=confirmation_rule,
        policy=policy,
        encode=lambda series: explore.encode_target(
            series, classification, classes
        ).to_numpy(dtype=float),
        withdrawal=withdrawal,
    )
    if run_dir is not None:
        folder = Path(run_dir)
        confirmation.write(folder / confirm.CLAIMS_FILE)
        (folder / CONTROL_FILE).write_text(
            json.dumps(control.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return GovernedRun(
        ledger=ledger,
        plan=plan,
        control=control,
        confirmation=confirmation,
        scope="global" if strata is None else "strata",
    )
