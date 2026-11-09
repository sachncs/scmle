"""Typed run result."""

import dataclasses
import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from scmle.artifacts import bundle as bundle_module
from scmle.contracts import data as data_contracts
from scmle.contracts import experiments
from scmle.data.readers import interchange
from scmle.evaluation import fit_health
from scmle.evaluation import parameter_estimation

LEADERBOARD_DETAILS = (
    "fold_scores",
    "observed_access",
    "evaluation_config",
    "runtime",
    "train_scores",
)


@dataclasses.dataclass(slots=True)
class RunResult:
    """Everything a completed run produced.

    Attributes:
        run_id: Run identifier.
        run_dir: Directory holding all artifacts.
        state: Terminal state name.
        plan: The executed experiment plan.
        profile: Dataset profile.
        trials: All trials in execution order.
        selected_model: Winning model name.
        evaluation: Final evaluation record (also saved as JSON).
        bundle: The packaged inference pipeline.
        diagnostics: Post-selection diagnostic reports by name.
        estimation: Separate parameter-estimation evidence, when explicitly
            declared on the project; never used for model selection.
    """

    run_id: str
    run_dir: pathlib.Path
    state: str
    plan: experiments.ExperimentPlan
    profile: data_contracts.DatasetProfile
    trials: list[experiments.TrialResult]
    selected_model: str
    evaluation: dict[str, Any]
    bundle: bundle_module.ModelBundle
    diagnostics: dict[str, dict[str, Any]] = dataclasses.field(
        default_factory=dict
    )
    estimation: parameter_estimation.EstimationResult | None = None

    def feature_importance(self) -> pd.DataFrame:
        """Returns permutation importance, most important feature first.

        Returns:
            One row per measured feature, ranked by mean metric drop, with the
            columns `feature`, `importance` and `std`. The diagnostic may cover
            only a random subset of the features when its time allowance ran
            out. The frame is empty if the diagnostic was not produced.
        """
        report = self.diagnostics.get("permutation_importance", {})
        return pd.DataFrame(report.get("features", []))

    def health_label(self, trial: experiments.TrialResult) -> str:
        """Gives the fit-health verdict of a trial.

        Args:
            trial: A trial of this run.

        Returns:
            The verdict, or an empty string for a trial that did not succeed or
            whose fit health cannot be assessed.
        """
        if trial.status is not experiments.TrialStatus.OK:
            return ""
        health = fit_health.assess_trial(
            trial, self.trials, self.evaluation["metric_greater_is_better"]
        )
        return "" if health is None else health.verdict

    def leaderboard(self) -> pd.DataFrame:
        """Returns trials ranked best-first, failed/skipped trials last.

        Returns:
            One row per trial: the trial's fields plus its fit-health verdict
            and whether it was selected. The columns in `LEADERBOARD_DETAILS`
            are left out and the parameters are JSON text. Successful trials
            come first, ordered by mean score; the others follow in execution
            order.
        """
        board = pd.DataFrame([t.to_dict() for t in self.trials])
        board["fit_health"] = [
            self.health_label(trial) for trial in self.trials
        ]
        board = board.drop(columns=list(LEADERBOARD_DETAILS))
        board["params"] = board["params"].map(json.dumps)
        board["selected"] = (
            board["trial_id"] == self.evaluation["selected_trial_id"]
        )
        succeeded = board["status"] == str(experiments.TrialStatus.OK)
        ranked = board[succeeded].sort_values(
            "score_mean",
            ascending=not self.evaluation["metric_greater_is_better"],
        )
        return pd.concat([ranked, board[~succeeded]]).reset_index(drop=True)

    def leaderboard_arrow(self) -> pa.Table:
        """Returns the leaderboard as an Arrow table.

        Returns:
            The table form of `leaderboard()`.
        """
        return interchange.to_arrow(self.leaderboard())

    def load_best_model(self) -> bundle_module.ModelBundle:
        """Returns the packaged best model.

        Returns:
            The model bundle of the selected model.
        """
        return self.bundle

    def save(
        self,
        directory: str | pathlib.Path,
        signing_key: str | bytes | None = None,
    ) -> pathlib.Path:
        """Exports the model bundle to `directory`.

        Args:
            directory: Destination directory.
            signing_key: HMAC key; defaults to `$SCMLE_SIGNING_KEY`. Signed
                bundles load without `trusted=True`.

        Returns:
            The directory the bundle was written to.
        """
        return self.bundle.save(directory, signing_key)

    def predict(self, data: str | pathlib.Path | pd.DataFrame) -> np.ndarray:
        """Predicts with the best model.

        Args:
            data: DataFrame or path to a supported tabular dataset.

        Returns:
            Predictions, one per row.
        """
        return self.bundle.predict(data)

    def predict_proba(
        self, data: str | pathlib.Path | pd.DataFrame
    ) -> np.ndarray:
        """Predicts class probabilities with the best model.

        Args:
            data: DataFrame or path to a supported tabular dataset.

        Returns:
            Class probabilities, one row per input row.
        """
        return self.bundle.predict_proba(data)

    def explain(
        self,
        data: str | pathlib.Path | pd.DataFrame,
        background: str | pathlib.Path | pd.DataFrame,
        class_label: Any = None,
    ) -> pd.DataFrame:
        """Explains predictions of the best model against a background.

        Args:
            data: DataFrame or path to the rows to explain.
            background: DataFrame or path to the reference rows.
            class_label: Class to explain for a classifier; if None, the
                positive class of a binary model (a multiclass model needs it).

        Returns:
            The explanation as a frame.
        """
        return self.bundle.explain(data, background, class_label=class_label)

    @property
    def meets_criteria(self) -> bool:
        """Whether every stated success criterion was met (True if none)."""
        return all(c["met"] for c in self.evaluation["success_criteria"])

    def unattested_suspects(self) -> list[str]:
        """Lists flagged columns that are neither quarantined nor attested.

        A column the leakage screen flagged stays a suspicion until the run's
        `admissibility_table.json` records it as quarantined or attested.
        Without that file, every flagged column is listed.

        Returns:
            Sorted column names; empty when nothing is outstanding.
        """
        suspects = set(map(str, self.profile.leakage.get("suspects", ())))
        path = pathlib.Path(self.run_dir) / "admissibility_table.json"
        if suspects and path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            for decision in record.get("decisions", ()):
                settled = decision.get("admission") == "quarantine" or (
                    decision.get("tier") == "attested"
                )
                if settled:
                    suspects.discard(str(decision.get("name")))
        return sorted(suspects)

    def claim_tiers(self) -> dict[str, int]:
        """Counts the claims in `confirmed_claims.json` by evidence tier.

        Returns:
            Tier name to count (a claim without a tier counts as
            `unverified`); empty if the run wrote no `confirmed_claims.json`.
        """
        path = pathlib.Path(self.run_dir) / "confirmed_claims.json"
        if not path.exists():
            return {}
        record = json.loads(path.read_text(encoding="utf-8"))
        counts: dict[str, int] = {}
        for claim in record.get("claims", ()):
            tier = str(claim.get("tier", "unverified"))
            counts[tier] = counts.get(tier, 0) + 1
        return counts

    def summary(self) -> str:
        """Returns a one-line human-readable result.

        Returns:
            The run id, selected model, cross-validation score, test metrics,
            whether the success criteria were met, claim tiers and any warning.
            A result with unattested flagged columns or an incomplete leakage
            screen is marked provisional.
        """
        evaluation = self.evaluation
        test = ", ".join(
            f"{name}={value:.4f}"
            for name, value in evaluation["test_metrics"].items()
        )
        unattested = self.unattested_suspects()
        warning = self.profile.screen_warning
        reasons = []
        if unattested:
            reasons.append(
                f"{len(unattested)} flagged column(s) not attested: "
                f"{', '.join(unattested)}"
            )
        if warning:
            reasons.append("leakage screen incomplete")
        label = f" (PROVISIONAL: {'; '.join(reasons)})" if reasons else ""
        text = (
            f"{self.run_id}: selected {self.selected_model} "
            f"(cv {evaluation['metric']}="
            f"{evaluation['selected_cv_score']:.4f}); test{label}: {test}"
        )
        missed = [
            f"{c['metric']} {c['achieved']:.4f} vs required {c['required']}"
            for c in evaluation["success_criteria"]
            if not c["met"]
        ]
        if evaluation["success_criteria"]:
            text += (
                "; success criteria met"
                if not missed
                else "; success criteria NOT met: " + ", ".join(missed)
            )
        tiers = self.claim_tiers()
        if tiers:
            text += "; claims: " + ", ".join(
                f"{count} {tier}" for tier, count in sorted(tiers.items())
            )
        if warning:
            text += f"; WARNING {warning}"
        return text
