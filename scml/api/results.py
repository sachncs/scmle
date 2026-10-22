"""Typed run result."""

import dataclasses
import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from scml.artifacts import bundle as bundle_module
from scml.contracts import data as data_contracts
from scml.contracts import experiments
from scml.data.readers import interchange
from scml.evaluation import fit_health
from scml.evaluation import parameter_estimation

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
        plan: The executed plan.
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
        """Returns permutation importance, most important feature first."""
        report = self.diagnostics.get("permutation_importance", {})
        return pd.DataFrame(report.get("features", []))

    def health_label(self, trial: experiments.TrialResult) -> str:
        """Fit-health verdict of a trial, or an empty string if unknown."""
        if trial.status is not experiments.TrialStatus.OK:
            return ""
        health = fit_health.assess_trial(
            trial, self.trials, self.evaluation["metric_greater_is_better"]
        )
        return "" if health is None else health.verdict

    def leaderboard(self) -> pd.DataFrame:
        """Returns trials ranked best-first, failed/skipped trials last."""
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
        """The leaderboard as an Arrow table."""
        return interchange.to_arrow(self.leaderboard())

    def load_best_model(self) -> bundle_module.ModelBundle:
        """Returns the packaged best model."""
        return self.bundle

    def save(
        self,
        directory: str | pathlib.Path,
        signing_key: str | bytes | None = None,
    ) -> pathlib.Path:
        """Exports the model bundle to `directory`.

        Args:
            directory: Destination directory.
            signing_key: HMAC key; defaults to `$SCML_SIGNING_KEY`. Signed
                bundles load without `trusted=True`.
        """
        return self.bundle.save(directory, signing_key)

    def predict(self, data: str | pathlib.Path | pd.DataFrame) -> np.ndarray:
        """Predicts with the best model."""
        return self.bundle.predict(data)

    def predict_proba(
        self, data: str | pathlib.Path | pd.DataFrame
    ) -> np.ndarray:
        """Predicts class probabilities with the best model."""
        return self.bundle.predict_proba(data)

    def explain(
        self,
        data: str | pathlib.Path | pd.DataFrame,
        background: str | pathlib.Path | pd.DataFrame,
        class_label: Any = None,
    ) -> pd.DataFrame:
        """Explains predictions of the best model against a background."""
        return self.bundle.explain(data, background, class_label=class_label)

    @property
    def meets_criteria(self) -> bool:
        """Whether every stated success criterion was met (True if none)."""
        return all(c["met"] for c in self.evaluation["success_criteria"])

    def summary(self) -> str:
        """Returns a one-line human-readable result."""
        evaluation = self.evaluation
        test = ", ".join(
            f"{name}={value:.4f}"
            for name, value in evaluation["test_metrics"].items()
        )
        text = (
            f"{self.run_id}: selected {self.selected_model} "
            f"(cv {evaluation['metric']}="
            f"{evaluation['selected_cv_score']:.4f}); test: {test}"
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
        return text
