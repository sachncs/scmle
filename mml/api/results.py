"""Typed run result."""

import dataclasses
import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd

from mml.artifacts import bundle as bundle_module
from mml.contracts import data as data_contracts
from mml.contracts import experiments


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

    def feature_importance(self) -> pd.DataFrame:
        """Returns permutation importance, most important feature first."""
        report = self.diagnostics.get("permutation_importance", {})
        return pd.DataFrame(report.get("features", []))

    def leaderboard(self) -> pd.DataFrame:
        """Returns trials ranked best-first, failed/skipped trials last."""
        board = pd.DataFrame([t.to_dict() for t in self.trials])
        board = board.drop(columns=["fold_scores"])
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

    def load_best_model(self) -> bundle_module.ModelBundle:
        """Returns the packaged best model."""
        return self.bundle

    def save(self, directory: str | pathlib.Path) -> pathlib.Path:
        """Exports the model bundle to `directory`."""
        return self.bundle.save(directory)

    def predict(self, data: str | pathlib.Path | pd.DataFrame) -> np.ndarray:
        """Predicts with the best model."""
        return self.bundle.predict(data)

    def predict_proba(
        self, data: str | pathlib.Path | pd.DataFrame
    ) -> np.ndarray:
        """Predicts class probabilities with the best model."""
        return self.bundle.predict_proba(data)

    def summary(self) -> str:
        """Returns a one-line human-readable result."""
        evaluation = self.evaluation
        test = ", ".join(
            f"{name}={value:.4f}"
            for name, value in evaluation["test_metrics"].items()
        )
        return (
            f"{self.run_id}: selected {self.selected_model} "
            f"(cv {evaluation['metric']}="
            f"{evaluation['selected_cv_score']:.4f}); test: {test}"
        )
