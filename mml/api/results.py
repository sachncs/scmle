"""Typed run result."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from mml.artifacts.bundle import ModelBundle
from mml.contracts.data import DatasetProfile
from mml.contracts.experiments import ExperimentPlan, TrialResult


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    state: str
    plan: ExperimentPlan
    profile: DatasetProfile
    trials: list[TrialResult]
    selected_model: str
    evaluation: dict
    bundle: ModelBundle
    decision_log: list[dict] = field(default_factory=list)

    def leaderboard(self) -> pd.DataFrame:
        rows = [t.to_dict() for t in self.trials]
        df = pd.DataFrame(rows).drop(columns=["fold_scores"])
        df["selected"] = df["model_name"] == self.selected_model
        gib = self.evaluation["metric_greater_is_better"]
        ok = df["status"] == "ok"
        ranked = df[ok].sort_values("score_mean", ascending=not gib)
        return pd.concat([ranked, df[~ok]]).reset_index(drop=True)

    def load_best_model(self) -> ModelBundle:
        return self.bundle

    def save(self, directory: str | Path) -> Path:
        return self.bundle.save(directory)

    def predict(self, data: str | Path | pd.DataFrame) -> np.ndarray:
        return self.bundle.predict(data)

    def predict_proba(self, data: str | Path | pd.DataFrame) -> np.ndarray:
        return self.bundle.predict_proba(data)

    def summary(self) -> str:
        e = self.evaluation
        test = ", ".join(f"{k}={v:.4f}" for k, v in e["test_metrics"].items())
        return (
            f"{self.run_id}: selected {self.selected_model} "
            f"(cv {e['metric']}={e['selected_cv_score']:.4f}); test: {test}"
        )
