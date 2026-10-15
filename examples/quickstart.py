"""Quickstart: plan, run and predict on a synthetic churn-like dataset."""

import numpy as np
import pandas as pd

import mml


def make_dataset(rows: int = 800) -> pd.DataFrame:
    """Builds a small binary-classification dataset."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "tenure": rng.integers(1, 72, rows),
            "monthly_spend": rng.normal(60, 20, rows),
            "plan": rng.choice(["basic", "plus", "pro"], rows),
        }
    )
    risk = -0.03 * frame.tenure + 0.02 * frame.monthly_spend
    risk += (frame.plan == "basic") * 0.8
    frame["churn"] = (risk + rng.normal(scale=0.7, size=rows) > 0.5).astype(int)
    return frame


def main() -> None:
    """Runs the full workflow and prints a summary."""
    data = make_dataset()
    project = mml.MLProject(data, "churn", metric="roc_auc", time_budget=120)
    print(project.plan().summary())
    result = project.run()
    print(result.summary())
    print(result.leaderboard()[["model_name", "status", "score_mean"]])
    print(result.predict(data.drop(columns="churn").head(5)))


if __name__ == "__main__":
    main()
