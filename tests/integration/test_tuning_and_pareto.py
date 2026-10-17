"""Hyperband tuning and Pareto selection in real runs."""

import numpy as np
import pandas as pd
import support

from mml.contracts import tasks


def frame(rows=800, seed=0):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({f"x{i}": rng.normal(size=rows) for i in range(4)})
    data["y"] = (data.x0 + data.x1**2 + rng.normal(0, 0.5, rows) > 0.5).astype(
        int
    )
    return data


def test_hyperband_with_a_grid_tunes_and_prunes_cheaply(tmp_path):
    result = support.make_project(
        frame(),
        "y",
        time_budget=300,
        output_dir=tmp_path,
        folds=5,
        search=tasks.SearchSpec(
            top_models=1,
            configs_per_model=8,
            optimizer="grid",
            pruning="hyperband",
        ),
    ).run()

    tuned = [t for t in result.trials if t.params]
    assert len(tuned) >= 4
    statuses = {t.status.value for t in tuned}
    assert "ok" in statuses
    assert result.state == "COMPLETED"


def test_the_pareto_front_is_reported_and_the_fast_model_wins_a_tie(tmp_path):
    result = support.make_project(
        frame(),
        "y",
        time_budget=300,
        output_dir=tmp_path,
        selection=tasks.SelectionSpec(pareto=True, pareto_tolerance=0.05),
    ).run()

    front = result.evaluation["pareto_front"]
    chosen = [p for p in front if p["chosen"]]
    assert len(chosen) == 1 and len(front) >= 3
    near = [
        p
        for p in front
        if not p["violations"]
        and max(q["score"] for q in front) - p["score"] <= 0.05
    ]
    assert chosen[0]["p95_latency_ms"] == min(p["p95_latency_ms"] for p in near)
    assert any(p["dominated"] for p in front) or len(front) >= 3


def test_pareto_is_off_by_default(tmp_path):
    result = support.make_project(
        frame(300), "y", time_budget=300, output_dir=tmp_path
    ).run()

    assert result.evaluation["pareto_front"] is None
