"""The time column can be offered to the model, but only on request."""

import numpy as np
import pandas as pd
import support


def drifting_frame(rows=700, seed=0):
    rng = np.random.default_rng(seed)
    stamp = np.arange(rows, dtype=float)
    x = rng.normal(size=rows)
    drift = (stamp / rows) * 2 - 1
    y = (x + 1.5 * drift + rng.normal(0, 0.5, rows) > 0).astype(int)
    return pd.DataFrame({"t": stamp, "x": x, "y": y})


def run(tmp_path, **kwargs):
    return support.make_project(
        drifting_frame(),
        "y",
        time_budget=300,
        output_dir=tmp_path,
        time_column="t",
        **kwargs,
    ).run()


def test_the_time_column_is_split_only_by_default(tmp_path):
    assert "t" not in run(tmp_path).bundle.features


def test_the_time_column_becomes_a_feature_on_request(tmp_path):
    result = run(tmp_path, time_as_feature=True)

    assert "t" in result.bundle.features
    assert result.evaluation["split_strategy"] == "temporal"
