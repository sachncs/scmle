"""Calibration and coverage reports written by a run."""

import numpy as np
import pandas as pd
import support


def three_class_frame(rows=900, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {"a": rng.normal(size=rows), "b": rng.normal(size=rows)}
    )
    scores = np.column_stack([frame.a, frame.b, -frame.a]) + rng.normal(
        0, 1.0, (rows, 3)
    )
    frame["y"] = scores.argmax(axis=1)
    return frame


def noisy_line(rows=2500, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, rows)
    return pd.DataFrame({"x": x, "y": 3 * x + rng.normal(0, 1 + 0.4 * x, rows)})


def test_a_multiclass_run_reports_its_reliability(tmp_path):
    result = support.make_project(
        three_class_frame(),
        "y",
        task="multiclass_classification",
        time_budget=300,
        output_dir=tmp_path,
    ).run()

    report = result.diagnostics["calibration"]

    assert 0.0 <= report["expected_calibration_error"] <= 1.0
    assert report["curve"] and report["brier_score"] > 0


def test_quantile_coverage_tracks_the_nominal_level_on_locked_rows(tmp_path):
    result = support.make_project(
        noisy_line(), "y", time_budget=300, output_dir=tmp_path
    ).run()

    curve = result.evaluation["quantile_coverage"]
    tests = result.evaluation["n_test"]

    assert len(curve) >= 3
    for point in curve:
        spread = 3 * np.sqrt(point["nominal"] * (1 - point["nominal"]) / tests)
        assert abs(point["empirical"] - point["nominal"]) <= spread + 0.03
