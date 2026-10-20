"""Tests for the dataset fingerprint dimensions."""

import numpy as np
import pandas as pd
import pytest

from scml.contracts import tasks
from scml.data.profiling import profiler
from scml.data.readers import tabular
from scml.data.validation import columns
from scml.intelligence import fingerprint


def profile_of(frame, task=tasks.TaskType.BINARY, **data_overrides):
    spec = tasks.ProjectSpec(
        data=tasks.DataSpec(uri="m", target="y", **data_overrides)
    )
    manifest = tabular.build_manifest(frame, "y")
    return profiler.DatasetProfiler(spec).profile(frame, manifest, task, 0.25)


def test_entropy_of_known_distributions():
    assert columns.entropy(pd.Series(["a", "b"] * 50)) == pytest.approx(1.0)
    assert columns.entropy(pd.Series(["a"] * 10)) == pytest.approx(0.0)
    assert columns.entropy(pd.Series(list("abcd") * 5)) == pytest.approx(2.0)
    assert columns.entropy(pd.Series([], dtype=object)) == 0.0


def test_outlier_rate_uses_tukey_fences():
    values = pd.Series([*range(1, 101), 10_000.0, -10_000.0])
    assert columns.outlier_rate(values) == pytest.approx(2 / 102)
    assert columns.outlier_rate(pd.Series([5.0] * 20)) == 0.0
    assert columns.outlier_rate(pd.Series([1.0, 2.0])) == 0.0


def test_redundant_pairs_detects_near_duplicates_only():
    rng = np.random.default_rng(0)
    base = rng.normal(size=500)
    frame = pd.DataFrame(
        {
            "a": base,
            "b": base * 3 + rng.normal(scale=0.01, size=500),
            "c": rng.normal(size=500),
        }
    )
    pairs = fingerprint.redundant_pairs(frame, ("a", "b", "c"))
    assert [(p["a"], p["b"]) for p in pairs] == [("a", "b")]
    assert pairs[0]["correlation"] > 0.99
    assert fingerprint.redundant_pairs(frame, ("a",)) == []


def test_order_drift_flags_trending_data_not_stationary_data():
    rng = np.random.default_rng(1)
    stationary = pd.DataFrame({"x": rng.normal(size=600)})
    trending = pd.DataFrame(
        {"x": rng.normal(size=600) + np.linspace(0, 6, 600)}
    )
    assert fingerprint.order_drift(stationary, ("x",)) < 0.1
    assert fingerprint.order_drift(trending, ("x",)) > 0.25
    assert fingerprint.order_drift(stationary.head(40), ("x",)) == 0.0


def frame_with_everything(rows=600):
    rng = np.random.default_rng(2)
    frame = pd.DataFrame(
        {
            "num": rng.normal(size=rows),
            "twin": 0.0,
            "cat": rng.choice(["a", "b", "c", "d"], rows),
            "grp": rng.integers(0, 30, rows),
            "when": pd.date_range("2024-01-01", periods=rows, freq="h"),
        }
    )
    frame["twin"] = frame["num"] * 2
    frame.loc[::10, "num"] = np.nan
    frame["y"] = (rng.random(rows) < 0.2).astype(int)
    return pd.concat([frame, frame.head(30)], ignore_index=True)


def test_fingerprint_covers_every_dimension():
    result = profile_of(
        frame_with_everything(), group_column="grp", time_column="when"
    ).dataset_statistics
    assert set(result) == {
        "shape",
        "types",
        "missingness",
        "target",
        "features",
        "relationships",
        "ml_difficulty",
        "structure",
        "quality",
        "operational",
        "risk",
    }
    assert (
        result["shape"]["rows"] == 630
        and result["shape"]["effective_samples"] == 600
    )
    assert (
        result["types"]["datetime"] == 1 and result["types"]["categorical"] == 1
    )
    assert result["missingness"]["columns_with_missing"] == 1
    assert (
        result["target"]["imbalance_ratio"] > 2
        and result["target"]["entropy_bits"] < 1
    )
    assert result["features"]["max_cardinality"] == 4
    assert result["features"]["mean_categorical_entropy_bits"] == pytest.approx(
        2.0, abs=0.05
    )
    assert [
        (p["a"], p["b"])
        for p in result["relationships"]["redundant_numeric_pairs"]
    ] == [("num", "twin")]
    assert result["structure"]["groups"]["n_groups"] == 30
    assert (
        result["structure"]["time"]["monotonic"] is False
    )  # duplicates appended last
    assert result["quality"]["duplicate_rows"] == 30
    assert (
        result["operational"]["read_seconds"] == 0.25
        and result["operational"]["memory_mb"] > 0
    )
    assert result["risk"]["order_drift_psi"] >= 0.0


def test_regression_target_fingerprint():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"x": rng.normal(size=300)})
    frame["y"] = np.exp(frame.x * 2)
    result = profile_of(frame, tasks.TaskType.REGRESSION).dataset_statistics
    assert result["target"]["skew"] > 1 and result["target"]["outlier_rate"] > 0
    assert "structure" in result and result["structure"] == {}


def test_profile_serializes_the_fingerprint():
    payload = profile_of(frame_with_everything()).to_dict()
    assert payload["dataset"]["shape"]["columns"] == 6
    assert payload["columns"]["num"]["outlier_rate"] >= 0


def test_ml_difficulty_reports_baselines_and_headroom():
    rng = np.random.default_rng(0)
    rows = 800
    frame = pd.DataFrame(
        {"a": rng.normal(size=rows), "b": rng.normal(size=rows)}
    )
    frame["y"] = (frame.a * frame.b > 0).astype(int)  # purely non-linear
    stats = profile_of(frame).dataset_statistics["ml_difficulty"]
    assert stats["available"] and stats["metric"] == "roc_auc"
    assert stats["constant_baseline"] == pytest.approx(0.5, abs=0.02)
    assert stats["linear_baseline"] < 0.6  # a line cannot separate XOR
    assert stats["nonlinearity_gain"] > 0.3  # trees can
    assert stats["separability"] == stats["linear_baseline"]


def test_ml_difficulty_fails_closed_when_scores_are_all_undefined(
    monkeypatch,
):
    """An undefined baseline is reported instead of becoming NaN."""
    frame = pd.DataFrame({"x": range(30), "y": [0, 1] * 15})

    def undefined_scores(*args, **kwargs):
        del args, kwargs
        return np.array([np.nan, np.nan])

    monkeypatch.setattr(
        fingerprint.model_selection, "cross_val_score", undefined_scores
    )
    spec = tasks.ProjectSpec(data=tasks.DataSpec(uri="m", target="y"))
    manifest = tabular.build_manifest(frame, "y")
    task = tasks.TaskType.BINARY
    plan = profiler.DatasetProfiler(spec).profile(
        frame, manifest, task, 0.25
    ).feature_plan

    result = fingerprint.ml_difficulty(frame, spec, task, plan)

    assert result == {
        "available": False,
        "reason": "constant baseline produced no finite scores",
    }


def test_missingness_patterns_types_and_scale_are_described():
    rng = np.random.default_rng(1)
    rows = 600
    frame = pd.DataFrame(
        {
            "small": rng.normal(size=rows),
            "huge": rng.normal(size=rows) * 1e6,
            "other": rng.normal(size=rows),
            "note": [f"free text comment number {i} " * 3 for i in range(rows)],
        }
    )
    blank = np.arange(rows) % 7 == 0
    frame.loc[blank, ["small", "other"]] = np.nan  # they go missing together
    frame["y"] = (rng.random(rows) < 0.5).astype(int)
    result = profile_of(frame).dataset_statistics
    patterns = result["missingness"]["patterns"]
    assert patterns["pattern_count"] == 2
    assert patterns["co_missing"] == 1
    assert patterns["top_pattern_share"] == pytest.approx(
        blank.mean(), abs=0.01
    )
    assert result["types"]["text_like"] == 1
    assert result["features"]["scale_orders_of_magnitude"] > 5
