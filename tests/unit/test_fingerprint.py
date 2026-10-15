"""Tests for the dataset fingerprint dimensions."""

import numpy as np
import pandas as pd
import pytest

from mml.contracts import tasks
from mml.data import columns
from mml.data import fingerprint
from mml.data import profiling
from mml.data import readers


def profile_of(frame, task=tasks.TaskType.BINARY, **data_overrides):
    spec = tasks.ProjectSpec(
        data=tasks.DataSpec(uri="m", target="y", **data_overrides)
    )
    manifest = readers.build_manifest(frame, "y")
    return profiling.DatasetProfiler(spec).profile(frame, manifest, task, 0.25)


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
