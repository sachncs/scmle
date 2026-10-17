"""A weight column steers training and is optional at prediction time."""

import numpy as np
import pandas as pd
import pytest
import support

from mml.contracts import errors
from mml.contracts import tasks
from mml.models import weighting


def conflicting_frame(rows=800, seed=0):
    """Same feature, opposite labels in two groups; weights pick the winner."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=rows)
    other = rng.normal(size=rows)
    first = np.arange(rows) % 2 == 0
    y = np.where(first, x > 0, x <= 0).astype(int)
    return pd.DataFrame({"x": x, "other": other, "first": first, "y": y})


def run(frame, tmp_path, **kwargs):
    return support.make_project(
        frame,
        "y",
        time_budget=300,
        output_dir=tmp_path,
        models=tasks.ModelSpec(exclude_families=["deep_learning"]),
        **kwargs,
    ).run()


def test_weights_are_training_only_and_not_needed_to_predict(tmp_path):
    frame = conflicting_frame()
    frame["w"] = np.where(frame["first"], 1.0, 0.01)
    probe = pd.DataFrame({"x": [2.0, -2.0], "other": [0.0, 0.0]})

    result = run(frame, tmp_path, weight_column="w", excluded_columns=["first"])

    assert isinstance(result.bundle.pipeline, weighting.WeightedClassifier)
    assert result.bundle.pipeline.column == "w"
    assert "w" not in result.bundle.features
    assert len(result.bundle.predict(probe)) == 2


def test_bad_weights_are_refused_with_a_clear_message(tmp_path):
    frame = conflicting_frame(200)
    frame["w"] = -1.0

    with pytest.raises(errors.DataError, match="non-negative"):
        run(frame, tmp_path, weight_column="w")


def test_weighted_test_metrics_are_reported_beside_the_plain_ones(tmp_path):
    frame = conflicting_frame()
    frame["w"] = np.where(frame["first"], 1.0, 0.01)

    weighted = run(
        frame, tmp_path / "w", weight_column="w", excluded_columns=["first"]
    )
    plain = run(frame, tmp_path / "p", excluded_columns=["first", "w"])

    assert set(weighted.evaluation["weighted_test_metrics"]) == set(
        weighted.evaluation["test_metrics"]
    )
    assert plain.evaluation["weighted_test_metrics"] is None
