"""Adversarial leakage checks."""

import json

import numpy as np
import pandas as pd
import pytest

import mml
from mml.contracts import errors


def test_pure_noise_is_not_inflated_by_the_pipeline(tmp_path):
    rng = np.random.default_rng(5)
    rows = 600
    frame = pd.DataFrame({f"f{i}": rng.normal(size=rows) for i in range(10)})
    frame["cat"] = rng.choice(list("abcdefgh"), rows)
    frame["y"] = rng.integers(0, 2, rows)
    result = mml.MLProject(frame, "y", output_dir=tmp_path, time_budget=120).run()
    assert abs(result.evaluation["test_metrics"]["roc_auc"] - 0.5) < 0.12
    assert abs(result.evaluation["selected_cv_score"] - 0.5) < 0.12


def test_leaked_feature_is_flagged_and_can_block(binary_frame, tmp_path):
    leaky = binary_frame.assign(leak=binary_frame.y * 3.0 + 1)
    result = mml.MLProject(leaky, "y", output_dir=tmp_path).run()
    assert any("'leak'" in w for w in result.profile.risk_findings)
    with pytest.raises(errors.BlockedError, match="leakage"):
        mml.MLProject(
            leaky, "y", output_dir=tmp_path, leakage_policy="block"
        ).run()


def test_groups_never_straddle_the_test_boundary(binary_frame, tmp_path):
    frame = binary_frame.assign(g=np.arange(len(binary_frame)) % 40)
    result = mml.MLProject(
        frame, "y", group_column="g", output_dir=tmp_path
    ).run()
    manifest = json.loads((result.run_dir / "validation_manifest.json").read_text())
    train_groups = set(frame.g.iloc[manifest["train_ids"]])
    test_groups = set(frame.g.iloc[manifest["test_ids"]])
    assert train_groups.isdisjoint(test_groups)
    assert "g" not in result.bundle.features


def test_test_set_is_opened_exactly_once_per_run(binary_frame, tmp_path):
    result = mml.MLProject(binary_frame, "y", output_dir=tmp_path).run()
    assert result.evaluation["test_set_opened"] is True
    events = (result.run_dir / "events.jsonl").read_text().splitlines()
    testing = [e for e in events if json.loads(e)["kind"] == "state.TESTING"]
    assert len(testing) == 1
