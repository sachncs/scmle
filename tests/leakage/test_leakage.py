import numpy as np
import pandas as pd
import pytest

from mml import MLProject
from mml.contracts.errors import BlockedError, LeakageError
from mml.data.splitting import LockedTestSet


def test_pure_noise_is_not_inflated_by_pipeline(tmp_path):
    rng = np.random.default_rng(5)
    n = 600
    df = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(10)})
    df["cat"] = rng.choice(list("abcdefgh"), n)
    df["y"] = rng.integers(0, 2, n)
    r = MLProject(df, "y", output_dir=tmp_path, time_budget=120).run()
    assert abs(r.evaluation["test_metrics"]["roc_auc"] - 0.5) < 0.12
    assert abs(r.evaluation["selected_cv_score"] - 0.5) < 0.12


def test_leaked_feature_is_flagged_and_can_block(binary_df, tmp_path):
    df = binary_df.assign(leak=binary_df.y * 3.0 + 1)
    r = MLProject(df, "y", output_dir=tmp_path).run()
    assert any("'leak'" in w for w in r.profile.risk_findings)
    with pytest.raises(BlockedError, match="leakage"):
        MLProject(df, "y", output_dir=tmp_path, leakage_policy="block").run()


def test_group_leakage_prevented_across_test_boundary(binary_df, tmp_path):
    df = binary_df.assign(g=np.arange(len(binary_df)) % 40)
    r = MLProject(df, "y", group_column="g", output_dir=tmp_path).run()
    import json

    vm = json.loads((r.run_dir / "validation_manifest.json").read_text())
    tr = set(df.g.iloc[vm["train_ids"]])
    te = set(df.g.iloc[vm["test_ids"]])
    assert tr.isdisjoint(te)


def test_group_column_is_not_a_feature(binary_df, tmp_path):
    df = binary_df.assign(g=np.arange(len(binary_df)) % 40)
    r = MLProject(df, "y", group_column="g", output_dir=tmp_path).run()
    assert "g" not in r.bundle.features


def test_test_set_cannot_be_reopened_after_run(binary_df, tmp_path):
    ts = LockedTestSet(binary_df.drop(columns="y"), binary_df.y, (0,))
    ts.open(ts.issue_token())
    with pytest.raises(LeakageError):
        ts.open(ts.issue_token())
