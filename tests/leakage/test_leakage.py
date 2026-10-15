"""Adversarial leakage checks."""

import json

import numpy as np
import pandas as pd
import pytest
import support

import mml
from mml.contracts import errors
from mml.contracts import tasks


def test_pure_noise_is_not_inflated_by_the_pipeline(tmp_path):
    rng = np.random.default_rng(5)
    rows = 600
    frame = pd.DataFrame({f"f{i}": rng.normal(size=rows) for i in range(10)})
    frame["cat"] = rng.choice(list("abcdefgh"), rows)
    frame["y"] = rng.integers(0, 2, rows)
    result = support.make_project(
        frame, "y", output_dir=tmp_path, time_budget=120
    ).run()
    assert abs(result.evaluation["test_metrics"]["roc_auc"] - 0.5) < 0.12
    assert abs(result.evaluation["selected_cv_score"] - 0.5) < 0.12


def test_leaked_feature_is_flagged_and_can_block(binary_frame, tmp_path):
    leaky = binary_frame.assign(leak=binary_frame.y * 3.0 + 1)
    result = support.make_project(leaky, "y", output_dir=tmp_path).run()
    assert any("'leak'" in w for w in result.profile.risk_findings)
    with pytest.raises(errors.BlockedError, match="leakage"):
        support.make_project(
            leaky, "y", output_dir=tmp_path, leakage_policy="block"
        ).run()


def test_groups_never_straddle_the_test_boundary(binary_frame, tmp_path):
    frame = binary_frame.assign(g=np.arange(len(binary_frame)) % 40)
    result = support.make_project(
        frame, "y", group_column="g", output_dir=tmp_path
    ).run()
    manifest = json.loads(
        (result.run_dir / "validation_manifest.json").read_text()
    )
    train_groups = set(frame.g.iloc[manifest["train_ids"]])
    test_groups = set(frame.g.iloc[manifest["test_ids"]])
    assert train_groups.isdisjoint(test_groups)
    assert "g" not in result.bundle.features


def test_test_set_is_opened_exactly_once_per_run(binary_frame, tmp_path):
    result = support.make_project(binary_frame, "y", output_dir=tmp_path).run()
    assert result.evaluation["test_set_opened"] is True
    events = (result.run_dir / "events.jsonl").read_text().splitlines()
    testing = [e for e in events if json.loads(e)["kind"] == "state.TESTING"]
    assert len(testing) == 1


def run_default(frame, directory):
    return mml.MLProject(
        frame,
        "y",
        time_budget=600,
        output_dir=directory,
        validation="random",  # Stratified splits legitimately use labels.
        search=tasks.SearchSpec(top_models=1, configs_per_model=4),
        strategies=tasks.StrategySpec(top_models=1),
        ensemble=tasks.EnsembleSpec(size=2),
    ).run()


def test_search_is_invariant_to_test_labels_and_test_features(
    imbalanced_frame, tmp_path
):
    """Nothing the search sees may depend on the locked test partition."""
    first = run_default(imbalanced_frame, tmp_path / "a")
    manifest = json.loads(
        (first.run_dir / "validation_manifest.json").read_text()
    )
    test_ids = np.array(manifest["test_ids"])

    altered = imbalanced_frame.copy()
    rng = np.random.default_rng(99)
    altered.loc[test_ids, "y"] = rng.permutation(
        altered.loc[test_ids, "y"].to_numpy()
    )
    for column in ("spend", "age", "visits"):
        altered.loc[test_ids, column] = rng.normal(500, 100, len(test_ids))
    second = run_default(altered, tmp_path / "b")

    assert (
        second.evaluation["split_fingerprint"]
        == first.evaluation["split_fingerprint"]
    )
    assert [
        (t.trial_id, t.model_name, t.strategy, t.params) for t in second.trials
    ] == [
        (t.trial_id, t.model_name, t.strategy, t.params) for t in first.trials
    ]
    for a, b in zip(first.trials, second.trials, strict=True):
        assert a.fold_scores == pytest.approx(b.fold_scores)
    assert second.selected_model == first.selected_model
    assert (
        second.evaluation["selected_params"]
        == first.evaluation["selected_params"]
    )
    assert second.evaluation["selected_cv_score"] == pytest.approx(
        first.evaluation["selected_cv_score"]
    )
    # Sanity: the test partition really was different.
    assert second.evaluation["test_metrics"] != first.evaluation["test_metrics"]
