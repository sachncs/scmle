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


def without_timings(statistics):
    """Removes wall-clock measurements, which differ between any two runs."""
    cleaned = json.loads(json.dumps(statistics, default=str))
    cleaned["operational"].pop("read_seconds")
    cleaned["ml_difficulty"].pop("seconds")
    return cleaned


def test_profile_and_feature_plan_never_see_the_locked_test_rows(
    binary_frame, tmp_path
):
    base = support.make_project(binary_frame, "y", output_dir=tmp_path / "a")
    base.plan()
    from mml.data.splitting import splitters

    split = splitters.SplitManager(base.spec, tasks.TaskType.BINARY).split(
        binary_frame
    )
    test_positions = list(split.manifest.test_ids)

    altered = binary_frame.copy()
    altered.loc[altered.index[test_positions], "a"] = np.nan  # all missing
    altered.loc[altered.index[test_positions], "b"] = 1e9  # absurd values
    altered.loc[altered.index[test_positions], "c"] = "never-seen"
    other = support.make_project(altered, "y", output_dir=tmp_path / "b")
    other.plan()

    before, after = base.profile, other.profile
    assert before.column_statistics == after.column_statistics
    assert before.feature_plan == after.feature_plan
    assert without_timings(before.dataset_statistics) == without_timings(
        after.dataset_statistics
    )
    rows = before.dataset_statistics["shape"]
    assert rows["rows"] == len(split.manifest.train_ids)
    assert rows["total_rows"] == len(binary_frame)
    assert rows["rows"] + len(test_positions) == rows["total_rows"]


def test_a_leak_present_only_in_the_test_rows_is_not_a_profile_finding(
    binary_frame, tmp_path
):
    from mml.data.splitting import splitters

    project = support.make_project(binary_frame, "y", output_dir=tmp_path)
    split = splitters.SplitManager(project.spec, tasks.TaskType.BINARY).split(
        binary_frame
    )
    test_positions = list(split.manifest.test_ids)
    leaky = binary_frame.assign(late=np.random.default_rng(1).normal(size=500))
    leaky.loc[leaky.index[test_positions], "late"] = (
        leaky.loc[leaky.index[test_positions], "y"] * 5.0
    )
    probe = support.make_project(leaky, "y", output_dir=tmp_path / "p")
    probe.plan()
    assert not any("'late'" in r for r in probe.profile.risk_findings)


def test_hard_enforcement_moves_the_test_rows_out_of_reach(
    tmp_path, monkeypatch
):
    from mml.core import limits

    rng = np.random.default_rng(21)
    frame = pd.DataFrame(
        {
            "sentinel": np.arange(400, dtype=float),
            "x": rng.normal(size=400),
        }
    )
    frame["y"] = (frame.x + rng.normal(0, 0.5, 400) > 0).astype(int)
    seen = []
    original = limits.TrialSupervisor.run

    def record(self, function, argument):
        seen.append(set(argument.features["sentinel"]))
        return original(self, function, argument)

    monkeypatch.setattr(limits.TrialSupervisor, "run", record)
    result = support.make_project(
        frame,
        "y",
        time_budget=300,
        output_dir=tmp_path,
        enforcement="hard",
    ).run()

    manifest = json.loads(
        (result.run_dir / "validation_manifest.json").read_text()
    )
    test_ids = set(manifest["test_ids"])
    test_values = {float(frame.loc[i, "sentinel"]) for i in test_ids}
    events = [
        json.loads(line)
        for line in (result.run_dir / "events.jsonl").read_text().splitlines()
    ]
    created = next(e for e in events if e["kind"] == "split.created")

    assert created["data"]["test_set_isolation"] == "separate vault process"
    assert seen and all(values.isdisjoint(test_values) for values in seen)
    assert result.evaluation["n_test"] == len(test_ids)
    assert result.evaluation["test_set_opened"] is True


def test_cooperative_mode_says_it_relies_on_the_in_process_protocol(
    binary_frame, tmp_path
):
    result = support.make_project(
        binary_frame, "y", time_budget=300, output_dir=tmp_path
    ).run()

    events = [
        json.loads(line)
        for line in (result.run_dir / "events.jsonl").read_text().splitlines()
    ]
    created = next(e for e in events if e["kind"] == "split.created")
    assert created["data"]["test_set_isolation"] == "in-process protocol"
