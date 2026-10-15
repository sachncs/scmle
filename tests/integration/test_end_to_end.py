import json

import numpy as np
import pytest

from mml import MLProject, ModelBundle
from mml.contracts.errors import BlockedError


def run(df, tmp_path, **kw):
    return MLProject(df, "y", time_budget=120, output_dir=tmp_path, **kw).run()


def test_binary_end_to_end_artifacts_and_inference(binary_df, tmp_path):
    r = run(binary_df, tmp_path)
    e = r.evaluation
    assert e["test_metrics"]["roc_auc"] > 0.8
    assert e["selected_cv_score"] > e["baseline_cv_score"]
    assert e["test_set_opened"] == 1 and e["n_test"] == 100
    for f in [
        "config.json",
        "environment.lock",
        "dataset_manifest.json",
        "dataset_profile.json",
        "validation_manifest.json",
        "execution_plan.json",
        "events.jsonl",
        "leaderboard.parquet",
        "evaluation.json",
        "report.html",
        "model/pipeline.joblib",
        "model/schema.json",
        "model/model_manifest.json",
        "diagnostics/decision_log.json",
    ]:
        assert (r.run_dir / f).exists(), f
    states = [
        json.loads(l)["kind"]
        for l in (r.run_dir / "events.jsonl").read_text().splitlines()
        if json.loads(l)["kind"].startswith("state.")
    ]
    assert states == [
        f"state.{s}"
        for s in [
            "CREATED",
            "VALIDATING",
            "PROFILING",
            "PLANNING",
            "TRAINING",
            "SELECTING",
            "TESTING",
            "PACKAGING",
            "COMPLETED",
        ]
    ]
    assert r.leaderboard().iloc[0]["selected"]

    saved = r.save(tmp_path / "exported")
    loaded = ModelBundle.load(saved)
    X = binary_df.drop(columns="y")
    np.testing.assert_allclose(r.predict_proba(X), loaded.predict_proba(X))
    assert r.predict(X.head(7)).shape == (7,)


def test_multiclass_string_labels(multiclass_df, tmp_path):
    r = run(multiclass_df, tmp_path, task="classification")
    assert r.evaluation["task"] == "multiclass_classification"
    assert set(r.predict(multiclass_df.drop(columns="y"))) <= {"lo", "mid", "hi"}
    assert r.predict_proba(multiclass_df.drop(columns="y")).shape[1] == 3


def test_regression(regression_df, tmp_path):
    r = run(regression_df, tmp_path, metric="rmse")
    assert r.evaluation["test_metrics"]["r2"] > 0.9
    assert r.evaluation["selected_cv_score"] < r.evaluation["baseline_cv_score"]  # lower rmse
    lb = r.leaderboard()
    assert lb.iloc[0]["score_mean"] == lb[lb.status == "ok"]["score_mean"].min()


def test_csv_input_and_plan_without_training(binary_df, tmp_path):
    binary_df.to_csv(tmp_path / "d.csv", index=False)
    p = MLProject(tmp_path / "d.csv", "y", output_dir=tmp_path / "runs")
    plan = p.plan()
    assert not (tmp_path / "runs").exists()  # plan() trains/writes nothing
    assert p.profile is not None and "dummy" in plan.candidate_models
    r = p.run(plan)
    assert r.plan.plan_id == plan.plan_id


def test_approved_plan_restricts_models(binary_df, tmp_path):
    p = MLProject(binary_df, "y", output_dir=tmp_path)
    plan = p.plan()
    from dataclasses import replace

    r = p.run(replace(plan, candidate_models=("dummy", "logistic_regression")))
    assert {t.model_name for t in r.trials} == {"dummy", "logistic_regression"}


def test_group_and_temporal_runs(binary_df, tmp_path):
    df = binary_df.assign(g=np.arange(len(binary_df)) % 50, t=np.arange(len(binary_df)))
    assert run(df, tmp_path, group_column="g").evaluation["split_strategy"] == "group"
    assert run(df, tmp_path, time_column="t").evaluation["split_strategy"] == "temporal"


def test_blocked_run_records_state(binary_df, tmp_path):
    df = binary_df.assign(y=np.arange(len(binary_df)) % 5)  # ambiguous integer target
    with pytest.raises(BlockedError) as e:
        run(df, tmp_path)
    run_dir = e.value.details["run_dir"]
    kinds = [json.loads(l)["kind"] for line in open(f"{run_dir}/events.jsonl")]
    assert kinds[-1] == "state.BLOCKED"


def test_runs_do_not_collide(binary_df, tmp_path):
    a, b = run(binary_df, tmp_path), run(binary_df, tmp_path)
    assert a.run_dir != b.run_dir
    assert a.evaluation["split_fingerprint"] == b.evaluation["split_fingerprint"]
    assert a.evaluation["selected_cv_score"] == pytest.approx(b.evaluation["selected_cv_score"])
