"""Imbalance and feature strategies, evaluated on evidence."""

import json

import numpy as np
import pytest
import support

import mml
from mml.contracts import tasks

STRATEGIES = tasks.StrategySpec(top_models=2)
NO_SEARCH = tasks.SearchSpec(top_models=0)


def run(frame, tmp_path, strategies=STRATEGIES, search=NO_SEARCH, **kwargs):
    project = support.make_project(
        frame,
        "y",
        time_budget=600,
        output_dir=tmp_path,
        search=search,
        strategies=strategies,
        **kwargs,
    )
    return project.run()


def decision_log(result):
    return json.loads(
        (result.run_dir / "diagnostics" / "decision_log.json").read_text()
    )


def test_imbalanced_data_triggers_strategy_trials(imbalanced_frame, tmp_path):
    ratio = (imbalanced_frame.y == 0).sum() / (imbalanced_frame.y == 1).sum()
    assert ratio > 3
    result = run(imbalanced_frame, tmp_path)
    tried = {t.strategy for t in result.trials if t.strategy != "default"}
    assert {"class_weight", "random_oversample", "random_undersample"} <= tried
    ids = [t.trial_id for t in result.trials]
    assert len(ids) == len(set(ids))
    log = decision_log(result)
    verdicts = [o for o in log["outcomes"] if "@" in o["technique"]]
    assert verdicts and all(
        " vs default " in o["reason"] or "not applicable" in o["reason"]
        for o in verdicts
    )
    assert log["selection"]["model"] == result.selected_model


def test_every_strategy_is_compared_on_the_original_validation_folds(
    imbalanced_frame, tmp_path
):
    result = run(imbalanced_frame, tmp_path)
    fold_counts = {
        len(t.fold_scores) for t in result.trials if t.status == "ok"
    }
    assert fold_counts == {5}
    default = next(
        t
        for t in result.trials
        if t.model_name == "logistic_regression" and t.strategy == "default"
    )
    weighted = next(
        t
        for t in result.trials
        if t.model_name == "logistic_regression"
        and t.strategy == "class_weight"
    )
    assert default.fold_scores != weighted.fold_scores


def test_adopted_strategy_is_used_for_the_final_model(
    imbalanced_frame, tmp_path
):
    result = run(imbalanced_frame, tmp_path)
    assert result.evaluation["selected_strategy"] == result.bundle.info.strategy
    selected = next(
        t
        for t in result.trials
        if t.model_name == result.selected_model
        and t.strategy == result.evaluation["selected_strategy"]
        and t.params == result.evaluation["selected_params"]
    )
    assert selected.score_mean == pytest.approx(
        result.evaluation["selected_cv_score"]
    )
    features = imbalanced_frame.drop(columns="y")
    loaded = mml.ModelBundle.load(
        result.save(tmp_path / "exported"), trusted=True
    )
    np.testing.assert_allclose(
        result.predict_proba(features), loaded.predict_proba(features)
    )
    assert loaded.info.strategy == result.evaluation["selected_strategy"]


def test_strategies_can_be_forced_to_win_and_are_then_used(
    imbalanced_frame, tmp_path
):
    only = tasks.StrategySpec(names=["random_undersample"], top_models=1)
    result = run(
        imbalanced_frame, tmp_path, strategies=only, metric="balanced_accuracy"
    )
    assert {t.strategy for t in result.trials} <= {
        "default",
        "random_undersample",
    }
    best_under = max(
        t.score_mean
        for t in result.trials
        if t.strategy == "random_undersample" and t.status == "ok"
    )
    best_default = max(
        t.score_mean
        for t in result.trials
        if t.strategy == "default"
        and t.model_name != "dummy"
        and t.status == "ok"
    )
    assert (
        best_under > best_default
    )  # undersampling helps balanced accuracy here
    assert result.evaluation["selected_strategy"] == "random_undersample"


def test_balanced_data_skips_imbalance_strategies_with_logged_reasons(
    binary_frame, tmp_path
):
    frame = binary_frame.assign(y=np.arange(len(binary_frame)) % 2)
    result = run(frame, tmp_path)
    tried = {t.strategy for t in result.trials}
    assert not tried & {
        "class_weight",
        "random_oversample",
        "random_undersample",
    }
    rejected = [
        o for o in decision_log(result)["outcomes"] if o["action"] == "reject"
    ]
    assert any("balanced enough" in o["reason"] for o in rejected)


def test_feature_strategies_apply_to_skewed_linear_models(
    imbalanced_frame, tmp_path
):
    result = run(imbalanced_frame, tmp_path)
    linear = {
        t.strategy
        for t in result.trials
        if t.model_name == "logistic_regression"
    }
    assert {"log_skewed", "interactions"} <= linear
    trees = {
        t.strategy for t in result.trials if t.model_name == "random_forest"
    }
    assert not trees & {"log_skewed", "interactions"}


def test_tuning_runs_under_the_winning_strategy(imbalanced_frame, tmp_path):
    search = tasks.SearchSpec(top_models=1, configs_per_model=4)
    result = run(imbalanced_frame, tmp_path, search=search)
    tuned = [t for t in result.trials if t.params]
    assert tuned and len({t.strategy for t in tuned}) == 1
    ids = [t.trial_id for t in result.trials]
    assert len(ids) == len(set(ids))


def test_strategy_run_resumes_without_recomputation(
    imbalanced_frame, tmp_path, monkeypatch
):
    import shutil

    from mml.core import trials as trials_module

    original = run(
        imbalanced_frame,
        tmp_path,
        search=tasks.SearchSpec(top_models=1, configs_per_model=2),
    )
    (original.run_dir / "evaluation.json").unlink()
    shutil.rmtree(original.run_dir / "model")
    calls = []
    real = trials_module.evaluate_payload
    monkeypatch.setattr(
        trials_module,
        "evaluate_payload",
        lambda p: calls.append(p.job.trial_id) or real(p),
    )
    resumed = mml.MLProject.resume(original.run_dir, data=imbalanced_frame)
    assert calls == []
    assert (
        resumed.evaluation["selected_strategy"]
        == original.evaluation["selected_strategy"]
    )
    assert [(t.trial_id, t.strategy) for t in resumed.trials] == [
        (t.trial_id, t.strategy) for t in original.trials
    ]


def test_leaderboard_marks_exactly_the_selected_trial(
    imbalanced_frame, tmp_path
):
    search = tasks.SearchSpec(top_models=1, configs_per_model=4)
    result = run(imbalanced_frame, tmp_path, search=search)
    board = result.leaderboard()
    assert board["selected"].sum() == 1
    winner = board[board["selected"]].iloc[0]
    assert winner["trial_id"] == result.evaluation["selected_trial_id"]
    assert winner["model_name"] == result.selected_model
    assert winner["score_mean"] == pytest.approx(
        result.evaluation["selected_cv_score"]
    )
