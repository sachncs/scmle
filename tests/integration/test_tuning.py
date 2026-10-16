"""Hyperparameter search with pruning, end to end."""

import json

import numpy as np
import pytest
import support

import mml
from mml.contracts import tasks

SEARCH = tasks.SearchSpec(top_models=1, configs_per_model=6, prune_fraction=0.5)


def run(frame, tmp_path, search=SEARCH, **kwargs):
    project = support.make_project(
        frame,
        "y",
        time_budget=300,
        output_dir=tmp_path,
        search=search,
        **kwargs,
    )
    return project.run()


def test_tuning_adds_pruned_and_completed_trials(binary_frame, tmp_path):
    result = run(binary_frame, tmp_path)
    tuned = [t for t in result.trials if t.params]
    pruned = [t for t in tuned if t.status == "pruned"]
    completed = [t for t in tuned if t.status == "ok"]
    assert len(pruned) == 3 and len(completed) == 3
    assert len({t.model_name for t in tuned}) == 1
    assert all(len(t.fold_scores) == 1 for t in pruned)
    assert all(len(t.fold_scores) > 1 for t in completed)
    trial_ids = [t.trial_id for t in result.trials]
    assert len(trial_ids) == len(set(trial_ids))
    assert (result.run_dir / "trials" / f"{tuned[0].trial_id}.json").exists()


def test_selection_can_pick_a_tuned_configuration(regression_frame, tmp_path):
    result = run(regression_frame, tmp_path, task="regression")
    best_default = min(
        t.score_mean
        for t in result.trials
        if t.status == "ok" and not t.params and t.model_name != "dummy"
    )
    assert result.evaluation["selected_cv_score"] <= best_default + 1e-12
    selected = result.evaluation["selected_params"]
    assert result.bundle.info.model_params == selected


def test_params_survive_in_leaderboard_and_bundle(binary_frame, tmp_path):
    result = run(binary_frame, tmp_path)
    board = result.leaderboard()
    assert (
        board["params"]
        .map(lambda value: isinstance(json.loads(value), dict))
        .all()
    )
    loaded = mml.ModelBundle.load(
        result.save(tmp_path / "exported"), trusted=True
    )
    assert loaded.info.model_params == result.bundle.info.model_params
    features = binary_frame.drop(columns="y")
    np.testing.assert_allclose(
        result.predict_proba(features), loaded.predict_proba(features)
    )


def test_tuning_can_be_disabled(binary_frame, tmp_path):
    result = run(binary_frame, tmp_path, search=tasks.SearchSpec(top_models=0))
    assert not any(t.params for t in result.trials)
    assert any(
        d.technique == "hyperparameter_tuning" and d.action == "defer"
        for d in result.plan.decisions
    )


def test_plan_documents_the_search(binary_frame):
    plan = support.make_project(binary_frame, "y", search=SEARCH).plan()
    decision = next(
        d for d in plan.decisions if d.technique == "hyperparameter_tuning"
    )
    assert decision.action == "include" and "random search" in decision.reason


def test_search_stops_when_budget_is_exhausted(binary_frame, tmp_path):
    result = support.make_project(
        binary_frame, "y", time_budget=1, output_dir=tmp_path, search=SEARCH
    ).run()
    assert result.state == "COMPLETED"
    assert sum(1 for t in result.trials if t.params) <= 6


def test_tuning_is_reproducible(binary_frame, tmp_path):
    first, second = run(binary_frame, tmp_path), run(binary_frame, tmp_path)
    assert [t.params for t in first.trials] == [t.params for t in second.trials]
    assert (
        first.evaluation["selected_params"]
        == second.evaluation["selected_params"]
    )


def test_optuna_backend(binary_frame, tmp_path):
    pytest.importorskip("optuna")
    search = tasks.SearchSpec(
        top_models=1, configs_per_model=6, optimizer="optuna"
    )
    result = run(binary_frame, tmp_path, search=search)
    assert any(t.params for t in result.trials) and result.state == "COMPLETED"


def test_cli_exposes_search_options(binary_frame, tmp_path, caplog):
    from mml import cli

    path = tmp_path / "d.csv"
    binary_frame.to_csv(path, index=False)
    code = cli.main(
        [
            "run",
            str(path),
            "--target",
            "y",
            "--output-dir",
            str(tmp_path / "r"),
            "--top-models",
            "1",
            "--configs-per-model",
            "2",
            "--time-budget",
            "300",
        ]
    )
    assert code == 0 and "selected" in caplog.text
