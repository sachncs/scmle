"""Imbalance and feature strategies, evaluated on evidence."""

import importlib.util
import json

import numpy
import pandas
import pytest
import support

import scmle.api.project
import scmle.artifacts.bundle
from scmle.contracts import tasks

STRATEGIES = tasks.Strategies(top_models=2)
NO_SEARCH = tasks.Search(top_models=0)


def run(frame, tmp_path, strategies=STRATEGIES, search=NO_SEARCH, **kwargs):
    project = support.make_project(
        frame,
        "y",
        budget=600,
        directory=tmp_path,
        search=search,
        strategies=strategies,
        **kwargs,
    )
    return project.run()


def decision_log(result):
    return json.loads(
        (result.directory / "diagnostics" / "decision_log.json").read_text()
    )


def test_imbalanced_data_triggers_strategy_trials(imbalanced_frame, tmp_path):
    ratio = (imbalanced_frame.y == 0).sum() / (imbalanced_frame.y == 1).sum()
    assert ratio > 3
    result = run(imbalanced_frame, tmp_path)
    tried = {t.strategy for t in result.trials if t.strategy != "default"}
    assert {"class_weight", "random_oversample", "random_undersample"} <= tried
    ids = [t.identifier for t in result.trials]
    assert len(ids) == len(set(ids))
    log = decision_log(result)
    verdicts = [o for o in log["outcomes"] if "@" in o["technique"]]
    assert verdicts and all(
        " vs default " in o["reason"] or "not applicable" in o["reason"]
        for o in verdicts
    )
    assert log["selection"]["model"] == result.model


def test_every_strategy_is_compared_on_the_original_validation_folds(
    imbalanced_frame, tmp_path
):
    result = run(imbalanced_frame, tmp_path)
    fold_counts = {
        len(t.folds) for t in result.trials if t.status == "ok"
    }
    assert fold_counts == {5}
    default = next(
        t
        for t in result.trials
        if t.model == "logistic_regression" and t.strategy == "default"
    )
    weighted = next(
        t
        for t in result.trials
        if t.model == "logistic_regression"
        and t.strategy == "class_weight"
    )
    assert default.folds != weighted.folds


def test_adopted_strategy_is_used_for_the_final_model(
    imbalanced_frame, tmp_path
):
    result = run(imbalanced_frame, tmp_path)
    assert (
        result.evaluation["selected"]["strategy"] == result.bundle.info.strategy
    )
    selected = next(
        t
        for t in result.trials
        if t.model == result.model
        and t.strategy == result.evaluation["selected"]["strategy"]
        and t.params == result.evaluation["selected"]["params"]
    )
    assert selected.score == pytest.approx(
        result.evaluation["selected"]["score"]
    )
    features = imbalanced_frame.drop(columns="y")
    loaded = scmle.artifacts.bundle.Bundle.load(
        result.save(tmp_path / "exported"), trusted=True
    )
    numpy.testing.assert_allclose(
        result.predict_proba(features), loaded.predict_proba(features)
    )
    assert loaded.info.strategy == result.evaluation["selected"]["strategy"]


def test_strategies_can_be_forced_to_win_and_are_then_used(
    imbalanced_frame, tmp_path
):
    only = tasks.Strategies(names=["random_undersample"], top_models=1)
    result = run(
        imbalanced_frame, tmp_path, strategies=only, metric="balanced_accuracy"
    )
    assert {t.strategy for t in result.trials} <= {
        "default",
        "random_undersample",
    }
    best_under = max(
        t.score
        for t in result.trials
        if t.strategy == "random_undersample" and t.status == "ok"
    )
    best_default = max(
        t.score
        for t in result.trials
        if t.strategy == "default"
        and t.model != "dummy"
        and t.status == "ok"
    )
    assert (
        best_under > best_default
    )  # undersampling helps balanced accuracy here, by a little
    # It is used only when the shared folds make it credibly better.
    verdicts = {
        o["technique"]: o["action"] for o in decision_log(result)["outcomes"]
    }
    adopted = any(
        technique.startswith("random_undersample@") and action == "include"
        for technique, action in verdicts.items()
    )
    chosen = result.evaluation["selected"]["strategy"]
    assert (chosen == "random_undersample") is adopted


def test_balanced_data_skips_imbalance_strategies_with_logged_reasons(
    binary_frame, tmp_path
):
    frame = binary_frame.assign(y=numpy.arange(len(binary_frame)) % 2)
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
        if t.model == "logistic_regression"
    }
    assert {"log_skewed", "interactions"} <= linear
    trees = {
        t.strategy for t in result.trials if t.model == "random_forest"
    }
    assert not trees & {"log_skewed", "interactions"}


def test_tuning_runs_under_the_winning_strategy(imbalanced_frame, tmp_path):
    search = tasks.Search(top_models=1, configs_per_model=4)
    result = run(imbalanced_frame, tmp_path, search=search)
    tuned = [t for t in result.trials if t.params]
    assert tuned and len({t.strategy for t in tuned}) == 1
    ids = [t.identifier for t in result.trials]
    assert len(ids) == len(set(ids))


def test_strategy_run_resumes_without_recomputation(
    imbalanced_frame, tmp_path, monkeypatch
):
    from scmle.pipeline import trials

    search = tasks.Search(top_models=1, configs_per_model=2)
    original = run(imbalanced_frame, tmp_path, search=search)
    directory = support.interrupt(
        support.make_project(
            imbalanced_frame,
            "y",
            budget=600,
            directory=tmp_path,
            search=search,
            strategies=STRATEGIES,
        )
    )
    calls = []
    real = trials.evaluate
    monkeypatch.setattr(
        trials,
        "evaluate",
        lambda p: calls.append(p.job.identifier) or real(p),
    )
    resumed = scmle.api.project.Project.resume(directory, data=imbalanced_frame)
    assert calls == []
    assert (
        resumed.evaluation["selected"]["strategy"]
        == original.evaluation["selected"]["strategy"]
    )
    assert [(t.identifier, t.strategy) for t in resumed.trials] == [
        (t.identifier, t.strategy) for t in original.trials
    ]


def test_leaderboard_marks_exactly_the_selected_trial(
    imbalanced_frame, tmp_path
):
    search = tasks.Search(top_models=1, configs_per_model=4)
    result = run(imbalanced_frame, tmp_path, search=search)
    board = result.leaderboard()
    assert board["selected"].sum() == 1
    winner = board[board["selected"]].iloc[0]
    assert winner["identifier"] == result.evaluation["selected"]["trial"]
    assert winner["model"] == result.model
    assert winner["score"] == pytest.approx(
        result.evaluation["selected"]["score"]
    )


def numeric_imbalanced(rows=900, seed=11):
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame(
        {"a": rng.normal(size=rows), "b": rng.normal(size=rows)}
    )
    frame["y"] = (rng.random(rows) < 0.07).astype(int)
    frame.loc[frame.y == 1, ["a", "b"]] += 1.2
    return frame


def test_smote_variants_are_evaluated_on_the_original_folds_when_installed(
    tmp_path,
):
    pytest.importorskip("imblearn")
    only = tasks.Strategies(
        names=["smote", "borderline_smote", "adasyn"], top_models=1
    )
    result = run(
        numeric_imbalanced(), tmp_path, strategies=only, metric="pr_auc"
    )
    tried = {t.strategy for t in result.trials if t.strategy != "default"}
    assert tried == {"smote", "borderline_smote", "adasyn"}
    assert {len(t.folds) for t in result.trials if t.status == "ok"} == {
        5
    }
    features = numeric_imbalanced().drop(columns="y")
    assert len(result.predict(features)) == len(features)


def test_missing_sampling_extra_is_explained_in_the_plan(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    plan = scmle.api.project.Project(
        numeric_imbalanced(),
        "y",
        strategies=tasks.Strategies(names=["smote"]),
    ).plan()
    decision = next(d for d in plan.decisions if d.technique == "smote")
    assert decision.action == "reject" and "sampling" in decision.reason


def test_jitter_augmentation_must_be_requested_explicitly(tmp_path):
    frame = numeric_imbalanced()
    default_plan = scmle.api.project.Project(frame, "y").plan()
    assert not any(
        d.technique == "jitter_augment" for d in default_plan.decisions
    )
    opted = tasks.Strategies(names=["jitter_augment"], top_models=1)
    result = run(frame, tmp_path, strategies=opted, metric="pr_auc")
    assert any(t.strategy == "jitter_augment" for t in result.trials)
    verdicts = json.loads(
        (result.directory / "diagnostics" / "decision_log.json").read_text()
    )["outcomes"]
    assert any(o["technique"].startswith("jitter_augment@") for o in verdicts)


def test_copula_augmentation_is_opt_in_and_judged_on_original_folds(tmp_path):
    frame = numeric_imbalanced()
    assert not any(
        d.technique == "copula_augment"
        for d in scmle.api.project.Project(frame, "y").plan().decisions
    )
    opted = tasks.Strategies(names=["copula_augment"], top_models=1)
    result = run(frame, tmp_path, strategies=opted, metric="pr_auc")
    tried = [t for t in result.trials if t.strategy == "copula_augment"]
    assert tried and tried[0].status == "ok" and len(tried[0].folds) == 5
    verdicts = json.loads(
        (result.directory / "diagnostics" / "decision_log.json").read_text()
    )["outcomes"]
    assert any(o["technique"].startswith("copula_augment@") for o in verdicts)
    features = frame.drop(columns="y")
    assert len(result.predict(features)) == len(features)
