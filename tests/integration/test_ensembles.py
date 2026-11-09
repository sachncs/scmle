"""Stacking ensembles, end to end."""

import numpy
import pytest
import support

import scmle
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.models import ensembles
import scmle.models.registry

ENSEMBLE = tasks.EnsembleSpec(size=3)


def run(frame, tmp_path, **kwargs):
    return support.make_project(
        frame,
        "y",
        time_budget=600,
        output_dir=tmp_path,
        ensemble=ENSEMBLE,
        **kwargs,
    ).run()


def test_stack_is_evaluated_against_its_members(binary_frame, tmp_path):
    result = run(binary_frame, tmp_path)
    stacks = [t for t in result.trials if t.model_name == "stacking"]
    assert len(stacks) == 1 and stacks[0].status == "ok"
    members = stacks[0].params["members"]
    assert 2 <= len(members) <= 3
    assert len({m["model"] for m in members}) == len(members)
    assert "dummy" not in {m["model"] for m in members}
    log = (result.run_dir / "diagnostics" / "decision_log.json").read_text()
    assert "stacking of" in log and "blending of" in log
    assert "best member" in log
    blends = [t for t in result.trials if t.model_name == "blending"]
    assert len(blends) == 1 and blends[0].status == "ok"
    assert blends[0].params["members"] == members


def test_stack_is_cross_fitted_inside_each_training_fold(binary_frame):
    registry = scmle.models.registry.ModelRegistry.with_builtins()
    adapter = registry.get("stacking")
    params = {
        "members": [
            {"model": "logistic_regression"},
            {"model": "random_forest"},
        ]
    }
    stack = adapter.build(tasks.TaskType.BINARY, 0, params)
    assert stack.cv == ensembles.STACK_FOLDS
    assert [pair[0].split("_", 1)[1] for pair in stack.estimators] == [
        "logistic_regression",
        "random_forest",
    ]


def test_regression_stack(regression_frame, tmp_path):
    result = run(regression_frame, tmp_path, task="regression")
    stacks = [t for t in result.trials if t.model_name == "stacking"]
    assert stacks and stacks[0].status == "ok"


def test_a_winning_stack_is_selected_refit_and_packaged(
    regression_frame, tmp_path
):
    result = run(regression_frame, tmp_path, task="regression")
    stack = next(t for t in result.trials if t.model_name == "stacking")
    best_other = min(
        t.score_mean
        for t in result.trials
        if t.status == "ok" and t.model_name != "stacking"
    )
    if stack.score_mean < best_other:
        assert result.selected_model == "stacking"
        assert result.bundle.info.model_params["members"]
        features = regression_frame.drop(columns="y")
        loaded = scmle.ModelBundle.load(
            result.save(tmp_path / "bundle"), trusted=True
        )
        numpy.testing.assert_allclose(
            result.predict(features), loaded.predict(features)
        )


def test_stacking_is_never_a_screening_candidate(binary_frame):
    registry = scmle.models.registry.ModelRegistry.with_builtins()
    names = {a.name for a in registry.candidates(tasks.TaskType.BINARY)[0]}
    assert "stacking" not in names
    rejected = registry.candidates(tasks.TaskType.BINARY)[1]
    assert "stacking" not in rejected


def test_stacking_needs_two_members():
    registry = scmle.models.registry.ModelRegistry.with_builtins()
    with pytest.raises(errors.ConfigError, match="at least two"):
        registry.get("stacking").build(
            tasks.TaskType.BINARY, 0, {"members": [{"model": "ridge"}]}
        )


def test_ensemble_can_be_disabled_and_is_planned(binary_frame, tmp_path):
    off = support.make_project(binary_frame, "y", output_dir=tmp_path).run()
    assert not any(t.model_name == "stacking" for t in off.trials)
    plan = scmle.MLProject(binary_frame, "y").plan()
    decision = next(d for d in plan.decisions if d.technique == "stacking")
    assert decision.action == "include"


def test_stack_resumes_without_recomputation(
    binary_frame, tmp_path, monkeypatch
):
    import shutil

    from scmle.experimentation import trials

    original = run(binary_frame, tmp_path)
    (original.run_dir / "evaluation.json").unlink()
    shutil.rmtree(original.run_dir / "model")
    calls = []
    real = trials.evaluate_payload
    monkeypatch.setattr(
        trials,
        "evaluate_payload",
        lambda p: calls.append(p.job.trial_id) or real(p),
    )
    resumed = scmle.MLProject.resume(original.run_dir, data=binary_frame)
    assert calls == [] and resumed.state == "COMPLETED"
