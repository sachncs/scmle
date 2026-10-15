"""Failure isolation, cancellation, budget handling and extensibility."""

import json
import os

import pytest
from sklearn import dummy

import mml
from mml.contracts import errors
from mml.contracts import models
from mml.contracts import tasks
from mml.core import orchestrator as orchestrator_module
from mml.core import stages as stages_module
from mml.models import registry as registry_module

BINARY = frozenset({tasks.TaskType.BINARY})


class ExplodingEstimator(dummy.DummyClassifier):
    def fit(self, features, target):
        raise RuntimeError("boom")


class ExplodingAdapter(models.ModelAdapter):
    name = "exploding"
    family = "test"
    capabilities = models.Capabilities(BINARY, cost=1)

    def build(self, task, seed):
        return ExplodingEstimator()


def read_events(run_dir):
    with open(f"{run_dir}/events.jsonl") as handle:
        return [json.loads(line) for line in handle]


def project_with(adapter, frame, tmp_path, **kwargs):
    registry = registry_module.ModelRegistry.with_builtins()
    registry.register(adapter)
    project = mml.MLProject(frame, "y", output_dir=tmp_path, **kwargs)
    project.orchestrator.registry = registry
    return project


def test_failed_trial_is_isolated(binary_frame, tmp_path):
    result = project_with(ExplodingAdapter(), binary_frame, tmp_path).run()
    failed = next(t for t in result.trials if t.model_name == "exploding")
    assert failed.status == "failed" and "boom" in failed.error
    assert result.selected_model != "exploding"
    assert failed.trial_id in result.leaderboard()["trial_id"].tolist()
    assert "trial.failed" in [e["kind"] for e in read_events(result.run_dir)]


def test_all_trials_failing_ends_in_failed_state(binary_frame, tmp_path):
    registry = registry_module.ModelRegistry((ExplodingAdapter(),))
    project = mml.MLProject(binary_frame, "y", output_dir=tmp_path)
    project.orchestrator.registry = registry
    with pytest.raises(
        errors.MagicMLError, match="all trials failed"
    ) as caught:
        project.run()
    assert (
        read_events(caught.value.details["run_dir"])[-1]["kind"]
        == "state.FAILED"
    )


def test_cancellation_keeps_completed_trials(binary_frame, tmp_path):
    holder = {}

    class CancellingEstimator(dummy.DummyClassifier):
        def fit(self, features, target):
            holder["project"].cancel()
            return super().fit(features, target)

    class CancellingAdapter(ExplodingAdapter):
        name = "canceller"

        def build(self, task, seed):
            return CancellingEstimator()

    project = project_with(CancellingAdapter(), binary_frame, tmp_path)
    holder["project"] = project
    with pytest.raises(errors.CancelledError) as caught:
        project.run()
    run_dir = caught.value.details["run_dir"]
    assert any(
        name.endswith(".json") for name in os.listdir(f"{run_dir}/trials")
    )
    assert read_events(run_dir)[-1]["kind"] == "state.CANCELLED"


def test_budget_exhaustion_skips_later_trials(binary_frame, tmp_path):
    result = mml.MLProject(
        binary_frame, "y", time_budget=1, output_dir=tmp_path
    ).run()
    assert result.trials[0].status == "ok" and result.state == "COMPLETED"
    skipped = [t for t in result.trials if t.status == "skipped"]
    assert all("budget" in t.error for t in skipped)


def test_custom_stage_sequence_is_polymorphic(binary_frame, tmp_path):
    class Recording(stages_module.PackageStage):
        calls = []

        def execute(self, context):
            Recording.calls.append(context.selected.model_name)
            super().execute(context)

    stages = (*stages_module.default_stages()[:-1], Recording())
    spec = mml.MLProject(binary_frame, "y").spec
    result = orchestrator_module.Orchestrator(
        spec, binary_frame, stages=stages
    ).run(tmp_path)
    assert Recording.calls == [result.selected_model]
