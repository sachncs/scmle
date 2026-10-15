"""Resource limits in real runs."""

import dataclasses
import time

import pytest
from sklearn import dummy
import support

from mml.contracts import errors
from mml.contracts import models
from mml.contracts import tasks
from mml.core import limits
from mml.core import orchestrator as orchestrator_module
from mml.models import registry as registry_module


def run_with(frame, tmp_path, **resources):
    spec = tasks.ProjectSpec(
        data=tasks.DataSpec(uri="<in-memory>", target="y"),
        resources=tasks.ResourceSpec(**resources),
        search=support.NO_TUNING,
        strategies=support.NO_STRATEGIES,
        ensemble=support.NO_ENSEMBLE,
    )
    return orchestrator_module.Orchestrator(spec, frame).run(tmp_path)


def test_every_run_reports_how_limits_were_applied(binary_frame, tmp_path):
    report = run_with(
        binary_frame, tmp_path, time_budget_seconds=600
    ).evaluation["resource_limits"]
    assert report["time_budget"]["within_limit"] is True
    assert report["memory"]["observed_peak_gb"] > 0
    assert set(report) == {"time_budget", "cpu", "memory", "gpu"}


def test_cpu_limit_caps_the_worker_pool(binary_frame, tmp_path):
    result = run_with(
        binary_frame, tmp_path, time_budget_seconds=600, workers=4, cpu_limit=2
    )
    cpu = result.evaluation["resource_limits"]["cpu"]
    assert cpu["workers"] == 2 and cpu["threads_per_worker"] == 1


def test_memory_limit_stops_new_trials_but_keeps_the_first(
    binary_frame, tmp_path, monkeypatch
):
    monkeypatch.setattr(limits, "peak_memory_gb", lambda: 50.0)
    result = run_with(
        binary_frame, tmp_path, time_budget_seconds=600, memory_limit_gb=1.0
    )
    assert result.trials[0].status == "ok"
    skipped = [t for t in result.trials[1:] if t.status == "skipped"]
    assert skipped and all("memory limit" in t.error for t in skipped)
    assert (
        result.evaluation["resource_limits"]["memory"]["within_limit"] is False
    )


def test_required_gpu_blocks_with_an_explanation(binary_frame, tmp_path):
    with pytest.raises(errors.BlockedError, match="gpu"):
        run_with(
            binary_frame, tmp_path, time_budget_seconds=600, gpu="required"
        )


def test_disabled_gpu_is_accepted(binary_frame, tmp_path):
    result = run_with(
        binary_frame, tmp_path, time_budget_seconds=600, gpu="disabled"
    )
    assert result.evaluation["resource_limits"]["gpu"]["mode"] == "disabled"


def test_hard_enforcement_supervises_trials_end_to_end(binary_frame, tmp_path):
    result = run_with(
        binary_frame, tmp_path, time_budget_seconds=600, workers=2
    )
    assert result.state == "COMPLETED"
    assert all(t.status == "ok" for t in result.trials)
    report = result.evaluation["resource_limits"]
    assert "killed" in report["time_budget"]["enforcement"]


class SleepyEstimator(dummy.DummyClassifier):
    def fit(self, features, target):
        time.sleep(120)
        return super().fit(features, target)


class SleepyAdapter(models.ModelAdapter):
    name = "sleepy"
    family = "test"
    capabilities = models.Capabilities(
        frozenset({tasks.TaskType.BINARY}), cost=2
    )

    def build(self, task, seed, params=None):
        return SleepyEstimator()


def test_a_trial_past_the_hard_ceiling_is_killed_and_the_run_continues(
    binary_frame, tmp_path
):
    registry = registry_module.ModelRegistry.with_builtins()
    registry.register(SleepyAdapter())
    project = support.make_project(
        binary_frame,
        "y",
        time_budget=6,
        enforcement="hard",
        output_dir=tmp_path,
    )
    project.orchestrator.registry = registry
    plan = dataclasses.replace(
        project.plan(), candidate_models=("dummy", "sleepy")
    )
    started = time.monotonic()
    result = project.run(plan)
    assert time.monotonic() - started < 100
    assert result.state == "COMPLETED"
    sleepy = [t for t in result.trials if t.model_name == "sleepy"]
    assert sleepy and sleepy[0].status == "failed"
    assert "hard time ceiling" in sleepy[0].error
    assert result.selected_model != "sleepy"
