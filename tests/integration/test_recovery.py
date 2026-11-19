"""Failure isolation, cancellation, budget handling and extensibility."""

import json
import pathlib

import pytest
from sklearn import dummy
import support

from scmle.contracts import errors
from scmle.contracts import models
from scmle.contracts import tasks
import scmle.models.registry
from scmle.pipeline import orchestrator
import scmle.pipeline.stages
from scmle.pipeline.stages import selecting

BINARY = frozenset({tasks.Task.BINARY})


class ExplodingEstimator(dummy.DummyClassifier):
    def fit(self, features, target):
        raise RuntimeError("boom")


class ExplodingAdapter(models.Model):
    name = "exploding"
    family = "test"
    capabilities = models.Capabilities(BINARY, cost=1)

    def build(self, task, seed, params=None):
        return ExplodingEstimator()


def read_events(run_dir):
    with pathlib.Path(f"{run_dir}/events.jsonl").open(
        encoding="utf-8"
    ) as handle:
        return [json.loads(line) for line in handle]


def project_with(adapter, frame, tmp_path, **kwargs):
    registry = scmle.models.registry.Catalog.stock()
    registry.register(adapter)
    project = support.make_project(frame, "y", output_dir=tmp_path, **kwargs)
    project.orchestrator.registry = registry
    return project


def test_failed_trial_is_isolated(binary_frame, tmp_path):
    result = project_with(ExplodingAdapter(), binary_frame, tmp_path).run()
    failed = next(t for t in result.trials if t.model_name == "exploding")
    assert failed.status == "failed" and "boom" in failed.error
    assert result.selected_model != "exploding"
    assert failed.trial_id in result.leaderboard()["identifier"].tolist()
    assert "trial.failed" in [e["kind"] for e in read_events(result.run_dir)]


def test_all_trials_failing_ends_in_failed_state(binary_frame, tmp_path):
    registry = scmle.models.registry.Catalog((ExplodingAdapter(),))
    project = support.make_project(binary_frame, "y", output_dir=tmp_path)
    project.orchestrator.registry = registry
    with pytest.raises(errors.ScmleError, match="all trials failed") as caught:
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

        def build(self, task, seed, params=None):
            return CancellingEstimator()

    project = project_with(CancellingAdapter(), binary_frame, tmp_path)
    holder["project"] = project
    with pytest.raises(errors.CancelledError) as caught:
        project.run()
    run_dir = caught.value.details["run_dir"]
    assert any(
        path.name.endswith(".json")
        for path in pathlib.Path(f"{run_dir}/trials").iterdir()
    )
    assert read_events(run_dir)[-1]["kind"] == "state.CANCELLED"


def test_a_budget_below_the_minimum_run_is_refused_before_work(
    binary_frame, tmp_path
):
    # The first trial no longer runs past a spent budget: a run whose
    # minimum cannot fit is refused, naming the smallest budget that fits.
    with pytest.raises(errors.BlockedError, match="smallest budget") as caught:
        support.make_project(
            binary_frame, "y", time_budget=1, output_dir=tmp_path
        ).run()
    smallest = caught.value.details["smallest_budget_seconds"]
    result = support.make_project(
        binary_frame, "y", time_budget=smallest, output_dir=tmp_path / "b"
    ).run()
    assert result.trials[0].status == "ok" and result.state == "COMPLETED"
    skipped = [t for t in result.trials if t.status == "skipped"]
    assert all("trial window" in t.error for t in skipped)


def test_custom_stage_sequence_is_polymorphic(binary_frame, tmp_path):
    class Recording(scmle.pipeline.stages.PackageStage):
        calls = []

        def execute(self, board):
            choice = board.get(selecting.Choice)
            Recording.calls.append(choice.selected.model_name)
            return super().execute(board)

    stages = (*scmle.pipeline.stages.default_stages()[:-1], Recording())
    spec = support.make_project(binary_frame, "y").spec
    result = orchestrator.Orchestrator(spec, binary_frame, stages=stages).run(
        tmp_path
    )
    assert Recording.calls == [result.selected_model]


def test_failed_and_blocked_runs_list_their_recoverable_trials(
    binary_frame, tmp_path
):
    holder = {}

    class CancellingEstimator(dummy.DummyClassifier):
        def fit(self, features, target):
            holder["project"].cancel()
            return super().fit(features, target)

    class CancellingAdapter(ExplodingAdapter):
        name = "late_canceller"

        def build(self, task, seed, params=None):
            return CancellingEstimator()

    project = project_with(CancellingAdapter(), binary_frame, tmp_path)
    holder["project"] = project
    with pytest.raises(errors.CancelledError) as caught:
        project.run()
    recoverable = caught.value.details["recoverable_trials"]
    assert recoverable and all(
        name.startswith("trial_") for name in recoverable
    )
    assert len(recoverable) == len(set(recoverable))

    ambiguous = binary_frame.assign(y=range(len(binary_frame)))
    ambiguous["y"] = ambiguous["y"] % 5
    with pytest.raises(errors.BlockedError) as blocked:
        support.make_project(ambiguous, "y", output_dir=tmp_path / "b").run()
    assert blocked.value.details["recoverable_trials"] == []
