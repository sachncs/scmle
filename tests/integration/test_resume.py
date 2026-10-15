"""Resume, checkpoint reuse and experiment tracking."""

import dataclasses
import json
import pathlib
import shutil

import pytest
from sklearn import dummy
import support

import mml
from mml.contracts import errors
from mml.contracts import models
from mml.contracts import tasks
from mml.core import orchestrator as orchestrator_module
from mml.core import trials as trials_module
from mml.models import registry as registry_module
from mml.tracking import stores

BINARY = frozenset({tasks.TaskType.BINARY})


class Counting(dummy.DummyClassifier):
    fits = 0

    def fit(self, features, target):
        type(self).fits += 1
        return super().fit(features, target)


class Canceller(Counting):
    fits = 0
    orchestrator = None
    armed = False

    def fit(self, features, target):
        if type(self).armed:
            type(self).orchestrator.cancel()
        return super().fit(features, target)


class CountedAdapter(models.ModelAdapter):
    name = "counted"
    family = "test"
    capabilities = models.Capabilities(BINARY, cost=2)

    def build(self, task, seed, params=None):
        return Counting()


class CancellerAdapter(CountedAdapter):
    name = "canceller"

    def build(self, task, seed, params=None):
        return Canceller()


def read_events(run_dir):
    with open(f"{run_dir}/events.jsonl") as handle:
        return [json.loads(line) for line in handle]


def test_cancelled_run_resumes_and_reuses_completed_trials(
    binary_frame, tmp_path
):
    counting, canceller = Counting, Canceller
    counting.fits = canceller.fits = 0
    canceller.armed = True
    registry = registry_module.ModelRegistry.with_builtins()
    registry.register(CountedAdapter())
    registry.register(CancellerAdapter())
    project = support.make_project(binary_frame, "y", output_dir=tmp_path)
    project.orchestrator.registry = registry
    canceller.orchestrator = project.orchestrator
    plan = dataclasses.replace(
        project.plan(), candidate_models=("dummy", "canceller", "counted")
    )

    with pytest.raises(errors.CancelledError) as caught:
        project.run(plan)
    run_dir = caught.value.details["run_dir"]
    assert canceller.fits == 5 and counting.fits == 0
    saved = sorted(p.name for p in (pathlib.Path(run_dir) / "trials").iterdir())
    assert saved == ["trial_001.json", "trial_002.json"]

    canceller.armed = False
    resumed = orchestrator_module.Orchestrator.for_run_directory(
        run_dir, frame=binary_frame, registry=registry
    ).resume(run_dir)
    assert resumed.state == "COMPLETED"
    # 5 CV fits were reused; the single new fit is the final refit of the
    # selected model (ties go to the alphabetically first equal-cost model).
    assert resumed.selected_model == "canceller"
    assert canceller.fits == 5 + 1, "the cancelled run's trial must be reused"
    assert counting.fits >= 5, "the unfinished trial must be computed"
    assert [t.model_name for t in resumed.trials] == [
        "dummy",
        "canceller",
        "counted",
    ]

    events = read_events(run_dir)
    kinds = [e["kind"] for e in events]
    assert "state.CANCELLED" in kinds and kinds[-1] == "state.COMPLETED"
    created = [e["reason"] for e in events if e["kind"] == "state.CREATED"]
    assert created == ["run created", "run resumed"]


def test_resume_of_tuned_run_recomputes_nothing(
    binary_frame, tmp_path, monkeypatch
):
    search = tasks.SearchSpec(top_models=1, configs_per_model=4)
    original = support.make_project(
        binary_frame, "y", time_budget=300, output_dir=tmp_path, search=search
    ).run()
    # Simulate a crash after the last trial: drop everything after TRAINING.
    (original.run_dir / "evaluation.json").unlink()
    shutil.rmtree(original.run_dir / "model")
    calls = []
    real = trials_module.evaluate_payload
    monkeypatch.setattr(
        trials_module,
        "evaluate_payload",
        lambda *a, **k: calls.append(1) or real(*a, **k),
    )
    resumed = mml.MLProject.resume(original.run_dir, data=binary_frame)
    assert calls == [], "all trials were checkpointed, none may be recomputed"
    assert [(t.trial_id, t.status, t.params) for t in resumed.trials] == [
        (t.trial_id, t.status, t.params) for t in original.trials
    ]
    assert (
        resumed.evaluation["selected_params"]
        == original.evaluation["selected_params"]
    )
    assert resumed.evaluation["selected_cv_score"] == pytest.approx(
        original.evaluation["selected_cv_score"]
    )


def test_cannot_resume_completed_run_or_changed_config(binary_frame, tmp_path):
    result = support.make_project(binary_frame, "y", output_dir=tmp_path).run()
    with pytest.raises(errors.ConfigError, match="already completed"):
        mml.MLProject.resume(result.run_dir, data=binary_frame)
    (result.run_dir / "evaluation.json").unlink()
    changed = support.make_project(binary_frame, "y", seed=1).orchestrator
    with pytest.raises(errors.ConfigError, match="configuration differs"):
        changed.resume(result.run_dir)
    with pytest.raises(FileNotFoundError):
        mml.MLProject.resume(tmp_path / "nope")


def test_tracking_database_records_runs_trials_and_lineage(
    binary_frame, tmp_path
):
    first = support.make_project(binary_frame, "y", output_dir=tmp_path).run()
    support.make_project(binary_frame, "y", output_dir=tmp_path, seed=3).run()
    store = stores.SqliteExperimentStore(tmp_path / "experiments.sqlite")
    runs = store.runs()
    assert len(runs) == 2 and {r["state"] for r in runs} == {"COMPLETED"}
    row = next(r for r in runs if r["run_id"] == first.run_id)
    assert row["dataset_id"] == first.profile.manifest.dataset_id
    assert row["data_hash"] == first.profile.manifest.data_hash
    assert row["split_fingerprint"] == first.evaluation["split_fingerprint"]
    assert row["selected_model"] == first.selected_model
    assert json.loads(row["test_metrics"])["roc_auc"] == pytest.approx(
        first.evaluation["test_metrics"]["roc_auc"]
    )
    assert len(store.trials(first.run_id)) == len(first.trials)
    assert {t["strategy"] for t in store.trials(first.run_id)} >= {"default"}
    config = json.loads(row["config_json"])
    assert (
        config["data"]["target"] == "y" and config["validation"]["seed"] == 42
    )
    lineage = json.loads(row["artifacts"])
    from mml.artifacts import bundle as bundle_module

    assert {
        "config.json",
        "evaluation.json",
        f"model/{bundle_module.MANIFEST_FILE}",
    } <= set(lineage)

    for name, digest in lineage.items():
        assert bundle_module.file_sha256(first.run_dir / name) == digest


def test_custom_tracker_is_honoured(binary_frame, tmp_path):
    class Recording(stores.NullExperimentStore):
        def __init__(self):
            self.seen = []

        def record_trial(self, run_id, trial):
            self.seen.append(trial.trial_id)

    recorder = Recording()
    spec = support.make_project(binary_frame, "y").spec
    orchestrator = orchestrator_module.Orchestrator(
        spec, binary_frame, tracker=recorder
    )
    result = orchestrator.run(tmp_path)
    assert recorder.seen == [t.trial_id for t in result.trials]
    assert not (tmp_path / "experiments.sqlite").exists()


def test_resume_refuses_checkpoints_computed_on_different_data(
    binary_frame, tmp_path
):
    counting, canceller = Counting, Canceller
    counting.fits = canceller.fits = 0
    canceller.armed = True
    registry = registry_module.ModelRegistry.with_builtins()
    registry.register(CancellerAdapter())
    project = support.make_project(binary_frame, "y", output_dir=tmp_path)
    project.orchestrator.registry = registry
    canceller.orchestrator = project.orchestrator
    plan = dataclasses.replace(
        project.plan(), candidate_models=("dummy", "canceller")
    )
    with pytest.raises(errors.CancelledError) as caught:
        project.run(plan)
    run_dir = caught.value.details["run_dir"]
    canceller.armed = False
    changed = binary_frame.copy()
    changed.iloc[1, 1] = changed.iloc[1, 1] + 1000.0
    orchestrator = orchestrator_module.Orchestrator.for_run_directory(
        run_dir, frame=changed, registry=registry
    )
    with pytest.raises(errors.ConfigError, match="data changed"):
        orchestrator.resume(run_dir)


def test_every_trial_records_its_evaluation_config_and_runtime(
    binary_frame, tmp_path
):
    result = support.make_project(
        binary_frame, "y", output_dir=tmp_path, folds=4, seed=7
    ).run()
    for trial in result.trials:
        assert trial.evaluation_config == {
            "metric": "roc_auc",
            "folds": 4,
            "seed": 7,
        }
        assert trial.runtime["python"].count(".") == 2
    record = json.loads(
        (result.run_dir / "trials" / "trial_002.json").read_text()
    )
    assert record["evaluation_config"]["folds"] == 4
    lightgbm = [t for t in result.trials if t.model_name == "lightgbm"]
    for trial in lightgbm:
        assert "lightgbm" in trial.runtime["libraries"]
