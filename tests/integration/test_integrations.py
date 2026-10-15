"""Optional integrations: MLflow tracking and the Ray execution backend."""

import statistics

import pytest
import support

from mml.contracts import errors
from mml.contracts import tasks
from mml.core import execution
from mml.core import orchestrator as orchestrator_module


def test_mlflow_store_records_runs_trials_and_lineage(binary_frame, tmp_path):
    pytest.importorskip("mlflow")
    from mml.tracking import mlflow_store

    store = mlflow_store.MlflowExperimentStore(
        f"sqlite:///{tmp_path / 'mlflow.db'}", "magicml-test"
    )
    spec = support.make_project(binary_frame, "y", time_budget=300).spec
    result = orchestrator_module.Orchestrator(
        spec, binary_frame, tracker=store
    ).run(tmp_path / "runs")

    runs = store.runs()
    assert len(runs) == 1 and runs[0]["run_id"] == result.run_id
    assert runs[0]["state"] == "COMPLETED" and "cv_score" in runs[0]
    assert any(key.startswith("test.") for key in runs[0])
    trials = store.trials(result.run_id)
    assert [t["trial_id"] for t in trials] == [
        t.trial_id for t in result.trials
    ]
    assert {t["model_name"] for t in trials} == {
        t.model_name for t in result.trials
    }
    history = store.history()
    assert (
        len(history) == 1 and history[0].selected_model == result.selected_model
    )
    assert set(history[0].fingerprint) >= {"log_rows", "log_imbalance"}


def test_mlflow_store_reuses_the_parent_run_on_resume(binary_frame, tmp_path):
    pytest.importorskip("mlflow")
    from mml.tracking import mlflow_store

    store = mlflow_store.MlflowExperimentStore(
        f"sqlite:///{tmp_path / 'mlflow.db'}", "magicml-resume"
    )
    store.start_run("run_x", "hash", "/tmp/x", "2026-10-15T00:00:00+00:00")
    again = mlflow_store.MlflowExperimentStore(
        f"sqlite:///{tmp_path / 'mlflow.db'}", "magicml-resume"
    )
    again.start_run("run_x", "hash", "/tmp/x", "2026-10-15T01:00:00+00:00")
    assert len(again.runs()) == 1


def test_mlflow_without_the_library_is_a_config_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("mlflow"):
            raise ImportError("no mlflow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    from mml.tracking import mlflow_store

    with pytest.raises(errors.ConfigError, match="tracking"):
        mlflow_store.MlflowExperimentStore()


def test_ray_backend_maps_in_order():
    pytest.importorskip("ray")
    backend = execution.create_backend(2, "ray")
    try:
        pairs = [[1, 3], [2, 4], [10, 20]]
        assert backend.map(statistics.fmean, pairs) == [2.0, 3.0, 15.0]
        assert backend.map(statistics.fmean, []) == []
    finally:
        backend.ray.shutdown()


def test_ray_run_matches_sequential(binary_frame, tmp_path):
    pytest.importorskip("ray")

    def spec_with(workers, executor):
        return tasks.ProjectSpec(
            data=tasks.DataSpec(uri="<in-memory>", target="y"),
            resources=tasks.ResourceSpec(
                time_budget_seconds=300, workers=workers, executor=executor
            ),
            search=tasks.SearchSpec(top_models=0),
            strategies=support.NO_STRATEGIES,
            ensemble=support.NO_ENSEMBLE,
        )

    sequential = orchestrator_module.Orchestrator(
        spec_with(1, "process"), binary_frame
    ).run(tmp_path / "seq")
    distributed = orchestrator_module.Orchestrator(
        spec_with(2, "ray"), binary_frame
    ).run(tmp_path / "ray")
    try:
        assert [t.score_mean for t in distributed.trials] == pytest.approx(
            [t.score_mean for t in sequential.trials]
        )
        assert distributed.selected_model == sequential.selected_model
    finally:
        import ray

        ray.shutdown()


def test_unknown_executor_is_rejected():
    with pytest.raises(errors.ConfigError, match="unknown executor"):
        execution.create_backend(2, "spark")
