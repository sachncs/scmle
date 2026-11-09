"""Meta-learning from the experiment history, end to end."""

import numpy
import pandas
import support

import scmle
from scmle.experimentation import stores


def make_frame(rows, seed):
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame(
        {"a": rng.normal(size=rows), "b": rng.normal(size=rows)}
    )
    frame["y"] = (frame.a + frame.b + rng.normal(size=rows) > 0).astype(int)
    return frame


def run(frame, directory):
    return support.make_project(
        frame, "y", time_budget=300, output_dir=directory
    ).run()


def test_runs_store_fingerprints_and_winning_strategies(tmp_path):
    result = run(make_frame(400, 0), tmp_path)
    history = stores.SqliteExperimentStore(
        tmp_path / "experiments.sqlite"
    ).history()
    assert len(history) == 1
    record = history[0]
    assert (
        record.run_id == result.run_id
        and record.selected_model == result.selected_model
    )
    assert set(record.fingerprint) >= {"log_rows", "log_imbalance"}
    assert record.task == "binary_classification"


def test_plan_prioritizes_models_that_won_on_similar_datasets(tmp_path):
    for seed in (1, 2, 3):
        run(make_frame(400, seed), tmp_path)
    winners = {
        r.selected_model
        for r in stores.SqliteExperimentStore(
            tmp_path / "experiments.sqlite"
        ).history()
    }
    project = support.make_project(make_frame(420, 9), "y", output_dir=tmp_path)
    plan = project.plan()
    decision = next(d for d in plan.decisions if d.technique == "meta_learning")
    assert decision.action == "include"
    models = list(plan.candidate_models)
    assert models[0] == "dummy"  # baselines stay first
    preferred = [m for m in models[1:] if m in winners]
    assert models[1 : 1 + len(preferred)] == preferred
    assert set(models) >= winners  # nothing is dropped, only reordered


def test_same_dataset_history_is_never_used(tmp_path):
    frame = make_frame(400, 0)
    run(frame, tmp_path)
    plan = support.make_project(frame, "y", output_dir=tmp_path).plan()
    decision = next(d for d in plan.decisions if d.technique == "meta_learning")
    assert (
        decision.action == "defer"
        and "no comparable history" in decision.reason
    )


def test_planning_without_a_database_creates_nothing(tmp_path):
    plan = support.make_project(
        make_frame(300, 4), "y", output_dir=tmp_path / "runs"
    ).plan()
    assert not (tmp_path / "runs").exists()
    assert any(
        d.technique == "meta_learning" and d.action == "defer"
        for d in plan.decisions
    )


def test_old_databases_are_migrated(tmp_path):
    import sqlite3

    path = tmp_path / "experiments.sqlite"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE runs (run_id TEXT PRIMARY KEY, config_hash TEXT NOT NULL, "
        "dataset_id TEXT, data_hash TEXT, split_fingerprint TEXT, state TEXT NOT NULL, "
        "started_at TEXT NOT NULL, updated_at TEXT NOT NULL, selected_model TEXT, "
        "metric TEXT, cv_score REAL, test_metrics TEXT, run_dir TEXT)"
    )
    connection.commit()
    connection.close()
    store = stores.SqliteExperimentStore(path)
    store.start_run("r", "h", "d", "2026-10-15T00:00:00+00:00")
    store.update_run(
        "r", fingerprint={"a": 1.0}, task="regression", selected_strategy="x"
    )
    assert store.runs()[0]["selected_strategy"] == "x"
    assert isinstance(scmle.MLProject, type)
