"""Tests for experiment stores."""

import pytest

from scmle.contracts import events
from scmle.contracts import experiments
from scmle.experimentation import stores


def make_trial(
    trial_id="trial_001", status=experiments.TrialStatus.OK, **kwargs
):
    return experiments.TrialResult(
        trial_id,
        "ridge",
        status,
        "rmse",
        1.5,
        0.1,
        [1.4, 1.6],
        0.2,
        params={"alpha": 0.5},
        **kwargs,
    )


@pytest.fixture
def store(tmp_path):
    return stores.SqliteExperimentStore(tmp_path / "db" / "experiments.sqlite")


def test_run_lifecycle_and_lineage(store):
    store.start_run("run_1", "abc", "/tmp/run_1", "2026-10-15T00:00:00+00:00")
    store.update_run(
        "run_1",
        dataset_id="ds_1",
        data_hash="h",
        selected_model="ridge",
        cv_score=1.5,
        test_metrics={"rmse": 1.2},
    )
    [row] = store.runs()
    assert row["dataset_id"] == "ds_1" and row["config_hash"] == "abc"
    assert row["test_metrics"] == '{"rmse": 1.2}'
    assert row["state"] == "CREATED"


def test_event_listener_mirrors_state(store):
    store.start_run("run_1", "abc", "d", "2026-10-15T00:00:00+00:00")
    store.listen("run_1", events.Event("run_1", "state.TRAINING"))
    store.listen("run_1", events.Event("run_1", "trial.ok"))
    assert store.runs()[0]["state"] == "TRAINING"


def test_trials_roundtrip_and_upsert(store):
    store.start_run("run_1", "abc", "d", "2026-10-15T00:00:00+00:00")
    store.record_trial("run_1", make_trial())
    store.record_trial(
        "run_1",
        make_trial(
            "trial_002",
            experiments.TrialStatus.FAILED,
        ),
    )
    store.record_trial("run_1", make_trial())  # replaces trial_001
    rows = store.trials("run_1")
    assert [r["trial_id"] for r in rows] == ["trial_001", "trial_002"]
    assert (
        rows[0]["params"] == '{"alpha": 0.5}' and rows[1]["status"] == "failed"
    )


def test_resume_reregisters_run_without_duplicates(store):
    store.start_run("run_1", "abc", "d", "2026-10-15T00:00:00+00:00")
    store.update_run("run_1", state="TRAINING")
    store.start_run("run_1", "abc", "d", "2026-10-15T01:00:00+00:00")
    assert len(store.runs()) == 1 and store.runs()[0]["state"] == "CREATED"


def test_unknown_columns_are_rejected(store):
    store.start_run("run_1", "abc", "d", "2026-10-15T00:00:00+00:00")
    with pytest.raises(ValueError, match="unknown run columns"):
        store.update_run("run_1", bogus=1)


def test_null_store_is_a_safe_noop():
    null = stores.NullExperimentStore()
    null.start_run("r", "h", "d", "t")
    null.update_run("r", state="X")
    null.record_trial("r", make_trial())
    null.listen("r", events.Event("r", "state.X"))
    assert null.runs() == [] and null.trials("r") == []
    assert isinstance(null, stores.ExperimentStore)


def test_every_operation_closes_its_connection(tmp_path):
    import gc
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        store = stores.SqliteExperimentStore(tmp_path / "leak.sqlite")
        store.start_run("r", "h", "d", "2026-10-15T00:00:00+00:00")
        store.update_run("r", state="TRAINING", fingerprint={"a": 1.0})
        store.record_trial("r", make_trial())
        store.runs()
        store.trials("r")
        store.history()
        gc.collect()


def test_a_failed_update_rolls_back_and_still_closes(tmp_path):
    store = stores.SqliteExperimentStore(tmp_path / "rollback.sqlite")
    store.start_run("r", "h", "d", "2026-10-15T00:00:00+00:00")
    with pytest.raises(ValueError):
        store.update_run("r", bogus=1)
    assert store.runs()[0]["state"] == "CREATED"
