import json

import pytest
from sklearn.dummy import DummyClassifier

from mml import MLProject
from mml.contracts.errors import MagicMLError
from mml.contracts.models import Capabilities, ModelSpec
from mml.contracts.tasks import TaskType
from mml.core.orchestrator import RunCancelled
from mml.models import registry
from mml.models.registry import register_model

BIN = frozenset({TaskType.BINARY})


class Exploding(DummyClassifier):
    def fit(self, X, y):
        raise RuntimeError("boom")


@pytest.fixture
def cleanup():
    added = []
    yield added
    for n in added:
        registry._REGISTRY.pop(n, None)


def test_failed_trial_is_isolated(binary_df, tmp_path, cleanup):
    register_model(ModelSpec("exploding", "x", Capabilities(BIN, cost=1), lambda t, s: Exploding()))
    cleanup.append("exploding")
    r = MLProject(binary_df, "y", output_dir=tmp_path).run()
    bad = [t for t in r.trials if t.model_name == "exploding"][0]
    assert bad.status == "failed" and "boom" in bad.error
    assert r.selected_model != "exploding"
    assert bad.trial_id in r.leaderboard()["trial_id"].tolist()


def test_all_trials_failing_reports_failed_state(binary_df, tmp_path, cleanup):
    from mml.models.registry import _REGISTRY

    saved = dict(_REGISTRY)
    _REGISTRY.clear()
    register_model(ModelSpec("exploding", "x", Capabilities(BIN), lambda t, s: Exploding()))
    try:
        with pytest.raises(MagicMLError, match="all trials failed") as e:
            MLProject(binary_df, "y", output_dir=tmp_path).run()
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(saved)
    last = json.loads(open(f"{e.value.details['run_dir']}/events.jsonl").readlines()[-1])
    assert last["kind"] == "state.FAILED"


def test_cancellation_keeps_completed_trials(binary_df, tmp_path, cleanup):
    project = MLProject(binary_df, "y", output_dir=tmp_path)

    class Canceller(DummyClassifier):
        def fit(self, X, y):
            project.cancel()
            return super().fit(X, y)

    register_model(ModelSpec("canceller", "x", Capabilities(BIN, cost=1), lambda t, s: Canceller()))
    cleanup.append("canceller")
    with pytest.raises(RunCancelled) as e:
        project.run()
    run_dir = e.value.details["run_dir"]
    import os

    assert any(f.endswith(".json") for f in os.listdir(f"{run_dir}/trials"))
    assert json.loads(open(f"{run_dir}/events.jsonl").readlines()[-1])["kind"] == "state.CANCELLED"


def test_budget_exhaustion_skips_later_trials(binary_df, tmp_path):
    r = MLProject(binary_df, "y", time_budget=1, output_dir=tmp_path).run()
    assert r.trials[0].status == "ok"
    # a 1s budget can't cover every candidate; skipped trials are reported, run still completes
    assert r.state == "COMPLETED"
    assert all(t.error for t in r.trials if t.status == "skipped")
