"""A real MLProject run as a job: durable workdir, restore and resume."""

import functools
import json
import pathlib
import tempfile
import threading

import numpy as np
import pandas as pd

from scmle.core import retry
from scmle.data.store import objectstore
from scmle.runners import base
from scmle.runners import local
from scmle.runners import project

OPTIONS = {
    "time_budget": 20,
    "search": {"top_models": 0},
    "models": {"exclude_families": ["boosting", "deep_learning"]},
}


def make_frame():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    noise = rng.normal(scale=0.5, size=200)
    frame["y"] = ((frame.a + 0.5 * frame.b + noise) > 0).astype(int)
    return frame


def test_workdir_sync_pushes_changes_and_restores(tmp_path):
    store = objectstore.LocalObjectStore(tmp_path / "store")
    source = tmp_path / "work"
    (source / "sub").mkdir(parents=True)
    (source / "a.txt").write_text("one")
    (source / "sub" / "b.txt").write_text("two")
    sync = project.WorkdirSync(store, "runs/x/workdir/", source)
    assert sync.push() == 2 and sync.push() == 0
    (source / "a.txt").write_text("changed")
    assert sync.push() == 1
    target = tmp_path / "restored"
    target.mkdir()
    other = project.WorkdirSync(store, "runs/x/workdir/", target)
    assert other.restore() == 2
    assert (target / "a.txt").read_text() == "changed"
    assert (target / "sub" / "b.txt").read_text() == "two"
    assert project.saved_run(target) is None


def spec_of(run_id="churn-1"):
    return base.JobSpec(
        run_id=run_id,
        name="churn",
        project={"data": "data/train.csv", "target": "y", "options": OPTIONS},
        identity="arn:train",
        retries=retry.RetryPolicy(3, 0.1, 1.0, 1.0),
    )


def runner_for(store, workload):
    return local.LocalRunner(
        store, workload, serving_identity="arn:serve", poll_seconds=0
    )


def test_a_project_runs_as_a_job(tmp_path):
    store = objectstore.LocalObjectStore(tmp_path / "store")
    store.put("data/train.csv", make_frame().to_csv(index=False).encode())
    manifest = runner_for(store, project.run_project).run(spec_of())
    assert manifest.state == "succeeded", manifest.attempts[-1].reason
    result = json.loads(store.get("runs/churn-1/result.json")[0])
    assert result["resumed"] is False and result["selected_model"]
    assert any(
        m.key.endswith(".json")
        for m in store.list_objects("runs/churn-1/workdir/")
    )
    assert not pathlib.Path(tempfile.gettempdir(), "scmle-job-churn-1").exists()


def test_a_cancelled_project_is_interrupted_and_the_retry_resumes_it(tmp_path):
    store = objectstore.LocalObjectStore(tmp_path / "store")
    store.put("data/train.csv", make_frame().to_csv(index=False).encode())
    attempts = []

    def cancel_first_attempt(started):
        attempts.append(started)
        if len(attempts) == 1:
            threading.Timer(3.0, started.cancel).start()

    workload = functools.partial(
        project.run_project, sync_seconds=1.0, on_start=cancel_first_attempt
    )
    manifest = runner_for(store, workload).run(spec_of())
    assert manifest.state == "succeeded", manifest.attempts
    assert [a.state for a in manifest.attempts] == ["interrupted", "succeeded"]
    result = json.loads(store.get("runs/churn-1/result.json")[0])
    assert result["resumed"] is True and result["selected_model"]
    assert len(attempts) == 1
