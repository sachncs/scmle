"""Model registry, serving adapters and lifecycle automation."""

import json
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest
import support

from mml.contracts import errors
from mml.deployment import lifecycle
from mml.deployment import registry as registry_module
from mml.deployment import serving


def make_frame(rows, seed, relabel=False):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=rows),
            "b": rng.normal(size=rows),
            "c": rng.choice(["x", "y", "z"], rows),
        }
    )
    signal = frame.a + 0.5 * frame.b + (frame.c == "x")
    signal = -signal if relabel else signal
    frame["y"] = (signal + rng.normal(scale=0.5, size=rows) > 0.5).astype(int)
    return frame


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    directory = tmp_path_factory.mktemp("deploy")
    return support.make_project(
        make_frame(700, 0), "y", time_budget=300, output_dir=directory
    ).run()


def post(url, payload):
    request = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_registry_versions_aliases_and_audit(trained, tmp_path):
    store = registry_module.ModelRegistryStore(tmp_path / "registry")
    bundle_dir = trained.save(tmp_path / "export")
    first = store.register("churn", bundle_dir, {"roc_auc": 0.9}, run_id="r1")
    second = store.register("churn", bundle_dir, {"roc_auc": 0.91}, run_id="r2")
    assert (first.version, second.version) == (1, 2)
    store.promote("churn", 1)
    store.promote("churn", 2, "challenger")
    assert store.aliases("churn") == {"champion": 1, "challenger": 2}
    assert store.resolve("churn", "challenger").run_id == "r2"
    assert store.resolve("churn", 2).metrics == {"roc_auc": 0.91}
    store.promote("churn", 2)
    assert store.resolve("churn").version == 2
    audit = [
        json.loads(line)
        for line in (store.root / "audit.jsonl").read_text().splitlines()
    ]
    assert [a["action"] for a in audit] == [
        "register",
        "register",
        "promote",
        "promote",
        "promote",
    ]
    assert audit[-1]["previous"] == 1
    features = make_frame(20, 1).drop(columns="y")
    np.testing.assert_array_equal(
        store.load("churn").predict(features), trained.predict(features)
    )
    reopened = registry_module.ModelRegistryStore(tmp_path / "registry")
    assert [v.version for v in reopened.versions("churn")] == [1, 2]


def test_registry_rejects_bad_requests(trained, tmp_path):
    store = registry_module.ModelRegistryStore(tmp_path / "registry")
    with pytest.raises(errors.ArtifactError, match="unknown model"):
        store.versions("ghost")
    store.register("m", trained.save(tmp_path / "export"))
    with pytest.raises(errors.ArtifactError, match="not registered"):
        store.promote("m", 9)
    with pytest.raises(errors.ArtifactError, match="no alias"):
        store.resolve("m", "champion")
    broken = tmp_path / "broken"
    broken.mkdir()
    with pytest.raises(errors.ArtifactError):
        store.register("m", broken)


def test_registry_refuses_tampered_bundles(trained, tmp_path):
    bundle_dir = trained.save(tmp_path / "export")
    with (bundle_dir / "pipeline.joblib").open("ab") as handle:
        handle.write(b"x")
    store = registry_module.ModelRegistryStore(tmp_path / "registry")
    with pytest.raises(errors.ArtifactError, match="checksum"):
        store.register("m", bundle_dir)
    assert not (tmp_path / "registry" / "m").exists()


def test_http_serving_predict_health_schema_and_errors(trained):
    adapter = serving.HttpServing(trained.bundle)
    adapter.start()
    try:
        base = adapter.address
        with urllib.request.urlopen(f"{base}/health") as response:
            assert json.loads(response.read()) == {"status": "ok"}
        with urllib.request.urlopen(f"{base}/schema") as response:
            assert json.loads(response.read())["schema"]["target"] == "y"
        records = make_frame(5, 2).drop(columns="y").to_dict("records")
        status, body = post(f"{base}/predict", {"records": records})
        assert status == 200 and len(body["predictions"]) == 5
        assert body["schema_valid"] is True and body["classes"] == [0, 1]
        assert np.allclose(np.array(body["probabilities"]).sum(axis=1), 1.0)
        assert (
            body["predictions"]
            == trained.predict(pd.DataFrame(records)).tolist()
        )
        status, body = post(f"{base}/predict", {"records": [{"a": 1.0}]})
        assert (
            status == 400
            and body["category"] == "schema"
            and "missing" in body["details"]
        )
        status, body = post(f"{base}/predict", {"nope": 1})
        assert status == 400 and body["category"] == "request"
        status, body = post(f"{base}/other", {})
        assert status == 404
    finally:
        adapter.stop()
    with pytest.raises(errors.MagicMLError):
        print(adapter.address)


def test_http_serving_reports_diagnostics_for_messy_input(trained):
    adapter = serving.HttpServing(trained.bundle)
    adapter.start()
    try:
        records = [{"a": 0.1, "b": None, "c": "never_seen", "extra": 1}]
        status, body = post(f"{adapter.address}/predict", {"records": records})
        assert status == 200
        diagnostics = body["diagnostics"]
        assert diagnostics["ignored_columns"] == ["extra"]
        assert diagnostics["unseen_categories"] == {"c": ["never_seen"]}
        assert diagnostics["missing_values"] == {"b": 1}
    finally:
        adapter.stop()


def test_batch_serving_scores_files(trained, tmp_path):
    source = tmp_path / "in.csv"
    make_frame(40, 3).drop(columns="y").to_csv(source, index=False)
    adapter = serving.BatchServing(trained.bundle)
    assert adapter.score_file(source, tmp_path / "out.csv") == 40
    out = pd.read_csv(tmp_path / "out.csv")
    assert {"prediction", "proba_0", "proba_1"} <= set(out.columns)


def test_lifecycle_keeps_a_healthy_champion(trained, tmp_path):
    store = registry_module.ModelRegistryStore(tmp_path / "registry")
    manager = lifecycle.LifecycleManager(store, "churn", str(tmp_path / "work"))
    record = manager.register_run(trained)
    assert store.resolve("churn").version == record.version == 1
    outcome = manager.review(make_frame(600, 4), "y")
    assert not outcome.retrained and outcome.promoted_version is None
    assert outcome.monitoring.recommendation.action == "none"


def test_lifecycle_retrains_and_promotes_after_concept_drift(trained, tmp_path):
    store = registry_module.ModelRegistryStore(tmp_path / "registry")
    manager = lifecycle.LifecycleManager(store, "churn", str(tmp_path / "work"))
    manager.register_run(trained)
    drifted = make_frame(2400, 5, relabel=True)
    outcome = manager.review(
        drifted,
        "y",
        project_options={
            "time_budget": 300,
            "search": support.NO_TUNING,
            "strategies": support.NO_STRATEGIES,
            "ensemble": support.NO_ENSEMBLE,
        },
    )
    assert outcome.monitoring.recommendation.action == "retrain"
    assert outcome.retrained and outcome.comparison.decision == "promote"
    assert outcome.promoted_version == 2 and store.resolve("churn").version == 2
    assert [v.version for v in store.versions("churn")] == [1, 2]
    assert any("promoted challenger" in note for note in outcome.notes)
    fresh = make_frame(300, 6, relabel=True)
    new = store.load("churn")
    accuracy = (new.predict(fresh.drop(columns="y")) == fresh["y"]).mean()
    assert accuracy > 0.75


def test_review_can_be_report_only(trained, tmp_path):
    store = registry_module.ModelRegistryStore(tmp_path / "registry")
    manager = lifecycle.LifecycleManager(store, "churn", str(tmp_path / "work"))
    manager.register_run(trained)
    outcome = manager.review(
        make_frame(1000, 7, relabel=True), "y", retrain=False
    )
    assert outcome.monitoring.recommendation.action == "retrain"
    assert not outcome.retrained and store.resolve("churn").version == 1
