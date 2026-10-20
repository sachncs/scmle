"""Model registry, serving adapters and lifecycle automation."""

import copy
import dataclasses
import json
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest
import support

import scml
from scml.contracts import errors
from scml.governance import lifecycle
from scml.governance import registry as registry_module
from scml.inference import serving
from scml.monitoring import champion as champion_module

APPROVAL_KEY = "promotion-review-key"


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


def approval_for(record):
    return registry_module.PromotionApproval(
        model=record.model,
        version=record.version,
        content_id=record.content_id,
        reviewer="integration-reviewer",
        decision=registry_module.PromotionDecision.APPROVED,
        reason="verified integration fixture",
        evidence=("integration-evidence.json",),
        approved_at="2026-10-20T00:00:00+00:00",
    ).sign(APPROVAL_KEY)


def register_champion(manager, store, result):
    record = manager.register_run(result)
    store.promote(
        manager.model,
        record.version,
        approval=approval_for(record),
        signing_key=APPROVAL_KEY,
    )
    return record


def test_registry_versions_aliases_and_audit(trained, tmp_path):
    store = registry_module.ModelRegistryStore(tmp_path / "registry")
    bundle_dir = trained.save(tmp_path / "export")
    first = store.register(
        "churn", bundle_dir, {"roc_auc": 0.9}, run_id="r1", trusted=True
    )
    newer = copy.copy(trained.bundle)
    newer.info = dataclasses.replace(trained.bundle.info, model_version="v2")
    changed = newer.save(tmp_path / "export2")
    second = store.register(
        "churn", changed, {"roc_auc": 0.91}, run_id="r2", trusted=True
    )
    assert (first.version, second.version) == (1, 2)
    assert first.content_id != second.content_id
    store.promote(
        "churn", 1, approval=approval_for(first), signing_key=APPROVAL_KEY
    )
    store.promote("churn", 2, "challenger")
    assert store.aliases("churn") == {"champion": 1, "challenger": 2}
    assert store.resolve("churn", "challenger").run_id == "r2"
    assert store.resolve("churn", 2).metrics == {"roc_auc": 0.91}
    store.promote(
        "churn", 2, approval=approval_for(second), signing_key=APPROVAL_KEY
    )
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
    assert audit[-1]["approval_record"] == approval_for(second).to_dict()
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
    store.register("m", trained.save(tmp_path / "export"), trusted=True)
    with pytest.raises(errors.ArtifactError, match="not registered"):
        store.promote("m", 9)
    with pytest.raises(errors.ArtifactError, match="no alias"):
        store.resolve("m", "champion")
    broken = tmp_path / "broken"
    broken.mkdir()
    with pytest.raises(errors.ArtifactError):
        store.register("m", broken, trusted=True)


def test_champion_promotion_requires_matching_independent_approval(
    trained, tmp_path
):
    store = registry_module.ModelRegistryStore(tmp_path / "registry")
    record = store.register(
        "m", trained.save(tmp_path / "export"), trusted=True
    )
    with pytest.raises(errors.ArtifactError, match="independent approval"):
        store.promote("m", record.version)
    with pytest.raises(errors.ArtifactError, match="bundle identity"):
        store.promote(
            "m",
            record.version,
            approval=dataclasses.replace(
                approval_for(record), content_id="wrong-content"
            ),
            signing_key=APPROVAL_KEY,
        )
    with pytest.raises(errors.ArtifactError, match="rejected"):
        store.promote(
            "m",
            record.version,
            approval=dataclasses.replace(
                approval_for(record),
                decision=registry_module.PromotionDecision.REJECTED,
            ),
            signing_key=APPROVAL_KEY,
        )
    with pytest.raises(errors.ArtifactError, match="valid signature"):
        store.promote(
            "m",
            record.version,
            approval=dataclasses.replace(approval_for(record), signature=None),
            signing_key=APPROVAL_KEY,
        )
    with pytest.raises(errors.ArtifactError, match="valid signature"):
        store.promote(
            "m",
            record.version,
            approval=dataclasses.replace(
                approval_for(record), reason="tampered review"
            ),
            signing_key=APPROVAL_KEY,
        )
    assert store.aliases("m") == {}


def test_registry_refuses_tampered_bundles(trained, tmp_path):
    bundle_dir = trained.save(tmp_path / "export")
    with (bundle_dir / "pipeline.joblib").open("ab") as handle:
        handle.write(b"x")
    store = registry_module.ModelRegistryStore(tmp_path / "registry")
    with pytest.raises(errors.ArtifactError, match="checksum"):
        store.register("m", bundle_dir, trusted=True)
    assert not (tmp_path / "registry" / "m").exists()


def test_http_serving_predict_health_schema_and_errors(trained):
    adapter = serving.HttpServing(trained.bundle)
    adapter.start()
    try:
        base = adapter.address
        with urllib.request.urlopen(f"{base}/health") as response:
            health = json.loads(response.read())
            assert health["status"] == "ok"
            assert health["task"] == "binary_classification"
            assert health["model_version"] == trained.bundle.info.model_version
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
            status == 422
            and body["category"] == "schema"
            and "missing" in body["details"]
        )
        status, body = post(f"{base}/predict", {"nope": 1})
        assert status == 400 and body["category"] == "request"
        status, body = post(f"{base}/other", {})
        assert status == 404
    finally:
        adapter.stop()
    with pytest.raises(errors.ScmlError):
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
    record = register_champion(manager, store, trained)
    assert store.resolve("churn").version == record.version == 1
    outcome = manager.review(make_frame(600, 4), "y")
    assert not outcome.retrained and outcome.promoted_version is None
    assert outcome.monitoring.recommendation.action == "none"


def test_lifecycle_retrains_and_promotes_after_concept_drift(trained, tmp_path):
    store = registry_module.ModelRegistryStore(tmp_path / "registry")
    manager = lifecycle.LifecycleManager(store, "churn", str(tmp_path / "work"))
    register_champion(manager, store, trained)
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
    assert outcome.promoted_version is None
    candidate = store.resolve("churn", registry_module.CHALLENGER)
    manager.promote_challenger(
        approval_for(candidate), signing_key=APPROVAL_KEY
    )
    assert store.resolve("churn").version == 2
    assert [v.version for v in store.versions("churn")] == [1, 2]
    assert any("independent approval" in note for note in outcome.notes)
    fresh = make_frame(300, 6, relabel=True)
    new = store.load("churn")
    accuracy = (new.predict(fresh.drop(columns="y")) == fresh["y"]).mean()
    assert accuracy > 0.75


def test_review_can_be_report_only(trained, tmp_path):
    store = registry_module.ModelRegistryStore(tmp_path / "registry")
    manager = lifecycle.LifecycleManager(store, "churn", str(tmp_path / "work"))
    register_champion(manager, store, trained)
    outcome = manager.review(
        make_frame(1000, 7, relabel=True), "y", retrain=False
    )
    assert outcome.monitoring.recommendation.action == "retrain"
    assert not outcome.retrained and store.resolve("churn").version == 1


def test_registering_identical_content_is_idempotent(trained, tmp_path):
    store = registry_module.ModelRegistryStore(tmp_path / "registry")
    bundle_dir = trained.save(tmp_path / "export")
    first = store.register("churn", bundle_dir, run_id="r1", trusted=True)
    again = store.register("churn", bundle_dir, run_id="r2", trusted=True)
    copy = trained.save(tmp_path / "export_copy")
    third = store.register("churn", copy, run_id="r3", trusted=True)
    assert again.version == third.version == first.version == 1
    assert len(store.versions("churn")) == 1
    assert first.content_id == registry_module.content_id_of(bundle_dir)
    audit = (store.root / "audit.jsonl").read_text().splitlines()
    assert len(audit) == 1  # duplicates leave no extra audit noise


def test_registry_refuses_unsigned_bundles_unless_vouched_for(
    trained, tmp_path
):
    store = registry_module.ModelRegistryStore(tmp_path / "registry")
    bundle_dir = trained.save(tmp_path / "export")
    with pytest.raises(errors.ArtifactError, match="unauthenticated"):
        store.register("m", bundle_dir)
    assert not (tmp_path / "registry" / "m").exists()
    record = store.register("m", bundle_dir, trusted=True)
    assert record.signed is False


def test_registry_accepts_signed_bundles_and_reauthenticates_on_load(
    trained, tmp_path
):
    key = "registry-test-key"
    store = registry_module.ModelRegistryStore(tmp_path / "registry")
    bundle_dir = trained.bundle.save(tmp_path / "export", signing_key=key)
    record = store.register("m", bundle_dir, signing_key=key)
    assert record.signed is True
    store.promote(
        "m", record.version, approval=approval_for(record), signing_key=APPROVAL_KEY
    )
    features = make_frame(10, 3).drop(columns="y")
    loaded = store.load("m", signing_key=key)
    assert len(loaded.predict(features)) == 10
    with pytest.raises(errors.ArtifactError, match="unauthenticated"):
        store.load("m")  # a signed bundle needs its key at load time


def count_trainings(monkeypatch):
    runs = []
    original = scml.MLProject.run

    def counted(self, *args, **kwargs):
        runs.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(scml.MLProject, "run", counted)
    return runs


FAST_OPTIONS = {
    "time_budget": 300,
    "search": support.NO_TUNING,
    "strategies": support.NO_STRATEGIES,
    "ensemble": support.NO_ENSEMBLE,
}


def audit_actions(store):
    lines = (store.root / registry_module.AUDIT_FILE).read_text().splitlines()
    return [json.loads(line)["action"] for line in lines]


def test_repeated_checks_of_the_same_drift_retrain_exactly_once(
    trained, tmp_path, monkeypatch
):
    runs = count_trainings(monkeypatch)
    store = registry_module.ModelRegistryStore(tmp_path / "registry")
    manager = lifecycle.LifecycleManager(store, "churn", str(tmp_path / "work"))
    register_champion(manager, store, trained)
    runs.clear()
    drifted = make_frame(2400, 5, relabel=True)

    first = manager.review(drifted, "y", project_options=FAST_OPTIONS)
    second = manager.review(drifted, "y", project_options=FAST_OPTIONS)
    third = manager.review(drifted, "y", project_options=FAST_OPTIONS)

    assert len(runs) == 1
    assert first.promoted_version is None
    assert store.resolve("churn", registry_module.CHALLENGER).version == 2
    assert second.skipped in {None, "already_handled"}
    assert not second.retrained and not third.retrained


def test_a_rejected_challenger_is_recorded_and_not_retried_at_once(
    trained, tmp_path, monkeypatch
):
    runs = count_trainings(monkeypatch)
    store = registry_module.ModelRegistryStore(tmp_path / "registry")
    manager = lifecycle.LifecycleManager(
        store,
        "churn",
        str(tmp_path / "work"),
        lifecycle.RetrainPolicy(min_rows=200, cooldown_rows=1500),
    )
    register_champion(manager, store, trained)
    runs.clear()
    monkeypatch.setattr(
        champion_module.ChampionChallenger,
        "compare",
        lambda self, data, outcomes, metric_name=None: (
            champion_module.Comparison(
                "roc_auc", len(data), 0.9, 0.9, 0.0, (-0.01, 0.01), 0.5, "keep"
            )
        ),
    )

    first = manager.review(
        make_frame(1200, 5, relabel=True), "y", project_options=FAST_OPTIONS
    )
    soon = manager.review(
        make_frame(400, 8, relabel=True), "y", project_options=FAST_OPTIONS
    )
    later = manager.review(
        make_frame(1400, 9, relabel=True), "y", project_options=FAST_OPTIONS
    )

    assert first.retrained and first.promoted_version is None
    assert store.resolve("churn").version == 1
    assert soon.skipped == "cooldown" and not soon.retrained
    assert later.retrained and len(runs) == 2
    actions = audit_actions(store)
    assert actions.count("lifecycle.challenger_rejected") == 2
    assert "lifecycle.retrain_skipped" in actions
    lines = (store.root / registry_module.AUDIT_FILE).read_text().splitlines()
    rejected = next(
        json.loads(line) for line in lines if "challenger_rejected" in line
    )
    assert rejected["decision"] == "keep" and "interval" in rejected


def test_too_few_labeled_rows_do_not_trigger_training(
    trained, tmp_path, monkeypatch
):
    runs = count_trainings(monkeypatch)
    store = registry_module.ModelRegistryStore(tmp_path / "registry")
    manager = lifecycle.LifecycleManager(
        store,
        "churn",
        str(tmp_path / "work"),
        lifecycle.RetrainPolicy(min_rows=5000),
    )
    register_champion(manager, store, trained)
    runs.clear()

    outcome = manager.review(
        make_frame(2400, 5, relabel=True), "y", project_options=FAST_OPTIONS
    )

    assert outcome.skipped == "too_few_rows" and not runs
