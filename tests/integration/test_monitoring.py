"""Drift, delayed labels, retraining advice and champion-challenger."""

import numpy as np
import pandas as pd
import pytest
import support

from scml.contracts import errors
from scml.monitoring import champion
from scml.monitoring import events
from scml.monitoring import monitor as monitor_module


def make_frame(rows, seed, shift=0.0, relabel=False):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=rows) + shift,
            "b": rng.normal(size=rows),
            "c": rng.choice(["x", "y", "z"], rows),
        }
    )
    signal = frame.a - shift + 0.5 * frame.b + (frame.c == "x")
    if relabel:
        signal = -signal
    frame["y"] = (signal + rng.normal(scale=0.5, size=rows) > 0.5).astype(int)
    return frame


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    directory = tmp_path_factory.mktemp("monitoring")
    frame = make_frame(800, 0)
    return support.make_project(
        frame, "y", time_budget=300, output_dir=directory
    ).run()


def test_stable_data_is_stable(trained):
    monitor = monitor_module.ModelMonitor(trained.bundle)
    report = monitor.assess(make_frame(600, 1).drop(columns="y"))
    assert report.drift.output_status == "stable"
    assert all(f.status == "stable" for f in report.drift.features)
    assert report.recommendation.action == "none"


def test_a_shifted_feature_is_found_and_ranked_first(trained):
    shifted = make_frame(600, 2, shift=2.5).drop(columns="y")
    report = monitor_module.ModelMonitor(trained.bundle).assess(shifted)
    worst = report.drift.features[0]
    assert worst.feature == "a" and worst.status == "high" and worst.psi > 0.25
    assert report.recommendation.action in {"investigate", "retrain"}
    assert any("'a'" in reason for reason in report.recommendation.reasons)


def test_unseen_categories_trigger_investigation(trained):
    data = make_frame(500, 3).drop(columns="y")
    data["c"] = np.where(np.arange(500) % 3 == 0, "brand_new", data["c"])
    report = monitor_module.ModelMonitor(trained.bundle).assess(data)
    category = next(f for f in report.drift.features if f.feature == "c")
    assert category.unseen_rate > 0.2
    assert any("unseen" in r for r in report.recommendation.reasons)


def test_delayed_labels_are_handled_explicitly(trained):
    data = make_frame(600, 4)
    outcomes = data["y"].astype(float).copy()
    outcomes.iloc[100:] = np.nan  # labels still pending
    monitor = monitor_module.ModelMonitor(trained.bundle)
    report = monitor.assess(data.drop(columns="y"), outcomes)
    performance = report.performance
    assert (
        performance.labeled_rows == 100
        and performance.labeled_fraction == pytest.approx(100 / 600)
    )
    assert performance.status in {"ok", "degraded"}
    few = outcomes.copy()
    few.iloc[10:] = np.nan
    sparse = monitor.assess(data.drop(columns="y"), few).performance
    assert sparse.status == "insufficient_labels" and sparse.metrics == {}


def test_concept_drift_is_caught_by_performance_not_features(trained):
    flipped = make_frame(800, 5, relabel=True)
    monitor = monitor_module.ModelMonitor(trained.bundle)
    report = monitor.assess(flipped.drop(columns="y"), flipped["y"])
    assert all(f.status == "stable" for f in report.drift.features)
    assert report.performance.status == "degraded"
    assert report.recommendation.action == "retrain"
    assert any("degraded" in reason for reason in report.recommendation.reasons)


def test_prediction_log_joins_late_outcomes(trained, tmp_path):
    log = events.PredictionLog(tmp_path / "log" / "events.jsonl")
    frame = make_frame(60, 6)
    logged = log.log_predictions(trained.bundle, frame.drop(columns="y"))
    assert (
        len(logged) == 60
        and logged[0].model_version == trained.bundle.info.model_version
    )
    assert logged[0].probability is not None
    truth = {
        e.event_id: int(y)
        for e, y in zip(logged[:40], frame["y"][:40], strict=True)
    }
    log.record_outcomes(truth)
    features, predictions, outcomes = log.load()
    assert len(features) == len(predictions) == len(outcomes) == 60
    assert outcomes.notna().sum() == 40 and outcomes.isna().sum() == 20
    report = monitor_module.ModelMonitor(trained.bundle).assess(
        features, outcomes
    )
    assert report.performance.labeled_rows == 40


def test_empty_log_is_empty(tmp_path):
    features, predictions, outcomes = events.PredictionLog(
        tmp_path / "none.jsonl"
    ).load()
    assert features.empty and predictions.empty and outcomes.empty


def test_bundles_without_reference_cannot_be_monitored(trained):
    import copy

    bare = copy.copy(trained.bundle)
    bare.reference = None
    with pytest.raises(errors.ArtifactError, match="reference"):
        monitor_module.ModelMonitor(bare)


def test_advisor_combines_rules_and_keeps_all_reasons():
    drift = monitor_module.DriftReport(100, [], 0.4, "high")
    performance = monitor_module.PerformanceReport(
        100, 100, 1.0, {}, {}, {"roc_auc": -0.3}, "degraded"
    )
    verdict = monitor_module.advise(
        drift, performance, monitor_module.MonitoringThresholds()
    )
    assert verdict.action == "retrain" and len(verdict.reasons) == 2
    calm = monitor_module.advise(
        monitor_module.DriftReport(10, [], 0.01, "stable"),
        None,
        monitor_module.MonitoringThresholds(),
    )
    assert calm.action == "none" and calm.reasons == []


def test_champion_challenger_promotes_only_a_clearly_better_model(tmp_path):
    train = make_frame(1500, 7)
    strong = support.make_project(
        train, "y", time_budget=300, output_dir=tmp_path / "s"
    ).run()
    weak = support.make_project(
        train.drop(columns=["a"]),
        "y",
        time_budget=300,
        output_dir=tmp_path / "w",
    ).run()
    holdout = make_frame(1500, 8)
    features = holdout.drop(columns="y")
    promote = champion.ChampionChallenger(
        weak.bundle, strong.bundle, bootstraps=80
    ).compare(features, holdout["y"])
    assert (
        promote.decision == "promote"
        and promote.advantage > 0
        and promote.interval[0] > 0
    )
    keep = champion.ChampionChallenger(
        strong.bundle, weak.bundle, bootstraps=80
    ).compare(features, holdout["y"])
    assert keep.decision == "keep" and keep.advantage < 0
    same = champion.ChampionChallenger(
        strong.bundle, strong.bundle, bootstraps=80
    ).compare(features, holdout["y"])
    assert same.decision == "keep" and same.advantage == pytest.approx(0.0)


def test_comparison_validates_inputs(trained, tmp_path):
    regression = support.make_project(
        make_frame(300, 9).assign(y=lambda f: f.a * 2),
        "y",
        task="regression",
        output_dir=tmp_path,
    ).run()
    with pytest.raises(errors.ConfigError, match="different tasks"):
        champion.ChampionChallenger(trained.bundle, regression.bundle)
    pair = champion.ChampionChallenger(trained.bundle, trained.bundle)
    few = make_frame(10, 10)
    with pytest.raises(errors.ConfigError, match="at least"):
        pair.compare(few.drop(columns="y"), few["y"])


def test_comparison_resamples_whole_groups_and_time_blocks(
    trained, monkeypatch
):
    import copy
    import dataclasses

    from scml.evaluation import diagnostics

    seen = []
    real = diagnostics.resample_rows

    def spy(rows, groups, temporal, rng):
        seen.append((groups is not None, temporal))
        return real(rows, groups, temporal, rng)

    monkeypatch.setattr(diagnostics, "resample_rows", spy)
    holdout = make_frame(300, 12)
    holdout["entity"] = np.arange(300) % 20
    holdout["when"] = pd.date_range("2024-01-01", periods=300, freq="D")
    features = holdout.drop(columns="y")

    clustered = copy.copy(trained.bundle)
    clustered.schema = dataclasses.replace(
        trained.bundle.schema, group_column="entity"
    )
    champion.ChampionChallenger(clustered, clustered, bootstraps=5).compare(
        features, holdout["y"]
    )
    assert seen and all(group and not block for group, block in seen)

    seen.clear()
    timed = copy.copy(trained.bundle)
    timed.schema = dataclasses.replace(
        trained.bundle.schema, time_column="when"
    )
    champion.ChampionChallenger(timed, timed, bootstraps=5).compare(
        features, holdout["y"]
    )
    assert seen and all(block and not group for group, block in seen)
