"""The feature graph inside real runs: fitting, packaging and inference."""

import dataclasses
import json
import pathlib

import numpy as np
import pandas as pd
import pytest
import support

import mml
from mml.contracts import errors
from mml.contracts import tasks


def node(name, op, inputs, **params):
    return tasks.FeatureNode(
        name=name, op=op, inputs=list(inputs), params=params
    )


def graph_frame(rows=700, seed=0):
    rng = np.random.default_rng(seed)
    price = np.exp(rng.normal(3, 1, rows))
    area = np.exp(rng.normal(4, 0.5, rows))
    when = pd.date_range("2024-01-01", periods=rows, freq="6h")
    frame = pd.DataFrame(
        {
            "price": price,
            "area": area,
            "when": when,
            "seg": rng.choice(["x", "y"], rows),
        }
    )
    unit = price / area  # the signal is a *ratio*, invisible to simple models
    frame["y"] = (
        np.log(unit) + 0.3 * (when.month % 2) + rng.normal(0, 0.3, rows)
        > np.log(unit).mean()
    ).astype(int)
    return frame


GRAPH = tasks.FeatureGraphSpec(
    nodes=[
        node("unit_price", "ratio", ["price", "area"]),
        node("log_unit", "log1p", ["unit_price"]),
        node("month", "datepart", ["when"], part="month"),
        node("area_band", "bin", ["area"], bins=4),
        node("price_clip", "clip", ["price"], lower=0.02, upper=0.98),
    ]
)


def run(frame, tmp_path, graph=GRAPH, models=None, **kwargs):
    project = support.make_project(
        frame,
        "y",
        time_budget=300,
        output_dir=tmp_path,
        feature_graph=graph,
        **kwargs,
    )
    if models is None:
        return project.run()
    plan = dataclasses.replace(project.plan(), candidate_models=tuple(models))
    return project.run(plan)


def interaction_frame(rows=800, seed=3):
    """The label is the sign of x1 * x2: invisible to any linear model."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {"x1": rng.normal(size=rows), "x2": rng.normal(size=rows)}
    )
    frame["when"] = pd.date_range("2024-01-01", periods=rows, freq="h")
    frame["y"] = (frame.x1 * frame.x2 + rng.normal(0, 0.1, rows) > 0).astype(
        int
    )
    return frame


def test_derived_features_are_used_and_inference_needs_only_raw_columns(
    tmp_path,
):
    frame = interaction_frame()
    graph = tasks.FeatureGraphSpec(
        nodes=[
            node("cross", "product", ["x1", "x2"]),
            node("month", "datepart", ["when"], part="month"),
        ]
    )
    linear = ("dummy", "logistic_regression")
    result = run(frame, tmp_path / "with", graph=graph, models=linear)
    plain = run(
        frame,
        tmp_path / "without",
        graph=tasks.FeatureGraphSpec(),
        models=linear,
    )
    plan = result.profile.feature_plan
    assert {"cross", "month"} <= set(plan.numeric)
    assert "when" in plan.dropped and "when" in plan.extra_inputs
    assert set(result.bundle.features) == {"x1", "x2", "when"}
    assert "cross" not in result.bundle.features  # computed, not supplied
    with_graph = result.evaluation["test_metrics"]["roc_auc"]
    without = plain.evaluation["test_metrics"]["roc_auc"]
    assert without < 0.6 and with_graph > 0.95  # XOR is linear in the product
    raw = frame.drop(columns=["y"])
    assert len(result.predict(raw)) == len(frame)
    with pytest.raises(errors.SchemaError, match="missing"):
        result.predict(raw.drop(columns=["when"]))


def test_bundle_with_a_graph_roundtrips_and_records_its_fingerprint(tmp_path):
    frame = graph_frame()
    result = run(frame, tmp_path)
    loaded = mml.ModelBundle.load(
        result.save(tmp_path / "export"), trusted=True
    )
    raw = frame.drop(columns=["y"])
    np.testing.assert_allclose(
        loaded.predict_proba(raw), result.predict_proba(raw)
    )
    assert loaded.schema.graph_fingerprint == GRAPH.fingerprint()
    events = (result.run_dir / "events.jsonl").read_text().splitlines()
    graph_events = [
        json.loads(e)
        for e in events
        if json.loads(e)["kind"] == "features.graph"
    ]
    assert graph_events[0]["data"]["fingerprint"] == GRAPH.fingerprint()
    assert len(graph_events[0]["data"]["nodes"]) == 5


def test_fitted_nodes_are_learned_from_training_rows_only(tmp_path):
    frame = graph_frame()
    result = run(frame, tmp_path)
    manifest = json.loads(
        (result.run_dir / "validation_manifest.json").read_text()
    )
    development = frame.iloc[manifest["train_ids"]]
    final = result.bundle.pipeline
    while not hasattr(final, "named_steps") and hasattr(final, "estimator_"):
        final = final.estimator_
    if not hasattr(final, "named_steps") or "graph" not in final.named_steps:
        pytest.skip("final estimator hides the graph step")
    states = final.named_steps["graph"].states_
    low, high = np.quantile(development["price"], [0.02, 0.98])
    assert states["price_clip"] == pytest.approx((low, high))
    assert not np.allclose(
        states["price_clip"], np.quantile(frame["price"], [0.02, 0.98])
    )


def test_a_different_graph_is_a_different_configuration(tmp_path):
    frame = graph_frame()
    other = tasks.FeatureGraphSpec(
        nodes=[node("unit_price", "difference", ["price", "area"])]
    )
    one, two = (
        run(frame, tmp_path / "a"),
        run(frame, tmp_path / "b", graph=other),
    )
    first = json.loads((one.run_dir / "config.json").read_text())
    second = json.loads((two.run_dir / "config.json").read_text())
    assert first["feature_graph"] != second["feature_graph"]


def test_invalid_graphs_are_refused_before_training(tmp_path):
    frame = graph_frame()
    bad = tasks.FeatureGraphSpec(
        nodes=[node("oops", "log1p", ["not_a_column"])]
    )
    with pytest.raises(errors.ConfigError, match="unknown columns") as caught:
        run(frame, tmp_path, graph=bad)
    events_path = pathlib.Path(caught.value.details["run_dir"]) / "events.jsonl"
    last = json.loads(events_path.read_text().splitlines()[-1])
    assert last["kind"] == "state.FAILED"
    leaky = tasks.FeatureGraphSpec(nodes=[node("t", "square", ["y"])])
    with pytest.raises(errors.ConfigError, match="target"):
        run(frame, tmp_path / "leak", graph=leaky)


def series_frame(entities=25, steps=40, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for entity in range(entities):
        level = rng.normal(0, 1)
        for step in range(steps):
            level = 0.8 * level + rng.normal(0, 0.5)
            rows.append(
                {
                    "unit": f"u{entity}",
                    "day": pd.Timestamp("2024-01-01") + pd.Timedelta(days=step),
                    "load": level,
                    "extra": rng.normal(),
                }
            )
    frame = (
        pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)
    )
    ordered = frame.sort_values(["unit", "day"])
    next_load = ordered.groupby("unit")["load"].shift(
        -1
    )  # outcome known only afterwards
    frame["y"] = (
        next_load.reindex(frame.index)
        > ordered.groupby("unit")["load"].transform("mean").reindex(frame.index)
    ).astype(int)
    return frame.dropna(subset=["y"])


SERIES = tasks.FeatureGraphSpec(
    nodes=[
        node("load_prev", "lag", ["load"], k=1),
        node("load_change", "diff", ["load"], k=1),
        node("load_avg3", "rolling_mean", ["load"], window=3),
        node("hour_proxy", "datepart", ["day"], part="dayofweek"),
    ]
)


def run_series(frame, tmp_path, **kwargs):
    return support.make_project(
        frame,
        "y",
        time_budget=300,
        output_dir=tmp_path,
        time_column="day",
        group_column="unit",
        feature_graph=SERIES,
        validation="group",
        **kwargs,
    ).run()


def test_time_series_features_are_generated_within_groups_and_scored_with_history(
    tmp_path,
):
    frame = series_frame()
    result = run_series(frame, tmp_path)
    plan = result.profile.feature_plan
    assert {"load_prev", "load_change", "load_avg3"} <= set(plan.numeric)
    assert [n.name for n in plan.series] == [
        "load_prev",
        "load_change",
        "load_avg3",
    ]
    assert {"unit", "day"} <= set(result.bundle.schema.extra_inputs)
    raw = frame.drop(columns=["y"])
    predictions = result.predict(raw)  # batch carries its own history
    assert len(predictions) == len(raw)
    batch = raw[raw.unit.isin(["u0", "u1"])]
    assert len(result.predict(batch)) == len(batch)
    # Features of a row never depend on later rows of the same entity.
    truncated = batch[
        batch.day <= batch.day.sort_values().iloc[len(batch) // 2]
    ]
    full_view = result.bundle.prepare_frame(batch)
    cut_view = result.bundle.prepare_frame(truncated)
    shared = cut_view.index
    pd.testing.assert_frame_equal(cut_view, full_view.loc[shared])


def test_search_is_invariant_to_future_target_rewrites_with_a_graph(tmp_path):
    frame = series_frame()
    first = run_series(frame, tmp_path / "a")
    manifest = json.loads(
        (first.run_dir / "validation_manifest.json").read_text()
    )
    altered = frame.copy()
    rng = np.random.default_rng(5)
    test_rows = frame.index[manifest["test_ids"]]
    altered.loc[test_rows, "y"] = rng.permutation(
        altered.loc[test_rows, "y"].to_numpy()
    )
    altered.loc[test_rows, "load"] = rng.normal(9, 1, len(test_rows))
    second = run_series(altered, tmp_path / "b")
    assert [t.params for t in first.trials] == [t.params for t in second.trials]
    for a, b in zip(first.trials, second.trials, strict=True):
        assert a.fold_scores == pytest.approx(b.fold_scores)


def test_a_single_new_record_gets_lag_and_rolling_values_from_stored_history(
    tmp_path,
):
    frame = series_frame()
    result = run_series(frame, tmp_path)
    raw = frame.drop(columns=["y"])
    unit = raw[raw.unit == "u3"].sort_values("day")
    last, history = unit.iloc[[-1]], unit.iloc[:-1]
    # A new day after everything the model saw: the record arrives alone.
    new = last.copy()
    new["day"] = last.day + pd.Timedelta(days=1)
    new["load"] = 0.5
    alone = result.bundle.prepare_frame(new)
    assert alone[["load_prev", "load_change", "load_avg3"]].notna().all().all()
    # It equals the features computed with the full history in the batch.
    together = result.bundle.prepare_frame(pd.concat([unit, new]))
    expected = together.iloc[[-1]]
    pd.testing.assert_frame_equal(
        alone.reset_index(drop=True), expected.reset_index(drop=True)
    )
    assert alone["load_prev"].iloc[0] == pytest.approx(last["load"].iloc[0])
    assert len(history) > 0


def test_history_survives_save_and_load_and_tampering_is_detected(tmp_path):
    result = run_series(series_frame(), tmp_path / "run")
    from mml.artifacts import bundle as bundle_module

    directory = tmp_path / "saved"
    result.bundle.save(directory)
    assert (directory / bundle_module.HISTORY_FILE).exists()
    loaded = bundle_module.ModelBundle.load(directory, trusted=True)
    pd.testing.assert_frame_equal(loaded.history, result.bundle.history)
    (directory / bundle_module.HISTORY_FILE).write_bytes(b"tampered")
    with pytest.raises(errors.ArtifactError, match="series history"):
        bundle_module.ModelBundle.load(directory, trusted=True)


def test_serving_remembers_records_so_the_next_single_record_sees_them(
    tmp_path,
):
    from mml.inference import serving

    result = run_series(series_frame(), tmp_path)
    adapter = serving.BatchServing(result.bundle)
    raw = series_frame().drop(columns=["y"])
    last_day = raw.day.max()
    first = raw[raw.unit == "u1"].iloc[[0]].copy()
    first["day"] = last_day + pd.Timedelta(days=1)
    first["load"] = 7.0
    adapter.respond(first.to_dict(orient="records"))
    second = first.copy()
    second["day"] = last_day + pd.Timedelta(days=2)
    second["load"] = 1.0
    view = result.bundle.prepare_frame(second)
    assert view["load_prev"].iloc[0] == pytest.approx(7.0)
