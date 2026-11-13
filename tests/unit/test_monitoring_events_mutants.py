"""Pins for prediction-log behaviours the integration suite left unchecked.

A stub bundle stands in for a trained one, so each test names exactly what
the log is expected to write.
"""

import dataclasses
import hashlib
import json
import types

import numpy
import pandas
import pytest

import scmle.artifacts.bundle
from scmle.contracts import observability
from scmle.monitoring import events


def make_policy(retention_days=30, redacted=("features",)):
    return observability.Requirements(
        retention_days=retention_days,
        required_signals=(observability.SignalKind.LATENCY,),
        redacted_fields=redacted,
        alert_thresholds={},
    )


def make_bundle(task="binary_classification", fail=None, files=None):
    def prepare_frame(data):
        if fail is not None:
            raise fail
        return data.copy()

    def predict(data):
        return numpy.arange(len(data)) % 2

    def predict_proba(data):
        return numpy.column_stack(
            [numpy.full(len(data), 0.25), numpy.full(len(data), 0.75)]
        )

    return types.SimpleNamespace(
        prepare_frame=prepare_frame,
        predict=predict,
        predict_proba=predict_proba,
        schema=types.SimpleNamespace(task=task),
        info=types.SimpleNamespace(
            model_version="v7",
            files={} if files is None else files,
        ),
    )


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_the_sidecar_sits_beside_the_log_with_its_own_suffix(tmp_path):
    log = events.PredictionLog(tmp_path / "p.jsonl")
    assert log.observability_path == tmp_path / "p.observability.jsonl"
    assert log.observability_policy is None
    assert log.run_id is None


def test_append_writes_one_typed_json_line(tmp_path):
    log = events.PredictionLog(tmp_path / "p.jsonl")
    log.append("note", {"when": pandas.Timestamp("2026-01-02"), "n": 3})
    log.append("note", {"n": 4})
    text = (tmp_path / "p.jsonl").read_text()
    assert text.endswith("\n")
    lines = text.splitlines()
    assert json.loads(lines[0]) == {
        "type": "note",
        "when": "2026-01-02 00:00:00",
        "n": 3,
    }
    assert lines[1] == '{"type": "note", "n": 4}'


def test_predictions_are_logged_with_features_probability_and_version(
    tmp_path,
):
    log = events.PredictionLog(tmp_path / "p.jsonl")
    data = pandas.DataFrame({"a": [1.5, None], "b": ["x", "y"]})
    logged = log.log_predictions(make_bundle(), data, ["e1", "e2"])
    assert [e.event_id for e in logged] == ["e1", "e2"]
    assert [e.model_version for e in logged] == ["v7", "v7"]
    assert [e.prediction for e in logged] == [0, 1]
    assert [e.probability for e in logged] == [0.75, 0.75]
    assert logged[1].features == {"a": None, "b": "y"}
    assert logged[0].features == {"a": 1.5, "b": "x"}
    assert all(e.timestamp.endswith("+00:00") for e in logged)
    stored = records(tmp_path / "p.jsonl")
    assert [r["type"] for r in stored] == ["prediction", "prediction"]
    assert stored[0] == {"type": "prediction", **dataclasses.asdict(logged[0])}
    assert not (tmp_path / "p.observability.jsonl").exists()


def test_a_regression_bundle_logs_no_probability(tmp_path):
    log = events.PredictionLog(tmp_path / "p.jsonl")
    data = pandas.DataFrame({"a": [1.0, 2.0, 3.0]})
    logged = log.log_predictions(make_bundle("regression"), data, None)
    assert [e.probability for e in logged] == [None, None, None]
    assert len({e.event_id for e in logged}) == 3
    assert all(len(e.event_id) == 32 for e in logged)


def test_event_ids_must_match_the_rows(tmp_path):
    log = events.PredictionLog(tmp_path / "p.jsonl")
    data = pandas.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(
        ValueError, match="event_ids must have one id per input row"
    ):
        log.log_predictions(make_bundle(), data, ["only-one"])
    with pytest.raises(ValueError, match="one id per input row"):
        log.log_predictions(make_bundle(), data, ["a", "b", "c"])
    assert not (tmp_path / "p.jsonl").exists()


def test_the_sidecar_carries_one_redacted_success_event_per_request(tmp_path):
    log = events.PredictionLog(
        tmp_path / "p.jsonl",
        observability_policy=make_policy(redacted=("email", "features")),
        run_id="serve-9",
    )
    data = pandas.DataFrame({"a": [1.0, 2.0]})
    bundle = make_bundle(
        files={scmle.artifacts.bundle.MODEL_FILE: "model-digest"}
    )
    log.log_predictions(bundle, data, ["x", "y"])
    sidecar = records(log.observability_path)
    assert [r["request_id"] for r in sidecar] == ["x", "y"]
    for record in sidecar:
        assert record["type"] == "inference"
        assert record["run_id"] == "serve-9"
        assert record["model_fingerprint"] == "model-digest"
        assert record["outcome"] == "success"
        assert record["error_category"] is None
        assert record["redacted_fields"] == ["email", "features"]
        assert record["latency_ms"] >= 0.0
        assert record["timestamp"].endswith("+00:00")
    assert sidecar[0]["input_fingerprint"] == sidecar[1]["input_fingerprint"]
    expected = hashlib.sha256(
        pandas.util.hash_pandas_object(data, index=False)
        .to_numpy(dtype="uint64")
        .tobytes()
    ).hexdigest()
    assert sidecar[0]["input_fingerprint"] == expected
    assert log.observability_path.read_text().count("\n") == 2
    assert ", " not in log.observability_path.read_text().splitlines()[0]


def test_the_run_id_and_model_fingerprint_fall_back_to_the_model_version(
    tmp_path,
):
    log = events.PredictionLog(
        tmp_path / "p.jsonl", observability_policy=make_policy()
    )
    log.log_predictions(make_bundle(), pandas.DataFrame({"a": [1.0]}), ["only"])
    (record,) = records(log.observability_path)
    assert record["run_id"] == "v7"
    assert record["model_fingerprint"] == "v7"


@pytest.mark.parametrize("failure", [TypeError, ValueError])
def test_unhashable_cells_fall_back_to_a_json_fingerprint(
    tmp_path, monkeypatch, failure
):
    def refuse(*args, **kwargs):
        raise failure("unhashable")

    monkeypatch.setattr(pandas.util, "hash_pandas_object", refuse)
    log = events.PredictionLog(
        tmp_path / "p.jsonl", observability_policy=make_policy()
    )
    data = pandas.DataFrame({"a": [1.5, 2.5]})
    log.log_predictions(make_bundle(), data, ["u", "v"])
    (first, second) = records(log.observability_path)
    payload = data.to_json(
        orient="split", date_format="iso", default_handler=str
    ).encode("utf-8")
    assert second["input_fingerprint"] == first["input_fingerprint"]
    assert first["input_fingerprint"] == hashlib.sha256(payload).hexdigest()


def test_a_failure_is_logged_as_an_error_event_and_propagates(tmp_path):
    log = events.PredictionLog(
        tmp_path / "p.jsonl", observability_policy=make_policy()
    )
    data = pandas.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(KeyError):
        log.log_predictions(
            make_bundle(fail=KeyError("boom")), data, ["first", "second"]
        )
    (record,) = records(log.observability_path)
    assert record["request_id"] == "first"
    assert record["outcome"] == "error"
    assert record["error_category"] == "KeyError"
    assert record["input_fingerprint"] is None
    assert not (tmp_path / "p.jsonl").exists()


def test_a_failure_without_event_ids_still_gets_a_request_id(tmp_path):
    log = events.PredictionLog(
        tmp_path / "p.jsonl", observability_policy=make_policy()
    )
    with pytest.raises(ValueError, match="bad"):
        log.log_predictions(
            make_bundle(fail=ValueError("bad")),
            pandas.DataFrame({"a": [1.0]}),
        )
    (record,) = records(log.observability_path)
    assert len(record["request_id"]) == 32
    assert record["error_category"] == "ValueError"
    empty = events.PredictionLog(
        tmp_path / "q.jsonl", observability_policy=make_policy()
    )
    with pytest.raises(ValueError, match="bad"):
        empty.log_predictions(
            make_bundle(fail=ValueError("bad")),
            pandas.DataFrame({"a": []}),
        )
    (record,) = records(empty.observability_path)
    assert len(record["request_id"]) == 32


def test_log_observability_does_nothing_without_a_policy(tmp_path):
    log = events.PredictionLog(tmp_path / "p.jsonl")
    log.log_observability(
        make_bundle(), "r", 0.0, "success", input_fingerprint=None
    )
    assert not log.observability_path.exists()


def test_the_error_category_and_latency_are_recorded(tmp_path):
    log = events.PredictionLog(
        tmp_path / "p.jsonl", observability_policy=make_policy()
    )
    log.log_observability(
        make_bundle(),
        "r1",
        0.0,
        "error",
        input_fingerprint="fp",
        error_category="OSError",
    )
    (record,) = records(log.observability_path)
    assert record["error_category"] == "OSError"
    assert record["input_fingerprint"] == "fp"
    assert record["latency_ms"] > 0.0


def sidecar_line(timestamp):
    return json.dumps({"type": "inference", "timestamp": timestamp}) + "\n"


def test_retention_reports_the_window_and_keeps_the_boundary_event(tmp_path):
    log = events.PredictionLog(
        tmp_path / "p.jsonl", observability_policy=make_policy(2)
    )
    kept = sidecar_line("2026-10-18T10:00:00+00:00")
    log.observability_path.write_text(
        sidecar_line("2026-10-18T09:59:59+00:00")
        + kept
        + sidecar_line("2026-10-20T10:00:00+00:00"),
        encoding="utf-8",
    )
    result = log.enforce_retention("2026-10-20T10:00:00+00:00")
    assert result == {
        "observed_at": "2026-10-20T10:00:00+00:00",
        "cutoff": "2026-10-18T10:00:00+00:00",
        "retained": 2,
        "removed": 1,
    }
    assert log.observability_path.read_text(encoding="utf-8").startswith(kept)
    leftovers = [p.name for p in tmp_path.iterdir()]
    assert sorted(leftovers) == ["p.observability.jsonl"]


def test_retention_without_a_sidecar_reports_zero_counts(tmp_path):
    log = events.PredictionLog(
        tmp_path / "p.jsonl", observability_policy=make_policy(1)
    )
    assert log.enforce_retention("2026-10-20T10:00:00+00:00") == {
        "observed_at": "2026-10-20T10:00:00+00:00",
        "cutoff": "2026-10-19T10:00:00+00:00",
        "retained": 0,
        "removed": 0,
    }
    now = log.enforce_retention()
    assert now["observed_at"].endswith("+00:00")


def test_retention_demands_a_policy_and_a_valid_time(tmp_path):
    bare = events.PredictionLog(tmp_path / "p.jsonl")
    with pytest.raises(
        ValueError, match="retention enforcement requires an observability"
    ):
        bare.enforce_retention("2026-10-20T10:00:00+00:00")
    log = events.PredictionLog(
        tmp_path / "p.jsonl", observability_policy=make_policy()
    )
    with pytest.raises(ValueError, match="observed_at"):
        log.enforce_retention("not-a-time")


def test_retention_keeps_every_event_when_the_policy_is_not_expired(tmp_path):
    log = events.PredictionLog(
        tmp_path / "p.jsonl", observability_policy=make_policy(30)
    )
    original = sidecar_line("2026-10-19T10:00:00+00:00") * 2
    log.observability_path.write_text(original, encoding="utf-8")
    result = log.enforce_retention("2026-10-20T10:00:00+00:00")
    assert (result["retained"], result["removed"]) == (2, 0)
    assert log.observability_path.read_text(encoding="utf-8") == original


def test_inference_timestamp_accepts_only_inference_objects():
    parsed = events.inference_timestamp(
        '{"type":"inference","timestamp":"2026-10-18T10:00:00+00:00"}'
    )
    assert parsed.isoformat() == "2026-10-18T10:00:00+00:00"
    with pytest.raises(ValueError, match="must be JSON objects"):
        events.inference_timestamp("[1]")
    with pytest.raises(ValueError, match="must have type inference"):
        events.inference_timestamp('{"type":"error","timestamp":"x"}')
    with pytest.raises(ValueError, match="must have type inference"):
        events.inference_timestamp('{"timestamp":"2026-10-18T10:00:00+00:00"}')
    with pytest.raises(ValueError, match=r"event\.timestamp"):
        events.inference_timestamp('{"type":"inference","timestamp":"nope"}')


def test_outcomes_are_logged_in_order_and_joined_by_event_id(tmp_path):
    log = events.PredictionLog(tmp_path / "p.jsonl")
    data = pandas.DataFrame({"a": [1.0, 2.0, 3.0]})
    log.log_predictions(make_bundle(), data, ["a", "b", "c"])
    logged = log.record_outcomes({"c": numpy.int64(1), "a": 0})
    assert [e.event_id for e in logged] == ["c", "a"]
    assert [e.actual for e in logged] == [1, 0]
    assert type(logged[0].actual) is int
    assert logged[0].timestamp.endswith("+00:00")
    stored = records(tmp_path / "p.jsonl")
    assert [r["type"] for r in stored][-2:] == ["outcome", "outcome"]
    features, predicted, actual = log.load()
    assert list(features.index) == ["a", "b", "c"]
    assert list(features.columns) == ["a"]
    assert predicted.tolist() == [0, 1, 0]
    assert actual["a"] == 0 and actual["c"] == 1
    assert numpy.isnan(actual["b"])


def test_a_later_record_replaces_an_earlier_one_for_the_same_event(tmp_path):
    log = events.PredictionLog(tmp_path / "p.jsonl")
    log.record_outcomes({"a": 0})
    log.record_outcomes({"a": 1})
    log.append(
        "prediction",
        {"event_id": "a", "features": {"f": 1}, "prediction": "old"},
    )
    log.append(
        "prediction",
        {"event_id": "a", "features": {"f": 2}, "prediction": "new"},
    )
    features, predicted, actual = log.load()
    assert features["f"].tolist() == [2]
    assert predicted.tolist() == ["new"]
    assert actual.tolist() == [1]


def test_to_plain_unwraps_numpy_scalars_only():
    assert type(events.to_plain(numpy.float64(1.5))) is float
    assert events.to_plain("text") == "text"
    assert events.to_plain(None) is None
