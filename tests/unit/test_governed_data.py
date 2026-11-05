"""Snapshots, quality gates and incremental processing on every backend."""

import datetime
import json

import pandas as pd
import pytest

from scmle.contracts import errors
from scmle.core import retry
from scmle.data.readers import connectors
from scmle.data.store import cloud
from scmle.data.store import objectstore
from scmle.data.store import processing
from scmle.data.store import quality
from scmle.data.store import snapshot
from tests import support_cloud as fakes

CONTRACT = snapshot.DataContract(
    "orders",
    "2",
    {"id": "int", "amount": "float", "region": "string"},
    max_age_seconds=7200,
    key_columns=("id",),
)
CSV = "id,amount,region\n1,10.5,north\n2,20.0,south\n3,5.25,north\n"


def make_store(kind, tmp_path):
    if kind == "local":
        return objectstore.LocalObjectStore(tmp_path / "root")
    if kind == "s3":
        return cloud.S3ObjectStore(fakes.FakeS3Client(), "bkt", "t1")
    return cloud.GcsObjectStore(fakes.FakeGcsClient(), "bkt", "t1")


@pytest.fixture(params=["local", "s3", "gcs"])
def store(request, tmp_path):
    return make_store(request.param, tmp_path)


def reader_for(store, key="o.csv", hours=0.0):
    """A reader whose clock is `hours` after the object was modified."""
    modified = snapshot.parse_time(store.head(key).modified)
    now = modified + datetime.timedelta(hours=hours)
    return snapshot.SnapshotReader(store, clock=lambda: now)


def test_every_read_records_identity_schema_freshness_and_lineage(store):
    store.put("in/orders.csv", CSV.encode())
    frame, record = reader_for(store, "in/orders.csv").read(
        "in/orders.csv", CONTRACT
    )
    assert len(frame) == 3
    assert record.uri == store.uri("in/orders.csv")
    assert record.sha256 == objectstore.sha256_of(CSV.encode())
    assert record.size == len(CSV) and record.rows == 3
    assert record.schema_version == "2" and record.schema_matches
    assert record.schema_digest == CONTRACT.digest
    assert record.freshness == snapshot.FRESH
    assert record.snapshot_id.startswith("snap-")
    ignored, child = reader_for(store, "in/orders.csv").read(
        "in/orders.csv", CONTRACT, parents=[record]
    )
    assert child.lineage == (record.snapshot_id,)


def test_snapshot_ids_follow_the_content(store):
    store.put("o.csv", CSV.encode())
    reader = reader_for(store)
    ignored, first = reader.read("o.csv", CONTRACT)
    ignored, again = reader.read("o.csv", CONTRACT)
    store.put("o.csv", (CSV + "4,1.0,south\n").encode())
    ignored, changed = reader.read("o.csv", CONTRACT)
    assert first.snapshot_id == again.snapshot_id
    assert changed.snapshot_id != first.snapshot_id
    assert len(reader.log.records) == 3


def test_staleness_and_schema_drift_are_recorded_not_hidden(store):
    store.put("o.csv", CSV.encode())
    ignored, stale = reader_for(store, hours=5).read("o.csv", CONTRACT)
    assert stale.freshness == snapshot.STALE and stale.age_seconds > 7200
    store.put("d.csv", b"id,amount\n1,2.0\n")
    ignored, drifted = reader_for(store, "d.csv").read("d.csv", CONTRACT)
    assert not drifted.schema_matches
    assert drifted.schema_digest != CONTRACT.digest
    open_ended = snapshot.DataContract("x", "1", {"id": "int"})
    ignored, free = reader_for(store, "d.csv").read("d.csv", open_ended)
    assert free.freshness == snapshot.UNKNOWN


def test_the_read_log_persists_and_traces_lineage(store, tmp_path):
    log = snapshot.ReadLog(tmp_path / "reads.jsonl")
    store.put("raw.csv", CSV.encode())
    store.put("clean.csv", CSV.encode() + b"9,1.0,south\n")
    reader = snapshot.SnapshotReader(store, log)
    ignored, raw = reader.read("raw.csv", CONTRACT)
    ignored, clean = reader.read("clean.csv", CONTRACT, parents=[raw])
    lines = (tmp_path / "reads.jsonl").read_text().splitlines()
    assert [json.loads(line)["snapshot_id"] for line in lines] == [
        raw.snapshot_id,
        clean.snapshot_id,
    ]
    trace = log.trace(clean.snapshot_id)
    assert [r.snapshot_id for r in trace] == [
        clean.snapshot_id,
        raw.snapshot_id,
    ]
    with pytest.raises(snapshot.SnapshotError):
        log.trace("snap-unknown")


def test_unreadable_objects_fail_clearly(store):
    store.put("x.bin", b"\x00\x01")
    with pytest.raises(snapshot.SnapshotError):
        reader_for(store, "x.bin").read("x.bin", CONTRACT)
    with pytest.raises(objectstore.NotFoundError):
        snapshot.SnapshotReader(store).read("missing.csv", CONTRACT)


def test_the_phase_9_connector_reads_through_a_store(store):
    if store.scheme == "file":  # the connector admits cloud schemes only
        return
    store.put("in/orders.csv", CSV.encode())
    connector = connectors.ObjectStoreConnector(
        store.uri("in/orders.csv"), snapshot.connector_reader(store)
    )
    assert len(connector.load_frame()) == 3


def test_a_contract_is_validated():
    for bad in (
        {"name": "", "schema_version": "1", "columns": {"a": "int"}},
        {"name": "n", "schema_version": "1", "columns": {"a": "decimal"}},
        {
            "name": "n",
            "schema_version": "1",
            "columns": {"a": "int"},
            "key_columns": ("b",),
        },
        {
            "name": "n",
            "schema_version": "1",
            "columns": {"a": "int"},
            "max_age_seconds": 0,
        },
    ):
        with pytest.raises(errors.ConfigError):
            snapshot.DataContract(**bad)


# ---- quality gates -------------------------------------------------------


def prepared(store, text=CSV, hours=0.0):
    store.put("o.csv", text.encode())
    return reader_for(store, hours=hours).read("o.csv", CONTRACT)


def gate_for(store, *rules):
    return quality.QualityGate(rules, store)


def test_a_clean_dataset_passes_both_stages(store):
    frame, record = prepared(store)
    gate = gate_for(
        store,
        quality.SchemaRule(),
        quality.FreshnessRule(),
        quality.MinRowsRule(3),
        quality.UniqueKeyRule(),
    )
    for stage in quality.Stage:
        result = gate.run(frame, record, CONTRACT, stage)
        assert result.report.decision == "passed" and len(result.frame) == 3
    assert store.exists(f"quality/{record.snapshot_id}-train.json")


def test_fail_policy_stops_the_run_with_the_report(store):
    frame, record = prepared(store, hours=9)
    gate = gate_for(store, quality.FreshnessRule(policy=quality.Policy.FAIL))
    with pytest.raises(quality.QualityGateError) as raised:
        gate.run(frame, record, CONTRACT, quality.Stage.TRAIN)
    assert not raised.value.quarantined
    assert raised.value.report.decision == "failed"
    assert "freshness" in str(raised.value)


def test_row_level_quarantine_removes_and_keeps_the_rows(store):
    frame, record = prepared(
        store, CSV + "2,20.0,south\n4,-5.0,north\n5,5000.0,north\n"
    )
    gate = gate_for(
        store,
        quality.UniqueKeyRule(policy=quality.Policy.QUARANTINE),
        quality.RangeRule("amount", 0, 1000, policy=quality.Policy.QUARANTINE),
    )
    result = gate.run(frame, record, CONTRACT, quality.Stage.INFERENCE)
    assert list(result.frame["id"]) == [1, 2, 3]
    assert list(result.quarantined["id"]) == [2, 4, 5]
    assert result.report.decision == "rows_quarantined"
    assert result.report.rows_in == 6 and result.report.rows_out == 3
    held = store.get(f"quarantine/{record.snapshot_id}/rows-inference.csv")[0]
    assert held.decode().count("\n") == 4


def test_dataset_level_quarantine_holds_the_snapshot_until_released(store):
    frame, record = prepared(store, "id,amount,region\n1,1.0,north\n")
    gate = gate_for(
        store, quality.MinRowsRule(5, policy=quality.Policy.QUARANTINE)
    )
    with pytest.raises(quality.QualityGateError) as raised:
        gate.run(frame, record, CONTRACT, quality.Stage.TRAIN)
    assert raised.value.quarantined
    permissive = gate_for(store, quality.MinRowsRule(1))
    with pytest.raises(quality.QualityGateError, match="is quarantined"):
        permissive.run(frame, record, CONTRACT, quality.Stage.TRAIN)
    permissive.release(record.snapshot_id)
    assert permissive.run(frame, record, CONTRACT, quality.Stage.TRAIN)


def test_warn_policy_records_and_continues(store):
    frame, record = prepared(store, "id,amount,region\n1,,north\n2,2.0,south\n")
    gate = gate_for(
        store,
        quality.NullFractionRule("amount", 0.1, policy=quality.Policy.WARN),
        quality.NoNullsRule("amount", policy=quality.Policy.WARN),
    )
    result = gate.run(frame, record, CONTRACT, quality.Stage.TRAIN)
    assert result.report.decision == "passed_with_warnings"
    assert len(result.frame) == 2
    failed = [f.rule for f in result.report.findings if not f.passed]
    assert failed == ["null_fraction", "no_nulls"]


def test_rules_apply_only_to_their_stages(store):
    frame, record = prepared(store, "id,amount,region\n1,1.0,north\n")
    only_train = quality.MinRowsRule(10, stages=[quality.Stage.TRAIN])
    gate = gate_for(store, only_train)
    with pytest.raises(quality.QualityGateError):
        gate.run(frame, record, CONTRACT, quality.Stage.TRAIN)
    result = gate.run(frame, record, CONTRACT, quality.Stage.INFERENCE)
    assert result.report.findings == ()


def test_schema_drift_and_allowed_values_are_caught(store):
    store.put("d.csv", b"id,amount\n1,2.0\n")
    frame, record = reader_for(store, "d.csv").read("d.csv", CONTRACT)
    with pytest.raises(quality.QualityGateError, match="missing column region"):
        gate_for(store, quality.SchemaRule()).run(
            frame, record, CONTRACT, quality.Stage.INFERENCE
        )
    frame, record = prepared(store, CSV + "6,1.0,mars\n")
    gate = gate_for(
        store,
        quality.AllowedValuesRule(
            "region", ["north", "south"], policy=quality.Policy.QUARANTINE
        ),
    )
    result = gate.run(frame, record, CONTRACT, quality.Stage.TRAIN)
    assert list(result.quarantined["region"]) == ["mars"]


def test_a_gate_needs_rules_and_valid_policies():
    with pytest.raises(errors.ConfigError):
        quality.QualityGate([])
    with pytest.raises(ValueError):
        quality.MinRowsRule(1, policy="explode")


# ---- incremental processing ------------------------------------------------


def upper(frame):
    return frame.assign(region=frame["region"].str.upper())


def processor(store, **options):
    options.setdefault("sleep", lambda seconds: None)
    return processing.IncrementalProcessor(store, CONTRACT, upper, **options)


def seed(store):
    store.put("input/a.csv", CSV.encode())
    store.put(
        "input/b.csv",
        b"id,amount,region\n3,5.25,north\n4,8.0,south\n4,8.0,south\n5,1.0,north\n",
    )


def outputs(store):
    return {m.key: store.get(m.key)[0] for m in store.list_objects("output/")}


def test_incremental_runs_process_each_object_once_and_deduplicate(store):
    seed(store)
    first = processor(store).run()
    assert first.processed == ("input/a.csv", "input/b.csv")
    assert first.duplicate_rows == 2 and first.rows_out == 5
    assert first.failed is None
    rows = outputs(store)["output/b.csv"].decode().strip().splitlines()
    assert rows == ["id,amount,region", "4,8.0,SOUTH", "5,1.0,NORTH"]
    second = processor(store).run()
    assert second.processed == () and second.skipped == first.processed
    store.put("input/c.csv", b"id,amount,region\n6,2.0,south\n1,10.5,north\n")
    third = processor(store).run()
    assert third.processed == ("input/c.csv",)
    assert third.duplicate_rows == 1 and third.rows_out == 1


def test_a_repeated_upload_is_skipped_and_a_changed_input_is_refused(store):
    seed(store)
    processor(store).run()
    store.put("input/a-copy.csv", CSV.encode())
    result = processor(store).run()
    assert result.duplicate_objects == ("input/a-copy.csv",)
    assert result.processed == ()
    store.put("input/a.csv", (CSV + "7,1.0,south\n").encode())
    with pytest.raises(processing.ChangedInputError):
        processor(store).run()


def test_the_result_is_deterministic(tmp_path):
    first = make_store("local", tmp_path / "one")
    second = make_store("s3", tmp_path / "two")
    for target in (first, second):
        seed(target)
        processor(target).run()
    assert outputs(first) == outputs(second)


def test_a_crash_between_output_and_checkpoint_recovers_without_duplicates(
    store,
    tmp_path,
):
    seed(store)

    class Crash(BaseException):
        pass

    def crash_on_b(key):
        if key == "input/b.csv":
            raise Crash()

    crashing = processor(store, before_commit=crash_on_b)
    with pytest.raises(Crash):
        crashing.run()
    # A killed process cannot release its lock.
    store.put("state/lock", b"locked")
    assert "output/b.csv" in outputs(store)
    state = processing.Checkpoint.from_bytes(
        store.get("state/checkpoint.json")[0]
    )
    assert list(state.processed) == ["input/a.csv"]
    with pytest.raises(processing.ConcurrentRunError):
        processor(store).run()
    processor(store).break_lock()
    resumed = processor(store).run()
    assert resumed.processed == ("input/b.csv",)
    final = processing.Checkpoint.from_bytes(
        store.get("state/checkpoint.json")[0]
    )
    assert set(final.processed) == {"input/a.csv", "input/b.csv"}
    reference = make_store("local", tmp_path / "reference")
    seed(reference)
    processor(reference).run()
    assert outputs(store) == outputs(reference)


def test_transient_failures_are_retried_with_backoff(store):
    seed(store)
    calls = {"n": 0}

    def flaky(frame):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise retry.TransientError("warehouse busy")
        return upper(frame)

    waits = []
    run = processing.IncrementalProcessor(
        store,
        CONTRACT,
        flaky,
        policy=retry.RetryPolicy(4, 1.0, 2.0),
        sleep=waits.append,
    ).run()
    assert run.failed is None and run.processed == (
        "input/a.csv",
        "input/b.csv",
    )
    assert waits == [1.0, 2.0] and run.retries == (1.0, 2.0)


def test_exhausted_or_permanent_failures_stop_at_the_object_and_keep_progress(
    store,
    tmp_path,
):
    seed(store)

    def broken_on_b(frame):
        if 4 in frame["id"].tolist():
            raise errors.DataError("bad batch")
        return upper(frame)

    run = processing.IncrementalProcessor(
        store, CONTRACT, broken_on_b, sleep=lambda s: None
    ).run()
    assert run.processed == ("input/a.csv",)
    assert run.failed == "input/b.csv" and "bad batch" in run.error
    assert not store.exists("output/b.csv")
    assert not store.exists("state/lock")
    fixed = processor(store).run()
    assert fixed.processed == ("input/b.csv",)

    def always_busy(frame):
        raise retry.TransientError("down")

    other = make_store("local", tmp_path / "reference")
    seed(other)
    run = processing.IncrementalProcessor(
        other,
        CONTRACT,
        always_busy,
        policy=retry.RetryPolicy(2, 0.0),
        sleep=lambda s: None,
    ).run()
    assert run.failed == "input/a.csv" and "RetryExhaustedError" in run.error


def test_the_gate_runs_before_processing_and_quarantine_is_respected(
    store, tmp_path
):
    store.put("input/a.csv", (CSV + "9,-1.0,north\n").encode())
    gate = quality.QualityGate(
        [quality.RangeRule("amount", 0, 100, policy=quality.Policy.QUARANTINE)],
        store,
    )
    run = processor(store, gate=gate).run()
    assert run.rows_out == 3
    strict = quality.QualityGate([quality.RangeRule("amount", 0, 1)], store)
    other = make_store("local", tmp_path / "reference")
    other.put("input/a.csv", CSV.encode())
    failed = processing.IncrementalProcessor(
        other,
        CONTRACT,
        upper,
        gate=quality.QualityGate([quality.RangeRule("amount", 0, 1)], other),
        sleep=lambda s: None,
    ).run()
    assert failed.failed == "input/a.csv" and "QualityGateError" in failed.error
    assert strict


def test_a_damaged_checkpoint_is_never_silently_reset(store):
    seed(store)
    processor(store).run()
    data = store.get("state/checkpoint.json")[0]
    store.put(
        "state/checkpoint.json", data.replace(b"input/a.csv", b"input/z.csv")
    )
    with pytest.raises(processing.CheckpointCorruptError):
        processor(store).run()
    store.put("state/checkpoint.json", b"not json")
    with pytest.raises(processing.CheckpointCorruptError):
        processor(store).run()
    assert not store.exists("state/lock")


def test_a_second_run_cannot_share_the_checkpoint(store):
    seed(store)
    store.put("state/lock", b"someone")
    with pytest.raises(processing.ConcurrentRunError):
        processor(store).run()
    assert store.exists("state/lock")


def test_max_objects_bounds_a_run_and_the_next_continues(store):
    seed(store)
    first = processor(store).run(max_objects=1)
    assert first.processed == ("input/a.csv",)
    second = processor(store).run()
    assert second.processed == ("input/b.csv",) and second.skipped == (
        "input/a.csv",
    )


def test_processing_needs_key_columns():
    contract = snapshot.DataContract("x", "1", {"id": "int"})
    with pytest.raises(errors.ConfigError):
        processing.IncrementalProcessor(
            objectstore.LocalObjectStore("/tmp/unused"), contract, upper
        )


def test_the_frame_dtype_kinds_are_reader_independent():
    frame = pd.DataFrame({"a": [1], "b": [1.5], "c": ["x"], "d": [True]})
    assert snapshot.observed_schema(frame) == {
        "a": "int",
        "b": "float",
        "c": "string",
        "d": "bool",
    }
