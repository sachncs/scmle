"""Guarantees of docs/GOVERNED_DATA.md that no other test enforced."""

import hashlib

import pytest

from scmle.data.store import objectstore
from scmle.data.store import processing
from scmle.data.store import quality
from tests.unit import test_governed_data


@pytest.fixture
def store(tmp_path):
    return objectstore.LocalObjectStore(tmp_path / "root")


def test_metadata_carries_a_hash_of_the_bytes_read(store):
    """Enforces docs/GOVERNED_DATA.md:10."""
    data = b"id,amount\n1,2.0\n"
    assert store.put("a.csv", data).sha256 == hashlib.sha256(data).hexdigest()
    assert store.head("a.csv").sha256 == hashlib.sha256(data).hexdigest()
    changed = b"id,amount\n1,9.0\n"
    store.locate("a.csv").write_bytes(changed)
    body, meta = store.get("a.csv")
    assert body == changed
    assert meta.sha256 == hashlib.sha256(changed).hexdigest()


def failing(rule, frame):
    """The rule with a quarantine policy and a frame that breaks it."""
    if rule == "schema":
        return quality.SchemaCheck(policy=quality.Severity.QUARANTINE), (
            frame.drop(columns=["region"])
        )
    if rule == "freshness":
        return quality.Freshness(policy=quality.Severity.QUARANTINE), frame
    if rule == "rows":
        return quality.MinRowsCheck(
            50, policy=quality.Severity.QUARANTINE
        ), frame
    broken = frame.assign(amount=[None] * len(frame))
    return quality.NullFractionCheck(
        "amount", 0.1, policy=quality.Severity.QUARANTINE
    ), broken


@pytest.mark.parametrize("rule", ["schema", "freshness", "rows", "nulls"])
def test_a_dataset_rule_holds_the_snapshot_until_released(store, rule):
    """Enforces docs/GOVERNED_DATA.md:53 (dataset-level rules)."""
    hours = 9.0 if rule == "freshness" else 0.0
    frame, record = test_governed_data.prepared(store, hours=hours)
    check, bad = failing(rule, frame)
    with pytest.raises(quality.QualityGateError) as raised:
        quality.QualityGate([check], store).run(
            bad, record, test_governed_data.CONTRACT, quality.Stage.TRAIN
        )
    assert raised.value.quarantined
    assert store.list_objects(f"quarantine/{record.snapshot_id}")
    permissive = quality.QualityGate([quality.MinRowsCheck(1)], store)
    with pytest.raises(quality.QualityGateError, match="is quarantined"):
        permissive.run(
            frame, record, test_governed_data.CONTRACT, quality.Stage.TRAIN
        )
    permissive.release(record.snapshot_id)
    assert permissive.run(
        frame, record, test_governed_data.CONTRACT, quality.Stage.TRAIN
    )


def test_quarantined_rows_and_the_report_are_written_apart(store):
    """Enforces docs/GOVERNED_DATA.md:53 (row-level rules)."""
    frame, record = test_governed_data.prepared(
        store, test_governed_data.CSV + "4,-5.0,north\n"
    )
    check = quality.RangeCheck(
        "amount", 0, 1000, policy=quality.Severity.QUARANTINE
    )
    result = quality.QualityGate([check], store).run(
        frame, record, test_governed_data.CONTRACT, quality.Stage.TRAIN
    )
    assert list(result.quarantined["id"]) == [4]
    held = f"quarantine/{record.snapshot_id}/rows-train.csv"
    assert "4,-5.0,north" in store.get(held)[0].decode()
    assert store.exists(f"quality/{record.snapshot_id}-train.json")


def test_after_a_crash_the_transform_runs_again_on_that_object(store):
    """Enforces docs/GOVERNED_DATA.md:74."""
    test_governed_data.seed(store)
    seen = []

    def counted(frame):
        seen.append(tuple(frame["id"]))
        return test_governed_data.upper(frame)

    class Crash(BaseException):
        pass

    def crash_on_b(key):
        if key == "input/b.csv":
            raise Crash

    def build(**options):
        return processing.IncrementalProcessor(
            store,
            test_governed_data.CONTRACT,
            counted,
            sleep=lambda seconds: None,
            **options,
        )

    with pytest.raises(Crash):
        build(before_commit=crash_on_b).run()
    build().break_lock()
    assert build().run().processed == ("input/b.csv",)
    b_rows = [ids for ids in seen if 5 in ids]
    assert len(b_rows) == 2
