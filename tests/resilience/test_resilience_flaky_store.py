"""Transient and permanent store failures, torn objects and repeated uploads.

`FaultyStore` raises the repo's `TransientStoreError` for a window of calls.
Within the `Retries` budget the run must finish with the same bytes as a
clean run; past it, or on a permanent error, the run stops at that object
with the checkpoint as it was.
"""

import failpoints
import faulty_store
import pytest

from scmle.contracts import errors
from scmle.core import retry
from scmle.data.store import objectstore
from scmle.data.store import processing

POLICY = retry.Retries(max_attempts=4, base_delay_seconds=0.5, factor=2.0)
KEYS = failpoints.input_keys()
REAL = [key for key in KEYS if key != failpoints.copy_key()]

# Targets the processor retries itself: where the filter lands on the third
# object (index 2) unless the target is the listing.
TARGETS = {
    "listing": {"operations": ("list_objects",), "prefix": "input/"},
    "read": {"operations": ("get",), "prefix": "input/", "skip": 2},
    "output": {"operations": ("put",), "prefix": "output/", "skip": 2},
    "checkpoint": {
        "operations": ("put",),
        "prefix": failpoints.STATE_KEY,
        "skip": 2,
    },
}


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    return failpoints.reference_run(tmp_path_factory.mktemp("clean") / "root")


@pytest.fixture(scope="module")
def after_two(tmp_path_factory):
    """The checkpoint a run that committed two objects leaves."""
    store = objectstore.LocalObjectStore(
        tmp_path_factory.mktemp("two") / "root"
    )
    failpoints.seed(store)
    failpoints.processor(store).run(max_objects=2)
    return store.get(failpoints.STATE_KEY)[0]


@pytest.fixture
def store(tmp_path):
    local = objectstore.LocalObjectStore(tmp_path / "root")
    failpoints.seed(local)
    return local


def flaky_processor(local, failures, target, **options):
    faulty = faulty_store.FaultyStore(local, failures, **TARGETS[target])
    return faulty, failpoints.processor(faulty, policy=POLICY, **options)


def waits_of(result):
    return result.retries


@pytest.mark.parametrize("target", ["listing", "read", "output", "checkpoint"])
@pytest.mark.parametrize("failures", [1, 2, 3])
def test_failures_within_the_budget_give_identical_output(
    store, reference, target, failures
):
    target_skip = 0 if target == "listing" else TARGETS[target]["skip"]
    faulty, run = flaky_processor(store, failures, target)
    result = run.run()
    assert result.failed is None and result.error is None
    assert result.processed == tuple(REAL)
    assert result.retries == tuple(POLICY.delays()[:failures])
    assert faulty.matching > target_skip + failures
    assert failpoints.files_of(store) == reference.files
    assert store.get(failpoints.STATE_KEY)[0] == reference.checkpoint
    assert not store.exists(failpoints.LOCK_KEY)


def test_a_transient_transform_failure_within_the_budget_is_retried(
    store, reference
):
    calls = {"n": 0}

    def flaky(frame):
        calls["n"] += 1
        if calls["n"] in (2, 3):
            raise retry.TransientError("warehouse busy")
        return failpoints.upper(frame)

    result = processing.IncrementalProcessor(
        store,
        failpoints.CONTRACT,
        flaky,
        policy=POLICY,
        sleep=lambda seconds: None,
    ).run()
    assert result.failed is None
    assert result.retries == (0.5, 1.0)
    assert failpoints.files_of(store) == reference.files


@pytest.mark.parametrize("target", ["read", "output", "checkpoint"])
def test_failures_past_the_budget_stop_at_that_object_and_resume_cleanly(
    store, reference, after_two, target
):
    faulty, run = flaky_processor(store, POLICY.max_attempts, target)
    result = run.run()
    assert result.failed == "input/o2.csv"
    assert result.error.startswith("RetryExhaustedError")
    assert result.processed == ("input/o0.csv", "input/o1.csv")
    assert result.retries == tuple(POLICY.delays())
    assert faulty.matching == TARGETS[target]["skip"] + POLICY.max_attempts
    assert store.get(failpoints.STATE_KEY)[0] == after_two
    assert store.exists("output/o2.csv") is (target == "checkpoint")
    assert not store.exists("output/o3.csv")
    assert not store.exists(failpoints.LOCK_KEY)

    faulty.heal()
    resumed = failpoints.processor(faulty, policy=POLICY).run()
    assert resumed.failed is None
    assert resumed.processed == ("input/o2.csv", "input/o3.csv")
    assert resumed.skipped == ("input/o0.csv", "input/o1.csv")
    assert failpoints.files_of(store) == reference.files
    assert store.get(failpoints.STATE_KEY)[0] == reference.checkpoint


def test_a_listing_that_keeps_failing_raises_and_changes_nothing(
    store, after_two
):
    failpoints.processor(store).run(max_objects=2)
    before = failpoints.files_of(store)
    faulty, run = flaky_processor(store, POLICY.max_attempts, "listing")
    with pytest.raises(retry.RetryExhaustedError) as raised:
        run.run()
    assert raised.value.attempts == POLICY.max_attempts
    assert faulty.matching == POLICY.max_attempts
    assert store.get(failpoints.STATE_KEY)[0] == after_two
    assert failpoints.files_of(store) == before
    assert not store.exists(failpoints.LOCK_KEY)


def test_a_transform_that_keeps_failing_stops_at_that_object(store, after_two):
    def busy_on_third(frame):
        if 8 in frame["id"].tolist():
            raise retry.TransientError("down")
        return failpoints.upper(frame)

    result = processing.IncrementalProcessor(
        store,
        failpoints.CONTRACT,
        busy_on_third,
        policy=POLICY,
        sleep=lambda seconds: None,
    ).run()
    assert result.failed == "input/o2.csv"
    assert result.error.startswith("RetryExhaustedError")
    assert store.get(failpoints.STATE_KEY)[0] == after_two
    assert not store.exists("output/o2.csv")


PERMANENT = {
    "store_error": lambda: objectstore.StoreError("denied"),
    "corrupt_object": lambda: objectstore.CorruptObjectError("torn read"),
    "not_found": lambda: objectstore.NotFoundError("vanished"),
    "framework_error": lambda: errors.DataError("bad bytes"),
}


@pytest.mark.parametrize("target", ["read", "output", "checkpoint"])
@pytest.mark.parametrize("kind", sorted(PERMANENT))
def test_a_permanent_error_stops_at_once_with_the_checkpoint_unchanged(
    store, after_two, target, kind
):
    faulty = faulty_store.FaultyStore(
        store, 1, error=PERMANENT[kind], **TARGETS[target]
    )
    result = failpoints.processor(faulty, policy=POLICY).run()
    assert result.failed == "input/o2.csv"
    assert result.error.startswith(type(PERMANENT[kind]()).__name__)
    assert result.retries == ()
    assert faulty.matching == TARGETS[target]["skip"] + 1
    assert result.processed == ("input/o0.csv", "input/o1.csv")
    assert store.get(failpoints.STATE_KEY)[0] == after_two
    assert not store.exists(failpoints.LOCK_KEY)


@pytest.mark.parametrize("target", ["listing", "read", "output", "checkpoint"])
def test_an_error_outside_the_framework_propagates_at_once(
    store, reference, target
):
    faulty = faulty_store.FaultyStore(
        store, 1, error=lambda: OSError("disk full"), **TARGETS[target]
    )
    run = failpoints.processor(faulty, policy=POLICY)
    with pytest.raises(OSError, match="disk full"):
        run.run()
    assert run.waits == []
    assert faulty.matching == TARGETS[target].get("skip", 0) + 1
    assert not store.exists(failpoints.LOCK_KEY)
    faulty.heal()
    assert failpoints.processor(faulty).run().failed is None
    assert failpoints.files_of(store) == reference.files


# ---- the retrying store protects the calls the processor does not retry ---

GUARDED = {
    "lock_acquire": {"operations": ("put",), "prefix": failpoints.LOCK_KEY},
    "checkpoint_read": {
        "operations": ("get",),
        "prefix": failpoints.STATE_KEY,
    },
    "lock_release": {"operations": ("delete",), "prefix": failpoints.LOCK_KEY},
}


@pytest.mark.parametrize("target", sorted(GUARDED))
@pytest.mark.parametrize("failures", [1, 3])
def test_a_retrying_store_absorbs_failures_in_every_call_within_budget(
    store, reference, target, failures
):
    waits = []
    faulty = faulty_store.FaultyStore(store, failures, **GUARDED[target])
    guarded = objectstore.RetryingStore(faulty, POLICY, sleep=waits.append)
    result = failpoints.processor(guarded).run()
    assert result.failed is None
    assert waits == POLICY.delays()[:failures]
    assert failpoints.files_of(store) == reference.files
    assert store.get(failpoints.STATE_KEY)[0] == reference.checkpoint
    assert not store.exists(failpoints.LOCK_KEY)


def test_a_lock_that_cannot_be_taken_within_budget_stops_before_any_work(
    store,
):
    waits = []
    faulty = faulty_store.FaultyStore(
        store, POLICY.max_attempts, **GUARDED["lock_acquire"]
    )
    guarded = objectstore.RetryingStore(faulty, POLICY, sleep=waits.append)
    with pytest.raises(retry.RetryExhaustedError) as raised:
        failpoints.processor(guarded).run()
    assert raised.value.attempts == POLICY.max_attempts
    assert waits == POLICY.delays()
    assert faulty.matching == POLICY.max_attempts
    assert failpoints.files_of(store) == {}
    assert not store.exists(failpoints.STATE_KEY)
    assert not store.exists(failpoints.LOCK_KEY)


def test_a_lock_release_that_fails_past_budget_leaves_finished_work_intact(
    store, reference
):
    faulty = faulty_store.FaultyStore(
        store, POLICY.max_attempts, **GUARDED["lock_release"]
    )
    guarded = objectstore.RetryingStore(
        faulty, POLICY, sleep=lambda seconds: None
    )
    with pytest.raises(retry.RetryExhaustedError):
        failpoints.processor(guarded).run()
    assert failpoints.files_of(store) == reference.files
    assert store.get(failpoints.STATE_KEY)[0] == reference.checkpoint
    assert store.exists(failpoints.LOCK_KEY)
    with pytest.raises(processing.ConcurrentRunError):
        failpoints.processor(store).run()
    failpoints.processor(store).break_lock()
    again = failpoints.processor(store).run()
    assert again.processed == () and again.skipped == tuple(KEYS)


@pytest.mark.parametrize("failures", [0, 1, 2, 3, 4, 5])
def test_the_retrying_store_makes_exactly_the_budgeted_attempts(
    store, failures
):
    waits = []
    faulty = faulty_store.FaultyStore(store, failures, operations=("get",))
    guarded = objectstore.RetryingStore(faulty, POLICY, sleep=waits.append)
    if failures < POLICY.max_attempts:
        assert guarded.get("input/o0.csv")[0] == failpoints.object_bytes(0)
        assert faulty.matching == failures + 1
        assert waits == POLICY.delays()[:failures]
    else:
        with pytest.raises(retry.RetryExhaustedError) as raised:
            guarded.get("input/o0.csv")
        assert raised.value.attempts == faulty.matching == POLICY.max_attempts
        assert waits == POLICY.delays()


# ---- torn and unreadable inputs -------------------------------------------


@pytest.mark.parametrize(
    "torn",
    [b"id,amount", b"\x00\x01\x02\xff\xfe", b"", b"id,amount,region\n1,"],
    ids=["header_cut", "binary", "empty", "row_cut_after_comma"],
)
def test_an_unusable_input_stops_at_it_and_leaves_the_checkpoint_alone(
    store, reference, after_two, torn
):
    store.put("input/o2-bad.csv", torn)
    failure = None
    result = None
    try:
        result = failpoints.processor(store, policy=POLICY).run()
    except Exception as error:  # noqa: BLE001 -- any stop is acceptable; the test checks which one and that the checkpoint survives
        failure = error
    if failure is not None:
        assert not isinstance(failure, processing.CheckpointCorruptError)
        stopped_by = type(failure).__name__
    else:
        assert result.failed == "input/o2-bad.csv"
        stopped_by = result.error
    assert stopped_by
    assert store.get(failpoints.STATE_KEY)[0] == after_two
    assert not store.exists("output/o2-bad.csv")
    assert not store.exists(failpoints.LOCK_KEY)
    store.delete("input/o2-bad.csv")
    resumed = failpoints.processor(store, policy=POLICY).run()
    assert resumed.failed is None
    assert failpoints.files_of(store) == reference.files


# ---- repeated uploads -----------------------------------------------------


def test_repeated_uploads_are_skipped_counted_and_never_written(
    store, reference
):
    store.put("input/o7-again.csv", failpoints.object_bytes(0))
    store.put("input/o8-again.csv", failpoints.object_bytes(2))
    result = failpoints.processor(store).run()
    assert result.duplicate_objects == (
        failpoints.copy_key(),
        "input/o7-again.csv",
        "input/o8-again.csv",
    ) or sorted(result.duplicate_objects) == sorted(
        [failpoints.copy_key(), "input/o7-again.csv", "input/o8-again.csv"]
    )
    assert result.processed == tuple(REAL)
    assert result.outputs == tuple(failpoints.output_key(k) for k in REAL)
    assert result.rows_out == reference.result.rows_out
    assert result.duplicate_rows == reference.result.duplicate_rows
    assert failpoints.files_of(store) == reference.files
    stored = failpoints.checkpoint_of(store)
    assert "input/o7-again.csv" in stored.processed
    again = failpoints.processor(store).run()
    assert again.duplicate_objects == () and again.processed == ()
    assert len(again.skipped) == len(KEYS) + 2


def test_the_later_key_is_the_duplicate_whichever_upload_came_first(store):
    store.put("input/a-first.csv", failpoints.object_bytes(2))
    result = failpoints.processor(store).run()
    assert "input/a-first.csv" in result.processed
    assert "input/o2.csv" in result.duplicate_objects
    assert "output/a-first.csv" in failpoints.files_of(store)
    assert not store.exists("output/o2.csv")
    ids = failpoints.all_ids(failpoints.files_of(store))
    assert len(ids) == len(set(ids))
