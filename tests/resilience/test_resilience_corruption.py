"""A damaged checkpoint stops the run; a stale lock needs `break_lock`.

Also: two processors cannot hold the lock together, within one process or
across processes.
"""

import hashlib
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time

import failpoints
import pytest

from scmle.data.store import objectstore
from scmle.data.store import processing

CHILD = pathlib.Path(__file__).with_name("failpoint_child.py")
ROOT = pathlib.Path(__file__).resolve().parents[2]


def environment():
    found = dict(os.environ)
    found["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), found.get("PYTHONPATH", "")]
    )
    found["OMP_NUM_THREADS"] = "1"
    return found


def redigested(document):
    """JSON bytes for a checkpoint body with a correct digest."""
    body = {k: v for k, v in document.items() if k != "digest"}
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()
    ).hexdigest()
    return json.dumps({**body, "digest": digest}, sort_keys=True).encode()


def damage(kind, good):
    document = json.loads(good)
    if kind == "empty":
        return b""
    if kind == "one_byte":
        return good[:1]
    if kind == "half":
        return good[: len(good) // 2]
    if kind == "all_but_the_last_byte":
        return good[:-1]
    if kind == "garbage":
        return b"\x00\xff\xfe not json"
    if kind == "bad_digest":
        flipped = "0" if document["digest"][0] != "0" else "1"
        document["digest"] = flipped + document["digest"][1:]
        return json.dumps(document, sort_keys=True).encode()
    if kind == "edited_processed":
        return good.replace(b"input/o0.csv", b"input/z9.csv")
    if kind == "edited_seen":
        document["seen"] = document["seen"][:-1]
        return json.dumps(document, sort_keys=True).encode()
    if kind == "other_format":
        document["format"] = 2
        return redigested(document)
    if kind == "missing_seen":
        del document["seen"]
        return redigested(document)
    if kind == "processed_not_a_mapping":
        document["processed"] = 5
        return redigested(document)
    if kind == "not_an_object":
        return b"[1, 2]"
    raise AssertionError(kind)


KINDS = (
    "empty",
    "one_byte",
    "half",
    "all_but_the_last_byte",
    "garbage",
    "bad_digest",
    "edited_processed",
    "edited_seen",
    "other_format",
    "missing_seen",
    "processed_not_a_mapping",
    "not_an_object",
)


@pytest.fixture
def finished(tmp_path):
    """A store whose run finished: every input processed and checkpointed."""
    store = objectstore.LocalObjectStore(tmp_path / "root")
    failpoints.seed(store)
    failpoints.processor(store).run()
    return store


@pytest.mark.parametrize("kind", KINDS)
def test_a_damaged_checkpoint_stops_the_run_and_is_not_reset(finished, kind):
    store = finished
    good = store.get(failpoints.STATE_KEY)[0]
    before = failpoints.files_of(store)
    store.put(failpoints.STATE_KEY, damage(kind, good))
    damaged = store.get(failpoints.STATE_KEY)[0]
    store.put("input/o9.csv", failpoints.object_bytes(9))
    for attempt in range(2):
        with pytest.raises(processing.CheckpointCorruptError):
            failpoints.processor(store).run()
        assert store.get(failpoints.STATE_KEY)[0] == damaged, attempt
        assert failpoints.files_of(store) == before, attempt
        assert not store.exists(failpoints.LOCK_KEY), attempt
    store.put(failpoints.STATE_KEY, good)
    resumed = failpoints.processor(store).run()
    assert resumed.processed == ("input/o9.csv",)
    assert resumed.skipped == tuple(failpoints.input_keys())


def test_a_damaged_checkpoint_stops_a_bounded_run_too(finished):
    store = finished
    store.put(failpoints.STATE_KEY, b"{")
    with pytest.raises(processing.CheckpointCorruptError):
        failpoints.processor(store).run(max_objects=1)
    assert store.get(failpoints.STATE_KEY)[0] == b"{"


def test_a_checkpoint_that_never_existed_starts_from_nothing(tmp_path):
    store = objectstore.LocalObjectStore(tmp_path / "root")
    failpoints.seed(store)
    result = failpoints.processor(store).run()
    assert result.skipped == () and len(result.processed) == 4


def test_a_deleted_checkpoint_rewrites_the_same_outputs(finished):
    store = finished
    before = failpoints.files_of(store)
    checkpoint = store.get(failpoints.STATE_KEY)[0]
    store.delete(failpoints.STATE_KEY)
    result = failpoints.processor(store).run()
    assert result.processed == tuple(
        k for k in failpoints.input_keys() if k != failpoints.copy_key()
    )
    assert failpoints.files_of(store) == before
    assert store.get(failpoints.STATE_KEY)[0] == checkpoint


def test_a_stale_lock_from_a_dead_process_is_released_only_by_break_lock(
    tmp_path,
):
    root = tmp_path / "root"
    store = objectstore.LocalObjectStore(root)
    failpoints.seed(store)
    child = subprocess.run(
        [
            sys.executable,
            str(CHILD),
            "kill",
            str(root),
            "lock_acquire",
            "after",
            "0",
            "plain",
        ],
        env=environment(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert child.returncode == -signal.SIGKILL, child.stderr[-2000:]
    assert store.exists(failpoints.LOCK_KEY)
    for attempt in range(3):
        with pytest.raises(processing.ConcurrentRunError):
            failpoints.processor(store).run()
        assert store.exists(failpoints.LOCK_KEY), attempt
        assert failpoints.files_of(store) == {}, attempt
    # Waiting does not release it, and neither does a run that bounds itself.
    time.sleep(0.2)
    with pytest.raises(processing.ConcurrentRunError):
        failpoints.processor(store).run(max_objects=1)
    failpoints.processor(store).break_lock()
    assert not store.exists(failpoints.LOCK_KEY)
    result = failpoints.processor(store).run()
    assert result.failed is None and len(result.processed) == 4
    assert not store.exists(failpoints.LOCK_KEY)


def test_breaking_a_lock_that_is_not_there_is_harmless(tmp_path):
    store = objectstore.LocalObjectStore(tmp_path / "root")
    failpoints.processor(store).break_lock()
    failpoints.seed(store)
    assert failpoints.processor(store).run().failed is None


def test_a_refused_run_leaves_the_holders_lock_alone(tmp_path):
    store = objectstore.LocalObjectStore(tmp_path / "root")
    failpoints.seed(store)
    store.put(failpoints.LOCK_KEY, b"someone else")
    with pytest.raises(processing.ConcurrentRunError):
        failpoints.processor(store).run()
    assert store.get(failpoints.LOCK_KEY)[0] == b"someone else"


def race_for_lock(store, racers):
    """Every racer tries a conditional put of the lock at the same moment."""
    barrier = threading.Barrier(racers)
    winners = []
    losers = []

    def race(number):
        barrier.wait()
        try:
            store.put(failpoints.LOCK_KEY, str(number).encode(), if_absent=True)
        except objectstore.AlreadyExistsError:
            losers.append(number)
        else:
            winners.append(number)

    threads = [threading.Thread(target=race, args=(n,)) for n in range(racers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return winners, losers


def test_only_one_of_many_racing_writers_takes_the_lock(tmp_path):
    store = objectstore.LocalObjectStore(tmp_path / "root")
    racers = 8
    for round_number in range(15):
        winners, losers = race_for_lock(store, racers)
        assert len(winners) == 1, round_number
        assert len(losers) == racers - 1
        assert store.get(failpoints.LOCK_KEY)[0] == str(winners[0]).encode()
        store.delete(failpoints.LOCK_KEY)


def test_racing_processors_never_work_at_the_same_time(tmp_path):
    root = tmp_path / "root"
    store = objectstore.LocalObjectStore(root)
    failpoints.seed(store)
    racers = 6
    barrier = threading.Barrier(racers)
    guard = threading.Lock()
    inside = {"now": 0, "most": 0}
    outcomes = []

    def slow_upper(frame):
        with guard:
            inside["now"] += 1
            inside["most"] = max(inside["most"], inside["now"])
        time.sleep(0.05)
        with guard:
            inside["now"] -= 1
        return failpoints.upper(frame)

    def race():
        run = processing.IncrementalProcessor(
            objectstore.LocalObjectStore(root),
            failpoints.CONTRACT,
            slow_upper,
        )
        barrier.wait()
        try:
            outcomes.append(run.run())
        except processing.ConcurrentRunError as error:
            outcomes.append(error)

    threads = [threading.Thread(target=race) for number in range(racers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(outcomes) == racers
    assert inside["most"] == 1
    done = [o for o in outcomes if isinstance(o, processing.Processing)]
    assert done and sum(len(o.processed) for o in done) == 4
    assert sum(len(o.duplicate_objects) for o in done) == 1
    reference = failpoints.reference_run(tmp_path / "root2")
    assert failpoints.files_of(store, ("output/",)) == {
        k: v for k, v in reference.files.items() if k.startswith("output/")
    }
    assert store.get(failpoints.STATE_KEY)[0] == reference.checkpoint
    assert not store.exists(failpoints.LOCK_KEY)


def test_a_second_process_is_refused_while_the_first_holds_the_lock(tmp_path):
    root = tmp_path / "root"
    store = objectstore.LocalObjectStore(root)
    failpoints.seed(store)
    marker = tmp_path / "marker"
    release = tmp_path / "release"
    holder = subprocess.Popen(
        [
            sys.executable,
            str(CHILD),
            "hold",
            str(root),
            str(marker),
            str(release),
        ],
        env=environment(),
    )
    try:
        deadline = time.monotonic() + 100
        while not marker.exists():
            assert holder.poll() is None, "holder ended early"
            assert time.monotonic() < deadline, "holder never took the lock"
            time.sleep(0.02)
        assert store.exists(failpoints.LOCK_KEY)
        with pytest.raises(processing.ConcurrentRunError):
            failpoints.processor(store).run()
        assert store.exists(failpoints.LOCK_KEY)
        release.write_text("go")
        assert holder.wait(timeout=100) == 0
    finally:
        if holder.poll() is None:
            holder.kill()
    assert not store.exists(failpoints.LOCK_KEY)
    done = failpoints.processor(store).run()
    assert done.processed == () and done.skipped == tuple(
        failpoints.input_keys()
    )
    reference = failpoints.reference_run(tmp_path / "root2")
    assert failpoints.files_of(store, ("output/",)) == {
        k: v for k, v in reference.files.items() if k.startswith("output/")
    }
