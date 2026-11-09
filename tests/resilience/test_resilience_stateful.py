"""A stateful model of `IncrementalProcessor` under crashes and corruption.

The machine adds objects (and repeated uploads), runs, crashes a run at a
failpoint inside the same process, resumes, and corrupts the checkpoint, and
compares the store with a reference function of the objects added so far.
"""

import hashlib
import pathlib
import tempfile

import failpoints
from hypothesis import HealthCheck
from hypothesis import settings
from hypothesis import strategies
from hypothesis.stateful import invariant
from hypothesis.stateful import precondition
from hypothesis.stateful import rule
from hypothesis.stateful import RuleBasedStateMachine
from hypothesis.stateful import run_state_machine_as_test
import pytest

from scmle.data.store import objectstore
from scmle.data.store import processing

LABELS = ("lock_acquire", "output_write", "checkpoint_write", "lock_release")
MUTATIONS = ("truncate", "empty", "garbage", "flip_digest", "drop_tail")


def csv_of(ids):
    lines = ["id,amount,region"]
    for row_id in ids:
        region = failpoints.REGIONS[row_id % 3]
        lines.append(f"{row_id},{row_id * 0.5},{region}")
    return ("\n".join(lines) + "\n").encode()


def reference_outputs(objects):
    """Outputs of all objects, from first principles and in key order.

    An object whose bytes were seen earlier is skipped; a row whose id was
    seen (in an earlier object or earlier in this one) is dropped; the region
    is upper-cased.
    """
    outputs = {}
    seen_objects = set()
    seen_ids = set()
    for key in sorted(objects):
        digest = hashlib.sha256(objects[key]).hexdigest()
        if digest in seen_objects:
            continue
        seen_objects.add(digest)
        lines = objects[key].decode().splitlines()
        text = "id,amount,region\n"
        fresh = set()
        for line in lines[1:]:
            row_id, amount, region = line.split(",")
            if row_id in seen_ids or row_id in fresh:
                continue
            fresh.add(row_id)
            text += f"{row_id},{amount},{region.upper()}\n"
        seen_ids |= fresh
        name = key.removeprefix("input/").rsplit(".", 1)[0]
        outputs[f"output/{name}.csv"] = text.encode()
    return outputs


def mutate(kind, good):
    if kind == "truncate":
        return good[: len(good) // 2]
    if kind == "empty":
        return b""
    if kind == "garbage":
        return b"\x00garbage"
    if kind == "drop_tail":
        return good[:-2]
    last = good.rindex(b'"digest": "') + len(b'"digest": "')
    flipped = b"0" if good[last : last + 1] != b"0" else b"1"
    return good[:last] + flipped + good[last + 1 :]


class ProcessorMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.directory = tempfile.TemporaryDirectory()
        self.store = objectstore.LocalObjectStore(
            pathlib.Path(self.directory.name) / "root"
        )
        self.objects = {}
        self.corrupt = False
        self.good_checkpoint = None
        self.stale_lock = False

    def teardown(self):
        self.directory.cleanup()

    def next_key(self):
        return f"input/n{len(self.objects):03d}.csv"

    def outputs(self):
        return failpoints.files_of(self.store, ("output/",))

    def check_after_success(self, result):
        expected = reference_outputs(self.objects)
        assert result.failed is None and result.error is None, result
        assert self.outputs() == expected
        ids = failpoints.all_ids(self.outputs())
        assert len(ids) == len(set(ids))
        if self.objects:
            checkpoint = failpoints.checkpoint_of(self.store)
            assert sorted(checkpoint.processed) == sorted(self.objects)
            assert checkpoint.seen == set(ids)
        else:
            assert not self.store.exists(failpoints.STATE_KEY)
        parts = (*result.processed, *result.skipped, *result.duplicate_objects)
        assert sorted(parts) == sorted(self.objects)
        assert not self.store.exists(failpoints.LOCK_KEY)

    def make_processor(self, store=None):
        return failpoints.processor(store or self.store)

    @rule(
        ids=strategies.lists(strategies.integers(0, 5), min_size=1, max_size=3)
    )
    def add_object(self, ids):
        key = self.next_key()
        data = csv_of(ids)
        self.store.put(key, data)
        self.objects[key] = data

    @precondition(lambda self: self.objects)
    @rule(which=strategies.integers(0, 1000))
    def add_duplicate(self, which):
        keys = sorted(self.objects)
        key = self.next_key()
        data = self.objects[keys[which % len(keys)]]
        self.store.put(key, data)
        self.objects[key] = data

    @rule()
    def run(self):
        if self.corrupt:
            self.expect_refusal()
        elif self.stale_lock:
            with pytest.raises(processing.ConcurrentRunError):
                self.make_processor().run()
            assert self.store.exists(failpoints.LOCK_KEY)
        else:
            result = self.make_processor().run()
            self.check_after_success(result)
            again = self.make_processor().run()
            assert again.processed == () and again.duplicate_objects == ()
            assert self.outputs() == reference_outputs(self.objects)

    @precondition(lambda self: not self.corrupt and not self.stale_lock)
    @rule(
        label=strategies.sampled_from(LABELS),
        phase=strategies.sampled_from(("before", "after")),
        index=strategies.integers(0, 3),
    )
    def crash_run(self, label, phase, index):
        trigger = failpoints.Trigger(
            label, phase, index, failpoints.raise_crash
        )
        store = failpoints.FailpointStore(self.store, trigger)
        try:
            result = self.make_processor(store).run()
        except failpoints.Crash:
            self.stale_lock = self.store.exists(failpoints.LOCK_KEY)
        else:
            self.check_after_success(result)

    @precondition(lambda self: self.stale_lock and not self.corrupt)
    @rule()
    def resume(self):
        with pytest.raises(processing.ConcurrentRunError):
            self.make_processor().run()
        self.make_processor().break_lock()
        self.stale_lock = False
        result = self.make_processor().run()
        self.check_after_success(result)

    @precondition(
        lambda self: (
            not self.corrupt
            and not self.stale_lock
            and self.store.exists(failpoints.STATE_KEY)
        )
    )
    @rule(kind=strategies.sampled_from(MUTATIONS))
    def corrupt_checkpoint(self, kind):
        self.good_checkpoint = self.store.get(failpoints.STATE_KEY)[0]
        self.store.put(failpoints.STATE_KEY, mutate(kind, self.good_checkpoint))
        self.corrupt = True
        self.expect_refusal()

    def expect_refusal(self):
        damaged = self.store.get(failpoints.STATE_KEY)[0]
        outputs = self.outputs()
        with pytest.raises(processing.CheckpointCorruptError):
            self.make_processor().run()
        assert self.store.get(failpoints.STATE_KEY)[0] == damaged
        assert self.outputs() == outputs
        assert not self.store.exists(failpoints.LOCK_KEY)

    @precondition(lambda self: self.corrupt)
    @rule()
    def repair(self):
        self.store.put(failpoints.STATE_KEY, self.good_checkpoint)
        self.corrupt = False

    @invariant()
    def no_output_ever_differs_from_the_reference(self):
        expected = reference_outputs(self.objects)
        for key, data in self.outputs().items():
            assert expected[key] == data, key


def test_the_processor_follows_the_reference_under_crashes_and_corruption():
    run_state_machine_as_test(
        ProcessorMachine,
        settings=settings(
            max_examples=50,
            stateful_step_count=25,
            deadline=None,
            database=None,
            suppress_health_check=list(HealthCheck),
        ),
    )
