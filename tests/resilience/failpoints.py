"""Shared pieces of the resilience tests: inputs, a processor, failpoints.

A `FailpointStore` wraps a store and calls a hook before and after every
write that matters to `IncrementalProcessor`: taking the lock, writing an
output, writing the checkpoint, writing a quality report, releasing the
lock. The hook decides to kill the process (a child process killed with
SIGKILL, which runs no cleanup, like a lost machine) or to raise `Crash` (the
same process, for the stateful test).
"""

import dataclasses
import os
import signal
import time

from scmle.data.store import objectstore
from scmle.data.store import processing
from scmle.data.store import quality
from scmle.data.store import snapshot

LOCK_KEY = "state/lock"
STATE_KEY = "state/checkpoint.json"
LABELS = (
    "lock_acquire",
    "output_write",
    "checkpoint_write",
    "gate_write",
    "lock_release",
)
CONTRACT = snapshot.DataContract(
    "orders",
    "2",
    {"id": "int", "amount": "float", "region": "string"},
    key_columns=("id",),
)
REGIONS = ("north", "south", "east")
OBJECT_COUNT = 4
RESULT_PREFIXES = ("output/", "quality/", "quarantine/")


class Crash(BaseException):
    """Stands in for a killed process when the test stays in one process."""


def upper(frame):
    return frame.assign(region=frame["region"].str.upper())


def object_bytes(index, bad=False):
    """The index-th generated input: six rows, two ids shared with the next.

    The first row repeats at the end, so every object has a repeated key.
    With `bad`, the fifth row has a negative amount (a quarantine target).
    """
    ids = [3 * index + step for step in range(5)]
    ids.append(ids[0])
    lines = ["id,amount,region"]
    for position, row_id in enumerate(ids):
        amount = -1.5 if bad and position == 4 else row_id * 1.5
        lines.append(f"{row_id},{amount},{REGIONS[row_id % 3]}")
    return ("\n".join(lines) + "\n").encode()


def copy_key():
    return f"input/o{OBJECT_COUNT}-copy.csv"


def input_keys():
    """Every input key in the order the processor takes them."""
    keys = [f"input/o{n}.csv" for n in range(OBJECT_COUNT)]
    return sorted([*keys, copy_key()])


def seed(store, gated=False, count=OBJECT_COUNT):
    """Puts `count` objects and, when all are put, a copy of the second.

    With `gated` the third object carries a row the quality gate removes.
    """
    for n in range(count):
        store.put(f"input/o{n}.csv", object_bytes(n, bad=gated and n == 2))
    if count == OBJECT_COUNT:
        store.put(copy_key(), object_bytes(1))


def output_key(key):
    return "output/" + key.removeprefix("input/").rsplit(".", 1)[0] + ".csv"


def gate_for(store):
    return quality.QualityGate(
        [
            quality.RangeCheck(
                "amount", 0, 1000, policy=quality.Severity.QUARANTINE
            )
        ],
        store,
    )


def processor(store, gated=False, **options):
    options.setdefault("sleep", lambda seconds: None)
    if gated:
        options["gate"] = gate_for(store)
    return processing.IncrementalProcessor(store, CONTRACT, upper, **options)


def files_of(store, prefixes=RESULT_PREFIXES):
    """Key to bytes of every object under the result prefixes."""
    found = {}
    for prefix in prefixes:
        for meta in store.list_objects(prefix):
            found[meta.key] = store.get(meta.key)[0]
    return found


def read_rows(data):
    """Rows of an output CSV as (id, amount, region) strings."""
    lines = data.decode().splitlines()
    return [tuple(line.split(",")) for line in lines[1:]]


def expected_plan(gated=False):
    """Per real input key: (rows dropped as repeats, rows written).

    Computed in plain Python from the generated inputs, independently of
    pandas and of the processor. The copy is not a real object.
    """
    seen = set()
    plan = {}
    for key in input_keys():
        if key == copy_key():
            continue
        index = int(key.removeprefix("input/o").removesuffix(".csv"))
        rows = read_rows(object_bytes(index, bad=gated and index == 2))
        rows = [r for r in rows if not (gated and float(r[1]) < 0)]
        fresh = []
        for row in rows:
            if row[0] not in seen and row[0] not in [f[0] for f in fresh]:
                fresh.append(row)
        plan[key] = (len(rows) - len(fresh), len(fresh))
        seen.update(row[0] for row in fresh)
    return plan


@dataclasses.dataclass(frozen=True)
class Reference:
    """What an uninterrupted run produced.

    Attributes:
        files: Result objects (outputs, reports, quarantine) to bytes.
        checkpoint: The checkpoint bytes.
        result: What the run reported.
    """

    files: dict
    checkpoint: bytes
    result: processing.Processing


def reference_run(root, gated=False):
    store = objectstore.LocalObjectStore(root)
    seed(store, gated)
    result = processor(store, gated).run()
    return Reference(files_of(store), store.get(STATE_KEY)[0], result)


def kill_process():
    """Ends this process at once, with no cleanup, like a lost machine."""
    os.kill(os.getpid(), signal.SIGKILL)
    while True:
        time.sleep(1)


def raise_crash():
    raise Crash


class Trigger:
    """A hook that runs `action` at one boundary.

    Attributes:
        target: (label, phase, index) at which to act.
        action: Zero-argument callable.
    """

    def __init__(self, label, phase, index, action):
        self.target = (label, phase, index)
        self.action = action

    def __call__(self, label, phase, index):
        if (label, phase, index) == self.target:
            self.action()


class FailpointStore(objectstore.ObjectStore):
    """A store that tells a hook about the writes the processor makes.

    Once the hook has raised `Crash` the store is dead: every later call
    raises `Crash` too, as the operations of a killed process never happen
    (including the release of the lock in the processor's `finally`).

    Attributes:
        inner: The wrapped store.
        hook: Called as `hook(label, phase, index)` with phase `before` or
            `after`; `index` counts that label's operations from 0.
        counts: Operations seen so far per label.
        dead: Whether a crash has happened.
    """

    def __init__(self, inner, hook):
        super().__init__(inner.location)
        self.scheme = inner.scheme
        self.inner = inner
        self.hook = hook
        self.counts = dict.fromkeys(LABELS, 0)
        self.dead = False

    def label_of(self, operation, key, if_absent=False):
        if operation == "delete":
            return "lock_release" if key == LOCK_KEY else None
        if operation != "put":
            return None
        if if_absent and key == LOCK_KEY:
            return "lock_acquire"
        if key.startswith("output/"):
            return "output_write"
        if key == STATE_KEY:
            return "checkpoint_write"
        if key.startswith(("quality/", "quarantine/")):
            return "gate_write"
        return None

    def around(self, operation, key, action, if_absent=False):
        if self.dead:
            raise Crash
        label = self.label_of(operation, key, if_absent)
        if label is None:
            return action()
        index = self.counts[label]
        self.counts[label] += 1
        try:
            self.hook(label, "before", index)
            result = action()
            self.hook(label, "after", index)
        except Crash:
            self.dead = True
            raise
        return result

    def put(self, key, data, *, if_absent=False):
        return self.around(
            "put",
            key,
            lambda: self.inner.put(key, data, if_absent=if_absent),
            if_absent,
        )

    def get(self, key):
        return self.around("get", key, lambda: self.inner.get(key))

    def head(self, key):
        return self.around("head", key, lambda: self.inner.head(key))

    def list_objects(self, prefix=""):
        return self.around(
            "list", prefix, lambda: self.inner.list_objects(prefix)
        )

    def delete(self, key):
        return self.around("delete", key, lambda: self.inner.delete(key))


def checkpoint_of(store):
    """The verified checkpoint stored in `store`."""
    return processing.Checkpoint.from_bytes(store.get(STATE_KEY)[0])


def all_ids(files):
    """Every id written across the output objects, with repeats."""
    return [
        row[0]
        for key, data in files.items()
        if key.startswith("output/")
        for row in read_rows(data)
    ]
