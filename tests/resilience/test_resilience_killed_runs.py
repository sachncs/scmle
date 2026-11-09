"""A processor killed at every write boundary resumes to identical output.

A child process runs `IncrementalProcessor` over a generated input prefix
and kills itself with SIGKILL (no cleanup, no `finally`, like a lost machine)
before or after each lock, output, checkpoint and quality-report write, for
each object. The parent then removes the dead process's lock, as the
documented `break_lock` decision, runs a fresh processor and compares with an
uninterrupted run.
"""

import os
import pathlib
import shutil
import signal
import subprocess
import sys

import failpoints
import pytest

from scmle.data.store import objectstore
from scmle.data.store import processing

CHILD = pathlib.Path(__file__).with_name("failpoint_child.py")
ROOT = pathlib.Path(__file__).resolve().parents[2]
KEYS = failpoints.input_keys()
REAL = [key for key in KEYS if key != failpoints.copy_key()]
PHASES = ("before", "after")


def plain_cases():
    cases = [("lock_acquire", phase, 0) for phase in PHASES]
    cases += [
        ("output_write", phase, n)
        for n in range(failpoints.OBJECT_COUNT)
        for phase in PHASES
    ]
    cases += [
        ("checkpoint_write", phase, n)
        for n in range(len(KEYS))
        for phase in PHASES
    ]
    cases += [("lock_release", phase, 0) for phase in PHASES]
    return cases


# Gated runs write a quality report per object and a quarantine file for the
# third object: five gate writes, the fourth belonging to the third object.
GATE_OWNERS = (0, 1, 2, 2, 3)
GATED_CASES = [
    ("gate_write", phase, n)
    for n in range(len(GATE_OWNERS))
    for phase in PHASES
]


def committed_before(label, phase, index):
    """How many inputs (in key order) the checkpoint holds after the kill."""
    if label == "lock_acquire":
        return 0
    if label == "output_write":
        return index
    if label == "checkpoint_write":
        return index + (phase == "after")
    if label == "gate_write":
        return GATE_OWNERS[index]
    return len(KEYS)


def lock_remains(label, phase):
    if label == "lock_acquire":
        return phase == "after"
    if label == "lock_release":
        return phase == "before"
    return True


def run_child(*arguments):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), environment.get("PYTHONPATH", "")]
    )
    environment["OMP_NUM_THREADS"] = "1"
    return subprocess.run(
        [sys.executable, str(CHILD), *map(str, arguments)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """One fixed root for every case.

    The snapshot id of an input includes the store's URI, so quality reports
    are named after the root; the same root makes them comparable.
    """
    return tmp_path_factory.mktemp("workspace") / "root"


@pytest.fixture
def root(workspace):
    shutil.rmtree(workspace, ignore_errors=True)
    yield workspace
    shutil.rmtree(workspace, ignore_errors=True)


@pytest.fixture(scope="module")
def references(workspace):
    found = {}
    for gated in (False, True):
        shutil.rmtree(workspace, ignore_errors=True)
        found[gated] = failpoints.reference_run(workspace, gated)
    shutil.rmtree(workspace, ignore_errors=True)
    return found


def check_resume(root, references, label, phase, index, gated):
    store = objectstore.LocalObjectStore(root)
    failpoints.seed(store, gated)
    kind = "gate" if gated else "plain"
    child = run_child("kill", root, label, phase, index, kind)
    assert child.returncode == -signal.SIGKILL, child.stderr[-2000:]
    reference = references[gated]
    # Whatever the child managed to write is already what the reference wrote.
    for key, data in failpoints.files_of(store).items():
        assert data == reference.files[key], key
    assert store.exists(failpoints.LOCK_KEY) is lock_remains(label, phase)
    if lock_remains(label, phase):
        with pytest.raises(processing.ConcurrentRunError):
            failpoints.processor(store, gated).run()
        failpoints.processor(store, gated).break_lock()
    committed = committed_before(label, phase, index)
    stored = (
        processing.Checkpoint.from_bytes(store.get(failpoints.STATE_KEY)[0])
        if committed
        else processing.Checkpoint()
    )
    assert sorted(stored.processed) == KEYS[:committed]

    result = failpoints.processor(store, gated).run()

    pending = KEYS[committed:]
    real = [key for key in pending if key != failpoints.copy_key()]
    plan = failpoints.expected_plan(gated)
    assert result.failed is None and result.error is None
    assert result.retries == ()
    assert result.skipped == tuple(KEYS[:committed])
    assert result.processed == tuple(real)
    assert result.duplicate_objects == (
        (failpoints.copy_key(),) if failpoints.copy_key() in pending else ()
    )
    assert result.outputs == tuple(failpoints.output_key(k) for k in real)
    assert result.duplicate_rows == sum(plan[k][0] for k in real)
    assert result.rows_out == sum(plan[k][1] for k in real)

    assert failpoints.files_of(store) == reference.files
    assert store.get(failpoints.STATE_KEY)[0] == reference.checkpoint
    final = failpoints.checkpoint_of(store)
    assert sorted(final.processed) == KEYS
    ids = failpoints.all_ids(failpoints.files_of(store))
    assert len(ids) == len(set(ids)) == len(final.seen)
    assert set(ids) == final.seen
    assert not store.exists(failpoints.LOCK_KEY)

    again = failpoints.processor(store, gated).run()
    assert again.processed == () and again.skipped == tuple(KEYS)
    assert again.duplicate_objects == () and again.outputs == ()
    assert failpoints.files_of(store) == reference.files


def test_an_uninterrupted_child_matches_the_reference_byte_for_byte(
    root, references
):
    store = objectstore.LocalObjectStore(root)
    failpoints.seed(store)
    child = run_child("kill", root, "output_write", "before", 99, "plain")
    assert child.returncode == 0, child.stderr[-2000:]
    assert failpoints.files_of(store) == references[False].files
    assert store.get(failpoints.STATE_KEY)[0] == references[False].checkpoint
    assert not store.exists(failpoints.LOCK_KEY)


def test_the_reference_run_reports_what_the_plan_expects(references):
    result = references[False].result
    plan = failpoints.expected_plan()
    assert result.processed == tuple(REAL)
    assert result.duplicate_objects == (failpoints.copy_key(),)
    assert result.duplicate_rows == sum(drop for drop, written in plan.values())
    assert result.rows_out == sum(written for drop, written in plan.values())


@pytest.mark.parametrize(
    ("label", "phase", "index"),
    plain_cases(),
    ids=str,
)
def test_a_kill_at_each_boundary_resumes_to_identical_output(
    root, references, label, phase, index
):
    check_resume(root, references, label, phase, index, gated=False)


@pytest.mark.parametrize(
    ("label", "phase", "index"),
    GATED_CASES,
    ids=str,
)
def test_a_kill_while_the_gate_writes_resumes_to_identical_output(
    root, references, label, phase, index
):
    check_resume(root, references, label, phase, index, gated=True)
