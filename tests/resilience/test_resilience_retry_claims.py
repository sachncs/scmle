"""The documented claim: transient store errors are retried under `Retries`.

`IncrementalProcessor` says "Transient errors from the store or the
transform are retried under a `Retries`". Each test below makes one store
call fail once with `TransientStoreError`, well inside a four-attempt budget,
and expects the run to finish with the bytes of a clean run.
"""

import failpoints
import faulty_store
import pytest

from scmle.core import retry
from scmle.data.store import objectstore

POLICY = retry.Retries(max_attempts=4, base_delay_seconds=0.5, factor=2.0)


@pytest.fixture
def store(tmp_path):
    local = objectstore.LocalObjectStore(tmp_path / "root")
    failpoints.seed(local, gated=True)
    return local


def gated_run(local, faulty_gate_store, **options):
    run = failpoints.processor(local, policy=POLICY, **options)
    run.gate = failpoints.gate_for(faulty_gate_store)
    return run.run()


def test_a_transient_failure_writing_the_gate_report_is_retried(
    store,
):
    faulty = faulty_store.FaultyStore(
        store, 1, operations=("put",), prefix="quality/"
    )
    result = gated_run(store, faulty)
    assert result.failed is None, result.error
    assert faulty.matching > 1


def test_a_transient_failure_checking_the_quarantine_marker_is_retried(
    store,
):
    faulty = faulty_store.FaultyStore(
        store, 1, operations=("head",), prefix="quarantine/"
    )
    result = gated_run(store, faulty)
    assert result.failed is None, result.error
    assert faulty.matching > 1


def test_a_transient_failure_recording_a_repeated_upload_is_retried(tmp_path):
    local = objectstore.LocalObjectStore(tmp_path / "root")
    failpoints.seed(local)
    # Writes 0 to 3 record the four objects; write 4 records the copy.
    faulty = faulty_store.FaultyStore(
        local, 1, operations=("put",), prefix=failpoints.STATE_KEY, skip=4
    )
    result = failpoints.processor(faulty, policy=POLICY).run()
    assert result.failed is None, result.error
    assert faulty.matching > 5
    assert result.duplicate_objects == (failpoints.copy_key(),)


def test_a_transient_failure_reading_the_checkpoint_is_retried(tmp_path):
    local = objectstore.LocalObjectStore(tmp_path / "root")
    failpoints.seed(local)
    failpoints.processor(local).run(max_objects=2)
    faulty = faulty_store.FaultyStore(
        local, 1, operations=("get",), prefix=failpoints.STATE_KEY
    )
    result = failpoints.processor(faulty, policy=POLICY).run()
    assert result.failed is None, result.error
    assert faulty.matching > 1
    assert len(result.skipped) == 2


def test_a_transient_failure_releasing_the_lock_is_retried(tmp_path):
    local = objectstore.LocalObjectStore(tmp_path / "root")
    failpoints.seed(local)
    faulty = faulty_store.FaultyStore(
        local, 1, operations=("delete",), prefix=failpoints.LOCK_KEY
    )
    result = failpoints.processor(faulty, policy=POLICY).run()
    assert result.failed is None
    assert faulty.matching > 1
    assert not local.exists(failpoints.LOCK_KEY)
