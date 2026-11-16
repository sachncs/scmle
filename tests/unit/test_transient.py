"""Real transient failures are retried, and only the work that can be (2.6).

ERR-6: only `TransientError`, which nothing raised, was ever retried.
ERR-7: a step whose success record failed to save re-ran its handler.
"""

import errno
import io
import urllib.error

import pytest

from scmle.core import retry
from scmle.data.store import objectstore


def http(code):
    return urllib.error.HTTPError("http://x", code, "status", {}, io.BytesIO())


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (retry.TransientError("throttled"), True),
        (TimeoutError(), True),
        (ConnectionResetError(), True),
        (OSError(errno.EBUSY, "busy"), True),
        (OSError(errno.EAGAIN, "again"), True),
        (OSError(errno.ENOENT, "missing"), False),
        (OSError(errno.EACCES, "denied"), False),
        (OSError(errno.ENOSPC, "full"), False),
        (http(429), True),
        (http(503), True),
        (http(404), False),
        (urllib.error.URLError(TimeoutError()), True),
        (urllib.error.URLError("bad host"), False),
        (ValueError("bad"), False),
    ],
)
def test_failures_are_classified(error, expected):
    assert retry.transient(error) is expected


def test_the_store_wrapper_retries_a_busy_file(tmp_path):
    inner = objectstore.LocalObjectStore(tmp_path)
    calls = {"n": 0}
    real = inner.put

    def busy(key, data, **options):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(errno.EBUSY, "busy")
        return real(key, data, **options)

    inner.put = busy
    store = objectstore.RetryingStore(
        inner, retry.Retries(3, 0.0, 1.0, 0.0), sleep=lambda seconds: None
    )
    store.put("a", b"1")
    assert calls["n"] == 2 and inner.get("a")[0] == b"1"


def test_a_succeeded_handler_is_not_run_again_when_its_record_fails_once(
    tmp_path,
):
    from tests.unit import test_workflow as flows

    store = objectstore.LocalObjectStore(tmp_path / "state")
    clock = flows.Clock()
    recorder = flows.Recorder(clock)
    flow = flows.pipeline()
    executor = flows.executor_for(store, flow, recorder, clock)
    real = executor.save_step
    failed = {"done": False}

    def flaky(run_id, step, state, **fields):
        first = not failed["done"]
        if state.value == "succeeded" and step.name == "ingest" and first:
            failed["done"] = True
            raise objectstore.TransientStoreError("throttled")
        return real(run_id, step, state, **fields)

    executor.save_step = flaky
    executor.start("req-1", "alice")
    assert failed["done"]
    assert recorder.names().count("ingest") == 1
