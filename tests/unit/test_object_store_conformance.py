"""The conformance suite against the local backend, and what is specific to it."""

import pytest

from scmle.contracts import errors
from scmle.core import retry
from scmle.data.store import conformance
from scmle.data.store import objectstore


def make_local(tmp_path):
    return objectstore.LocalObjectStore(tmp_path / "root")


@pytest.fixture
def store(tmp_path):
    return make_local(tmp_path)


def test_the_conformance_suite_passes_on_the_local_backend(store):
    report = conformance.run_conformance(store, run_id="unit")
    failed = [(r.name, r.detail) for r in report.results if not r.passed]
    assert failed == []
    assert report.passed
    assert len(report.results) == len(conformance.CHECKS)


def test_the_suite_leaves_nothing_behind(store):
    conformance.run_conformance(store, run_id="clean")
    assert store.list_objects("") == []


def test_the_suite_catches_a_backend_that_breaks_a_promise(tmp_path):
    class Overwriting(objectstore.LocalObjectStore):
        def put(self, key, data, *, if_absent=False):
            return super().put(key, data, if_absent=False)

    report = conformance.run_conformance(Overwriting(tmp_path / "bad"))
    failed = {r.name for r in report.results if not r.passed}
    assert failed == {"if_absent", "if_absent_race"}
    assert not report.passed


def test_the_uri_names_the_backend_and_prefix(tmp_path):
    assert make_local(tmp_path).uri("a/b").startswith("file://")


def test_a_local_key_cannot_leave_the_root(tmp_path):
    store = make_local(tmp_path)
    for key in ("../outside", "a/../../b", "/etc/passwd"):
        with pytest.raises(objectstore.StoreError):
            store.put(key, b"x")
    assert not (tmp_path / "outside").exists()


class Flaky(objectstore.LocalObjectStore):
    """A local store whose first writes fail transiently, or permanently."""

    def __init__(self, root, failures):
        super().__init__(root)
        self.failures = list(failures)
        self.puts = 0

    def put(self, key, data, *, if_absent=False):
        self.puts += 1
        if self.failures:
            raise self.failures.pop(0)
        return super().put(key, data, if_absent=if_absent)


def test_transient_failures_are_retried_with_the_declared_backoff(tmp_path):
    waits = []
    inner = Flaky(
        tmp_path / "r",
        [
            objectstore.TransientStoreError("a"),
            objectstore.TransientStoreError("b"),
        ],
    )
    store = objectstore.RetryingStore(
        inner,
        retry.RetryPolicy(max_attempts=4, base_delay_seconds=1, factor=2),
        sleep=waits.append,
    )
    store.put("k", b"x")
    assert store.get("k")[0] == b"x"
    assert waits == [1, 2]


def test_retries_stop_at_the_budget_and_never_retry_a_permanent_error(
    tmp_path,
):
    waits = []
    transient = objectstore.TransientStoreError("slow")
    inner = Flaky(tmp_path / "r", [transient] * 3)
    store = objectstore.RetryingStore(
        inner,
        retry.RetryPolicy(max_attempts=3, base_delay_seconds=0.1),
        sleep=waits.append,
    )
    with pytest.raises(retry.RetryExhaustedError) as raised:
        store.put("k", b"x")
    assert raised.value.attempts == 3 and len(waits) == 2
    inner.puts = 0
    inner.failures = [objectstore.StoreError("denied")]
    with pytest.raises(objectstore.StoreError):
        store.put("k", b"x")
    assert inner.puts == 1


def test_retry_policy_is_validated_and_deterministic():
    policy = retry.RetryPolicy(5, 0.5, 2.0, 3.0)
    assert policy.delays() == [0.5, 1.0, 2.0, 3.0]
    for bad in (
        {"max_attempts": 0},
        {"factor": 0.5},
        {"base_delay_seconds": -1},
        {"max_delay_seconds": float("inf")},
    ):
        with pytest.raises(errors.ConfigError):
            retry.RetryPolicy(**bad)


def test_a_denied_link_onto_a_taken_path_means_already_exists(tmp_path):
    taken = tmp_path / "taken"
    taken.write_bytes(b"x")
    denied = PermissionError(13, "Access is denied")
    assert objectstore.refused_as_existing(denied, taken)
    assert objectstore.refused_as_existing(FileExistsError(), taken)


def test_a_denied_link_onto_a_missing_path_is_a_real_error(tmp_path):
    denied = PermissionError(13, "Access is denied")
    assert not objectstore.refused_as_existing(denied, tmp_path / "absent")
    assert not objectstore.refused_as_existing(
        OSError(28, "No space left"), tmp_path / "absent"
    )


def test_a_racer_that_dies_is_reported_with_its_error(tmp_path):
    class Broken(objectstore.LocalObjectStore):
        def put(self, key, data, *, if_absent=False):
            raise OSError("disk on fire")

    with pytest.raises(conformance.ConformanceFailure, match="disk on fire"):
        conformance.check_if_absent_race(Broken(tmp_path / "r"), "base")


def test_path_of_is_lexical_and_never_leaves_the_root(tmp_path):
    local = objectstore.LocalObjectStore(tmp_path / "root")
    deep = local.path_of("a/b/c.json")
    assert deep == local.root / "a" / "b" / "c.json"
    assert not deep.parent.exists()
    for bad in ("../x", "a/../../x", "/abs"):
        with pytest.raises(objectstore.StoreError):
            local.path_of(bad)
