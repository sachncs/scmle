"""One conformance suite, three backends; plus what is specific to each."""

import pytest

from scmle.contracts import errors
from scmle.core import retry
from scmle.data.store import cloud
from scmle.data.store import conformance
from scmle.data.store import objectstore
from tests import support_cloud as fakes


def make_local(tmp_path):
    return objectstore.LocalObjectStore(tmp_path / "root")


def make_s3(tmp_path):
    return cloud.S3ObjectStore(fakes.FakeS3Client(), "bucket", "team/data")


def make_gcs(tmp_path):
    return cloud.GcsObjectStore(fakes.FakeGcsClient(), "bucket", "team/data")


BACKENDS = {"local": make_local, "s3": make_s3, "gcs": make_gcs}


@pytest.fixture(params=sorted(BACKENDS))
def store(request, tmp_path):
    return BACKENDS[request.param](tmp_path)


def test_the_conformance_suite_passes_on_every_backend(store):
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


def test_uris_name_the_backend_and_prefix(tmp_path):
    assert make_s3(tmp_path).uri("a/b") == "s3://bucket/team/data/a/b"
    assert make_gcs(tmp_path).uri("a/b") == "gs://bucket/team/data/a/b"
    assert make_local(tmp_path).uri("a/b").startswith("file://")


def test_a_local_key_cannot_leave_the_root(tmp_path):
    store = make_local(tmp_path)
    for key in ("../outside", "a/../../b", "/etc/passwd"):
        with pytest.raises(objectstore.StoreError):
            store.put(key, b"x")
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize("name", ["s3", "gcs"])
def test_a_changed_object_is_detected_by_its_recorded_hash(name, tmp_path):
    store = BACKENDS[name](tmp_path)
    store.put("k", b"original")
    if name == "s3":
        stored = store.client.objects[("bucket", "team/data/k")]
    else:
        stored = store.bucket_handle.store["team/data/k"]
    stored["data"] = b"tampered"
    with pytest.raises(objectstore.CorruptObjectError):
        store.get("k")


def test_s3_errors_are_translated():
    client = fakes.FakeS3Client()
    store = cloud.S3ObjectStore(client, "bucket")
    client.fail_next("SlowDown")
    with pytest.raises(objectstore.TransientStoreError):
        store.put("k", b"x")
    client.fail_next("AccessDenied")
    with pytest.raises(objectstore.StoreError) as raised:
        store.put("k", b"x")
    assert not isinstance(raised.value, objectstore.TransientStoreError)


def test_gcs_errors_are_translated():
    client = fakes.FakeGcsClient()
    store = cloud.GcsObjectStore(client, "bucket")
    client.fail_next(fakes.FakeServiceUnavailable())
    with pytest.raises(objectstore.TransientStoreError):
        store.put("k", b"x")
    store.put("k", b"x")
    with pytest.raises(objectstore.AlreadyExistsError):
        store.put("k", b"y", if_absent=True)


@pytest.mark.parametrize("name", ["s3", "gcs"])
def test_transient_failures_are_retried_with_the_declared_backoff(
    name, tmp_path
):
    inner = BACKENDS[name](tmp_path)
    client = inner.client if name == "s3" else inner.bucket_handle.client
    waits = []
    store = objectstore.RetryingStore(
        inner,
        retry.RetryPolicy(max_attempts=4, base_delay_seconds=1, factor=2),
        sleep=waits.append,
    )
    if name == "s3":
        client.fail_next("SlowDown", "503")
    else:
        client.fail_next(
            fakes.FakeServiceUnavailable(), fakes.FakeServiceUnavailable()
        )
    store.put("k", b"x")
    assert store.get("k")[0] == b"x"
    assert waits == [1, 2]


def test_retries_stop_at_the_budget_and_never_retry_a_permanent_error(
    tmp_path,
):
    client = fakes.FakeS3Client()
    waits = []
    store = objectstore.RetryingStore(
        cloud.S3ObjectStore(client, "bucket"),
        retry.RetryPolicy(max_attempts=3, base_delay_seconds=0.1),
        sleep=waits.append,
    )
    client.fail_next("SlowDown", "SlowDown", "SlowDown")
    with pytest.raises(retry.RetryExhaustedError) as raised:
        store.put("k", b"x")
    assert raised.value.attempts == 3 and len(waits) == 2
    client.calls.clear()
    client.fail_next("AccessDenied")
    with pytest.raises(objectstore.StoreError):
        store.put("k", b"x")
    assert client.calls == ["put"]


def test_the_sdk_factories_import_lazily_and_explain_a_missing_sdk():
    def missing(name):
        raise ImportError(name)

    with pytest.raises(errors.ConfigError, match="boto3"):
        cloud.s3_store("b", importer=missing)
    with pytest.raises(errors.ConfigError, match="google-cloud-storage"):
        cloud.gcs_store("b", importer=missing)


def test_the_factories_pass_arguments_to_the_sdk_client():
    class Boto:
        @staticmethod
        def client(service, **arguments):
            Boto.seen = (service, arguments)
            return fakes.FakeS3Client()

    store = cloud.s3_store(
        "b", "p", importer=lambda name: Boto, endpoint_url="http://minio:9000"
    )
    assert Boto.seen == ("s3", {"endpoint_url": "http://minio:9000"})
    assert store.uri("k") == "s3://b/p/k"


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


def test_the_conformance_script_reports_and_sets_its_exit_status(
    tmp_path, capsys
):
    import importlib.util
    import json
    import pathlib

    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_object_store_conformance.py"
    )
    spec = importlib.util.spec_from_file_location("conformance_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "report.json"
    status = module.main(
        ["local", "--root", str(tmp_path / "root"), "--out", str(out)]
    )
    assert status == 0
    assert json.loads(out.read_text())["passed"] is True
    assert json.loads(capsys.readouterr().out)["backend"].startswith("file:")
    with pytest.raises(SystemExit):
        module.main(["s3"])


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
