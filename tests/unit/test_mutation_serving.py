"""Exact answers of the reference service: errors, readiness, manifests."""

import json
import pathlib
import threading
import urllib.error
import urllib.request

import pandas
import pytest

import scmle.artifacts.bundle
from scmle.contracts import errors
import scmle.contracts.serving
from scmle.inference import guard
from scmle.inference import parity
from scmle.serving import server

GOLDEN = pathlib.Path(__file__).resolve().parents[2] / "deploy" / "local"


@pytest.fixture(scope="module")
def golden():
    return parity.goldens(GOLDEN / "golden")


@pytest.fixture(scope="module")
def bundle(golden, tmp_path_factory):
    folder = tmp_path_factory.mktemp("golden")
    parity.handwrite(golden.spec).save(folder / "bundle")
    return scmle.artifacts.bundle.Bundle.load(folder / "bundle", trusted=True)


def call(url, payload=None, kind="application/json"):
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, body, {"Content-Type": kind})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return (
                response.status,
                json.loads(response.read()),
                response.headers,
            )
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read()), error.headers


def failing(bundle, monkeypatch, error):
    adapter = server.HttpServing(bundle)

    def respond(records):
        raise error

    monkeypatch.setattr(adapter, "respond", respond)
    return adapter


@pytest.mark.parametrize(
    ("error", "status", "category"),
    [
        (guard.DeadlineError("scoring ran late"), 504, "timeout"),
        (guard.UnreadyError("scorer is not ready"), 503, "not_ready"),
    ],
)
def test_a_scoring_failure_keeps_its_message(
    golden, bundle, monkeypatch, error, status, category
) -> None:
    adapter = failing(bundle, monkeypatch, error)
    adapter.start()
    try:
        found = call(f"{adapter.address}/predict", {"records": golden.records})
    finally:
        adapter.stop()
    assert found[:2] == (
        status,
        {
            "schema_version": scmle.contracts.serving.SCHEMA_VERSION,
            "error": str(error),
            "category": category,
            "request_id": found[1]["request_id"],
        },
    )


@pytest.mark.parametrize(
    "error",
    [RuntimeError("secret detail"), errors.OrderError("called too early")],
)
def test_an_unexpected_failure_is_internal(
    golden, bundle, monkeypatch, error
) -> None:
    adapter = failing(bundle, monkeypatch, error)
    adapter.start()
    try:
        status, body = call(
            f"{adapter.address}/predict", {"records": golden.records}
        )[:2]
    finally:
        adapter.stop()
    assert (status, body["error"], body["category"]) == (
        500,
        "internal serving failure",
        "internal",
    )


def test_a_body_that_is_not_json_is_refused_by_name(golden, bundle) -> None:
    adapter = server.HttpServing(bundle)
    adapter.start()
    try:
        status, body = call(
            f"{adapter.address}/predict", {"records": []}, "text/plain"
        )[:2]
    finally:
        adapter.stop()
    assert (status, body["error"], body["category"]) == (
        415,
        "Content-Type must be application/json",
        "request",
    )


def test_an_unready_service_refuses_with_a_retry_hint(golden, bundle) -> None:
    release = threading.Event()

    def loader(readiness):
        readiness.begin(guard.Phase.VERIFYING)
        release.wait(20)
        return bundle

    adapter = server.HttpServing(loader=loader)
    adapter.start()
    try:
        health = call(f"{adapter.address}/health")
        scored = call(f"{adapter.address}/predict", {"records": golden.records})
    finally:
        release.set()
        adapter.stop()
    for status, body, headers in (health, scored):
        assert (status, body["category"]) == (503, "not_ready")
        assert headers["Retry-After"] == "1"


def test_identity_names_the_model_version(bundle) -> None:
    adapter = server.HttpServing(bundle)
    assert adapter.identity() == {
        "model_version": "golden-1",
        "artifact_digest": adapter.artifact_digest,
    }


def test_a_warm_up_without_a_record_is_skipped_and_ready(
    bundle, monkeypatch
) -> None:
    adapter = server.HttpServing(bundle)
    monkeypatch.setattr(adapter, "synthesise", lambda: None)
    adapter.warm()
    found = adapter.readiness.snapshot()
    assert found["ready"]
    assert found["notes"]["warming"] == (
        "skipped: no synthetic record for series or extra inputs"
    )


def test_the_batch_manifest_names_its_schema_version(
    golden, bundle, tmp_path
) -> None:
    source = tmp_path / "records.csv"
    pandas.DataFrame(golden.records).to_csv(source, index=False)
    server.Batcher(bundle).score_file(source, tmp_path / "scored.csv")
    manifest = json.loads(
        (tmp_path / "scored.csv.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == scmle.contracts.serving.SCHEMA_VERSION
