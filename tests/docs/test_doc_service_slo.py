"""Guarantees of docs/SERVICE_SLO.md that no other test enforced."""

import json
import pathlib
import threading
import time
import urllib.error
import urllib.request

import pytest

import scmle.artifacts.bundle
from scmle.contracts import errors
from scmle.deploy import parity
from scmle.deploy import verify
from scmle.serving import server

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "deploy" / "local" / "golden"


@pytest.fixture(scope="module")
def golden():
    return parity.goldens(GOLDEN)


@pytest.fixture(scope="module")
def bundle(golden, tmp_path_factory):
    folder = tmp_path_factory.mktemp("golden")
    parity.handwrite(golden.spec).save(folder / "bundle")
    return scmle.artifacts.bundle.Bundle.load(folder / "bundle", trusted=True)


def call(url, payload=None):
    """Sends one request and returns the status and the JSON body.

    Args:
        url: The address to call.
        payload: A JSON body to post, or None for a GET.

    Returns:
        The HTTP status and the decoded body.
    """
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, body, {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def settle(adapter, seconds=10.0):
    """Waits until no scoring worker is busy.

    Args:
        adapter: The running service.
        seconds: How long to wait at most.

    Returns:
        Whether every worker became free in time.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if adapter.admission.state()["busy"] == 0:
            return True
        time.sleep(0.02)
    return False


def test_three_timeouts_in_a_row_wedge_livez_and_leave_history_alone(
    golden, bundle, monkeypatch
):
    """Enforces docs/SERVICE_SLO.md:71.

    Also enforces website/src/pages/docs/service.astro:26.
    """
    adapter = server.HttpServing(bundle, scoring_timeout_seconds=0.1)
    adapter.start()
    remembered = []
    gates = []
    original = bundle.score

    def slow(frame):
        if gates:
            gates[-1].wait(20)
        return original(frame)

    monkeypatch.setattr(bundle, "remember", remembered.append)
    monkeypatch.setattr(bundle, "score", slow)
    url = f"{adapter.address}/predict"
    payload = {"records": golden.records[:1]}
    try:
        for ignored in range(3):
            assert call(f"{adapter.address}/livez")[0] == 200
            gates.append(threading.Event())
            status, body = call(url, payload)
            assert status == 504 and body["category"] == "timeout"
            gates[-1].set()
            assert settle(adapter)
        status, body = call(f"{adapter.address}/livez")
        assert status == 503 and body["status"] == "wedged"
        assert body["misses"] == 3
        assert remembered == []
        gates.clear()
        assert call(url, payload)[0] == 200
        assert len(remembered) == 1
        assert call(f"{adapter.address}/livez")[0] == 200
    finally:
        for gate in gates:
            gate.set()
        adapter.stop()


def test_production_refuses_the_published_demo_key_unless_allowed(tmp_path):
    """Enforces docs/SERVICE_SLO.md:128.

    Also enforces website/src/pages/docs/service.astro:84.
    Also enforces website/src/pages/docs/security.astro:35.
    """
    demo = next(iter(verify.PUBLISHED_KEYS))
    production = {"SCMLE_ENV": "production", "SCMLE_SIGNING_KEY": demo}
    with pytest.raises(errors.ArtifactError, match="published demo fixture"):
        verify.inspect(tmp_path, production)
    verify.refuse(demo, production=False)
    verify.refuse(demo, production=True, allowed=True)
    assert verify.ALLOW_PUBLISHED_KEY == "SCMLE_ALLOW_PUBLISHED_KEY"
