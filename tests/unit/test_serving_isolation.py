"""A slow client holds its own connection, never a scoring slot (C7)."""

import json
import logging
import pathlib
import socket
import struct
import time
import urllib.request

import pytest

import scmle.artifacts.bundle
from scmle.deploy import parity
from scmle.serving import server

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "deploy" / "local" / "golden"


@pytest.fixture(scope="module")
def golden():
    return parity.goldens(GOLDEN)


@pytest.fixture(scope="module")
def bundle(golden, tmp_path_factory):
    folder = tmp_path_factory.mktemp("isolation")
    parity.handwrite(golden.spec).save(folder / "bundle")
    return scmle.artifacts.bundle.Bundle.load(folder / "bundle", trusted=True)


def test_a_body_sent_slowly_does_not_take_the_only_slot(golden, bundle):
    adapter = server.Service(bundle, max_concurrent=1, timeout=1.0)
    adapter.start()
    host, port = adapter.httpd.server_address
    slow = socket.create_connection((host, port), timeout=10)
    try:
        slow.sendall(
            b"POST /predict HTTP/1.0\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: 100\r\n\r\n{"
        )
        time.sleep(0.2)
        body = json.dumps({"records": golden.records[:1]}).encode()
        request = urllib.request.Request(
            f"{adapter.address}/predict",
            body,
            {"Content-Type": "application/json"},
        )
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=10) as response:
            assert response.status == 200
        assert time.monotonic() - started < 0.9
        answer = slow.makefile("rb").read()
        assert answer.startswith(b"HTTP/1.0 408 ")
        assert b"request body not received in time" in answer
    finally:
        slow.close()
        adapter.stop()


def test_a_body_cut_off_by_a_reset_is_logged_and_serving_goes_on(
    bundle, caplog
):
    # A reset while the body was read escaped `predict`'s catch-all once the
    # body moved ahead of admission: no log line, no handling.
    adapter = server.Service(bundle, timeout=10.0)
    adapter.start()
    host, port = adapter.httpd.server_address
    cut = socket.create_connection((host, port), timeout=10)
    try:
        with caplog.at_level(logging.INFO, logger="scmle.serving"):
            cut.sendall(
                b"POST /predict HTTP/1.0\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: 100\r\n\r\n{"
            )
            time.sleep(0.3)
            cut.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
            cut.close()
            deadline = time.monotonic() + 5
            while (
                "request body read failed" not in caplog.messages
                and time.monotonic() < deadline
            ):
                time.sleep(0.02)
        assert "request body read failed" in caplog.messages
        with urllib.request.urlopen(
            f"{adapter.address}/livez", timeout=10
        ) as response:
            assert response.status == 200
    finally:
        cut.close()
        adapter.stop()
