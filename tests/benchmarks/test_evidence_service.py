"""The C7 service harness: its counting rules, its clients and one SIGTERM."""

import json
import math
import pathlib
import signal
import socket
import struct
import threading
import time

import pandas
import pytest

from benchmarks.evidence import service
from benchmarks.evidence import traffic
import scmle.artifacts.bundle
from scmle.deploy import parity
from scmle.serving import server

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "deploy" / "local" / "golden"
POSIX = pytest.mark.skipif(
    not hasattr(signal, "SIGHUP"), reason="SIGTERM handling is POSIX only"
)


@pytest.fixture(scope="module")
def golden():
    return parity.goldens(GOLDEN)


@pytest.fixture(scope="module")
def folder(golden, tmp_path_factory):
    root = tmp_path_factory.mktemp("service")
    return pathlib.Path(parity.handwrite(golden.spec).save(root / "bundle"))


@pytest.fixture
def adapter(folder):
    bundle = scmle.artifacts.bundle.Bundle.load(folder, trusted=True)
    serving = server.Service(bundle)
    serving.start()
    yield serving
    serving.stop()


def test_the_minimum_bounds_the_failure_rate():
    assert traffic.minimum(0.001, 0.05) == 2995
    assert traffic.minimum(0.1, 0.05) == 29
    assert 1 - 0.05 ** (1 / 2995) <= 0.001 < 1 - 0.05 ** (1 / 2994)


def test_the_bound_is_the_binomial_order_statistic():
    assert traffic.bound([float(item) for item in range(100)], 0.95, 0.05) == (
        98.0
    )
    assert traffic.bound([], 0.95, 0.05) == math.inf
    assert traffic.bound([0.1] * 10, 0.95, 0.05) == math.inf


def test_tally_counts_answers_by_status_and_failures_by_kind():
    outcomes = [
        traffic.Outcome(0.0, 0.1, 200, "answered"),
        traffic.Outcome(0.0, 0.1, 200, "answered"),
        traffic.Outcome(0.0, 0.1, None, "reset"),
        traffic.Outcome(0.0, 0.1, 504, "answered"),
    ]
    assert traffic.tally(outcomes) == {"200": 2, "reset": 1, "504": 1}


def test_the_stall_step_hangs_only_on_the_sentinel():
    step = service.Stall(seconds=0.2).fit(None)
    frame = pandas.DataFrame({"a": [1.0, 2.0]})
    started = time.monotonic()
    assert step.transform(frame) is frame
    assert time.monotonic() - started < 0.1
    started = time.monotonic()
    step.transform(pandas.DataFrame({"a": [service.SENTINEL]}))
    assert time.monotonic() - started >= 0.2
    assert step.__sklearn_is_fitted__() is True


def test_send_tells_an_answer_from_a_refusal(adapter, golden):
    address = adapter.httpd.server_address
    found = traffic.send(address, "/predict", {"records": golden.records[:1]})
    assert (found.status, found.kind) == (200, "answered")
    closed = socket.socket()
    closed.bind(("127.0.0.1", 0))
    port = closed.getsockname()[1]
    closed.close()
    found = traffic.send(("127.0.0.1", port), "/livez", timeout=5)
    assert (found.status, found.kind) == (None, "refused")


def test_send_tells_a_reset_and_a_timeout():
    listener = socket.create_server(("127.0.0.1", 0))
    address = listener.getsockname()
    held = []

    def answer(reset):
        connection, ignored = listener.accept()
        connection.recv(1024)
        if reset:
            connection.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
            connection.close()
        else:
            held.append(connection)

    try:
        worker = threading.Thread(target=answer, args=(True,))
        worker.start()
        assert traffic.send(address, "/livez", timeout=5).kind == "reset"
        worker.join(5)
        worker = threading.Thread(target=answer, args=(False,))
        worker.start()
        assert traffic.send(address, "/livez", timeout=0.3).kind == "timeout"
        worker.join(5)
    finally:
        for connection in held:
            connection.close()
        listener.close()


def test_the_router_stops_routing_after_the_threshold(adapter):
    router = traffic.Router(adapter.httpd.server_address, 0.05, 1.0, 3)
    time.sleep(0.2)
    assert router.routes.is_set()
    adapter.draining = True
    router.join(5)
    assert not router.routes.is_set()
    assert router.probes[-3:] == [503, 503, 503]
    assert 200 in router.probes


def test_a_load_stops_once_its_gate_closes(adapter, golden):
    gate = threading.Event()
    gate.set()
    address = adapter.httpd.server_address
    with traffic.Load(
        address, {"records": golden.records[:1]}, 2, gate
    ) as load:
        time.sleep(0.2)
        gate.clear()
        time.sleep(0.2)
        count = len(load.outcomes)
        time.sleep(0.2)
        assert len(load.outcomes) == count > 0
    assert {item.status for item in load.outcomes} == {200}


def test_a_slow_body_is_answered_408(folder):
    bundle = scmle.artifacts.bundle.Bundle.load(folder, trusted=True)
    serving = server.Service(bundle, timeout=0.3)
    serving.start()
    try:
        slow = traffic.crawl(serving.httpd.server_address, 2)
        assert [traffic.answered(item) for item in slow] == [408, 408]
    finally:
        serving.stop()


def document(passed=True):
    """A stored measurement in the shape `measure` writes."""
    profile = {
        "records": 1,
        "requests": 2995,
        "failed": 0,
        "p95": 0.02,
        "p95_bound": 0.025,
        "p95_target": 0.1,
        "passed": passed,
    }
    return {
        "code": service.current(),
        "settings": service.settings(service.alpha()),
        "objectives": {"profiles": [profile], "passed": passed},
        "hang": {
            "hung": 504,
            "load": {"200": 4000},
            "livez_all_hung": 503,
            "passed": True,
        },
        "stop": {
            "rounds": [{}] * 29,
            "resets": 0,
            "refused": 12,
            "answered_after": 900,
            "default": [{}],
            "default_resets": 2,
            "default_answered_after": 0,
            "direct": [{}] * 3,
            "direct_resets": 18,
            "passed": True,
        },
    }


def test_the_summary_decides_on_every_measured_part():
    value, verdict = service.summary(document())
    assert verdict == "pass"
    assert value == (
        "1 record(s): p95 20.0 ms (bound 25.0 <= 100 ms), 0 of 2995 failed; "
        "a hung request answered 504 while 4000 others got ['200']; every "
        "worker hung: /livez 503; SIGTERM under load for callers routed by "
        "readiness, with SCMLE_LINGER_SECONDS 7.5 >= probe threshold x "
        "(period + timeout) x29: 0 resets, 12 refused after close, 900 "
        "answered after the signal; with the default linger 0 x1: 2 resets, "
        "0 answered after the signal (clients that ignore readiness: 18 "
        "resets in 3 rounds, not deciding); soak not measured"
    )
    assert service.summary(document(passed=False))[1] == "fail"


def test_problems_name_other_code_and_other_settings():
    assert service.problems(document()) == []
    stale = {**document(), "code": "other", "settings": {}}
    assert service.problems(stale) == [
        "service: measurements are for other code",
        "service: measurements are for other settings",
    ]


def test_settings_record_the_derived_counts():
    found = service.settings(0.05)
    assert found["requests_min"] == 2995
    assert found["rounds"] == 29
    assert found["period"] == 0.5
    assert found["probe_timeout"] == 2.0
    assert found["threshold"] == 3
    assert found["linger"] == 3 * (0.5 + 2.0)
    assert found["default_linger"] == 0.0
    assert found["default"] == 1
    assert found["clients"] == server.MAX_CONCURRENT_REQUESTS // 2
    assert json.loads(json.dumps(found)) == found


def test_a_soak_needs_a_positive_duration(capsys):
    with pytest.raises(SystemExit):
        service.main(["--soak", "0"])
    assert "positive" in capsys.readouterr().err


@POSIX
def test_scmle_serve_resets_nothing_on_sigterm_under_routed_load(
    folder, golden
):
    found = service.drained(folder, {"records": golden.records[:3]}, True)
    assert found["exit"] == 0
    assert found["resets"] == 0 and found["timeouts"] == 0
    assert found["before"].get("200", 0) > 0
    assert found["routed_at_signal"] is True
    assert found["answered_after"] > 0
    assert found["linger"] == service.LINGER
    assert found["passed"] is True


def test_the_soak_growth_is_flat_for_flat_memory():
    samples = [(float(t), 100 * t, 200.0 + (t % 2)) for t in range(90)]
    found, per = service.growth(samples, 90.0)
    assert found == 0.0 and per == 0.0


def test_the_soak_growth_finds_a_leak_of_20_kib_per_request():
    samples = [
        (float(t), 100 * t, 200.0 + 100 * t * 20 / 1024) for t in range(90)
    ]
    found, per = service.growth(samples, 90.0)
    assert per == pytest.approx(20.0)
    assert per > service.LEAK
    assert found == pytest.approx(40 * 100 * 20 / 1024)


def test_the_soak_growth_needs_samples_and_requests():
    assert service.growth([(50.0, 10, 200.0), (60.0, 20, 210.0)], 90.0) == (
        None,
        None,
    )
    early = [(float(t), 100 * t, 200.0) for t in range(29)]
    assert service.growth(early, 90.0) == (None, None)
    unread = [(float(t), 100 * t, None) for t in range(90)]
    assert service.growth(unread, 90.0) == (None, None)
    idle = [(float(t), 0, 200.0 + t) for t in range(90)]
    assert service.growth(idle, 90.0) == (40.0, None)


@POSIX
def test_a_hung_request_does_not_wedge_scmle_serve(tmp_path):
    found = service.hang(tmp_path, timeout=1.0)
    assert found["hung"] == 504
    assert found["scoring_timeout_seconds"] == 1.0
    assert set(found["load"]) == {"200"}
    assert found["after_hung"] > 0
    limit = server.MAX_CONCURRENT_REQUESTS
    assert found["slow"] == {"408": limit}
    assert found["all_hung"] == {"504": limit - 1}
    assert found["livez_all_hung"] == 503
    assert found["readyz_all_hung"] == 503
    assert found["exit"] == 0
    assert found["passed"] is True
