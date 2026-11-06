"""The generated service feeds the operations signals from real requests."""

import json
import pathlib
import urllib.error
import urllib.request

import pytest

from scmle.deploy import runtime
from scmle.inference import parity
from scmle.inference import serving
from scmle.operations import signals

ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def reference():
    golden = parity.load_golden(ROOT / "deploy" / "local" / "golden")
    return parity.reference_bundle(golden.spec), golden


def call(base, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        base + path, data, {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as reply:
            return reply.status
    except urllib.error.HTTPError as error:
        return error.code


def records_of(golden):
    return [dict(row) for row in golden.records[:3]]


def test_every_finished_request_is_recorded_with_route_status_and_time(
    reference,
):
    bundle, golden = reference
    monitor = signals.SignalMonitor()
    adapter = serving.HttpServing(bundle, monitor=monitor)
    adapter.start()
    try:
        base = adapter.address
        assert call(base, "/livez") == 200
        assert call(base, "/predict", {"records": records_of(golden)}) == 200
        assert call(base, "/nope") == 404
        assert call(base, "/predict", {"records": "bad"}) == 400
    finally:
        adapter.stop()
    seen = [
        (route, status) for ignored, route, status, ignored in monitor.requests
    ]
    assert ("/livez", 200) in seen and ("/predict", 200) in seen
    assert ("/nope", 404) in seen and ("/predict", 400) in seen
    assert all(
        seconds >= 0 for ignored, ignored, ignored, seconds in monitor.requests
    )


def test_scored_features_are_sampled_into_the_monitor(reference):
    bundle, golden = reference
    monitor = signals.SignalMonitor(
        signals.WindowPolicy(sample_rate=1.0, seed=1)
    )
    adapter = serving.HttpServing(bundle, monitor=monitor)
    adapter.start()
    try:
        base = adapter.address
        assert call(base, "/predict", {"records": records_of(golden)}) == 200
    finally:
        adapter.stop()
    assert sum(len(frame) for ignored, frame in monitor.features) == 3


def test_a_failing_monitor_never_breaks_serving(reference):
    bundle, golden = reference

    class Broken:
        def record_request(self, *args):
            raise RuntimeError("monitor down")

        def observe_features(self, *args):
            raise RuntimeError("monitor down")

    adapter = serving.HttpServing(bundle, monitor=Broken())
    adapter.start()
    try:
        base = adapter.address
        assert call(base, "/predict", {"records": records_of(golden)}) == 200
    finally:
        adapter.stop()


def test_the_runtime_writes_evaluated_signals_as_json_lines(tmp_path):
    monitor = signals.SignalMonitor()
    for second in range(50):
        monitor.record_request(1000.0 + second, "/predict", 200, 0.01)
    target = tmp_path / "signals.jsonl"
    written = runtime.emit_signals(monitor, 1060.0, target)
    line = json.loads(target.read_text().splitlines()[0])
    assert line["at"] == 1060.0 and line["signals"]
    assert written == len(line["signals"])
    names = {item["name"] for item in line["signals"]}
    assert names  # at least the service signals exist without a profile
    runtime.emit_signals(monitor, 1061.0, target)
    assert len(target.read_text().splitlines()) == 2


def test_the_settings_choose_the_signal_file_and_interval():
    chosen = runtime.signal_settings(
        {
            "SCMLE_SIGNALS_FILE": "/var/run/signals.jsonl",
            "SCMLE_SIGNALS_INTERVAL": "15",
        }
    )
    assert chosen == {"path": "/var/run/signals.jsonl", "interval": 15.0}
    assert runtime.signal_settings({}) == {"path": None, "interval": 60.0}
