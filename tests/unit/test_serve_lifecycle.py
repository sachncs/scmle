"""Serving lifecycle: drain on stop, reload without downtime, signals."""

import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import pandas
import pytest

from scmle import cli
import scmle.artifacts.bundle
from scmle.contracts import errors
from scmle.deploy import parity
from scmle.inference import guard
from scmle.operations import signals
from scmle.serving import runtime
from scmle.serving import server
from tests import support_ops

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "deploy" / "local" / "golden"
POSIX = pytest.mark.skipif(
    not hasattr(signal, "SIGHUP"), reason="SIGHUP is POSIX only"
)


@pytest.fixture(scope="module")
def golden():
    return parity.goldens(GOLDEN)


@pytest.fixture(scope="module")
def folders(golden, tmp_path_factory):
    """Two saved bundles of the golden model: version 1 and version 2."""
    root = tmp_path_factory.mktemp("lifecycle")
    first = parity.handwrite(golden.spec).save(root / "one")
    second = parity.handwrite(
        {**golden.spec, "model_version": "golden-2", "intercept": 1.3}
    ).save(root / "two")
    return pathlib.Path(first), pathlib.Path(second)


@pytest.fixture
def bundle(folders):
    return scmle.artifacts.bundle.Bundle.load(folders[0], trusted=True)


def call(url, payload=None, timeout=20):
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, body, {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_the_drain_time_is_derived_from_the_request_bounds(bundle):
    adapter = server.HttpServing(
        bundle, request_timeout_seconds=4.0, scoring_timeout_seconds=7.0
    )
    assert adapter.drain_seconds == 15.0
    assert server.HttpServing(bundle, drain_seconds=2.5).drain_seconds == 2.5
    with pytest.raises(errors.InputError, match="drain time"):
        server.HttpServing(bundle, drain_seconds=0)


def test_stop_lets_a_request_in_flight_finish_then_refuses_connections(
    golden, bundle, monkeypatch
):
    adapter = server.HttpServing(bundle)
    entered, release = threading.Event(), threading.Event()
    original = adapter.respond

    def slow(records):
        entered.set()
        release.wait(20)
        return original(records)

    monkeypatch.setattr(adapter, "respond", slow)
    adapter.start()
    url = adapter.address
    results = []
    worker = threading.Thread(
        target=lambda: results.append(
            call(f"{url}/predict", {"records": golden.records[:1]})
        )
    )
    worker.start()
    assert entered.wait(20)
    stopper = threading.Thread(target=adapter.stop)
    stopper.start()
    time.sleep(0.3)
    assert stopper.is_alive() and adapter.draining
    health = adapter.health()
    assert health["ready"] is False and health["reason"] == "draining"
    # Refused, or reset if it reached the backlog before the socket closed.
    with pytest.raises(OSError):
        urllib.request.urlopen(f"{url}/livez", timeout=2)
    release.set()
    worker.join(20)
    stopper.join(20)
    assert not stopper.is_alive()
    assert results[0][0] == 200 and len(results[0][1]["predictions"]) == 1


def test_stop_gives_up_on_a_request_after_the_grace(
    golden, bundle, monkeypatch, caplog
):
    adapter = server.HttpServing(bundle)
    entered, release = threading.Event(), threading.Event()

    def stuck(records):
        entered.set()
        release.wait(20)
        return {}

    monkeypatch.setattr(adapter, "respond", stuck)
    adapter.start()
    worker = threading.Thread(
        target=call,
        args=(f"{adapter.address}/predict", {"records": golden.records[:1]}),
    )
    worker.start()
    try:
        assert entered.wait(20)
        started = time.monotonic()
        adapter.stop(grace=0.3)
        assert 0.3 <= time.monotonic() - started < 5
        assert "1 request(s) still in flight after 0.3 s" in caplog.text
    finally:
        release.set()
        worker.join(20)


def test_reload_swaps_the_bundle_while_requests_keep_succeeding(
    golden, bundle, folders
):
    adapter = server.HttpServing(bundle)
    adapter.start()
    try:
        url = f"{adapter.address}/predict"
        before = call(url, {"records": golden.records[:1]})[1]
        assert before["model_version"] == "golden-1"
        statuses = []
        stop = threading.Event()

        def hammer():
            while not stop.is_set():
                statuses.append(call(url, {"records": golden.records[:1]})[0])

        threads = [threading.Thread(target=hammer) for number in range(3)]
        for thread in threads:
            thread.start()
        time.sleep(0.2)
        digest = adapter.reload(runtime.fetch(folders[1], trusted=True))
        time.sleep(0.2)
        stop.set()
        for thread in threads:
            thread.join(20)
        after = call(url, {"records": golden.records[:1]})[1]
        assert after["model_version"] == "golden-2"
        assert after["artifact_digest"] == digest != before["artifact_digest"]
        assert after["probabilities"] != before["probabilities"]
        assert statuses and set(statuses) == {200}
        assert adapter.readiness.snapshot()["notes"]["loading"] == str(
            folders[1]
        )
    finally:
        adapter.stop()


def test_a_failed_reload_keeps_the_old_bundle_serving(golden, bundle):
    adapter = server.HttpServing(bundle)
    adapter.start()
    try:

        def broken(readiness):
            readiness.begin(guard.Phase.VERIFYING)
            raise errors.ArtifactError("signature mismatch")

        with pytest.raises(errors.ArtifactError, match="signature mismatch"):
            adapter.reload(broken)
        status, body = call(
            f"{adapter.address}/predict", {"records": golden.records[:1]}
        )
        assert status == 200 and body["model_version"] == "golden-1"
        assert adapter.health()["ready"] is True
        with pytest.raises(errors.ConfigError, match="no loader"):
            adapter.reload()
    finally:
        adapter.stop()
    adapter.draining = True
    with pytest.raises(guard.UnreadyError, match="draining"):
        adapter.reload(runtime.fetch(pathlib.Path("unused"), trusted=True))


def test_reload_recovers_a_service_whose_first_load_failed(folders):
    def refuse(readiness):
        readiness.begin(guard.Phase.VERIFYING)
        raise errors.ArtifactError("not yet")

    adapter = server.HttpServing(loader=refuse)
    adapter.start()
    try:
        adapter.loading.join(20)
        assert adapter.health()["ready"] is False
        adapter.reload(runtime.fetch(folders[0], trusted=True))
        status, body = call(f"{adapter.address}/readyz")
        assert status == 200 and body["model_version"] == "golden-1"
    finally:
        adapter.stop()


@POSIX
def test_hold_reloads_on_sighup_and_drains_on_sigterm(
    bundle, folders, tmp_path
):
    monitor = signals.SignalMonitor()
    adapter = server.HttpServing(bundle, monitor=monitor)
    written = tmp_path / "signals.jsonl"
    before = signal.getsignal(signal.SIGTERM)
    reloaded = threading.Event()
    loader = runtime.fetch(folders[1], trusted=True)

    def tracked(readiness):
        model = loader(readiness)
        reloaded.set()
        return model

    def drive():
        deadline = time.monotonic() + 20
        while adapter.httpd is None and time.monotonic() < deadline:
            time.sleep(0.02)
        os.kill(os.getpid(), signal.SIGHUP)
        assert reloaded.wait(20)
        time.sleep(0.1)
        os.kill(os.getpid(), signal.SIGTERM)

    driver = threading.Thread(target=drive)
    driver.start()
    runtime.hold(adapter, monitor, written, 60.0, tracked)
    driver.join(20)
    assert adapter.loaded.info.model_version == "golden-2"
    assert adapter.httpd is None and adapter.draining
    lines = written.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and "signals" in json.loads(lines[0])
    assert signal.getsignal(signal.SIGTERM) is before


def test_hold_refuses_a_bad_interval_and_a_worker_thread(bundle):
    adapter = server.HttpServing(bundle)
    monitor = signals.SignalMonitor()
    with pytest.raises(errors.ConfigError, match="interval"):
        runtime.hold(adapter, monitor, None, 0)
    found = []

    def elsewhere():
        try:
            runtime.hold(adapter, monitor, None, 1.0)
        except errors.ConfigError as error:
            found.append(str(error))

    thread = threading.Thread(target=elsewhere)
    thread.start()
    thread.join(20)
    assert found == ["serve from the main thread"]


def test_a_failed_refresh_is_logged_and_serving_continues(bundle, caplog):
    adapter = server.HttpServing(bundle)
    runtime.refresh(adapter, None)
    assert "reload failed; the previous bundle keeps serving" in caplog.text
    assert adapter.loaded is bundle


def test_the_drain_time_comes_from_the_environment():
    assert runtime.settings({})["drain_seconds"] is None
    assert runtime.settings({"SCMLE_DRAIN_SECONDS": "12"})[
        "drain_seconds"
    ] == pytest.approx(12.0)


@POSIX
def test_the_serve_command_reloads_and_drains_as_a_process(
    golden, folders, tmp_path
):
    live = tmp_path / "live"
    live.symlink_to(folders[0], target_is_directory=True)
    written = tmp_path / "signals.jsonl"
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; from scmle import cli; sys.exit(cli.main())",
            "serve",
            str(live),
            "--trusted",
            "--port",
            "0",
            "--signals-file",
            str(written),
            "--signals-interval",
            "0.5",
        ],
        stderr=subprocess.PIPE,
        text=True,
        cwd=ROOT,
    )
    try:
        address = None
        for line in process.stderr:
            found = re.search(r"listening on (http://\S+)", line)
            if found:
                address = found.group(1)
                break
        assert address is not None
        status, body = call(
            f"{address}/predict", {"records": golden.records[:2]}
        )
        assert status == 200 and body["model_version"] == "golden-1"
        live.unlink()
        live.symlink_to(folders[1], target_is_directory=True)
        process.send_signal(signal.SIGHUP)
        deadline = time.monotonic() + 30
        version = None
        while time.monotonic() < deadline and version != "golden-2":
            version = call(
                f"{address}/predict", {"records": golden.records[:1]}
            )[1]["model_version"]
            time.sleep(0.1)
        assert version == "golden-2"
        process.send_signal(signal.SIGTERM)
        assert process.wait(60) == cli.EXIT_OK
        rest = process.stderr.read()
        assert "stopping: draining the requests in flight" in rest
        assert written.read_text(encoding="utf-8").count("\n") >= 1
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(20)


def test_the_monitor_keeps_the_newest_rows_up_to_its_capacity():
    monitor = signals.SignalMonitor(signals.Window(min_samples=5, capacity=50))
    for step in range(5):
        frame = pandas.DataFrame({"age": range(step * 20, step * 20 + 20)})
        assert monitor.sense(1000.0 + step, frame) == 20
    kept = [row["age"] for at, rows in monitor.features for row in rows]
    assert monitor.held == 50
    assert kept == list(range(50, 100))
    assert [at for at, rows in monitor.features] == [1002.0, 1003.0, 1004.0]
    monitor.prune(1003.5 + monitor.policy.window_seconds)
    assert monitor.held == 20
    assert [at for at, rows in monitor.features] == [1004.0]
    assert signals.CAPACITY == 9604
    assert signals.Window().capacity == signals.CAPACITY
    with pytest.raises(errors.InputError, match="capacity"):
        signals.Window(min_samples=30, capacity=29)


def test_the_monitor_follows_the_reference_of_the_bundle_served(
    golden, tmp_path
):
    first = parity.handwrite(golden.spec)
    first.reference = support_ops.profile(seed=1)
    second = parity.handwrite({**golden.spec, "model_version": "golden-2"})
    second.reference = support_ops.profile(seed=2)
    folder = pathlib.Path(second.save(tmp_path / "two"))
    monitor = signals.SignalMonitor()
    adapter = server.HttpServing(first, monitor=monitor)
    assert monitor.profile is first.reference
    adapter.reload(runtime.fetch(folder, trusted=True))
    assert adapter.loaded.info.model_version == "golden-2"
    assert monitor.profile.to_dict() == second.reference.to_dict()
    monitor.sense(support_ops.START, support_ops.frame(400, seed=3, shift=25))
    names = {item.name for item in monitor.evaluate(support_ops.START + 1)}
    assert "feature_psi_max" in names


def test_a_lazily_loaded_bundle_gives_the_monitor_its_reference(
    golden, tmp_path
):
    model = parity.handwrite(golden.spec)
    model.reference = support_ops.profile()
    folder = pathlib.Path(model.save(tmp_path / "one"))
    monitor = signals.SignalMonitor()
    adapter = server.HttpServing(
        loader=runtime.fetch(folder, trusted=True), monitor=monitor
    )
    assert monitor.profile is None
    adapter.boot()
    assert adapter.health()["ready"] is True
    assert monitor.profile.to_dict() == model.reference.to_dict()


def test_a_monitor_failing_on_a_new_bundle_is_logged(bundle, caplog):
    class Broken:
        def tally(self, *args):
            raise RuntimeError("monitor down")

        def sense(self, *args):
            raise RuntimeError("monitor down")

        def adopt(self, profile):
            raise RuntimeError("monitor down")

    adapter = server.HttpServing(bundle, monitor=Broken())
    assert adapter.loaded is bundle
    assert "the signal monitor failed on a new bundle" in caplog.text


def test_the_serve_command_takes_the_signals_window():
    parser = cli.build_parser()
    args = parser.parse_args(["serve", "model", "--signals-window", "45"])
    assert args.signals_window == 45.0
    assert parser.parse_args(["serve", "model"]).signals_window == 300.0


def test_a_request_accepted_while_draining_is_still_scored(golden, bundle):
    adapter = server.HttpServing(bundle)
    adapter.start()
    try:
        # Drain has begun but this connection was already accepted: it is
        # finished, while the probe takes the service out of rotation.
        adapter.draining = True
        status, body = call(
            f"{adapter.address}/predict", {"records": golden.records[:1]}
        )
        assert status == 200 and body["model_version"] == "golden-1"
        status, body = call(f"{adapter.address}/readyz")
        assert status == 503 and body["reason"] == "draining"
    finally:
        adapter.stop()
