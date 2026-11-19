"""Soak: `scmle serve` under steady load, reloads and a drained stop.

Marked `soak` so CI selects it on its own (`pytest -m soak`). It runs for
`SCMLE_SOAK_SECONDS` (minutes, not hours) against the real command in a
child process: client threads keep scoring, SIGHUP swaps the bundle every
`RELOAD` seconds, and SIGTERM stops it while requests are in flight.

Thresholds and their derivations:

- `FAILURE`, `ALPHA`: no request may fail, and enough must be sent that
  zero failures bounds the failure rate. With zero failures in n requests
  the exact (Clopper-Pearson) one-sided upper bound at level 1 - ALPHA is
  1 - ALPHA ** (1 / n); it falls to FAILURE once
  n >= log(ALPHA) / log(1 - FAILURE), which is `MINIMUM` (2995 at the
  defaults, the "rule of three").
- `LEAK`: after the warm-up (two reloads, so both bundles have been
  loaded, and longer than the monitor's `WINDOW`, so its kept traffic is
  at its steady size) resident memory may grow by at most LEAK KiB per
  request answered, comparing the medians of the first and last third of
  the samples (medians ignore the garbage collector's sawtooth).
  Engineering budget: the Python heap stays bounded (checked with
  tracemalloc), but each connection gets a thread and the free-threaded
  allocator gives back the heaps of finished threads lazily, which was
  measured at about 4 KiB per request over five minutes; keeping one
  scored frame per request (the monitor's former defect) cost about
  23 KiB per request. LEAK sits between, so the soak catches a leak of a
  scored frame or more per request; finer leaks need hours.
"""

import collections
import json
import math
import os
import pathlib
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

from scmle import cli
from scmle.deploy import parity
from scmle.rehearsal.demo import client

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "deploy" / "local" / "golden"
# Seconds the service is soaked for at least; minutes, so CI can afford
# it, and never under twice WARM, so memory is sampled as long after the
# warm-up as before. The soak goes on past it until MINIMUM requests are
# answered, for at most SLACK times as long.
SLACK = 3
# Client threads scoring at the same time.
CLIENTS = 4
# Seconds between two reloads (SIGHUP).
RELOAD = 15.0
# Seconds between two evaluations of the operations signals.
INTERVAL = 5.0
# Seconds of traffic each evaluation covers (`--signals-window`).
WINDOW = 20.0
# Seconds before memory is compared: two reloads, longer than WINDOW.
WARM = max(2 * RELOAD, WINDOW)
SECONDS = max(float(os.environ.get("SCMLE_SOAK_SECONDS", "120")), 2 * WARM)
# Largest failure rate the run must bound, at confidence 1 - ALPHA.
FAILURE = 0.001
ALPHA = 0.05
MINIMUM = math.ceil(math.log(ALPHA) / math.log(1 - FAILURE))
# KiB resident memory may grow by per request after the warm-up.
LEAK = 12.0

pytestmark = [
    pytest.mark.soak,
    pytest.mark.skipif(
        not hasattr(signal, "SIGHUP"), reason="SIGHUP is POSIX only"
    ),
    pytest.mark.timeout(int(SLACK * SECONDS) + 300),
]


def call(url, payload):
    """Scores one request; the HTTP status, or None if never answered."""
    request = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read())
            return response.status, body["model_version"]
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")[:300]
    except OSError:
        return None, None


class Load:
    """Client threads scoring until stopped, recording every outcome.

    Attributes:
        outcomes: (status, model version, sent after SIGTERM) per request.
        stop: Set to end the clients.
        stopping: Set once SIGTERM has been sent.
    """

    def __init__(self, url, records):
        self.url = url
        self.records = records
        self.outcomes = []
        self.stop = threading.Event()
        self.stopping = threading.Event()
        self.threads = [
            threading.Thread(target=self.score, args=(offset,))
            for offset in range(CLIENTS)
        ]

    def score(self, offset):
        index = offset
        while not self.stop.is_set():
            late = self.stopping.is_set()
            row = self.records[index % len(self.records)]
            status, version = call(self.url, {"records": [row]})
            self.outcomes.append((status, version, late))
            index += CLIENTS

    def start(self):
        for thread in self.threads:
            thread.start()

    def join(self):
        self.stop.set()
        for thread in self.threads:
            thread.join(60)


def test_the_minimum_bounds_the_failure_rate():
    assert MINIMUM == 2995
    assert 1 - ALPHA ** (1 / MINIMUM) <= FAILURE
    assert 1 - ALPHA ** (1 / (MINIMUM - 1)) > FAILURE


def test_serve_survives_load_reloads_and_drains(tmp_path):
    golden = parity.goldens(GOLDEN)
    first = parity.handwrite(golden.spec).save(tmp_path / "one")
    second = parity.handwrite(
        {**golden.spec, "model_version": "golden-2", "intercept": 1.3}
    ).save(tmp_path / "two")
    folders = [pathlib.Path(first), pathlib.Path(second)]
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
            str(INTERVAL),
            "--signals-window",
            str(WINDOW),
        ],
        stderr=subprocess.PIPE,
        text=True,
        cwd=ROOT,
    )
    lines = []
    listening = threading.Event()

    def read():
        # Drained all along: a full pipe would block the service's logging.
        for line in process.stderr:
            lines.append(line)
            if "listening on" in line:
                listening.set()

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    load = None
    try:
        assert listening.wait(60), "".join(lines)
        address = next(
            found.group(1)
            for line in lines
            if (found := re.search(r"listening on (http://\S+)", line))
        )
        load = Load(f"{address}/predict", golden.records)
        load.start()
        started = time.monotonic()
        reloads, samples = 0, []
        while (elapsed := time.monotonic() - started) < SECONDS or (
            len(load.outcomes) < MINIMUM and elapsed < SLACK * SECONDS
        ):
            if elapsed >= RELOAD * (reloads + 1):
                reloads += 1
                live.unlink()
                live.symlink_to(folders[reloads % 2], target_is_directory=True)
                process.send_signal(signal.SIGHUP)
            resident = client.footprint(process)
            assert resident is not None
            if elapsed >= WARM:
                samples.append((len(load.outcomes), resident))
            time.sleep(1.0)
        load.stopping.set()
        process.send_signal(signal.SIGTERM)
        assert process.wait(120) == cli.EXIT_OK
        load.join()
        reader.join(30)
    finally:
        if load is not None:
            load.join()
        if process.poll() is None:
            process.kill()
            process.wait(30)
    log = "".join(lines)
    served = [outcome for outcome in load.outcomes if not outcome[2]]
    late = [outcome for outcome in load.outcomes if outcome[2]]
    assert len(served) >= MINIMUM, len(served)
    failed = collections.Counter(
        (status, version) for status, version, after in served if status != 200
    )
    assert not failed, (failed, log[-4000:])
    # After SIGTERM a request is answered in full or refused unanswered.
    assert {status for status, version, after in late} <= {200, None}
    versions = {version for status, version, after in served}
    assert versions == {"golden-1", "golden-2"}
    assert reloads >= 2
    assert log.count("serving sha256:") == reloads
    assert "reload failed" not in log
    assert "stopping: draining the requests in flight" in log
    third = len(samples) // 3
    assert third >= 5, samples
    first, last = samples[:third], samples[-third:]
    answered = statistics.median(n for n, mib in last) - statistics.median(
        n for n, mib in first
    )
    growth = statistics.median(mib for n, mib in last) - statistics.median(
        mib for n, mib in first
    )
    assert answered > 0 and 1024 * growth / answered <= LEAK, samples
    signals = written.read_text(encoding="utf-8").splitlines()
    assert len(signals) >= int(elapsed // INTERVAL) - 1
    assert all("signals" in json.loads(line) for line in signals)
