"""Scenarios where a dependency goes away: the object store, a metrics sink."""

import dataclasses
import http
import time

from scmle.contracts import errors
from scmle.contracts import observability
from scmle.deploy import mocks
from scmle.deploy import provenance
from scmle.operations import recovery
from scmle.operations import spool
from scmle.rehearsal import demo
from scmle.rehearsal.faults import harness


def object_store_loss() -> recovery.Outcome:
    """The store goes away; the service keeps serving; access returns.

    Returns:
        The recovery result of this fault: its checks, its RTO and RPO
        measurements and the details observed.

    Raises:
        ScmleError: If the service did not become ready.
    """
    with harness.stage(
        "store", "lost_object_store_access", "local emulator + subprocess"
    ) as (fleet, base, result):
        service = harness.export_service(base / "service", harness.specify())
        port = demo.unused()
        store = mocks.Depot(base / "objects", port=port)
        fleet.serve(store)
        url = store.address
        etags = demo.sync(url, service, "models/m/v1")
        fetched = base / "fetched"
        demo.download(url, "models/m/v1", etags, fetched)
        running = fleet.start(fetched)
        if harness.boot(running) is None:
            raise errors.ScmleError("the service did not become ready")
        store.stop()
        fetch_failed = False
        try:
            mocks.get_object(url, "models/m/v1/serve.py", timeout=2.0)
        except OSError:
            # handled: recorded as fetch_failed check
            fetch_failed = True
        served_during = harness.predict(running.base)[0] == http.HTTPStatus.OK
        restored = time.monotonic()
        again = mocks.Depot(base / "objects", port=port)
        fleet.serve(again)
        intact = {
            name: again.root.joinpath("models", "m", "v1", name).exists()
            for name in etags
        }
        lost = sum(not found for found in intact.values())
        second = base / "refetched"
        demo.download(url, "models/m/v1", etags, second)
        same = provenance.bundle_digest(
            second / "model"
        ) == provenance.bundle_digest(service / "model")
        replacement = fleet.start(second)
        seconds = harness.boot(replacement)
        elapsed = time.monotonic() - restored
        victim = again.root.joinpath("models", "m", "v1", "serve.py")
        victim.write_bytes(victim.read_bytes() + b"# altered")
        mismatch = False
        try:
            mocks.get_object(
                url, "models/m/v1/serve.py", expected=etags["serve.py"]
            )
        except errors.ArtifactError:
            # handled: recorded as mismatch check
            mismatch = True
        result.checks = {
            "the fetch failed while the store was down": fetch_failed,
            "the running service kept serving": served_during,
            "the bundle fetched after recovery has the original digest": same,
            "an altered object is refused by its digest": mismatch,
        }
        result.measurements = [
            recovery.Measurement(
                "object_store",
                elapsed if seconds is not None else None,
                lost,
                "from the emulator being started again on the same port and "
                "root to a refetched, digest-verified bundle serving; RPO is "
                "uploaded objects missing from the root afterwards. "
                + harness.METHOD_CLOCK,
            )
        ]
        result.detail = {"objects": len(etags)}
    return result


def blackout() -> recovery.Outcome:
    """The metrics sink is down; samples are spooled and drained later.

    Returns:
        The recovery result of this fault: its checks, its RTO and RPO
        measurements and the details observed.

    Raises:
        ScmleError: If the service did not become ready.
    """
    with harness.stage("spool", "dependency_outage", "local emulator") as (
        fleet,
        base,
        result,
    ):
        service = harness.export_service(base / "service", harness.specify())
        running = fleet.start(service)
        if harness.boot(running) is None:
            raise errors.ScmleError("the service did not become ready")
        port = demo.unused()
        url = f"http://127.0.0.1:{port}"

        def send(samples: list[observability.Sample]) -> int:
            return mocks.transmit(url, samples, timeout=2.0)

        stamp = "2026-11-05T10:00:00+00:00"
        samples = [
            observability.Sample(
                f"probe_{n}",
                observability.Indicator.THROUGHPUT,
                float(n),
                "count",
                stamp,
            )
            for n in range(harness.SPOOLED_SAMPLES)
        ]
        first = spool.MetricsSpool(base / "spool.jsonl", send, batch_size=20)
        first.add(samples)
        down = first.flush()
        serving_during = (
            harness.predict(running.base)[0] == http.HTTPStatus.OK
            and demo.retrieve(f"{running.base}/readyz")[0] == http.HTTPStatus.OK
        )
        restarted = spool.MetricsSpool(
            base / "spool.jsonl", send, batch_size=20
        )
        survived = restarted.pending()
        sink = mocks.MetricsSink(base / "sink.jsonl", port=port)
        fleet.serve(sink)
        back = time.monotonic()
        done = restarted.flush()
        elapsed = time.monotonic() - back
        stored = {s["name"] for s in sink.samples()}
        lost = len({s.name for s in samples} - stored)
        result.checks = {
            "the failed flush kept every sample": down.sent == 0
            and down.remaining == harness.SPOOLED_SAMPLES
            and down.error is not None,
            "serving and readiness were unaffected": serving_during,
            "a restarted spool still held every sample": survived
            == harness.SPOOLED_SAMPLES,
            "the spool drained": done.remaining == 0 and done.error is None,
        }
        result.measurements = [
            recovery.Measurement(
                "telemetry_spool",
                elapsed,
                lost,
                "50 samples spooled while the sink was down, the spool object "
                "recreated (a restart), the sink started, then flushed; RTO is "
                "the flush time, RPO is sample names absent from the sink",
            )
        ]
        result.detail = {"flush_while_down": dataclasses.asdict(down)}
    return result
