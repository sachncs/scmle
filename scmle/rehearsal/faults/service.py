"""Scenarios where the service must refuse: bad bundles, keys and schemas."""

import http
import json
import shutil
import threading
import time

from scmle.contracts import errors
from scmle.operations import classify
from scmle.operations import recovery
from scmle.operations import signals
from scmle.rehearsal import demo
from scmle.rehearsal.faults import harness
from scmle.security import audit

DRAIN_REQUESTS = 3
DRAIN_SECONDS = 10.0


class Client(threading.Thread):
    """Sends requests to whatever instance the router points at."""

    def __init__(self, router: dict[str, harness.Instance]) -> None:
        """Starts idle; ``start`` begins sending.

        Args:
            router: Maps ``current`` to the instance that receives requests.
        """
        super().__init__(daemon=True)
        self.router = router
        self.stop_flag = threading.Event()
        self.sent = 0
        self.failed = 0
        self.longest_gap = 0.0

    def run(self) -> None:
        """Sends one request every 50 ms until stopped."""
        gap_start: float | None = None
        while not self.stop_flag.is_set():
            status = harness.predict(self.router["current"].base, 1)[0]
            self.sent += 1
            now = time.monotonic()
            if status == http.HTTPStatus.OK:
                if gap_start is not None:
                    self.longest_gap = max(self.longest_gap, now - gap_start)
                    gap_start = None
            else:
                self.failed += 1
                gap_start = gap_start or now
            time.sleep(0.05)

    def drain(self) -> None:
        """Waits until the requests in flight at a switch have finished.

        An orchestrator drains the old instance before stopping it: a request
        already sent to it must finish, which is certain once two requests
        that began after the switch have been answered. The wait ends after
        `DRAIN_REQUESTS` more requests or after `DRAIN_SECONDS`.
        """
        drained = self.sent + DRAIN_REQUESTS
        deadline = time.monotonic() + DRAIN_SECONDS
        while self.sent < drained and time.monotonic() < deadline:
            time.sleep(0.05)


def restart() -> recovery.Outcome:
    """Replaces an instance under client load: a tampered copy, then a good one.

    The tampered replacement must stay alive but unready and never receive
    traffic. The good replacement takes over once its ``/readyz`` is 200; the
    old instance is then terminated and the failures a client saw are counted.

    Returns:
        The recovery result of this fault: its checks, its RTO and RPO
        measurements and the details observed.

    Raises:
        ScmleError: If the first instance did not become ready.
    """
    with harness.stage(
        "rolling", "unhealthy_container_rolling_restart", "local subprocess"
    ) as (fleet, base, result):
        service = harness.export_service(base / "service", harness.specify())
        broken = harness.corrupt(service, base / "broken")
        old = fleet.start(service)
        if harness.boot(old) is None:
            raise errors.ScmleError("the first instance did not become ready")
        router = {"current": old}
        client = Client(router)
        try:
            client.start()
            time.sleep(0.5)
            bad = fleet.start(broken)
            verdict = harness.failing(bad)
            admitted = verdict["readyz"] == http.HTTPStatus.OK
            bad.process.kill()
            started = time.monotonic()
            new = fleet.start(service)
            became_ready = harness.boot(new)
            if became_ready is not None:
                router["current"] = new
                client.drain()
            old.process.terminate()
            old.process.wait(timeout=15)
            time.sleep(0.8)
            client.stop_flag.set()
            client.join(timeout=10)
        finally:
            client.stop_flag.set()
        window = time.monotonic() - started
        result.checks = {
            "the unready replacement stayed alive and unready": (
                harness.unready(verdict)
            ),
            "the unready replacement was never given traffic": not admitted,
            "the ready replacement took over": became_ready is not None,
            "no request failed during the replacement": client.failed == 0,
        }
        result.measurements = [
            recovery.Measurement(
                "service",
                client.longest_gap,
                client.failed,
                "a client sends one /predict every 50 ms to the instance "
                "the runner has declared current; RTO is the longest run of "
                "failures, RPO is failed requests; the runner stands in for "
                "the orchestrator: it switches only after /readyz is 200",
            )
        ]
        result.detail = {
            "requests_sent": client.sent,
            "requests_failed": client.failed,
            "unready_replacement": verdict,
            "replacement_ready_seconds": None
            if became_ready is None
            else round(became_ready, 2),
            "window_seconds": round(window, 2),
        }
    return result


def invalid_credentials() -> recovery.Outcome:
    """A wrong signing key leaves the service unready until corrected.

    Returns:
        The recovery result of this fault: its checks, its RTO and RPO
        measurements and the details observed.
    """
    with harness.stage(
        "credentials", "invalid_credentials", "local subprocess"
    ) as (fleet, base, result):
        service = harness.export_service(base / "service", harness.specify())
        wrong = fleet.start(service, key=harness.WRONG_KEY)
        verdict = harness.failing(wrong)
        detected = time.monotonic()
        wrong.process.kill()
        wrong.process.wait(timeout=15)
        right = fleet.start(service, key=demo.DEMO_SIGNING_KEY)
        seconds = harness.boot(right)
        elapsed = time.monotonic() - detected
        served = (
            harness.predict(right.base)[0] == http.HTTPStatus.OK
            if seconds is not None
            else False
        )
        result.checks = {
            "the wrong key leaves the service alive and unready": (
                harness.unready(verdict)
            ),
            "the failure reason is reported without the key": harness.WRONG_KEY
            not in json.dumps(verdict),
            "the corrected key serves": served,
        }
        result.measurements = [
            recovery.Measurement(
                "secrets",
                elapsed if seconds is not None else None,
                0,
                "from the service reporting the credential failure to the "
                "first 200 from /predict under the corrected key (restart "
                "included). " + harness.METHOD_CLOCK,
            )
        ]
        result.detail = {"with_wrong_key": verdict}
    return result


def rekey() -> recovery.Outcome:
    """A bundle re-signed under a rotated key is refused under the old key.

    The service is shown to run under the old key, the re-signed bundle to be
    unready under the old key and to serve under the rotated key, and the
    rotation to be recorded in an audit ledger.

    Returns:
        The recovery result of this fault: its checks, its RTO and RPO
        measurements and the details observed.
    """
    with harness.stage(
        "rotation", "secret_rotation_recovery", "local subprocess"
    ) as (fleet, base, result):
        old_service = harness.export_service(
            base / "old", harness.specify(), demo.DEMO_SIGNING_KEY
        )
        first = fleet.start(old_service, key=demo.DEMO_SIGNING_KEY)
        before = harness.boot(first)
        first.process.kill()
        first.process.wait(timeout=15)
        new_service = harness.export_service(
            base / "new", harness.specify(), harness.ROTATED_KEY
        )
        stale = fleet.start(new_service, key=demo.DEMO_SIGNING_KEY)
        verdict = harness.failing(stale)
        detected = time.monotonic()
        stale.process.kill()
        stale.process.wait(timeout=15)
        fresh = fleet.start(new_service, key=harness.ROTATED_KEY)
        seconds = harness.boot(fresh)
        elapsed = time.monotonic() - detected
        log = audit.Log.open(base / "audit.jsonl")
        log.admin("operator-1", "key_rotation", from_key="old", to_key="new")
        log.ledger.verify()
        result.checks = {
            "the service ran under the old key": before is not None,
            "the rotated bundle is refused under the old key": (
                verdict["readyz"] == http.HTTPStatus.SERVICE_UNAVAILABLE
                and verdict["predict"] == http.HTTPStatus.SERVICE_UNAVAILABLE
            ),
            "the rotated key serves": seconds is not None
            and harness.predict(fresh.base)[0] == http.HTTPStatus.OK,
            "the rotation is in the audit ledger": bool(log.rows("admin")),
        }
        result.measurements = [
            recovery.Measurement(
                "secrets",
                elapsed if seconds is not None else None,
                0,
                "from the stale key being refused to the first 200 under "
                "the rotated key with the re-signed bundle. "
                + harness.METHOD_CLOCK,
            )
        ]
        result.detail = {"stale_key_verdict": verdict}
    return result


def corrupted_bundle() -> recovery.Outcome:
    """A tampered bundle never serves; a digest-checked copy restores it.

    Returns:
        The recovery result of this fault: its checks, its RTO and RPO
        measurements and the details observed.
    """
    with harness.stage("corrupt", "corrupted_bundle", "local subprocess") as (
        fleet,
        base,
        result,
    ):
        service = harness.export_service(base / "service", harness.specify())
        expected = harness.digest(service)
        pristine = base / "pristine"
        shutil.copytree(service, pristine)
        tampered = harness.corrupt(service, base / "tampered")
        broken = fleet.start(tampered)
        verdict = harness.failing(broken)
        detected = time.monotonic()
        broken.process.kill()
        broken.process.wait(timeout=15)
        differs = harness.digest(tampered) != expected
        shutil.rmtree(tampered)
        shutil.copytree(pristine, tampered)
        restored = harness.digest(tampered) == expected
        fixed = fleet.start(tampered)
        seconds = harness.boot(fixed)
        elapsed = time.monotonic() - detected
        result.checks = {
            "the tampered bundle was alive but never ready": (
                harness.unready(verdict)
            ),
            "the tampered bundle's digest differs": differs,
            "the restored copy has the recorded digest": restored,
            "the restored bundle serves": seconds is not None
            and harness.predict(fixed.base)[0] == http.HTTPStatus.OK,
        }
        result.measurements = [
            recovery.Measurement(
                "service",
                elapsed if seconds is not None else None,
                0,
                "from the service reporting the tampered bundle to the first "
                "200 after restoring a copy whose digest equals the recorded "
                "one. " + harness.METHOD_CLOCK,
            )
        ]
        result.detail = {"tampered_verdict": verdict}
    return result


def refuse(url: str) -> tuple[float, list[int], classify.Classification]:
    """Sends 40 requests missing a required column and classifies them.

    Args:
        url: The ``/predict`` URL of the service.

    Returns:
        The `time.monotonic()` just before the first request, the status of
        each request and the classification of the signals they produced.
    """
    missing = [
        {k: v for k, v in r.items() if k != "age"} for r in harness.records(3)
    ]
    started = time.monotonic()
    watch = signals.Monitor(signals.Window(minimum=30))
    statuses = []
    for step in range(40):
        code = demo.post_json(url, {"records": missing})[0]
        statuses.append(code)
        watch.tally(100.0 + step, "/predict", code, 0.01)
    return started, statuses, classify.classify(watch.evaluate(140.0))


def incompatible_schema() -> recovery.Outcome:
    """Requests with a wrong schema digest, version or column are refused.

    A different expected digest and an unserved schema version are checked to
    be refused with 409, records missing a column with 422, and the refusals
    to classify as data quality. The service must stay ready, and a caller
    that reads ``/schema`` and sends the digest must get a 200.

    Returns:
        The recovery result of this fault: its checks, its RTO and RPO
        measurements and the details observed.

    Raises:
        ScmleError: If the service did not become ready.
    """
    with harness.stage("schema", "incompatible_schema", "local subprocess") as (
        fleet,
        base,
        result,
    ):
        service = harness.export_service(base / "service", harness.specify())
        running = fleet.start(service)
        if harness.boot(running) is None:
            raise errors.ScmleError("the service did not become ready")
        url = f"{running.base}/predict"
        wrong_digest = demo.post_json(
            url,
            {
                "records": harness.records(1),
                "expected_schema_digest": "sha256:" + "0" * 64,
            },
        )[0]
        wrong_version = demo.post_json(
            url, {"records": harness.records(1), "schema_version": "9"}
        )[0]
        started, statuses, verdict = refuse(url)
        digest = demo.retrieve(f"{running.base}/schema")[1]["schema_digest"]
        adapted = demo.post_json(
            url,
            {"records": harness.records(1), "expected_schema_digest": digest},
        )[0]
        elapsed = time.monotonic() - started
        ready = demo.retrieve(f"{running.base}/readyz")[0] == http.HTTPStatus.OK
        result.checks = {
            "a different expected schema digest is refused with 409": (
                wrong_digest == http.HTTPStatus.CONFLICT
            ),
            "an unserved schema version is refused with 409": wrong_version
            == http.HTTPStatus.CONFLICT,
            "a record missing a required column is refused with 422": set(
                statuses
            )
            == {422},
            "the refusals classify as data quality, not infrastructure": (
                verdict.verdict is classify.Verdict.DATA_QUALITY
            ),
            "the caller recovers after reading /schema": adapted
            == http.HTTPStatus.OK,
            "the service stayed ready": ready,
        }
        result.measurements = [
            recovery.Measurement(
                "service",
                elapsed,
                0,
                "from the first refused request to a 200 after the client "
                "read /schema and sent its digest; the service itself was "
                "never down. " + harness.METHOD_CLOCK,
            )
        ]
        result.detail = {
            "classification": verdict.to_dict(),
            "statuses": {
                "expected_schema_digest": wrong_digest,
                "schema_version": wrong_version,
            },
        }
    return result
