"""The demo's steps, each measured and recorded in the transcript."""

import http
import json
import os
import pathlib
import shutil
import subprocess  # nosec B404 - the services the steps start and stop
import time
from typing import Any

import pandas

from scmle.artifacts import bundle
import scmle.artifacts.results
from scmle.deploy import packaging
from scmle.deploy import parity
from scmle.deploy import provenance
import scmle.governance.registry
from scmle.monitoring import monitor
from scmle.operations import lifecycle
from scmle.rehearsal.demo import client
from scmle.rehearsal.demo import fixture
from scmle.rehearsal.demo import journal
from scmle.serving import server


def prepare(
    workdir: pathlib.Path,
    data: pathlib.Path = fixture.DATA_DIRECTORY,
    transcript: journal.Transcript | None = None,
) -> dict[str, Any]:
    """Trains, evaluates, registers and packages.

    Writes `evaluation.json` and a model registry into `workdir`, registers the
    good and the bad model, promotes the good one, and exports the good model
    and the golden reference model as service folders.

    Args:
        workdir: Working directory; created as needed.
        data: Directory holding `train.csv` and `bad_release_train.csv`.
        transcript: Transcript to record the steps in; a new one if None.

    Returns:
        A dict with the paths `service` and `golden_service`.
    """
    log = transcript or journal.Transcript()
    workdir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    good = fixture.train("good", data / "train.csv")
    bad = fixture.train("bad", data / "bad_release_train.csv")
    evaluation = {
        "good": good.evaluation["test_metrics"],
        "bad_release": bad.evaluation["test_metrics"],
    }
    (workdir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    log.record(
        "train and evaluate",
        started,
        ok=all(
            metrics["roc_auc"] > fixture.MINIMUM_ROC_AUC
            for metrics in evaluation.values()
        ),
        test_metrics=evaluation,
    )
    return package(workdir, good, bad, log)


def package(
    workdir: pathlib.Path,
    good: scmle.artifacts.results.Run,
    bad: scmle.artifacts.results.Run,
    log: journal.Transcript,
) -> dict[str, pathlib.Path]:
    """Registers both models, promotes the good one and packages services.

    Args:
        workdir: Working directory; the registry and the service folders are
            written below it.
        good: The run of the good model.
        bad: The run of the bad release.
        log: Transcript the step is recorded in.

    Returns:
        A dict with the paths `service` and `golden_service`.
    """
    started = time.monotonic()
    registry = scmle.governance.registry.ModelRegistryStore(
        workdir / "registry"
    )
    records = []
    for label, result in (("v1 good", good), ("v2 bad release", bad)):
        exported = result.bundle.save(
            workdir / f"export-{label[:2]}", fixture.DEMO_SIGNING_KEY
        )
        records.append(
            registry.register(
                fixture.MODEL,
                exported,
                metrics=result.evaluation["test_metrics"],
                run_id=result.run_id,
                signing_key=fixture.DEMO_SIGNING_KEY,
            )
        )
    registry.promote(
        fixture.MODEL,
        records[0].version,
        approval=fixture.sanction(
            records[0], "initial release", "evaluation.json"
        ),
        signing_key=fixture.DEMO_SIGNING_KEY,
    )
    service = packaging.scaffold(
        good.bundle, workdir / "service", signing_key=fixture.DEMO_SIGNING_KEY
    )
    golden = parity.goldens(fixture.GOLDEN_DIRECTORY)
    golden_service = packaging.scaffold(
        parity.handwrite(golden.spec),
        workdir / "golden-service",
        signing_key=fixture.DEMO_SIGNING_KEY,
    )
    log.record(
        "register and package",
        started,
        ok=(service / "provenance.json").exists(),
        versions=[record.version for record in records],
        champion=registry.resolve(fixture.MODEL).version,
        bundle_digest=provenance.bundle_digest(service / "model"),
        files=sorted(p.name for p in service.iterdir()),
    )
    return {"service": service, "golden_service": golden_service}


def compare(
    base: str,
    model_directory: pathlib.Path,
    data: pathlib.Path,
    workdir: pathlib.Path,
) -> dict[str, Any]:
    """Compares online, batch and golden predictions with the bundle.

    Args:
        base: Base URL of the running service.
        model_directory: Directory of the model bundle the service serves.
        data: Directory holding `fresh.csv`.
        workdir: Directory where the batch output is written.

    Returns:
        A dict with whether the online and batch predictions match the bundle,
        the largest probability difference, the artifact digests seen in the
        response and in the batch manifest, and whether they are the same.
    """
    frame = pandas.read_csv(data / "fresh.csv").drop(columns=[fixture.TARGET])
    direct = bundle.Bundle.load(
        model_directory, signing_key=fixture.DEMO_SIGNING_KEY
    )
    rows = frame.head(60).to_dict("records")
    status, body = client.post_json(f"{base}/predict", {"records": rows})
    online_match = (
        status == http.HTTPStatus.OK
        and body["predictions"] == direct.predict(frame.head(60)).tolist()
    )
    worst = 0.0
    if status == http.HTTPStatus.OK:
        expected = direct.predict_proba(frame.head(60))
        worst = float(
            abs(
                expected - pandas.DataFrame(body["probabilities"]).to_numpy()
            ).max()
        )
    batch = server.Batcher(direct)
    batch_path = workdir / "batch-predictions.csv"
    batch.score_file(data / "fresh.csv", batch_path)
    scored = pandas.read_csv(batch_path)
    batch_match = (
        scored["prediction"].tolist() == direct.predict(frame).tolist()
    )
    manifest = json.loads(
        (workdir / "batch-predictions.csv.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "online_matches_bundle": online_match,
        "max_probability_difference": worst,
        "batch_matches_bundle": batch_match,
        "digest_in_response": body.get("artifact_digest"),
        "digest_in_batch_manifest": manifest["artifact_digest"],
        "same_digest": body.get("artifact_digest")
        == manifest["artifact_digest"],
    }


def tamper(service: pathlib.Path, workdir: pathlib.Path) -> dict[str, Any]:
    """Checks that a modified and an unsigned copy are alive but never ready.

    Args:
        service: Service folder to copy and damage.
        workdir: Directory in which the two copies are made.

    Returns:
        For `corrupted` and `unsigned`, the HTTP statuses of `/livez`, `/readyz`
        and `/predict` and the reason the service reports.
    """
    outcomes: dict[str, Any] = {}
    corrupt = workdir / "corrupt-service"
    unsigned = workdir / "unsigned-service"
    for target in (corrupt, unsigned):
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(service, target)
    with (corrupt / "model" / bundle.MODEL_FILE).open("ab") as handle:
        handle.write(b"tampered")
    (unsigned / "model" / bundle.SIGNATURE_FILE).unlink()
    for label, folder in (("corrupted", corrupt), ("unsigned", unsigned)):
        port = client.unused()
        process = client.launch(folder, port)
        base = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + 30
            ready_status = 0
            ready: dict[str, Any] = {}
            while time.monotonic() < deadline:
                try:
                    ready_status, ready = client.retrieve(
                        f"{base}/readyz", timeout=2.0
                    )
                except OSError:
                    # handled: not-yet-ready poll retried until deadline
                    time.sleep(0.2)
                    continue
                if ready.get("phase") == "failed":
                    break
                time.sleep(0.2)
            live_status = client.retrieve(f"{base}/livez")[0]
            predict_status = client.post_json(
                f"{base}/predict", {"records": [{"age": 1.0}]}
            )[0]
            outcomes[label] = {
                "livez": live_status,
                "readyz": ready_status,
                "predict": predict_status,
                "reason": ready.get("reason"),
            }
        finally:
            client.halt(process)
    return outcomes


def rollback(
    workdir: pathlib.Path, data: pathlib.Path = fixture.DATA_DIRECTORY
) -> dict[str, Any]:
    """Promotes the bad release, monitors it, rolls back, serves the result.

    The lifecycle manager loads registered bundles with the key from the
    environment, so the demo key is set for the duration of the drill only.

    Args:
        workdir: Working directory prepared by `prepare`.
        data: Directory holding `fresh.csv`.

    Returns:
        The findings of `drill`.
    """
    previous = os.environ.get("SCMLE_SIGNING_KEY")
    os.environ["SCMLE_SIGNING_KEY"] = fixture.DEMO_SIGNING_KEY
    try:
        return drill(workdir, data)
    finally:
        if previous is None:
            os.environ.pop("SCMLE_SIGNING_KEY", None)
        else:
            os.environ["SCMLE_SIGNING_KEY"] = previous


def drill(workdir: pathlib.Path, data: pathlib.Path) -> dict[str, Any]:
    """Runs the drill itself: promote, monitor, roll back, serve.

    Args:
        workdir: Working directory prepared by `prepare`.
        data: Directory holding `fresh.csv`.

    Returns:
        A dict with the champion version before and after rollback, the
        monitor's recommendation and reasons, whether the rollback happened and
        which version was restored, whether the restored model serves the same
        predictions as the original, whether the audit log has the rollback
        check, and the bad release's metrics on the fresh data.
    """
    registry = scmle.governance.registry.ModelRegistryStore(
        workdir / "registry"
    )
    manager = lifecycle.Maintainer(
        registry, fixture.MODEL, str(workdir / "work")
    )
    fresh = pandas.read_csv(data / "fresh.csv")
    good_record = registry.versions(fixture.MODEL)[0]
    bad_record = registry.versions(fixture.MODEL)[1]
    registry.promote(
        fixture.MODEL,
        bad_record.version,
        approval=fixture.sanction(bad_record, "bad release (drill)", "drill"),
        signing_key=fixture.DEMO_SIGNING_KEY,
    )
    champion_before = registry.resolve(fixture.MODEL).version
    features = fresh.drop(columns=[fixture.TARGET])
    report = monitor.Watcher(
        registry.load(fixture.MODEL, signing_key=fixture.DEMO_SIGNING_KEY)
    ).assess(features, fresh[fixture.TARGET])
    outcome = manager.revertible(
        fresh,
        fixture.TARGET,
        approval=fixture.sanction(
            good_record, "rollback to v1 (drill)", "drill"
        ),
        signing_key=fixture.DEMO_SIGNING_KEY,
    )
    restored = registry.load(
        fixture.MODEL, signing_key=fixture.DEMO_SIGNING_KEY
    )
    original = bundle.Bundle.load(
        workdir / "export-v1", signing_key=fixture.DEMO_SIGNING_KEY
    )
    adapter = server.HttpServing(restored)
    adapter.start()
    try:
        status, body = client.post_json(
            f"{adapter.address}/predict",
            {"records": features.head(40).to_dict("records")},
        )
    finally:
        adapter.stop()
    audit = [
        json.loads(line)["action"]
        for line in (registry.root / scmle.governance.registry.AUDIT_FILE)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    return {
        "champion_before_rollback": champion_before,
        "monitor_recommendation": report.recommendation.action,
        "monitor_reasons": report.recommendation.reasons,
        "rolled_back": outcome.rolled_back,
        "restored_version": outcome.restored_version,
        "champion_after_rollback": registry.resolve(fixture.MODEL).version,
        "serves_restored_model": status == http.HTTPStatus.OK
        and body["predictions"] == original.predict(features.head(40)).tolist(),
        "audit_has_rollback_check": "lifecycle.rollback_check" in audit,
        "bad_release_metrics_on_fresh": getattr(
            report.performance, "metrics", None
        ),
    }


def mirror(
    transcript: journal.Transcript,
    store_url: str,
    service: pathlib.Path,
    workdir: pathlib.Path,
) -> pathlib.Path:
    """Uploads the service folder, fetches it back and records the result.

    Args:
        transcript: Where the step is recorded.
        store_url: Base URL of the object store.
        service: The service folder to upload.
        workdir: Working directory; the copy is fetched into
            `fetched-service` below it, replacing an earlier copy.

    Returns:
        The folder the service was fetched into.
    """
    started = time.monotonic()
    etags = client.sync(store_url, service, "models/churn/v1")
    fetched = workdir / "fetched-service"
    if fetched.exists():
        shutil.rmtree(fetched)
    client.download(store_url, "models/churn/v1", etags, fetched)
    same = provenance.bundle_digest(
        fetched / "model"
    ) == provenance.bundle_digest(service / "model")
    transcript.record(
        "object store round trip",
        started,
        ok=same and len(etags) > 0,
        objects=len(etags),
    )
    return fetched


def corroborate(
    transcript: journal.Transcript,
    golden_service: pathlib.Path,
    processes: list[subprocess.Popen[bytes]],
) -> None:
    """Serves the golden-record service and checks parity against it.

    Args:
        transcript: Where the step is recorded.
        golden_service: The service folder holding the golden bundle.
        processes: Started services; the new process is appended so the
            caller stops it.
    """
    started = time.monotonic()
    golden_port = client.unused()
    golden_base = f"http://127.0.0.1:{golden_port}"
    golden_process = client.launch(golden_service, golden_port)
    processes.append(golden_process)
    golden_ready = client.settle(golden_base, golden_process)
    report: dict[str, Any] = {"passed": False}
    if golden_ready is not None:
        report = parity.crosscheck(
            golden_base, fixture.GOLDEN_DIRECTORY, "cross_environment", 30.0
        )
    transcript.record(
        "golden-record parity",
        started,
        ok=bool(report["passed"]),
        max_probability_error=report.get("max_probability_error"),
        tolerance=report.get("tolerance"),
    )


def instrument(
    transcript: journal.Transcript,
    *,
    base: str,
    process: subprocess.Popen[bytes],
    data: pathlib.Path,
    metrics_url: str,
    rolled_back: bool,
    startup: float | None,
) -> None:
    """Measures latency and memory, ships them to the sink and records it.

    Args:
        transcript: Where the step is recorded.
        base: Base URL of the service.
        process: The service process, whose memory is read.
        data: Directory holding the demo datasets.
        metrics_url: Base URL of the metrics sink.
        rolled_back: Whether the rollback drill rolled back.
        startup: Seconds the service took to become ready, if it did.
    """
    started = time.monotonic()
    frame = pandas.read_csv(data / "fresh.csv").drop(columns=[fixture.TARGET])
    single = client.measure(base, client.serialize(frame.head(1)))
    hundred = client.measure(base, client.serialize(frame.head(100)), 40)
    memory = client.footprint(process)
    accepted = client.push(
        metrics_url,
        {
            "latency_single_p95": (single["p95_ms"], "ms"),
            "latency_hundred_p95": (hundred["p95_ms"], "ms"),
            "memory_resident": (memory or 0.0, "MiB"),
            "drift_rollback_drill": (1.0 if rolled_back else 0.0, "bool"),
        },
        "v1",
    )
    count = client.retrieve(f"{metrics_url}/metrics")[1]["count"]
    transcript.record(
        "measure and ship metrics",
        started,
        ok=accepted == fixture.EXPECTED_SAMPLES
        and count >= fixture.EXPECTED_SAMPLES,
        single_record=single,
        hundred_records=hundred,
        resident_mib=memory,
        startup_seconds=None if startup is None else round(startup, 2),
        samples_in_sink=count,
    )


def serve(
    transcript: journal.Transcript,
    fetched: pathlib.Path,
    processes: list[subprocess.Popen[bytes]],
) -> tuple[str, subprocess.Popen[bytes], float]:
    """Serves the fetched folder under production rules and records it.

    Args:
        transcript: Where the step is recorded.
        fetched: The service folder fetched from the object store.
        processes: Started services; the new process is appended so the
            caller stops it.

    Returns:
        The base URL of the service, its process and the seconds it took to
        become ready.

    Raises:
        ArtifactError: If the service did not become ready, carrying the tail
            of its stderr.
    """
    started = time.monotonic()
    port = client.unused()
    base = f"http://127.0.0.1:{port}"
    process = client.launch(fetched, port)
    processes.append(process)
    startup = client.settle(base, process)
    status, ready = client.retrieve(f"{base}/readyz") if startup else (0, {})
    transcript.record(
        "serve under production rules",
        started,
        ok=startup is not None and status == http.HTTPStatus.OK,
        startup_seconds=None if startup is None else round(startup, 2),
        artifact_digest=ready.get("artifact_digest"),
        verify_note=ready.get("notes", {}).get("verifying"),
    )
    if startup is None:
        client.unready(process)
    return base, process, startup


def score(
    transcript: journal.Transcript,
    base: str,
    fetched: pathlib.Path,
    data: pathlib.Path,
    workdir: pathlib.Path,
) -> None:
    """Predicts online and in batch, compares with the bundle and records it.

    Args:
        transcript: Where the step is recorded.
        base: Base URL of the running service.
        fetched: The service folder the service runs from.
        data: Directory holding `fresh.csv`.
        workdir: Directory where the batch output is written.
    """
    started = time.monotonic()
    checks = compare(base, fetched / "model", data, workdir)
    transcript.record(
        "predict (online and batch)",
        started,
        ok=checks["online_matches_bundle"]
        and checks["batch_matches_bundle"]
        and checks["same_digest"]
        and checks["max_probability_difference"]
        < fixture.PROBABILITY_TOLERANCE,
        **checks,
    )


def reject(
    transcript: journal.Transcript,
    service: pathlib.Path,
    workdir: pathlib.Path,
) -> None:
    """Checks that damaged copies never become ready and records it.

    Args:
        transcript: Where the step is recorded.
        service: Service folder to copy and damage.
        workdir: Directory in which the copies are made.
    """
    started = time.monotonic()
    negatives = tamper(service, workdir)
    transcript.record(
        "corrupted and unsigned bundles never become ready",
        started,
        ok=all(
            item["livez"] == http.HTTPStatus.OK
            and item["readyz"] == http.HTTPStatus.SERVICE_UNAVAILABLE
            and item["predict"] == http.HTTPStatus.SERVICE_UNAVAILABLE
            for item in negatives.values()
        ),
        **negatives,
    )
