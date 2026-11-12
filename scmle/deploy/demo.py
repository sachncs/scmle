"""The local demo: train, evaluate, package, predict, monitor, roll back.

One command (`make demo`, or `python -m scmle.deploy.demo run`) runs the whole
lifecycle on the explicit files in `deploy/local/`, with no cloud account:

1. train a good model and a bad release from two datasets, and evaluate them;
2. package the good model as a hardened service folder, store it in the mock
   object store, fetch it back by digest and serve it with the production
   rules (signed bundle, signed provenance, no `SCMLE_TRUSTED`);
3. predict through the service and the batch path and compare both with the
   bundle, then check the golden records;
4. show that a corrupted copy and an unsigned copy never become ready;
5. register both models, promote the bad release, monitor it on fresh
   labelled data, and roll back to the good one (the rollback drill);
6. ship metrics to the mock metrics sink and measure startup and latency.

`prepare` and `verify` split the same work for the compose file, where the
service runs in a container and the test runner calls it over the network.
The demo signing key below is a published fixture, not a secret.
"""

import argparse
from collections.abc import Mapping
import dataclasses
import http
import json
import os
import pathlib
import shutil
import socket
import statistics
import subprocess  # nosec B404 - runs the generated serve.py and ps
import sys
import tempfile
import time
from typing import Any
from typing import NoReturn
import urllib.error
import urllib.request

import pandas

import scmle.api.project
import scmle.api.results
from scmle.artifacts import bundle
from scmle.contracts import errors
from scmle.contracts import observability
from scmle.contracts import tasks
from scmle.deploy import mocks
from scmle.deploy import provenance
from scmle.governance import lifecycle
import scmle.governance.registry
from scmle.inference import parity
from scmle.inference import serving
from scmle.monitoring import monitor
from scmle.security import tokens

DEMO_SIGNING_KEY = "scmle-local-demo-signing-key-published-fixture"
MODEL = "churn"
TARGET = "churned"
DATA_DIRECTORY = pathlib.Path("deploy/local/data")
GOLDEN_DIRECTORY = pathlib.Path("deploy/local/golden")
REVIEWER = "demo-reviewer"
READY_TIMEOUT_SECONDS = 120.0
LATENCY_REQUESTS = 120
EXPECTED_SAMPLES = 4  # metric samples the demo ships to the sink
MINIMUM_ROC_AUC = 0.8  # a trained demo model must beat this on the test set
PROBABILITY_TOLERANCE = 1e-9  # online and batch probabilities must agree
PROMOTED_VERSION = 2  # the bad release is registered second and promoted
FAST_OPTIONS: dict[str, Any] = {
    "search": tasks.Search(top_models=0),
    "strategies": tasks.Strategies(top_models=0),
    "ensemble": tasks.Stacking(size=0),
    "models": tasks.Candidates(
        exclude_families=["bagging", "deep_learning", "foundation"]
    ),
    "enforcement": "cooperative",
    "time_budget": 300,
    "seed": 7,
}


@dataclasses.dataclass
class Step:
    """One recorded step of the demo.

    Attributes:
        name: Description of the step.
        ok: Whether the step held.
        seconds: Wall-clock seconds the step took.
        detail: Measured values and findings of the step.
    """

    name: str
    ok: bool
    seconds: float
    detail: dict[str, Any]


class Transcript:
    """The ordered record of what the demo did and whether it held.

    Attributes:
        steps: The recorded steps, in order.
    """

    def __init__(self) -> None:
        """Starts empty."""
        self.steps: list[Step] = []

    def record(
        self, name: str, started: float, ok: bool, **detail: Any
    ) -> None:
        """Appends a step and prints it.

        Args:
            name: Description of the step.
            started: Value of `time.monotonic()` when the step began.
            ok: Whether the step held.
            **detail: Measured values and findings, kept on the step and
                printed.
        """
        step = Step(name, ok, round(time.monotonic() - started, 3), detail)
        self.steps.append(step)
        print(
            f"[{'ok' if ok else 'FAIL'}] {name} ({step.seconds}s)", flush=True
        )
        for key, value in detail.items():
            print(f"      {key}: {value}", flush=True)

    @property
    def passed(self) -> bool:
        """Whether at least one step was recorded and every step held."""
        return bool(self.steps) and all(step.ok for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        """Builds a JSON-compatible record.

        Returns:
            A dict with `passed` and `steps` (each step as a dict).
        """
        return {
            "passed": self.passed,
            "steps": [dataclasses.asdict(step) for step in self.steps],
        }


def now() -> str:
    """Formats the current time.

    Returns:
        The current UTC time as an ISO-8601 string with a `+00:00` offset and
        second resolution.
    """
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def train(name: str, data: pathlib.Path) -> scmle.api.results.Run:
    """Trains one model on a CSV with the demo's fixed, fast options.

    Args:
        name: Prefix of the temporary output directory.
        data: CSV file holding the features and the target column.

    Returns:
        The run result, including the model bundle and its evaluation.
    """
    frame = pandas.read_csv(data)
    return scmle.api.project.Project(
        frame,
        TARGET,
        output_dir=str(tempfile.mkdtemp(prefix=name)),
        **FAST_OPTIONS,
    ).run()


def approval_for(
    record: scmle.governance.registry.VersionRecord, reason: str, evidence: str
) -> scmle.governance.registry.PromotionApproval:
    """Builds an approval bound to one registered version.

    Args:
        record: The registered version being approved.
        reason: Why the reviewer approves.
        evidence: Name of the evidence the reviewer cites.

    Returns:
        An approved promotion by the demo reviewer, signed with the demo key.
    """
    return scmle.governance.registry.PromotionApproval(
        model=record.model,
        version=record.version,
        content_id=record.content_id,
        reviewer=REVIEWER,
        decision=scmle.governance.registry.PromotionDecision.APPROVED,
        reason=reason,
        evidence=(evidence,),
        approved_at=now(),
    ).sign(DEMO_SIGNING_KEY)


def start_service(
    folder: pathlib.Path, port: int, environment: str = "production"
) -> subprocess.Popen[bytes]:
    """Starts a generated `serve.py` as the container would.

    `SCMLE_TRUSTED` is removed from the environment, and the published demo key
    is allowed.

    Args:
        folder: Service folder holding `serve.py`.
        port: Local port to listen on, on 127.0.0.1.
        environment: Value of `SCMLE_ENV` for the service.

    Returns:
        The running process; its stderr is piped and its stdout discarded.
    """
    env = {
        **os.environ,
        "HOST": "127.0.0.1",
        "PORT": str(port),
        "SCMLE_ENV": environment,
        "SCMLE_SIGNING_KEY": DEMO_SIGNING_KEY,
        "SCMLE_ALLOW_PUBLISHED_KEY": "1",
    }
    env.pop("SCMLE_TRUSTED", None)
    folder = folder.resolve()
    return subprocess.Popen(  # noqa: S603 -- our own interpreter and script  # nosec B603
        [sys.executable, str(folder / "serve.py")],
        cwd=folder,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


class Authenticator(urllib.request.BaseHandler):
    """Adds one bearer token to requests for the listed services only.

    Attributes:
        token: The bearer token.
        prefixes: URL prefixes the token may be sent to.
    """

    handler_order = 100

    def __init__(self, token: str, prefixes: tuple[str, ...]) -> None:
        """Remembers the token and the URL prefixes it may be sent to.

        Args:
            token: The bearer token.
            prefixes: URL prefixes the token may be sent to.
        """
        self.token = token
        self.prefixes = prefixes

    def http_request(
        self, request: urllib.request.Request
    ) -> urllib.request.Request:
        """Attaches the token if the URL is one of the listed services.

        Args:
            request: The outgoing request.

        Returns:
            The same request, with an `Authorization` header (not carried over
            redirects) when its URL starts with one of the prefixes.
        """
        if request.full_url.startswith(self.prefixes):
            request.add_unredirected_header(
                "Authorization", f"Bearer {self.token}"
            )
        return request


def install_bearer(
    environ: Mapping[str, str], prefixes: tuple[str, ...]
) -> bool:
    """Makes `urllib` authenticate to the compose services.

    The token is minted from the published demo key in `deploy/local/demo.env`
    and sent only to URLs that start with one of `prefixes`.

    Args:
        environ: Environment holding `SCMLE_AUTH_HS256_SECRET` and, optionally,
            `SCMLE_AUTH_ISSUER` and `SCMLE_AUTH_AUDIENCE`.
        prefixes: URL prefixes the token may be sent to.

    Returns:
        Whether a token was installed (the demo key was configured).
    """
    secret = environ.get("SCMLE_AUTH_HS256_SECRET")
    if not secret:
        return False
    now = time.time()
    token = tokens.sign_hs256(
        secret.encode(),
        "env",
        {
            "iss": environ.get("SCMLE_AUTH_ISSUER", ""),
            "aud": environ.get("SCMLE_AUTH_AUDIENCE", ""),
            "sub": "demo-test-runner",
            "iat": int(now),
            "exp": int(now) + 3600,
            "scope": "model:read model:predict",
        },
    )
    opener = urllib.request.build_opener(Authenticator(token, prefixes))
    urllib.request.install_opener(opener)
    return True


def get_json(url: str, timeout: float = 10.0) -> tuple[int, dict[str, Any]]:
    """GETs JSON; HTTP errors return their status and body.

    Args:
        url: URL to request.
        timeout: Seconds to wait.

    Returns:
        The HTTP status and the parsed JSON body.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as reply:  # noqa: S310 -- URLs are http://127.0.0.1 addresses built by the demo  # nosec B310
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def post_json(
    url: str, payload: dict[str, Any], timeout: float = 30.0
) -> tuple[int, dict[str, Any]]:
    """POSTs JSON; HTTP errors return their status and body.

    Args:
        url: URL to post to.
        payload: Object sent as the JSON body.
        timeout: Seconds to wait.

    Returns:
        The HTTP status and the parsed JSON body.
    """
    request = urllib.request.Request(  # noqa: S310 -- URLs are http://127.0.0.1 addresses built by the demo
        url,
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:  # noqa: S310 -- URLs are http://127.0.0.1 addresses built by the demo  # nosec B310
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def wait_ready(
    base: str,
    process: subprocess.Popen[bytes] | None = None,
    timeout: float = READY_TIMEOUT_SECONDS,
) -> float | None:
    """Waits until `/readyz` answers 200.

    Args:
        base: Base URL of the service.
        process: The service process; if it exits, waiting stops.
        timeout: Seconds to wait.

    Returns:
        Seconds until `/readyz` was 200, or None if it never was in time or the
        process exited.
    """
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if process is not None and process.poll() is not None:
            return None
        try:
            status = get_json(f"{base}/readyz", timeout=2.0)[0]
        except OSError:
            status = 0
        if status == http.HTTPStatus.OK:
            return time.monotonic() - started
        time.sleep(0.2)
    return None


def stop_service(process: subprocess.Popen[bytes] | None) -> None:
    """Terminates a started service.

    The process is terminated, then killed if it has not exited after 15
    seconds, and its stderr pipe is closed.

    Args:
        process: The process to stop; None does nothing.
    """
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)
    if process.stderr is not None:
        process.stderr.close()


def free_port() -> int:
    """Finds a free local TCP port.

    Returns:
        A port on 127.0.0.1 that was free when probed.
    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def upload_folder(
    base: str, folder: pathlib.Path, prefix: str
) -> dict[str, str]:
    """Uploads every file of `folder`.

    Args:
        base: Base URL of the object store.
        folder: Folder whose files are uploaded recursively.
        prefix: Key prefix under which the files are stored.

    Returns:
        A mapping from each file's path relative to `folder` to the entity tag
        the store returned.
    """
    return {
        path.relative_to(folder).as_posix(): mocks.put_object(
            base,
            f"{prefix}/{path.relative_to(folder).as_posix()}",
            path.read_bytes(),
        )
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def download_folder(
    base: str, prefix: str, expected: dict[str, str], destination: pathlib.Path
) -> None:
    """Fetches the folder back, refusing any file whose digest differs.

    Args:
        base: Base URL of the object store.
        prefix: Key prefix under which the files are stored.
        expected: Mapping from relative file path to the expected entity tag.
        destination: Folder to write the files to; created as needed.
    """
    for name, etag in expected.items():
        data = mocks.get_object(base, f"{prefix}/{name}", expected=etag)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def prepare(
    workdir: pathlib.Path,
    data: pathlib.Path = DATA_DIRECTORY,
    transcript: Transcript | None = None,
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
    log = transcript or Transcript()
    workdir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    good = train("good", data / "train.csv")
    bad = train("bad", data / "bad_release_train.csv")
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
            metrics["roc_auc"] > MINIMUM_ROC_AUC
            for metrics in evaluation.values()
        ),
        test_metrics=evaluation,
    )

    started = time.monotonic()
    registry = scmle.governance.registry.ModelRegistryStore(
        workdir / "registry"
    )
    records = []
    for label, result in (("v1 good", good), ("v2 bad release", bad)):
        exported = result.bundle.save(
            workdir / f"export-{label[:2]}", DEMO_SIGNING_KEY
        )
        records.append(
            registry.register(
                MODEL,
                exported,
                metrics=result.evaluation["test_metrics"],
                run_id=result.run_id,
                signing_key=DEMO_SIGNING_KEY,
            )
        )
    registry.promote(
        MODEL,
        records[0].version,
        approval=approval_for(records[0], "initial release", "evaluation.json"),
        signing_key=DEMO_SIGNING_KEY,
    )
    service = good.bundle.export_service(
        workdir / "service", signing_key=DEMO_SIGNING_KEY
    )
    golden = parity.load_golden(GOLDEN_DIRECTORY)
    golden_service = parity.reference_bundle(golden.spec).export_service(
        workdir / "golden-service", signing_key=DEMO_SIGNING_KEY
    )
    log.record(
        "register and package",
        started,
        ok=(service / "provenance.json").exists(),
        versions=[record.version for record in records],
        champion=registry.resolve(MODEL).version,
        bundle_digest=provenance.bundle_digest(service / "model"),
        files=sorted(p.name for p in service.iterdir()),
    )
    return {"service": service, "golden_service": golden_service}


def predict_checks(
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
    frame = pandas.read_csv(data / "fresh.csv").drop(columns=[TARGET])
    direct = bundle.Bundle.load(model_directory, signing_key=DEMO_SIGNING_KEY)
    rows = frame.head(60).to_dict("records")
    status, body = post_json(f"{base}/predict", {"records": rows})
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
    batch = serving.BatchServing(direct)
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


def corrupted_and_unsigned_stay_unready(
    service: pathlib.Path, workdir: pathlib.Path
) -> dict[str, Any]:
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
        port = free_port()
        process = start_service(folder, port)
        base = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + 30
            ready_status = 0
            ready: dict[str, Any] = {}
            while time.monotonic() < deadline:
                try:
                    ready_status, ready = get_json(
                        f"{base}/readyz", timeout=2.0
                    )
                except OSError:
                    time.sleep(0.2)
                    continue
                if ready.get("phase") == "failed":
                    break
                time.sleep(0.2)
            live_status = get_json(f"{base}/livez")[0]
            predict_status = post_json(
                f"{base}/predict", {"records": [{"age": 1.0}]}
            )[0]
            outcomes[label] = {
                "livez": live_status,
                "readyz": ready_status,
                "predict": predict_status,
                "reason": ready.get("reason"),
            }
        finally:
            stop_service(process)
    return outcomes


def rollback_drill(
    workdir: pathlib.Path, data: pathlib.Path = DATA_DIRECTORY
) -> dict[str, Any]:
    """Promotes the bad release, monitors it, rolls back, serves the result.

    The lifecycle manager loads registered bundles with the key from the
    environment, so the demo key is set for the duration of the drill only.

    Args:
        workdir: Working directory prepared by `prepare`.
        data: Directory holding `fresh.csv`.

    Returns:
        The findings of `drill_steps`.
    """
    previous = os.environ.get("SCMLE_SIGNING_KEY")
    os.environ["SCMLE_SIGNING_KEY"] = DEMO_SIGNING_KEY
    try:
        return drill_steps(workdir, data)
    finally:
        if previous is None:
            os.environ.pop("SCMLE_SIGNING_KEY", None)
        else:
            os.environ["SCMLE_SIGNING_KEY"] = previous


def drill_steps(workdir: pathlib.Path, data: pathlib.Path) -> dict[str, Any]:
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
    manager = lifecycle.Maintainer(registry, MODEL, str(workdir / "work"))
    fresh = pandas.read_csv(data / "fresh.csv")
    good_record = registry.versions(MODEL)[0]
    bad_record = registry.versions(MODEL)[1]
    registry.promote(
        MODEL,
        bad_record.version,
        approval=approval_for(bad_record, "bad release (drill)", "drill"),
        signing_key=DEMO_SIGNING_KEY,
    )
    champion_before = registry.resolve(MODEL).version
    features = fresh.drop(columns=[TARGET])
    report = monitor.ModelMonitor(
        registry.load(MODEL, signing_key=DEMO_SIGNING_KEY)
    ).assess(features, fresh[TARGET])
    outcome = manager.check_rollback(
        fresh,
        TARGET,
        approval=approval_for(good_record, "rollback to v1 (drill)", "drill"),
        signing_key=DEMO_SIGNING_KEY,
    )
    restored = registry.load(MODEL, signing_key=DEMO_SIGNING_KEY)
    original = bundle.Bundle.load(
        workdir / "export-v1", signing_key=DEMO_SIGNING_KEY
    )
    adapter = serving.HttpServing(restored)
    adapter.start()
    try:
        status, body = post_json(
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
        "champion_after_rollback": registry.resolve(MODEL).version,
        "serves_restored_model": status == http.HTTPStatus.OK
        and body["predictions"] == original.predict(features.head(40)).tolist(),
        "audit_has_rollback_check": "lifecycle.rollback_check" in audit,
        "bad_release_metrics_on_fresh": getattr(
            report.performance, "metrics", None
        ),
    }


def records_of(frame: pandas.DataFrame) -> list[dict[str, Any]]:
    """Converts rows to JSON-ready dictionaries with string keys.

    Args:
        frame: The data frame to convert.

    Returns:
        One dict per row.
    """
    return [
        {str(key): value for key, value in row.items()}
        for row in frame.to_dict("records")
    ]


def process_memory_mb(process: subprocess.Popen[bytes] | None) -> float | None:
    """Reads the resident memory of a child process.

    Args:
        process: The process to measure; None gives None.

    Returns:
        The resident set size in MiB rounded to one decimal, or None if no
        process was given or the platform did not report a number (read with
        `ps` on POSIX and `psutil` on Windows).
    """
    if process is None:
        return None
    if sys.platform == "win32":
        import psutil  # noqa: PLC0415 -- Windows-only dependency  # pyright: ignore[reportMissingModuleSource]  # psutil is a Windows-only dependency and ships no stub source here

        try:
            return round(
                psutil.Process(process.pid).memory_info().rss / 1048576, 1
            )
        except psutil.Error:
            return None
    result = subprocess.run(  # noqa: S603 -- fixed arguments  # nosec B603 B607
        ["ps", "-o", "rss=", "-p", str(process.pid)],  # noqa: S607 -- ps is looked up on PATH, as on every POSIX host
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return round(int(result.stdout.strip()) / 1024, 1)
    except ValueError:
        return None


def measure_latency(
    base: str, rows: list[dict[str, Any]], count: int = LATENCY_REQUESTS
) -> dict[str, float]:
    """Measures latency for sequential requests of `rows`.

    Args:
        base: Base URL of the service.
        rows: Records sent in each request.
        count: Number of requests.

    Returns:
        The request count and the p50, p95, p99 and mean latency in milliseconds
        (percentiles by index into the sorted timings).

    Raises:
        ScmleError: If a request is not answered 200.
    """
    timings = []
    for attempt in range(count):
        started = time.perf_counter()
        status = post_json(f"{base}/predict", {"records": rows})[0]
        if status != http.HTTPStatus.OK:
            raise errors.ScmleError(
                f"measurement request {attempt} failed: {status}"
            )
        timings.append((time.perf_counter() - started) * 1000)
    timings.sort()

    def percentile(share: float) -> float:
        """Looks up a latency percentile.

        Args:
            share: Fraction in [0, 1].

        Returns:
            The timing at that share of the sorted timings, in milliseconds,
            rounded to two decimals.
        """
        return round(
            timings[min(len(timings) - 1, int(share * len(timings)))], 2
        )

    return {
        "requests": float(count),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "mean_ms": round(statistics.fmean(timings), 2),
    }


def push_metrics(
    sink: str, values: dict[str, tuple[float, str]], model_version: str
) -> int:
    """Sends named values to the metrics sink.

    Args:
        sink: Base URL of the metrics sink.
        values: Mapping from metric name to its value and unit. A name that
            starts with `latency` or `memory` is sent as that kind of signal;
            any other is sent as drift.
        model_version: Version label attached to every sample.

    Returns:
        How many samples the sink accepted.
    """
    kinds = {
        "latency": observability.SignalKind.LATENCY,
        "memory": observability.SignalKind.MEMORY,
    }
    samples = [
        observability.MetricSample(
            name=name,
            kind=kinds.get(name.split("_")[0], observability.SignalKind.DRIFT),
            value=value,
            unit=unit,
            timestamp=now(),
            labels={"model": MODEL, "model_version": model_version},
        )
        for name, (value, unit) in values.items()
    ]
    return mocks.post_metrics(sink, samples)


def round_trip_object_store(
    transcript: Transcript,
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
    etags = upload_folder(store_url, service, "models/churn/v1")
    fetched = workdir / "fetched-service"
    if fetched.exists():
        shutil.rmtree(fetched)
    download_folder(store_url, "models/churn/v1", etags, fetched)
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


def check_golden_parity(
    transcript: Transcript,
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
    golden_port = free_port()
    golden_base = f"http://127.0.0.1:{golden_port}"
    golden_process = start_service(golden_service, golden_port)
    processes.append(golden_process)
    golden_ready = wait_ready(golden_base, golden_process)
    report: dict[str, Any] = {"passed": False}
    if golden_ready is not None:
        report = parity.check_remote(
            golden_base, GOLDEN_DIRECTORY, "cross_environment", 30.0
        )
    transcript.record(
        "golden-record parity",
        started,
        ok=bool(report["passed"]),
        max_probability_error=report.get("max_probability_error"),
        tolerance=report.get("tolerance"),
    )


def fail_not_ready(process: subprocess.Popen[bytes]) -> NoReturn:
    """Raises the error for a service that did not become ready.

    Args:
        process: The service process; the tail of its stderr is the message.

    Raises:
        ArtifactError: Always, carrying the last 800 characters of stderr.
    """
    raise errors.ArtifactError(
        (process.stderr.read().decode() if process.stderr else "")[-800:]
    )


def measure_and_ship_metrics(
    transcript: Transcript,
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
    frame = pandas.read_csv(data / "fresh.csv").drop(columns=[TARGET])
    single = measure_latency(base, records_of(frame.head(1)))
    hundred = measure_latency(base, records_of(frame.head(100)), 40)
    memory = process_memory_mb(process)
    accepted = push_metrics(
        metrics_url,
        {
            "latency_single_p95": (single["p95_ms"], "ms"),
            "latency_hundred_p95": (hundred["p95_ms"], "ms"),
            "memory_resident": (memory or 0.0, "MiB"),
            "drift_rollback_drill": (1.0 if rolled_back else 0.0, "bool"),
        },
        "v1",
    )
    count = get_json(f"{metrics_url}/metrics")[1]["count"]
    transcript.record(
        "measure and ship metrics",
        started,
        ok=accepted == EXPECTED_SAMPLES and count >= EXPECTED_SAMPLES,
        single_record=single,
        hundred_records=hundred,
        resident_mib=memory,
        startup_seconds=None if startup is None else round(startup, 2),
        samples_in_sink=count,
    )


def run(
    workdir: pathlib.Path,
    data: pathlib.Path = DATA_DIRECTORY,
    store_url: str | None = None,
    metrics_url: str | None = None,
) -> Transcript:
    """Runs the whole demo natively.

    Args:
        workdir: Working directory; created as needed.
        data: Directory holding the demo datasets.
        store_url: Base URL of an object store; a local mock is started if None.
        metrics_url: Base URL of a metrics sink; a local mock is started if
            None.

    Returns:
        The transcript. An exception in any step is recorded as a failed final
        step instead of being raised.

    Raises:
        ArtifactError: Raised inside the run if the served model does not become
            ready, but caught by the run's own handler and recorded in the
            transcript; it does not reach the caller.
    """
    transcript = Transcript()
    own_store = own_sink = None
    processes: list[subprocess.Popen[bytes]] = []
    try:
        if store_url is None:
            own_store = mocks.ObjectStore(workdir / "object-store")
            own_store.start()
            store_url = own_store.address
        if metrics_url is None:
            own_sink = mocks.MetricsSink(workdir / "metrics.jsonl")
            own_sink.start()
            metrics_url = own_sink.address
        paths = prepare(workdir, data, transcript)
        service, golden_service = paths["service"], paths["golden_service"]

        fetched = round_trip_object_store(
            transcript, store_url, service, workdir
        )

        started = time.monotonic()
        port = free_port()
        base = f"http://127.0.0.1:{port}"
        process = start_service(fetched, port)
        processes.append(process)
        startup = wait_ready(base, process)
        status, ready = get_json(f"{base}/readyz") if startup else (0, {})
        transcript.record(
            "serve under production rules",
            started,
            ok=startup is not None and status == http.HTTPStatus.OK,
            startup_seconds=None if startup is None else round(startup, 2),
            artifact_digest=ready.get("artifact_digest"),
            verify_note=ready.get("notes", {}).get("verifying"),
        )
        if startup is None:
            fail_not_ready(process)

        started = time.monotonic()
        checks = predict_checks(base, fetched / "model", data, workdir)
        transcript.record(
            "predict (online and batch)",
            started,
            ok=checks["online_matches_bundle"]
            and checks["batch_matches_bundle"]
            and checks["same_digest"]
            and checks["max_probability_difference"] < PROBABILITY_TOLERANCE,
            **checks,
        )

        check_golden_parity(transcript, golden_service, processes)

        started = time.monotonic()
        negatives = corrupted_and_unsigned_stay_unready(service, workdir)
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

        started = time.monotonic()
        drill = rollback_drill(workdir, data)
        transcript.record(
            "monitor and rollback drill",
            started,
            ok=drill["rolled_back"]
            and drill["champion_before_rollback"] == PROMOTED_VERSION
            and drill["champion_after_rollback"] == 1
            and drill["serves_restored_model"]
            and drill["audit_has_rollback_check"],
            **drill,
        )

        measure_and_ship_metrics(
            transcript,
            base=base,
            process=process,
            data=data,
            metrics_url=metrics_url,
            rolled_back=drill["rolled_back"],
            startup=startup,
        )
    except Exception as error:  # noqa: BLE001 -- the transcript must show why it stopped
        transcript.record(
            "demo stopped", time.monotonic(), ok=False, error=repr(error)[:500]
        )
    finally:
        for process in processes:
            stop_service(process)
        for mock in (own_store, own_sink):
            if mock is not None:
                mock.stop()
    return transcript


def verify_remote(
    api: str,
    golden_api: str,
    workdir: pathlib.Path,
    data: pathlib.Path,
    metrics_url: str,
) -> Transcript:
    """Makes the checks the compose test runner makes against containers.

    Args:
        api: Base URL of the running service.
        golden_api: Base URL of the service running the golden reference model.
        workdir: Working directory prepared by `prepare`.
        data: Directory holding `fresh.csv`.
        metrics_url: Base URL of the metrics sink.

    Returns:
        The transcript; it stops after the first step if the services do not
        become ready.
    """
    transcript = Transcript()
    install_bearer(os.environ, (api, golden_api))
    started = time.monotonic()
    ready = wait_ready(api) is not None and wait_ready(golden_api) is not None
    transcript.record("services ready", started, ok=ready)
    if not ready:
        return transcript
    started = time.monotonic()
    checks = predict_checks(api, workdir / "service" / "model", data, workdir)
    transcript.record(
        "predict (online and batch)",
        started,
        ok=checks["online_matches_bundle"]
        and checks["batch_matches_bundle"]
        and checks["same_digest"],
        **checks,
    )
    started = time.monotonic()
    report = parity.check_remote(
        golden_api, GOLDEN_DIRECTORY, "cross_environment", 30.0
    )
    transcript.record(
        "golden-record parity", started, ok=bool(report["passed"])
    )
    started = time.monotonic()
    drill = rollback_drill(workdir, data)
    transcript.record(
        "monitor and rollback drill",
        started,
        ok=drill["rolled_back"] and drill["champion_after_rollback"] == 1,
        **drill,
    )
    started = time.monotonic()
    accepted = push_metrics(
        metrics_url, {"drift_rollback_drill": (1.0, "bool")}, "v1"
    )
    transcript.record("ship metrics", started, ok=accepted == 1)
    return transcript


def main(arguments: list[str] | None = None) -> int:
    """Runs the command line (`run`, `prepare` or `verify`).

    Writes `transcript-<action>.json` into the working directory.

    Args:
        arguments: Command-line arguments; None reads `sys.argv`.

    Returns:
        0 if every recorded step held, otherwise 1.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0]  # pyright: ignore[reportOptionalMemberAccess]  # the module docstring is always present
    )
    actions = parser.add_subparsers(dest="action", required=True)
    for name in ("run", "prepare", "verify"):
        action = actions.add_parser(name)
        action.add_argument("--workdir", type=pathlib.Path, required=True)
        action.add_argument("--data", type=pathlib.Path, default=DATA_DIRECTORY)
    actions.choices["run"].add_argument("--store-url")
    actions.choices["run"].add_argument("--metrics-url")
    actions.choices["verify"].add_argument("--api-url", required=True)
    actions.choices["verify"].add_argument("--golden-api-url", required=True)
    actions.choices["verify"].add_argument("--metrics-url", required=True)
    options = parser.parse_args(arguments)
    if options.action == "prepare":
        log = Transcript()
        prepare(options.workdir, options.data, log)
        transcript = log
    elif options.action == "verify":
        transcript = verify_remote(
            options.api_url,
            options.golden_api_url,
            options.workdir,
            options.data,
            options.metrics_url,
        )
    else:
        transcript = run(
            options.workdir,
            options.data,
            options.store_url,
            options.metrics_url,
        )
    options.workdir.mkdir(parents=True, exist_ok=True)
    (options.workdir / f"transcript-{options.action}.json").write_text(
        json.dumps(transcript.to_dict(), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
        newline="\n",
    )
    print("DEMO PASSED" if transcript.passed else "DEMO FAILED", flush=True)
    return 0 if transcript.passed else 1


if __name__ == "__main__":
    sys.exit(main())
