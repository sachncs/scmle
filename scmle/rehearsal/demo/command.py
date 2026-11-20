"""The local demo: train, evaluate, package, predict, monitor, roll back.

The command line (`run`, `prepare` or `verify`) and the two ways through the
demo: `run` natively, and `verify` against the services of the compose file.
"""

import argparse
import json
import os
import pathlib
import time
from typing import TYPE_CHECKING

from scmle.deploy import mocks
from scmle.deploy import parity
from scmle.rehearsal.demo import client
from scmle.rehearsal.demo import fixture
from scmle.rehearsal.demo import journal
from scmle.rehearsal.demo import stages

if TYPE_CHECKING:
    import subprocess  # nosec B404 - the services the demo starts and stops


def run(
    workdir: pathlib.Path,
    data: pathlib.Path = fixture.DATA_DIRECTORY,
    store_url: str | None = None,
    metrics_url: str | None = None,
) -> journal.Transcript:
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
    transcript = journal.Transcript()
    own_store = own_sink = None
    processes: list[subprocess.Popen[bytes]] = []
    try:
        if store_url is None:
            own_store = mocks.Depot(workdir / "object-store")
            own_store.start()
            store_url = own_store.address
        if metrics_url is None:
            own_sink = mocks.Collector(workdir / "metrics.jsonl")
            own_sink.start()
            metrics_url = own_sink.address
        paths = stages.prepare(workdir, data, transcript)
        service, golden_service = paths["service"], paths["golden_service"]

        fetched = stages.mirror(transcript, store_url, service, workdir)

        base, process, startup = stages.serve(transcript, fetched, processes)
        stages.score(transcript, base, fetched, data, workdir)
        stages.corroborate(transcript, golden_service, processes)
        stages.reject(transcript, service, workdir)
        started = time.monotonic()
        drill = stages.rollback(workdir, data)
        transcript.record(
            "monitor and rollback drill",
            started,
            ok=drill["rolled_back"]
            and drill["champion_before_rollback"] == fixture.PROMOTED_VERSION
            and drill["champion_after_rollback"] == 1
            and drill["serves_restored_model"]
            and drill["audit_has_rollback_check"],
            **drill,
        )
        stages.instrument(
            transcript,
            base=base,
            process=process,
            data=data,
            metrics_url=metrics_url,
            rolled_back=drill["rolled_back"],
            startup=startup,
        )
    except Exception as error:  # noqa: BLE001 -- the transcript must show why it stopped
        # handled: returned as failed finding/problem/refusal with error text
        transcript.record(
            "demo stopped", time.monotonic(), ok=False, error=repr(error)[:500]
        )
    finally:
        for process in processes:
            client.halt(process)
        for mock in (own_store, own_sink):
            if mock is not None:
                mock.stop()
    return transcript


def verify(
    api: str,
    golden_api: str,
    workdir: pathlib.Path,
    data: pathlib.Path,
    metrics_url: str,
) -> journal.Transcript:
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
    transcript = journal.Transcript()
    client.authenticate(os.environ, (api, golden_api))
    started = time.monotonic()
    ready = (
        client.settle(api) is not None and client.settle(golden_api) is not None
    )
    transcript.record("services ready", started, ok=ready)
    if not ready:
        return transcript
    started = time.monotonic()
    checks = stages.compare(api, workdir / "service" / "model", data, workdir)
    transcript.record(
        "predict (online and batch)",
        started,
        ok=checks["online_matches_bundle"]
        and checks["batch_matches_bundle"]
        and checks["same_digest"],
        **checks,
    )
    started = time.monotonic()
    report = parity.crosscheck(
        golden_api, fixture.GOLDEN_DIRECTORY, "cross_environment", 30.0
    )
    transcript.record(
        "golden-record parity", started, ok=bool(report["passed"])
    )
    started = time.monotonic()
    drill = stages.rollback(workdir, data)
    transcript.record(
        "monitor and rollback drill",
        started,
        ok=drill["rolled_back"] and drill["champion_after_rollback"] == 1,
        **drill,
    )
    started = time.monotonic()
    accepted = client.push(
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
        action.add_argument(
            "--data", type=pathlib.Path, default=fixture.DATA_DIRECTORY
        )
    actions.choices["run"].add_argument("--store-url")
    actions.choices["run"].add_argument("--metrics-url")
    actions.choices["verify"].add_argument("--api-url", required=True)
    actions.choices["verify"].add_argument("--golden-api-url", required=True)
    actions.choices["verify"].add_argument("--metrics-url", required=True)
    options = parser.parse_args(arguments)
    if options.action == "prepare":
        log = journal.Transcript()
        stages.prepare(options.workdir, options.data, log)
        transcript = log
    elif options.action == "verify":
        transcript = verify(
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
