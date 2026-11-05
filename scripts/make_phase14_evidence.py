"""Generates the Phase 14 evidence by running the real code.

    uv run python scripts/make_phase14_evidence.py plan/evidence/phase-14

Writes, under the output folder: the local demo transcript, the SBOM and the
signed provenance of the packaged service, the vulnerability scan and the gate
verdict (when the scanner and its database are reachable), the base-image
digest lookup (when the registry is reachable), the golden-record parity
reports, the measured startup, latency and memory, and a determinism check of
the generated folder. `summary.json` lists what ran and what did not.

Nothing here is edited by hand. The demo signing key is a published fixture.
Nothing was built or run in a container, and nothing touches a cloud.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess  # nosec B404
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from scmle.deploy import base_image  # noqa: E402
from scmle.deploy import demo  # noqa: E402
from scmle.deploy import provenance  # noqa: E402
from scmle.deploy import vuln_gate  # noqa: E402
from scmle.inference import parity  # noqa: E402

SOURCE_DATE_EPOCH = "1767225600"  # 2026-01-01T00:00:00Z


def write_json(path: Path, payload: object) -> None:
    """Write deterministic JSON."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def run_scan(service: Path, out: Path) -> dict:
    """Run pip-audit on the service lock and apply the gate."""
    scanner = shutil.which("pip-audit") or str(
        Path(sys.executable).with_name("pip-audit")
    )
    report = out / "scan.json"
    result = subprocess.run(  # nosec B603
        [
            scanner,
            "--requirement",
            str(service / "requirements.lock"),
            "--no-deps",
            "--disable-pip",
            "--format",
            "json",
            "--output",
            str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if not report.exists():
        return {"ran": False, "reason": result.stderr[-400:]}
    try:
        verdict = vuln_gate.evaluate(
            json.loads(report.read_text(encoding="utf-8")),
            json.loads(Path("deploy/vuln-policy.json").read_text()),
            json.loads((service / "sbom.cdx.json").read_text()),
        )
    except vuln_gate.GateError as error:
        verdict = {"passed": False, "error": str(error)}
    write_json(out / "vuln-gate-verdict.json", verdict)
    return {"ran": True, "scanner_exit": result.returncode, "verdict": verdict}


def run_parity(workdir: Path, out: Path) -> dict:
    """Serve the golden service and compare at both tolerances."""
    port = demo.free_port()
    process = demo.start_service(workdir / "golden-service", port)
    base = f"http://127.0.0.1:{port}"
    try:
        if demo.wait_ready(base, process) is None:
            return {"ran": False}
        reports = {}
        for level in ("same_process", "cross_environment"):
            report = parity.check_remote(base, demo.GOLDEN_DIRECTORY, level)
            write_json(out / f"parity-{level}.json", report)
            reports[level] = report["passed"]
        return {"ran": True, "passed": reports}
    finally:
        demo.stop_service(process)


def determinism(workdir: Path, out: Path) -> dict:
    """Package one bundle twice and compare every non-model file."""
    from scmle.artifacts import bundle as bundle_module

    os.environ["SOURCE_DATE_EPOCH"] = SOURCE_DATE_EPOCH
    bundle = bundle_module.ModelBundle.load(
        workdir / "export-v1", signing_key=demo.DEMO_SIGNING_KEY
    )
    digests = []
    for name in ("first", "second"):
        folder = bundle.export_service(
            Path(tempfile.mkdtemp(prefix=name)) / "service",
            signing_key=demo.DEMO_SIGNING_KEY,
        )
        digests.append(
            {
                path.relative_to(folder).as_posix(): provenance.file_digest(
                    path
                )
                for path in sorted(folder.rglob("*"))
                if path.is_file()
                and "model" not in path.relative_to(folder).parts
            }
        )
    differing = sorted(k for k in digests[0] if digests[0][k] != digests[1][k])
    write_json(
        out / "determinism.json",
        {
            "scope": "files of the generated folder other than model/, from "
            "the same bundle packaged twice with SOURCE_DATE_EPOCH fixed. "
            "This is not image reproducibility: no image was built.",
            "files_compared": len(digests[0]),
            "differing": differing,
            "first": digests[0],
        },
    )
    return {"files_compared": len(digests[0]), "differing": differing}


def main(arguments: list[str]) -> int:
    """Generate the evidence folder."""
    out = Path(arguments[0] if arguments else "plan/evidence/phase-14")
    out.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="phase14-"))
    summary: dict = {"signing_key": "published demonstration fixture"}
    transcript = demo.run(workdir)
    write_json(out / "demo-transcript.json", transcript.to_dict())
    summary["demo_passed"] = transcript.passed
    summary["demo_steps"] = {s.name: s.ok for s in transcript.steps}
    service = workdir / "service"
    shutil.copy(service / "sbom.cdx.json", out / "sbom.cdx.json")
    shutil.copy(service / "provenance.json", out / "provenance.json")
    shutil.copy(service / "requirements.lock", out / "requirements.lock")
    summary["scan"] = run_scan(service, out)
    try:
        lock = base_image.resolve("python:3.12-slim")
        write_json(out / "base-image-lookup.json", lock)
        summary["base_image_lookup"] = {"ran": True, "digest": lock["digest"]}
    except Exception as error:  # network or registry unavailable
        summary["base_image_lookup"] = {
            "ran": False,
            "reason": repr(error)[:200],
        }
    summary["parity"] = run_parity(workdir, out)
    summary["determinism"] = determinism(workdir, out)
    summary["not_run"] = [
        "docker build of the service image or the local image",
        "docker compose up (daemon not running)",
        "image attestation of a real build (needs an image digest)",
        "arm64 build",
        "two-build image digest comparison",
        "any AWS or GCP step",
    ]
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0 if summary["demo_passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
