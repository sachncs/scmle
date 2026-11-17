"""Guarantees of docs/OPERATING_LOOP.md that no other test enforced."""

import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
FLOWS = ROOT / ".github" / "workflows"


def commands(flow):
    """Every `run` command of a workflow, as one string each."""
    jobs = yaml.safe_load(flow.read_text(encoding="utf-8"))["jobs"]
    return [
        " ".join(str(step["run"]).split())
        for job in jobs.values()
        for step in job["steps"]
        if "run" in step
    ]


def test_pull_requests_enforce_75_and_the_nightly_run_85_on_core():
    """Enforces docs/OPERATING_LOOP.md:34."""
    ci = yaml.safe_load((FLOWS / "ci.yml").read_text(encoding="utf-8"))
    assert "pull_request" in ci[True]
    bounded = [c for c in commands(FLOWS / "ci.yml") if "--cov=scmle" in c]
    assert bounded and all("--cov-fail-under=75" in c for c in bounded)
    uploads = [
        step["with"]["path"]
        for job in ci["jobs"].values()
        for step in job["steps"]
        if "upload-artifact" in str(step.get("uses", ""))
    ]
    assert any("coverage.xml" in path for path in uploads)
    nightly = FLOWS / "full-suite.yml"
    assert (
        "schedule" in yaml.safe_load(nightly.read_text(encoding="utf-8"))[True]
    )
    core = [c for c in commands(nightly) if "coverage report" in c]
    assert core and "--fail-under=85" in core[0]
    assert "scmle/contracts/*" in core[0] and "scmle/governance/*" in core[0]
