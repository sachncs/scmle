"""Guarantees of docs/RELEASING.md that no other test enforced."""

import pathlib

import yaml

WORKFLOW = (
    pathlib.Path(__file__).resolve().parents[2]
    / ".github"
    / "workflows"
    / "release.yml"
)
PUBLISH = "pypa/gh-action-pypi-publish"


def jobs():
    """The jobs of the release workflow.

    Returns:
        A mapping from job name to its definition.
    """
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]


def needs(job):
    """The jobs that must succeed before a job starts.

    Args:
        job: A job definition.

    Returns:
        The names of the jobs it needs, as a set.
    """
    value = job.get("needs", [])
    return {value} if isinstance(value, str) else set(value)


def test_publishing_waits_for_every_check_and_the_evidence():
    """Enforces docs/RELEASING.md:3.

    PyPI upload is reachable only after the tag check, the platform tests and
    the preflight have passed, and only after the full suite, the skip check,
    the benchmark regression gate and the artifact checks in its own job.

    Also enforces website/src/pages/docs/releasing.astro:9.
    """
    found = jobs()
    assert needs(found["publish"]) == {"test", "release-checks"}
    assert needs(found["test"]) == {"verify"}
    assert needs(found["release-checks"]) == {"verify"}
    preflight = " ".join(
        s.get("run", "") for s in found["release-checks"]["steps"]
    )
    assert "./scripts/release_preflight.sh" in preflight
    steps = found["publish"]["steps"]
    upload = next(
        i for i, s in enumerate(steps) if s.get("uses", "").startswith(PUBLISH)
    )
    before = " ".join(s.get("run", "") for s in steps[:upload])
    for check in (
        "pytest -q",
        "check_pytest_skips.py",
        "benchmarks.check_regression",
        "twine check",
        "verify_release_artifacts.sh",
    ):
        assert check in before, check
    assert all(s.get("continue-on-error") is not True for s in steps[:upload])
    assert "|| true" not in before
