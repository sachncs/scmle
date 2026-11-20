"""Behaviour of the bad-release scenario and its process control plane."""

import pathlib

import pytest

from scmle.contracts import errors
from scmle.deploy import control
from scmle.rehearsal.faults import harness
from scmle.rehearsal.faults import release

ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def golden_paths(monkeypatch):
    """The scenario reads the golden files relative to the repository root."""
    monkeypatch.chdir(ROOT)


def test_an_empty_plane_has_nothing_deployed():
    plane = release.Processes(harness.Fleet())
    assert plane.provider == "local"
    assert (plane.images, plane.revisions, plane.digests) == ({}, {}, {})
    assert plane.primary is None
    assert plane.count == 0
    with pytest.raises(errors.ArtifactError, match="nothing is deployed"):
        plane.observe()
    with pytest.raises(errors.ArtifactError, match="nothing is deployed"):
        plane.base()


@pytest.mark.parametrize(
    "allocation",
    [{"a": 50, "b": 50}, {"a": 99}, {}],
)
def test_the_plane_moves_all_traffic_to_one_revision_only(allocation):
    plane = release.Processes(harness.Fleet())
    with pytest.raises(control.Unsupported, match="only all traffic"):
        plane.route(allocation)
    plane.route({"local-7": 100})
    assert plane.primary == "local-7"


def test_a_revision_that_was_never_started_is_not_ready():
    plane = release.Processes(harness.Fleet())
    plane.revisions["ghost"] = harness.Instance(
        None, harness.demo.unused(), pathlib.Path()
    )
    assert plane.ready("ghost") is False


def test_a_plane_deploys_and_observes_real_revisions(tmp_path):
    fleet = harness.Fleet()
    plane = release.Processes(fleet)
    try:
        folder = harness.export(tmp_path / "img", harness.specify())
        digest = plane.enroll(folder)
        assert digest.startswith("sha256:")
        assert plane.images == {digest: folder}
        first = plane.deploy(digest)
        second = plane.deploy(digest)
        assert (first, second) == ("local-1", "local-2")
        assert plane.primary == "local-1"
        assert harness.boot(plane.revisions[first]) is not None
        seen = plane.observe()
        assert seen.revision == "local-1"
        assert seen.image_digest == digest
        assert seen.ready is True
        assert seen.revisions == (("local-1", digest), ("local-2", digest))
        assert plane.base() == plane.revisions[first].base
    finally:
        assert fleet.close() == 0


def test_a_bad_release_is_detected_and_rolled_back():
    result = release.restore()
    assert result.checks, result
    assert all(result.checks.values()), result.checks
    assert result.fault == "bad_release"
    assert result.detail["processes_still_alive_after_close"] == 0
    assert all(m.lost == 0 for m in result.measurements)
    detail = result.detail
    assert detail["detected_after_labelled_cases"] % 3 == 0
    assert detail["rollback"]["done"] is True
    assert detail["rollback"]["version"] == 1
    assert detail["unready_release"]["done"] is False
    assert detail["unready_release"]["restored"] is True
    assert detail["control_plane"].startswith("Processes")
