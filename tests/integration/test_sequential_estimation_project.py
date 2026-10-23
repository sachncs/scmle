"""Integration tests for durable sequential estimation projects."""

import json

import pytest

from scml.api.sequential import SequentialEstimationProject
from tests.unit.test_sequential_project import make_estimand
from tests.unit.test_sequential_project import make_frame
from tests.unit.test_sequential_project import make_policy


def test_project_persists_atomic_report_and_verifies_exact_replay(
    tmp_path,
) -> None:
    """Run, persistence, load, and replay form one durable contract."""
    project = SequentialEstimationProject(
        make_frame(), make_estimand(), make_policy(), output_dir=tmp_path
    )

    result = project.run()
    envelope = json.loads(project.artifact_path.read_text(encoding="utf-8"))
    replay = project.verify_replay()

    assert project.result == result
    assert envelope["schema_version"] == 1
    assert envelope["digest"] == result.digest()
    assert replay.level == "exact"
    assert replay.reasons == []


def test_project_replay_rejects_a_different_input_identity(tmp_path) -> None:
    """Changing even one input value invalidates the persisted identity."""
    frame = make_frame()
    project = SequentialEstimationProject(
        frame, make_estimand(), make_policy(), output_dir=tmp_path
    )
    project.run()
    changed = frame.copy()
    changed.loc[0, "outcome"] += 1.0

    replay = project.verify_replay(changed)

    assert replay.level == "different"
    assert any("fingerprint" in reason for reason in replay.reasons)


def test_project_rejects_tampered_report_before_replay(tmp_path) -> None:
    """Digest verification prevents replay from consuming altered evidence."""
    project = SequentialEstimationProject(
        make_frame(), make_estimand(), make_policy(), output_dir=tmp_path
    )
    project.run()
    envelope = json.loads(project.artifact_path.read_text(encoding="utf-8"))
    envelope["report"]["rows_used"] += 1
    project.artifact_path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(ValueError, match="digest mismatch"):
        project.verify_replay()
