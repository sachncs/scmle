"""A run's test opening and checkpoint rejections outlive its own files.

Closes the C8 bypasses `checkpoint_resume`, `exam_resume_deleted` and
`exam_resume_edited` (benchmarks/evidence/bypass.py).
"""

import hashlib
import json
import pathlib
import shutil

import pytest
import support

import scmle.api.project
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.core import policies
from scmle.core import state
from scmle.data.custody import journal


def completed(frame, folder):
    return support.make_project(frame, "y", output_dir=folder).run()


def rehash(run):
    """Makes the declaration agree with an edited config.json."""
    config = tasks.Config.model_validate_json(
        (run / "config.json").read_text(encoding="utf-8")
    )
    path = run / "parameter_estimation_declaration.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record.pop("digest", None)
    record["project_config_hash"] = config.config_hash()
    digest = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps({**record, "digest": digest}), "utf-8")
    lock = run / "parameter_estimation_declaration.lock"
    lock.write_text(f"{digest}\n", encoding="utf-8")


def test_deleting_the_evaluation_does_not_reopen_the_test_set(
    binary_frame, tmp_path
):
    # Enforces docs/CLI.md:23.
    run = completed(binary_frame, tmp_path).run_dir
    (run / "evaluation.json").unlink()
    with pytest.raises(errors.BlockedError, match="already opened"):
        scmle.api.project.Project.resume(run, data=binary_frame)


def test_editing_the_config_does_not_score_another_model(
    binary_frame, tmp_path
):
    result = completed(binary_frame, tmp_path)
    run = result.run_dir
    record = json.loads((run / "config.json").read_text(encoding="utf-8"))
    record["models"]["exclude_models"].append(result.selected_model)
    (run / "config.json").write_text(json.dumps(record), encoding="utf-8")
    rehash(run)
    (run / "evaluation.json").unlink()
    (run / "execution_plan.json").unlink(missing_ok=True)
    with pytest.raises(errors.BlockedError, match="already opened"):
        scmle.api.project.Project.resume(run, data=binary_frame)


def test_a_copied_run_has_no_custody_record(binary_frame, tmp_path):
    run = completed(binary_frame, tmp_path).run_dir
    copy = tmp_path / "run_copy"
    shutil.copytree(run, copy)
    (copy / "evaluation.json").unlink()
    with pytest.raises(errors.BlockedError, match="no custody record"):
        scmle.api.project.Project.resume(copy, data=binary_frame)


def rejected(frame, folder):
    reject = policies.Callback(
        lambda review: policies.Approval(False, "alice", "no"),
        gates=frozenset({state.Progress.PLANNING}),
    )
    project = support.make_project(
        frame, "y", output_dir=folder, checkpoints=(reject,)
    )
    with pytest.raises(errors.BlockedError) as caught:
        project.run()
    return pathlib.Path(caught.value.details["run_dir"])


def test_a_rejected_run_does_not_resume_without_its_checkpoint(
    binary_frame, tmp_path
):
    run = rejected(binary_frame, tmp_path)
    with pytest.raises(errors.BlockedError, match="rejected by alice"):
        scmle.api.project.Project.resume(run, data=binary_frame)
    assert not (run / "trials").exists()


def test_deleting_the_events_does_not_forget_a_rejection(
    binary_frame, tmp_path
):
    run = rejected(binary_frame, tmp_path)
    (run / "events.jsonl").unlink()
    with pytest.raises(errors.BlockedError, match="rejected by alice"):
        scmle.api.project.Project.resume(run, data=binary_frame)


def test_a_rejected_run_resumes_once_a_checkpoint_approves(
    binary_frame, tmp_path
):
    run = rejected(binary_frame, tmp_path)
    approve = policies.Callback(lambda review: True)
    result = scmle.api.project.Project.resume(
        run, data=binary_frame, checkpoints=(approve,)
    )
    assert result.state == "COMPLETED"


def test_an_edited_journal_is_refused(tmp_path):
    record = journal.Journal(tmp_path / "custody" / "run_x.jsonl", "run_x")
    record.start("hash")
    record.decide(state.Progress.PLANNING.value, False, "alice", "no")
    lines = record.path.read_text(encoding="utf-8").splitlines()
    record.path.write_text(
        lines[0] + "\n" + lines[1].replace("alice", "bob") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(errors.BlockedError, match="edited"):
        record.admit()


def test_a_fresh_start_resets_a_reused_run_id(tmp_path):
    record = journal.Journal(tmp_path / "custody" / "run_x.jsonl", "run_x")
    record.start("one")
    record.witness()
    with pytest.raises(errors.BlockedError, match="already opened"):
        record.admit()
    record.start("two")
    record.admit()
    assert record.rejected() == {}
