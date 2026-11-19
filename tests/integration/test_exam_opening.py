"""The locked test set is opened exactly once per run, resumed or not."""

import json
import pathlib

import pytest
import support

from scmle.assurance.custody import journal
from scmle.contracts import errors
from scmle.pipeline import policies


@pytest.fixture
def openings(monkeypatch):
    """Counts every release of the locked rows through the custody wrapper."""
    seen = []
    original = journal.Custodian.open

    def counted(self, token, binding=None):
        seen.append(binding)
        return original(self, token, binding)

    monkeypatch.setattr(journal.Custodian, "open", counted)
    return seen


def kinds(run_dir):
    text = (pathlib.Path(run_dir) / "events.jsonl").read_text("utf-8")
    return [json.loads(line)["kind"] for line in text.splitlines()]


def test_a_run_opens_the_test_set_once(binary_frame, tmp_path, openings):
    result = support.make_project(
        binary_frame, "y", time_budget=300, output_dir=tmp_path
    ).run()
    assert len(openings) == 1
    assert kinds(result.run_dir).count("test.opened") == 1


def test_a_resumed_run_opens_the_test_set_once(
    binary_frame, tmp_path, openings
):
    gate = {"open": False}
    reviewer = policies.Callback(lambda review: gate["open"])
    first = support.make_project(
        binary_frame,
        "y",
        time_budget=300,
        output_dir=tmp_path,
        checkpoints=(reviewer,),
    )
    with pytest.raises(errors.BlockedError) as caught:
        first.run()
    assert openings == []
    gate["open"] = True
    resumed = first.orchestrator.resume(caught.value.details["run_dir"])
    assert resumed.state == "COMPLETED"
    assert len(openings) == 1
    assert kinds(resumed.run_dir).count("test.opened") == 1
