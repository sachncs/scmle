"""A run killed at every stage boundary resumes without reopening its exam.

A child process (`stage_child.py`) runs the standard tabular workflow and
kills itself with SIGKILL before or after each stage record is written (CON-3,
CON-4). The parent then resumes the run in this process and checks that:

- the killed run is never marked complete (`complete.json` is written last);
- a run killed before its locked test set opened resumes to the same
  selected model and test metrics as an uninterrupted run, with exactly one
  opening of the test set in the custody journal;
- a run killed after the opening is refused (`BlockedError`) and the journal
  gains no further opening;
- the seconds the killed run spent are carried, not restarted: every record
  written by the resumed run counts at least the elapsed time recorded
  before the kill;
- a finished stage whose product is kept (`products.Kept`) is not run
  again: its product is loaded and its record left as it was, and every
  other stage runs. Kept products are signed envelopes, so both processes
  share a signing key; without one nothing is unpickled and every stage
  runs again.
"""

import json
import os
import pathlib
import signal
import subprocess
import sys

import pytest
import stage_child

from scmle.assurance.custody import journal
from scmle.contracts import errors
from scmle.contracts import signing
from scmle.pipeline import archive
from scmle.pipeline import products
import scmle.pipeline.orchestrator
import scmle.pipeline.stages

CHILD = pathlib.Path(__file__).with_name("stage_child.py")
ROOT = pathlib.Path(__file__).resolve().parents[2]
PHASES = ("before", "after")
DEFAULT = scmle.pipeline.stages.default_stages()
STAGES = len(DEFAULT)
KEPT = [
    type(stage).__name__
    for stage in DEFAULT
    if stage.makes is not None and issubclass(stage.makes, products.Kept)
]
KEY = "a signing key shared by the killed and the resumed run"
# Metrics may move by this much between two runs of the same seeds: only
# floating-point order differs. The characterization goldens use the same.
TOLERANCE = 1e-6


def run_child(output, index, phase):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), environment.get("PYTHONPATH", "")]
    )
    environment["OMP_NUM_THREADS"] = "1"
    environment[signing.SIGNING_KEY_ENV] = KEY
    return subprocess.run(
        [sys.executable, str(CHILD), str(output), str(index), phase],
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def run_directory(output):
    (found,) = [p.parent for p in output.glob("*/config.json")]
    return found


def openings(directory):
    entries = journal.Journal.locate(directory).current()
    return sum(entry["kind"] == "opened" for entry in entries)


@pytest.fixture
def executed(monkeypatch):
    """The names of the stages whose `execute` runs in this process."""
    names = []
    monkeypatch.setenv(signing.SIGNING_KEY_ENV, KEY)
    for stage in DEFAULT:
        kind = type(stage)
        real = kind.execute

        def execute(self, context, real=real, name=kind.__name__):
            names.append(name)
            return real(self, context)

        monkeypatch.setattr(kind, "execute", execute)
    return names


def headline(directory):
    evaluation = json.loads((directory / "evaluation.json").read_text())
    return evaluation["selected"]["model"], evaluation["metrics"]


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    output = tmp_path_factory.mktemp("reference")
    child = run_child(output, -1, "before")
    assert child.returncode == 0, child.stderr[-2000:]
    directory = run_directory(output)
    assert (directory / archive.MARKER).is_file()
    assert len(list((directory / archive.FOLDER).glob("*.json"))) == STAGES
    return directory, headline(directory)


def cases():
    return [(i, phase) for i in range(STAGES) for phase in PHASES]


@pytest.mark.parametrize(("index", "phase"), cases(), ids=str)
def test_a_run_killed_at_a_stage_boundary_resumes_or_is_refused(
    reference, executed, tmp_path, index, phase
):
    child = run_child(tmp_path, index, phase)
    assert child.returncode == -signal.SIGKILL, child.stderr[-2000:]
    directory = run_directory(tmp_path)
    assert not (directory / archive.MARKER).exists()
    config = stage_child.spec().config_hash()
    before = archive.Directory(directory, config).read()
    assert len(before) == index + (phase == "after")
    spent = max((record["elapsed"] for record in before), default=0.0)
    finished = [record["stage"] for record in before]
    opened = openings(directory)
    assert opened in {0, 1}

    orchestrator = scmle.pipeline.orchestrator.Orchestrator.reload(
        directory, frame=stage_child.frame()
    )
    if opened:
        with pytest.raises(errors.BlockedError, match="already opened"):
            orchestrator.resume(directory)
        assert openings(directory) == 1
        assert not (directory / archive.MARKER).exists()
        return
    result = orchestrator.resume(directory)
    assert result.state == "COMPLETED"
    assert openings(directory) == 1
    assert (directory / archive.MARKER).is_file()
    restored = [name for name in KEPT if name in finished]
    assert orchestrator.pipeline.restored == restored
    assert executed == [
        type(stage).__name__
        for stage in DEFAULT
        if type(stage).__name__ not in restored
    ]
    after = archive.Directory(directory, config).read()
    assert len(after) == STAGES
    for record in after:
        if record["stage"] in restored:
            assert record in before
        else:
            assert record["elapsed"] >= spent
    model, metrics = headline(directory)
    expected, scores = reference[1]
    assert model == expected
    assert metrics.keys() == scores.keys()
    for name, score in scores.items():
        if isinstance(score, int | float):
            assert metrics[name] == pytest.approx(score, abs=TOLERANCE), name


def test_a_completed_run_is_not_resumed(reference):
    directory = reference[0]
    orchestrator = scmle.pipeline.orchestrator.Orchestrator.reload(
        directory, frame=stage_child.frame()
    )
    with pytest.raises(errors.ConfigError, match="already completed"):
        orchestrator.resume(directory)


def test_an_edited_stage_record_is_refused(tmp_path):
    child = run_child(tmp_path, 3, "before")
    assert child.returncode == -signal.SIGKILL, child.stderr[-2000:]
    directory = run_directory(tmp_path)
    first = min((directory / archive.FOLDER).glob("*.json"))
    body = json.loads(first.read_text())
    body["elapsed"] = 0.0
    first.write_text(json.dumps(body))
    orchestrator = scmle.pipeline.orchestrator.Orchestrator.reload(
        directory, frame=stage_child.frame()
    )
    with pytest.raises(errors.BlockedError, match="was edited"):
        orchestrator.resume(directory)
    assert openings(directory) == 0
