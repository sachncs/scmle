"""Claim C10 change cost: the scripted extension exercises and their verdict."""

import sys

import pytest

from benchmarks.evidence import extensibility
from scmle.contracts import errors


def tree(root, text="alpha = 1\n"):
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "core.py").write_text(text)
    (root / "pkg" / "other.py").write_text("beta = 1\n")
    return root


@pytest.fixture
def listed(monkeypatch):
    """Lists every file of a plain folder, as `git ls-files` would."""

    def every(root):
        return sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        )

    monkeypatch.setattr(extensibility, "files", every)


def exercise(edits, probe, tests=("check_me.py",)):
    return extensibility.Exercise(
        point="toy",
        name="toy_ext",
        sources={"toy.py": "value = 2\n"},
        entries="[scmle.models]\ntoy = toy:value\n",
        edits=edits,
        probe=probe,
        tests=tests,
        reason={},
    )


# The prelude imports scmle from the copy, which a toy tree lacks.
@pytest.fixture
def bare(monkeypatch):
    monkeypatch.setattr(extensibility, "PRELUDE", "")


def test_an_edit_applies_once_and_refuses_moved_code(tmp_path):
    root = tree(tmp_path)
    extensibility.Edit("pkg/core.py", "alpha = 1", "alpha = 2").apply(root)
    assert (root / "pkg" / "core.py").read_text() == "alpha = 2\n"
    with pytest.raises(errors.InputError, match="redone"):
        extensibility.Edit("pkg/core.py", "alpha = 1", "x").apply(root)
    with pytest.raises(errors.InputError, match="missing"):
        extensibility.Edit("pkg/gone.py", "a", "b").apply(root)


def test_twice_present_text_is_refused(tmp_path):
    root = tree(tmp_path, text="a = 1\na = 1\n")
    with pytest.raises(errors.InputError, match="2 times"):
        extensibility.Edit("pkg/core.py", "a = 1", "a = 2").apply(root)


def test_the_extension_is_an_installed_distribution(tmp_path):
    home = exercise((), "").install(tmp_path)
    assert (home / "toy.py").read_text() == "value = 2\n"
    info = home / "toy_ext-0.1.dist-info"
    assert "toy = toy:value" in (info / "entry_points.txt").read_text()
    assert "Name: toy_ext" in (info / "METADATA").read_text()


CHECK = """
import pathlib
import sys

text = pathlib.Path("pkg/core.py").read_text()
sys.exit(0 if "alpha = 2" in text else 1)
"""


def test_a_needed_edit_is_counted_and_shown_needed(
    tmp_path, listed, bare, monkeypatch
):
    root = tree(tmp_path / "source")
    (root / "check_me.py").write_text("def test_ok():\n    pass\n")
    edits = (extensibility.Edit("pkg/core.py", "alpha = 1", "alpha = 2"),)
    record = extensibility.attempt(exercise(edits, CHECK), root)
    assert record["files"] == ["pkg/core.py"]
    assert not record["alone"]
    assert record["works"], record["probe"]
    assert record["needed"] == {"pkg/core.py": True}
    assert record["tests"]["passed"], record["tests"]
    assert extensibility.decide(record, 3) == ("pass", [])
    # The source tree is never changed.
    assert (root / "pkg" / "core.py").read_text() == "alpha = 1\n"


def test_an_unneeded_edit_fails_the_point(tmp_path, listed, bare):
    root = tree(tmp_path / "source")
    (root / "check_me.py").write_text("def test_ok():\n    pass\n")
    edits = (
        extensibility.Edit("pkg/core.py", "alpha = 1", "alpha = 2"),
        extensibility.Edit("pkg/other.py", "beta = 1", "beta = 2"),
    )
    record = extensibility.attempt(exercise(edits, CHECK), root)
    assert record["needed"] == {"pkg/core.py": True, "pkg/other.py": False}
    verdict, reasons = extensibility.decide(record, 3)
    assert verdict == "fail"
    assert any("pkg/other.py" in reason for reason in reasons)


def test_failing_tests_fail_the_point(tmp_path, listed, bare):
    root = tree(tmp_path / "source")
    (root / "check_me.py").write_text("def test_bad():\n    assert False\n")
    edits = (extensibility.Edit("pkg/core.py", "alpha = 1", "alpha = 2"),)
    record = extensibility.attempt(exercise(edits, CHECK), root)
    assert not record["tests"]["passed"]
    assert extensibility.decide(record, 3)[0] == "fail"


def test_no_listed_test_is_not_a_pass(tmp_path, listed, bare):
    root = tree(tmp_path / "source")
    edits = (extensibility.Edit("pkg/core.py", "alpha = 1", "alpha = 2"),)
    record = extensibility.attempt(exercise(edits, CHECK, tests=()), root)
    assert not record["tests"]["passed"]


def record(files, works=True, tests=True):
    return {
        "point": "toy",
        "files": files,
        "works": works,
        "needed": dict.fromkeys(files, True),
        "tests": {"passed": tests, "tail": ""},
    }


def test_more_files_than_the_limit_fail():
    files = ["a.py", "b.py", "c.py", "d.py"]
    assert extensibility.decide(record(files[:3]), 3) == ("pass", [])
    verdict, reasons = extensibility.decide(record(files), 3)
    assert verdict == "fail"
    assert reasons == ["4 files outside the package > 3"]


def test_an_extension_that_does_not_work_fails():
    verdict, reasons = extensibility.decide(record([], works=False), 3)
    assert verdict == "fail"
    assert "does not work" in reasons[0]


def document(verdict="pass", complete=True):
    limit = int(extensibility.acceptance.load()[extensibility.VARIABLE].value)
    return {
        "code": extensibility.digest(),
        "limit": limit,
        "complete": complete,
        "verdict": verdict,
        "points": [{"point": "metric", "files": ["m.py"], "verdict": verdict}],
    }


def test_a_current_complete_document_decides():
    assert extensibility.problems(document()) == []
    value, verdict = extensibility.summary(document("fail"))
    assert verdict == "fail"
    assert "metric: 1 files (fail)" in value


def test_stale_or_partial_documents_are_not_measured():
    stale = document()
    stale["code"] = "0" * 64
    assert extensibility.summary(stale)[1] == "not measured"
    assert extensibility.summary(document(complete=False))[1] == (
        "not measured"
    )
    moved = document()
    moved["limit"] = 99
    assert extensibility.problems(moved)


def test_every_point_of_the_criterion_has_an_exercise():
    points = {exercise.point for exercise in extensibility.EXERCISES}
    assert {"model family", "metric", "reader", "task type"} <= points
    assert any(point.startswith("data kind") for point in points)
    for item in extensibility.EXERCISES:
        assert item.tests, item.point
        assert set(item.reason) == set(item.files), item.point
        for test in item.tests:
            assert (extensibility.ROOT / test).is_file(), test


def test_every_edit_still_applies_to_the_current_code(tmp_path):
    for item in extensibility.EXERCISES:
        copy = tmp_path / item.name
        for name in item.files:
            (copy / name).parent.mkdir(parents=True, exist_ok=True)
            source = extensibility.ROOT / name
            (copy / name).write_text(source.read_text(encoding="utf-8"))
        for edit in item.edits:
            edit.apply(copy)


def test_write_refuses_a_partial_measurement(capsys):
    with pytest.raises(SystemExit):
        extensibility.main(["--write", "--point", "metric"])
    assert "--point is for --show" in capsys.readouterr().err


def test_unknown_points_are_refused(capsys):
    with pytest.raises(SystemExit):
        extensibility.main(["--show", "--point", "nothing"])
    assert "unknown points" in capsys.readouterr().err


def test_the_probe_runs_with_the_copy_first_on_the_path(tmp_path):
    repo = tmp_path / "repo"
    site = tmp_path / "site"
    repo.mkdir()
    site.mkdir()
    code = (
        "import os, sys; first = os.environ['PYTHONPATH'].split(os.pathsep)[0]"
        "; sys.exit(0 if os.path.samefile(first, os.getcwd()) else 1)"
    )
    attempt = extensibility.launch([sys.executable, "-c", code], repo, site, 60)
    assert attempt.passed, attempt.tail


def test_running_out_of_time_is_a_failure(tmp_path):
    code = "import time; time.sleep(5)"
    attempt = extensibility.launch(
        [sys.executable, "-c", code], tmp_path, tmp_path, 1
    )
    assert not attempt.passed
    assert "timed out" in attempt.tail
