"""Tests for scripts/recite.py, which moves doc-line citations in tests."""

import importlib.util
import pathlib
import subprocess

SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "recite.py"


def load():
    """Imports the script as a module."""
    spec = importlib.util.spec_from_file_location("recite", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lines_follow_an_insertion_and_a_reworded_line_keeps_its_place():
    module = load()
    before = ["a", "b", "c", "d"]
    after = ["a", "new", "new", "b", "C", "d"]

    found = module.align(before, after)

    # "c" was reworded in place, so line 3 maps to its new position.
    assert found == {1: 1, 2: 4, 3: 5, 4: 6}


def test_a_citation_names_the_doc_and_the_line():
    module = load()

    # Built from parts, so recite.py does not move this sample citation.
    sample = "# Enforces docs/LEAKAGE" + ".md:172 and README" + ".md:9."
    found = module.CITATION.findall(sample)

    assert found == [("docs/LEAKAGE.md", "172"), ("README.md", "9")]


def cite(number):
    """A citation line, built from parts so recite.py does not move it."""
    return "# Enforces docs/X" + f".md:{number}.\n"


def test_a_moved_citation_is_not_moved_again_onto_another():
    module = load()
    # Two lines inserted above line 2: 2 -> 4 and 4 -> 6. The moved 2 now
    # reads 4, a number the committed file also cites.
    maps = {"docs/X.md": {1: 1, 2: 4, 3: 5, 4: 6}}
    committed = ["x = 1", cite(2).rstrip("\n"), cite(4).rstrip("\n")]
    text = "x = 1\n" + cite(2) + cite(4)

    once, moved = module.recite(text, committed, maps)
    twice, again = module.recite(once, committed, maps)

    assert once == "x = 1\n" + cite(4) + cite(6)
    assert moved == 2
    assert twice == once
    assert again == 0


def test_a_line_added_since_the_commit_is_kept_and_the_rest_move():
    module = load()
    maps = {"docs/X.md": {2: 4}}
    committed = [cite(2).rstrip("\n")]
    text = cite(2) + cite(2) + "# a new test\n"

    found, moved = module.recite(text, committed, maps)

    assert found == cite(4) + cite(2) + "# a new test\n"
    assert moved == 1


def test_running_the_script_twice_changes_nothing(tmp_path, monkeypatch):
    module = load()

    def git(*arguments):
        subprocess.run(
            ["git", *arguments], cwd=tmp_path, check=True, capture_output=True
        )

    (tmp_path / "docs").mkdir()
    (tmp_path / "tests").mkdir()
    doc = tmp_path / "docs" / "X.md"
    doc.write_text("a\nb\nc\nd\n", "utf-8")
    test = tmp_path / "tests" / "test_x.py"
    test.write_text("x = 1\n" + cite(2) + cite(4), "utf-8")
    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c")
    doc.write_text("new\nnew\na\nb\nc\nd\n", "utf-8")
    monkeypatch.setattr(module, "ROOT", tmp_path)

    module.main([])
    once = test.read_text("utf-8")
    module.main([])

    assert once == "x = 1\n" + cite(4) + cite(6)
    assert test.read_text("utf-8") == once
