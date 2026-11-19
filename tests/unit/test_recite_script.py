"""Tests for scripts/recite.py, which moves doc-line citations in tests."""

import difflib
import importlib.util
import itertools
import pathlib
import random
import subprocess

SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "recite.py"
ROOT = SCRIPT.parents[1]


def load():
    """Imports the script as a module."""
    spec = importlib.util.spec_from_file_location("recite", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cite(number, doc="docs/X"):
    """A citation line, built from parts so recite.py does not move it."""
    return f"# Enforces {doc}" + f".md:{number}.\n"


def git(folder, *arguments):
    """Runs git in a folder, failing the test when git fails."""
    subprocess.run(
        ["git", *arguments], cwd=folder, check=True, capture_output=True
    )


def commit(folder, message):
    """Commits everything in a folder."""
    git(folder, "add", ".")
    git(
        folder,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-q",
        "-m",
        message,
    )


def test_lines_follow_an_insertion_and_a_reworded_line_keeps_its_place():
    module = load()
    before = ["a", "b", "c", "d"]
    after = ["a", "new", "new", "b", "C", "d"]

    found = module.align(before, after)

    # "c" was reworded in place, so line 3 maps to its new position.
    assert found == {1: 1, 2: 4, 3: 5, 4: 6}


def test_a_line_reworded_in_a_block_of_another_size_follows_its_wording():
    module = load()
    line = "A blocked run exits with code 2 and leaves the test rows sealed."
    before = ["x", line, "y"]
    after = [
        "x",
        "<pre>scmle run data.csv</pre>",
        "Then: " + line.replace("sealed", "sealed and unopened"),
        "A wholly different sentence about serving models over HTTP.",
        "y",
    ]

    found = module.align(before, after)

    assert found == {1: 1, 2: 3, 3: 5}


def test_a_rewritten_line_has_no_target():
    module = load()
    before = ["x", "The screen flags a leak in every planted world.", "y"]
    after = ["x", "Install the extra first.", "Then run it.", "y"]

    assert module.align(before, after) == {1: 1, 3: 4}


def test_a_citation_names_the_doc_the_line_and_a_range():
    module = load()

    sample = (
        "# Enforces docs/LEAKAGE"
        ".md:172 and README"
        ".md:9 and website/src/learn/a"
        ".html#L4-L6."
    )
    found = module.CITATION.findall(sample)

    assert found == [
        ("docs/LEAKAGE.md", "172"),
        ("README.md", "9"),
        ("website/src/learn/a.html", "4-L6"),
    ]


def test_both_ends_of_a_range_move_and_an_unmapped_line_is_lost():
    module = load()
    maps = {"docs/X.md": {2: 4, 3: 5}}

    def chart(ref, doc):
        return maps.get(doc)

    line = "# Enforces docs/X" + ".md:2-3 and docs/X" + ".md:9.\n"
    found, moved, lost = module.restate(line, line, "HEAD", chart)

    assert found == "# Enforces docs/X" + ".md:4-5 and docs/X" + ".md:9.\n"
    assert moved == 1
    assert lost == ["docs/X" + ".md:9"]


def test_only_lines_from_the_origin_move_and_a_second_run_moves_nothing():
    module = load()
    maps = {"docs/X.md": {1: 1, 2: 4, 3: 5, 4: 6}}

    def chart(ref, doc):
        return maps[doc]

    text = "x = 1\n" + cite(2) + cite(4) + cite(2)
    # Line 4 is not committed: it was written against the working copy.
    committed = {2: ("HEAD", cite(2)), 3: ("HEAD", cite(4))}

    once, moved, lost = module.recite(text, committed.get, chart)
    after = {2: None, 3: None}
    twice, again, still = module.recite(once, after.get, chart)

    assert once == "x = 1\n" + cite(4) + cite(6) + cite(2)
    assert (moved, lost) == (2, [])
    assert (twice, again, still) == (once, 0, [])


def test_running_the_script_twice_changes_nothing(tmp_path, monkeypatch):
    module = load()
    (tmp_path / "docs").mkdir()
    (tmp_path / "tests").mkdir()
    doc = tmp_path / "docs" / "X.md"
    doc.write_text("a\nb\nc\nd\n", "utf-8")
    test = tmp_path / "tests" / "test_x.py"
    test.write_text("x = 1\n" + cite(2) + cite(4), "utf-8")
    git(tmp_path, "init", "-q")
    commit(tmp_path, "c")
    doc.write_text("new\nnew\na\nb\nc\nd\n", "utf-8")
    monkeypatch.setattr(module, "ROOT", tmp_path)

    assert module.main(["--check"]) == 1
    module.main([])
    once = test.read_text("utf-8")
    module.main([])

    assert once == "x = 1\n" + cite(4) + cite(6)
    assert test.read_text("utf-8") == once
    assert module.main(["--check"]) == 0


def test_history_undoes_a_wrong_move_committed_with_its_doc(tmp_path):
    module = load()
    (tmp_path / "docs").mkdir()
    (tmp_path / "tests").mkdir()
    doc = tmp_path / "docs" / "X.md"
    doc.write_text("a\nb\nc\nd\n", "utf-8")
    test = tmp_path / "tests" / "test_x.py"
    test.write_text(cite(3) + cite(4, "docs/Y"), "utf-8")
    (tmp_path / "docs" / "Y.md").write_text("p\nq\nr\ns\n", "utf-8")
    git(tmp_path, "init", "-q")
    commit(tmp_path, "first")
    # One line inserted above "c", but the citation was moved by two; the
    # other citation was corrected by hand with its doc unchanged.
    doc.write_text("new\na\nb\nc\nd\n", "utf-8")
    test.write_text(cite(5) + cite(2, "docs/Y"), "utf-8")
    commit(tmp_path, "second")

    head = module.survey(module.Atlas(tmp_path), write=False)
    traced = module.Atlas(tmp_path, history=True)
    moved, files, lost = module.survey(traced, write=True)

    assert head == (0, [], [])
    assert (moved, files, lost) == (1, ["tests/test_x.py"], [])
    assert test.read_text("utf-8") == cite(4) + cite(2, "docs/Y")


def test_kin_is_the_99th_percentile_of_unrelated_lines_in_the_docs():
    module = load()
    rng = random.Random(0)
    found = []
    docs = [
        *sorted((ROOT / "website" / "src").rglob("*.astro")),
        *sorted((ROOT / "website" / "src").rglob("*.html")),
        *sorted((ROOT / "docs").rglob("*.md")),
    ]
    for path in docs:
        lines = [
            line
            for line in path.read_text("utf-8").splitlines()
            if len(line.strip()) > 40
        ]
        if len(lines) < 2:
            continue
        draws = itertools.repeat(lines, 200)
        for one, other in map(rng.sample, draws, itertools.repeat(2)):
            if one != other:
                matcher = difflib.SequenceMatcher(
                    a=one, b=other, autojunk=False
                )
                found.append(matcher.ratio())
    found.sort()
    percentile = found[int(0.99 * len(found)) - 1]

    # Re-derived on the docs of today; a drift past 0.03 means KIN must be
    # re-derived and its comment updated.
    assert abs(percentile - module.KIN) < 0.03
