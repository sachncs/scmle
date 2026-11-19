"""C12 evidence: extraction of documented guarantees and their test links."""

import pytest

from benchmarks.evidence import guarantees
from benchmarks.evidence import links
from scmle.contracts import errors

SAMPLE = """# Title

The tool always writes a report. It is fast.

## Loading

* A bundle is refused unless signed.
* Nothing here is promised.

```python
never_counted = True  # a code block is not a guarantee
```

<!-- the tool must not count this comment -->

| Threat | What must hold | Test |
|---|---|---|
| Tampering | Edits are rejected | `tests/unit/test_bundle.py` |

Use `never` in code. See [What it cannot do](OTHER.md).
"""


def test_guarantee_words_are_found_outside_code_links_and_headers():
    # Enforces docs/preregistration/c12.md:14.
    found = guarantees.extract(SAMPLE, "docs/SAMPLE.md")
    assert [g["line"] for g in found] == [3, 7, 18]
    assert found[0]["text"] == "The tool always writes a report."
    assert found[1]["words"] == ["refused"]
    assert found[1]["section"] == "loading"
    assert found[2]["cited"] == ["tests/unit/test_bundle.py"]


def test_a_sentence_spanning_lines_keeps_its_first_and_last_line():
    text = "First part.\nThe gate\nblocks the run. Next one.\n"
    (found,) = guarantees.extract(text, "docs/SAMPLE.md")
    assert (found["line"], found["end"]) == (2, 3)
    assert found["text"] == "The gate blocks the run."


def test_a_cited_test_links_only_when_it_names_one_existing_test(tmp_path):
    # Enforces docs/preregistration/c12.md:62.
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "a.py").write_text(
        "def test_one():\n    pass\n\nclass TestTwo:\n"
        "    def test_two(self):\n        pass\n"
    )
    found = guarantees.extract(
        "It never fails (`tests/a.py::test_one`). "
        "It must hold (`tests/a.py::TestTwo::test_two`). "
        "It always works (`tests/a.py`). "
        "It never drifts (`tests/a.py::test_gone`). "
        "It must pass (`tests/b.py::test_one`).\n",
        "docs/SAMPLE.md",
    )
    cited = {c for g in found for c in g["cited"]}
    exists = links.resolve(tmp_path, cited)
    links.link(found, [], exists)
    assert found[0]["tests"] == ["tests/a.py::test_one"]
    assert found[1]["tests"] == ["tests/a.py::TestTwo::test_two"]
    # A whole test file names no test that enforces the statement.
    assert found[2]["tests"] == []
    assert found[2]["vague"] == ["tests/a.py"]
    assert found[3]["tests"] == []
    assert found[3]["missing"] == ["tests/a.py::test_gone"]
    assert found[4]["missing"] == ["tests/b.py::test_one"]


def test_a_test_naming_a_doc_line_or_anchor_links_that_guarantee():
    source = (
        "def test_one():\n"
        '    """docs/SAMPLE.md#L7 holds."""\n'
        "\n"
        "class TestTwo:\n"
        "    def test_two(self):\n"
        '        path = "docs/SAMPLE.md#loading"\n'
        "\n"
        "def test_three():\n"
        '    path = "docs/OTHER.md:3"\n'
    )
    records = links.mark(source, "tests/unit/test_x.py", {"docs/SAMPLE.md"})
    assert records == [
        {
            "doc": "docs/SAMPLE.md",
            "test": "tests/unit/test_x.py::test_one",
            "lines": [7, 7],
        },
        {
            "doc": "docs/SAMPLE.md",
            "test": "tests/unit/test_x.py::TestTwo::test_two",
            "anchor": "loading",
        },
    ]
    found = guarantees.extract(SAMPLE, "docs/SAMPLE.md")
    links.link(found, records, set())
    assert found[0]["tests"] == []
    # The anchor names a section with two guarantees (lines 7 and 18), not
    # one statement, so only the line mention links.
    assert found[1]["tests"] == ["tests/unit/test_x.py::test_one"]
    assert found[2]["tests"] == []


def test_a_mention_without_line_or_anchor_is_not_a_link():
    # Enforces docs/preregistration/c12.md:62.
    source = 'def test_x():\n    open("docs/SAMPLE.md")\n'
    assert links.mark(source, "tests/t.py", {"docs/SAMPLE.md"}) == []


def test_a_mention_outside_every_test_function_is_not_a_link():
    source = '"""Covers docs/SAMPLE.md:3 and docs/SAMPLE.md#loading."""\n'
    assert links.mark(source, "tests/t.py", {"docs/SAMPLE.md"}) == []


def test_an_anchor_links_only_a_section_with_one_guarantee():
    text = "## Rules\n\nIt never fails.\n\nIt always logs.\n"
    found = guarantees.extract(text, "docs/SAMPLE.md")
    record = {"doc": "docs/SAMPLE.md", "test": "t.py::test_x"}
    links.link(found, [{**record, "anchor": "rules"}], set())
    assert [g["tests"] for g in found] == [[], []]
    links.link(found, [{**record, "lines": [5, 5]}], set())
    assert [g["tests"] for g in found] == [[], ["t.py::test_x"]]


def test_the_spec_convention_needs_one_cited_doc():
    source = (
        '"""Checks docs/SAMPLE.md item by item."""\n'
        "def test_l2_l4_something():\n    pass\n"
        "def test_l7_to_l9_more():\n    pass\n"
        "def test_other():\n    pass\n"
    )
    records = links.specify(source, "tests/t.py", {"docs/SAMPLE.md"})
    assert [r["lines"] for r in records] == [[2, 4], [7, 9]]
    unnamed = source.replace("docs/SAMPLE.md", "the spec")
    assert links.specify(unnamed, "tests/t.py", {"docs/SAMPLE.md"}) == []


def test_the_verdict_passes_only_when_every_guarantee_is_linked(monkeypatch):
    # Enforces docs/preregistration/c12.md:8.
    # Enforces docs/preregistration/c12.md:36.
    monkeypatch.setattr(guarantees, "problems", lambda document: [])
    linked = {"tests": ["tests/a.py"]}
    document = {"counts": guarantees.tally([linked, linked])}
    assert guarantees.summary(document)[1] == "pass"
    document = {"counts": guarantees.tally([linked, {"tests": []}])}
    value, verdict = guarantees.summary(document)
    assert verdict == "fail"
    assert value.startswith("1/2 ")
    document = {"counts": guarantees.tally([])}
    assert guarantees.summary(document)[1] == "not measured"


def test_a_stale_document_is_not_measured():
    value, verdict = guarantees.summary({"content_hash": "0", "counts": {}})
    assert verdict == "not measured"
    assert "other docs" in value


PAGE = """---
import DocTable from "../components/DocTable.astro";
// a comment that isn't text: never counted
const rows = [
  ["Tampering", "Edits are <strong>rejected</strong>", "<code>never</code>"],
  ["Speed", "Fast", "none"],
];
const sample = `code that must not count`;
---

<Layout title="T">
  <h2 id="loading">Loading</h2>
  <p>A bundle is refused
  unless signed. It is small.</p>
  <pre><code>never_counted = True</code></pre>
  <table><tr><th>Threat</th><th>Rule</th></tr>
  <tr><td>Floods</td><td>Requests are blocked</td></tr></table>
  <DocTable head={["What"]} rows={rows} note={"The run always stops."} />
  <p>Use <code>never</code> here. See <a href="/x">what it cannot do</a>.</p>
  <style>p { color: red; }</style>
</Layout>
"""


def test_site_pages_yield_markup_rows_and_strings():
    found = guarantees.extract(PAGE, "website/src/pages/docs/x.astro")
    texts = [g["text"] for g in found]
    assert texts == [
        "Tampering | Edits are rejected | `never`",
        "A bundle is refused unless signed.",
        "Floods | Requests are blocked",
        "The run always stops.",
    ]
    assert [(g["line"], g["end"]) for g in found] == [
        (5, 5),
        (13, 14),
        (17, 17),
        (18, 18),
    ]
    assert found[1]["section"] == "loading"


def test_import_paths_and_header_arrays_are_not_text():
    page = (
        "---\n"
        'import body from "../../learn/what-it-cannot-do.html?raw";\n'
        "---\n"
        '<DocTable head={["Guarantee", "Never"]} rows={[["a", "b never"]]} />\n'
    )
    found = guarantees.extract(page, "website/src/pages/docs/x.astro")
    assert [g["text"] for g in found] == ["a | b never"]


def test_the_scan_covers_preregistration_and_the_site(tmp_path):
    # Enforces docs/preregistration/c12.md:14.
    (tmp_path / "docs" / "preregistration").mkdir(parents=True)
    (tmp_path / "docs" / "A.md").write_text("x\n")
    (tmp_path / "docs" / "preregistration" / "c1.md").write_text("x\n")
    (tmp_path / "website" / "src" / "pages").mkdir(parents=True)
    (tmp_path / "website" / "src" / "pages" / "i.astro").write_text("x\n")
    (tmp_path / "website" / "src" / "f.html").write_text("x\n")
    names = [
        p.relative_to(tmp_path).as_posix() for p in guarantees.collect(tmp_path)
    ]
    assert names == [
        "docs/A.md",
        "docs/preregistration/c1.md",
        "website/src/f.html",
        "website/src/pages/i.astro",
    ]
    source = 'def test_x():\n    "website/src/pages/i.astro:1 holds"\n'
    (record,) = links.mark(source, "t.py", {"website/src/pages/i.astro"})
    assert record["lines"] == [1, 1]


def test_bad_input_is_an_input_error(tmp_path):
    with pytest.raises(errors.InputError):
        guarantees.collect(tmp_path)
    with pytest.raises(errors.InputError):
        guarantees.problems(["not", "a", "mapping"])  # type: ignore[arg-type]


def test_the_stored_result_is_current():
    assert guarantees.problems() == []
