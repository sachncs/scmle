"""C12 evidence: extraction of documented guarantees and their test links."""

import pytest

from benchmarks.evidence import guarantees
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


def test_a_cited_test_links_only_when_the_path_exists():
    found = guarantees.extract(
        "It never fails (`tests/a.py`). It must pass (`tests/b.py`).\n",
        "docs/SAMPLE.md",
    )
    guarantees.link(found, [], {"tests/a.py"})
    assert found[0]["tests"] == ["tests/a.py"]
    assert found[1]["tests"] == []
    assert found[1]["missing"] == ["tests/b.py"]


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
    links = guarantees.mark(source, "tests/unit/test_x.py", {"docs/SAMPLE.md"})
    assert links == [
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
    guarantees.link(found, links, set())
    assert found[0]["tests"] == []
    assert found[1]["tests"] == [
        "tests/unit/test_x.py::TestTwo::test_two",
        "tests/unit/test_x.py::test_one",
    ]


def test_a_mention_without_line_or_anchor_is_not_a_link():
    source = 'def test_x():\n    open("docs/SAMPLE.md")\n'
    assert guarantees.mark(source, "tests/t.py", {"docs/SAMPLE.md"}) == []


def test_the_spec_convention_needs_one_cited_doc():
    source = (
        '"""Checks docs/SAMPLE.md item by item."""\n'
        "def test_l2_l4_something():\n    pass\n"
        "def test_l7_to_l9_more():\n    pass\n"
        "def test_other():\n    pass\n"
    )
    links = guarantees.specify(source, "tests/t.py", {"docs/SAMPLE.md"})
    assert [r["lines"] for r in links] == [[2, 4], [7, 9]]
    unnamed = source.replace("docs/SAMPLE.md", "the spec")
    assert guarantees.specify(unnamed, "tests/t.py", {"docs/SAMPLE.md"}) == []


def test_the_verdict_passes_only_when_every_guarantee_is_linked(monkeypatch):
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


def test_bad_input_is_an_input_error(tmp_path):
    with pytest.raises(errors.InputError):
        guarantees.collect(tmp_path)
    with pytest.raises(errors.InputError):
        guarantees.problems(["not", "a", "mapping"])  # type: ignore[arg-type]


def test_the_stored_result_is_current():
    assert guarantees.problems() == []
