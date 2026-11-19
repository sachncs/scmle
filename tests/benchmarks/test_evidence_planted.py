"""C12 evidence: a false statement planted in a copy of the docs is caught.

The third clause of docs/preregistration/c12.md. The checker under test is
`benchmarks.evidence.guarantees`; the planted statements here are the
builder's development set (the registration keeps the validator's own set
sealed).
"""

import pathlib
import shutil

import pytest

from benchmarks.evidence import guarantees

ROOT = pathlib.Path(__file__).resolve().parents[2]
# One false statement per kind the registration names: a changed default, a
# reversed refusal and a wrong limit. None names a test, so none is linked.
PLANTS = (
    "The default time budget is always 5 seconds.",
    "A bundle without a signature is never refused.",
    "The service must accept requests of up to 1 TB.",
)


def copy(target):
    """A copy of the scanned docs, with the real `tests/` beside it.

    Args:
        target: An empty directory.

    Returns:
        The copy's root.
    """
    shutil.copy2(ROOT / "README.md", target / "README.md")
    shutil.copytree(ROOT / "docs", target / "docs")
    shutil.copytree(ROOT / "website" / "src", target / "website" / "src")
    (target / "tests").symlink_to(ROOT / "tests", target_is_directory=True)
    return target


def plant(path, statement):
    """Appends a statement to a doc as a new paragraph.

    Args:
        path: A Markdown file or a site page.
        statement: The sentence.

    Returns:
        The 1-based line the statement is on.
    """
    text = path.read_text(encoding="utf-8").rstrip("\n")
    body = statement if path.suffix == ".md" else f"<p>{statement}</p>"
    path.write_text(text + "\n\n" + body + "\n", encoding="utf-8")
    return text.count("\n") + 3


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = copy(tmp_path_factory.mktemp("planted"))
    return root, guarantees.measure(root)


def test_a_false_statement_in_every_doc_with_a_guarantee_is_flagged(tree):
    # Enforces docs/preregistration/c12.md:17-18.
    # Enforces docs/preregistration/c12.md:44-46.
    # Enforces docs/preregistration/c12.md:78-81.
    # Enforces docs/preregistration/c12.md:84-85.
    root, before = tree
    docs = sorted({g["doc"] for g in before["guarantees"]})
    assert len(docs) >= 10
    planted = {}
    for index, doc in enumerate(docs):
        statement = PLANTS[index % len(PLANTS)]
        planted[(doc, plant(root / doc, statement))] = statement
    after = guarantees.measure(root)
    flagged = {
        (g["doc"], g["line"]): g["text"]
        for g in after["guarantees"]
        if not g["tests"]
    }
    for place, statement in planted.items():
        assert flagged.get(place) == statement, place
    assert after["counts"]["unlinked"] == (
        before["counts"]["unlinked"] + len(planted)
    )
    assert guarantees.summary(after)[1] != "pass"
    # Once re-measured, the stored result is current: `--check`, the CI
    # step, does not fail on an unlinked guarantee.
    assert guarantees.problems(after, root) == []


def test_a_stored_result_goes_stale_when_a_linked_statement_changes(
    tmp_path,
):
    # Enforces docs/preregistration/c12.md:17-18 (a reversed statement).
    # Enforces docs/preregistration/c12.md:81-84.
    root = copy(tmp_path)
    stored = guarantees.measure(root)
    for linked in stored["guarantees"]:
        path = root / linked["doc"]
        lines = path.read_text(encoding="utf-8").split("\n")
        index = linked["line"] - 1
        if linked["tests"] and " never " in lines[index]:
            break
    else:
        pytest.fail("no linked statement says never on its first line")
    lines[index] = lines[index].replace(" never ", " always ")
    path.write_text("\n".join(lines), encoding="utf-8")
    assert guarantees.problems(stored, root) == [
        "guarantees: result is for other docs, tests or code"
    ]
