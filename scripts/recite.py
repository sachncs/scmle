"""Moves test citations of doc lines after the doc is edited.

Tests cite the documented guarantee they enforce as `docs/X.md:LINE`
(benchmarks/evidence/guarantees.py counts these links). Editing a doc
shifts its lines, so the citations of every line below the edit go stale.
This maps each cited line of each changed doc from the committed version
to the working copy (lines that did not change keep their text) and
rewrites the citations in tests/. Only a citation still holding its
committed number moves, so running it twice moves nothing twice.

    python scripts/recite.py            # against HEAD
    python scripts/recite.py REF        # against another commit
"""

import difflib
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CITATION = re.compile(r"((?:docs/[A-Za-z_]+|README)\.md):(\d+)")


def lines(ref: str, doc: str) -> list[str]:
    """The lines of a doc at a commit, empty if it did not exist there."""
    done = subprocess.run(
        ["git", "show", f"{ref}:{doc}"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    return done.stdout.splitlines() if done.returncode == 0 else []


def align(before: list[str], after: list[str]) -> dict[int, int]:
    """Maps 1-based line numbers from before to after.

    Unchanged lines follow their text; a block replaced by one of the same
    size was reworded in place, so its lines keep their position in it.
    """
    found: dict[int, int] = {}
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    for tag, first, last, start, end in matcher.get_opcodes():
        if tag == "equal" or (tag == "replace" and last - first == end - start):
            for offset in range(last - first):
                found[first + offset + 1] = start + offset + 1
    return found


def blank(line: str) -> str:
    """A line with its line ending and its citations' numbers removed."""
    return CITATION.sub(r"\1:", line.rstrip("\r\n"))


def restate(
    before: str, line: str, maps: dict[str, dict[int, int]]
) -> tuple[str, int]:
    """Moves the citations of one line that still hold committed numbers.

    `before` is the committed line `line` descends from: the two differ at
    most in their citations' numbers, so their citations pair in order. A
    citation still holding its committed number is moved; one whose number
    differs was moved already or edited by hand and is kept.

    Args:
        before: The committed line.
        line: The working line, with the same citations in the same order.
        maps: Per doc, committed line number to working line number.

    Returns:
        The rewritten line and how many of its citations moved.
    """
    olds = iter(CITATION.findall(before))
    moved = 0

    def shift(match: re.Match[str]) -> str:
        nonlocal moved
        doc, number = match.group(1), int(match.group(2))
        old = int(next(olds)[1])
        if number != old:
            return match.group(0)
        target = maps.get(doc, {}).get(number, number)
        moved += target != number
        return f"{doc}:{target}"

    return CITATION.sub(shift, line), moved


def recite(
    text: str, committed: list[str], maps: dict[str, dict[int, int]]
) -> tuple[str, int]:
    """Moves the citations of a test file that use the committed numbering.

    The working file is aligned to the committed one line by line with
    citation numbers ignored, so a moved citation still pairs with the one
    it came from. On a line that pairs, a citation still holding its
    committed number is moved and any other is kept, so a second run moves
    nothing: every citation it finds is already at its target. A line added
    or edited since the commit is kept as it is, so a test file edited for
    other reasons is still kept in step on the lines it shares.

    Args:
        text: The working test file.
        committed: The test file's lines at the commit; empty when new.
        maps: Per doc, committed line number to working line number.

    Returns:
        The rewritten file and how many citations moved.
    """
    current = text.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(
        a=[blank(line) for line in committed],
        b=[blank(line) for line in current],
        autojunk=False,
    )
    moved = 0
    for tag, first, last, start, _ in matcher.get_opcodes():
        if tag != "equal":
            continue
        for offset in range(last - first):
            index = start + offset
            current[index], count = restate(
                committed[first + offset], current[index], maps
            )
            moved += count
    return "".join(current), moved


def main(arguments: list[str] | None = None) -> int:
    """Rewrites stale citations; prints how many moved.

    Args:
        arguments: An optional commit to compare against (default HEAD).

    Returns:
        0.
    """
    ref = (arguments if arguments is not None else sys.argv[1:]) or ["HEAD"]
    changed = subprocess.run(
        ["git", "diff", "--name-only", ref[0], "--", "docs", "README.md"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=True,
    ).stdout.split()
    maps = {
        doc: align(
            lines(ref[0], doc),
            (ROOT / doc).read_text("utf-8").splitlines(),
        )
        for doc in changed
        if doc.endswith(".md") and (ROOT / doc).exists()
    }
    moved = 0
    for path in sorted((ROOT / "tests").rglob("*.py")):
        text = path.read_text("utf-8")
        committed = lines(ref[0], str(path.relative_to(ROOT)))
        rewritten, count = recite(text, committed, maps)
        moved += count
        if rewritten != text:
            path.write_text(rewritten, "utf-8")
    print(f"{moved} citations moved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
