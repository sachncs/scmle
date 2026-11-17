"""Moves test citations of doc lines after the doc is edited.

Tests cite the documented guarantee they enforce as `docs/X.md:LINE`
(benchmarks/evidence/guarantees.py counts these links). Editing a doc
shifts its lines, so the citations of every line below the edit go stale.
This maps each cited line of each changed doc from the committed version
to the working copy (lines that did not change keep their text) and
rewrites the citations in tests/.

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
    """Maps 1-based line numbers of unchanged lines from before to after."""
    found: dict[int, int] = {}
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    for tag, first, last, start, _ in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(last - first):
                found[first + offset + 1] = start + offset + 1
    return found


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
        # A file whose citations differ from the committed file's was already
        # moved or edited by hand; its numbers are not the committed ones,
        # so it is left alone (running this twice moves nothing twice).
        committed = CITATION.findall(
            "\n".join(lines(ref[0], str(path.relative_to(ROOT))))
        )
        if CITATION.findall(text) != committed:
            if any(doc in maps for doc, _ in committed):
                print(f"left alone (already edited): {path.relative_to(ROOT)}")
            continue

        def shift(match: re.Match[str]) -> str:
            nonlocal moved
            doc, number = match.group(1), int(match.group(2))
            target = maps.get(doc, {}).get(number, number)
            moved += target != number
            return f"{doc}:{target}"

        rewritten = CITATION.sub(shift, text)
        if rewritten != text:
            path.write_text(rewritten, "utf-8")
    print(f"{moved} citations moved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
