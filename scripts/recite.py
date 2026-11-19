"""Moves test citations of doc lines after the doc is edited.

Tests cite the documented guarantee they enforce by its line:
`docs/X.md:12`, `docs/X.md:12-15` or `docs/X.md#L12-L15`, for `README.md`,
every Markdown file under `docs/` and every site source under `website/src`
(`benchmarks/evidence/guarantees.py` counts these links). Editing a doc
shifts its lines, so the citations of every line below the edit go stale.

A citation is right against the doc as it stood when it was written: its
origin. By default the origin of every committed line is HEAD, which holds
when each commit carries its docs and citations in step (the check in
`tests/docs/test_doc_citations.py` keeps it so). Each cited line is mapped
from the doc at the origin to the working copy, following unchanged text; a
block reworded in place keeps its lines' positions, and a line reworded
inside a block of another size follows its rewording (`KIN`). A citation on
a line not yet committed, or renumbered since HEAD, was written against the
working copy and stays, so running the script twice moves nothing twice.

`--history` repairs citations committed out of step: the origin is then the
commit that last wrote the line (`git blame`), followed back through every
commit that only renumbered the line's citations while editing a doc they
name, since such a move may have been wrong; a renumbering with the doc
unchanged, or from a line no move can follow, was a correction by hand and
is an origin. A citation written ahead of its commit (against doc edits
committed later) is moved wrongly by it, so check what it moves.

`--base REF` takes every citation, committed or not, as right against the
docs at REF: after a run of this script (or a commit), edit the docs again
and run it with `--base HEAD --docs` naming the docs edited since, when the
citations of those docs were in step with HEAD before the edit. `--docs`
limits any mode to the docs it names.

A cited line that was deleted, or reworded beyond `KIN`, has no target. It
is kept and named as lost, for a person to re-cite.

    python scripts/recite.py              # rewrite the stale citations
    python scripts/recite.py --check      # name them; exit 1 if any
    python scripts/recite.py --history    # repair from each line's commit
    python scripts/recite.py --base REF --docs DOC...
"""

import argparse
from collections.abc import Callable
import dataclasses
import difflib
import functools
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
# The scanned docs, as benchmarks/evidence/links.py names them (a test keeps
# the two equal).
SOURCE = r"README\.md|docs/[\w/-]+\.md|website/src/[\w/.-]+?\.(?:astro|html)"
CITATION = re.compile(
    r"(?<![\w/.-])(" + SOURCE + r")(?::|#L)(\d+(?:-L?\d+)?)(?!\d)"
)
NUMBER = re.compile(r"\d+")
# A reworded line must read more like its old self than unrelated lines
# do: KIN is the 99th percentile of the ratio between two different lines
# (over 40 characters) drawn from the same scanned doc, 0.587 over 22,586
# pairs (200 per doc, seed 0, on the docs of 2026-11-19); a test re-derives
# it.
KIN = 0.59
HEADER = re.compile(r"^([0-9a-f]{40}) (\d+) (\d+)")
UNCOMMITTED = "0" * 40

Chart = Callable[[str, str], dict[int, int] | None]
Trace = Callable[[int], tuple[str, str] | None]


@dataclasses.dataclass(frozen=True)
class Origin:
    """Where one line of a file was last written.

    Attributes:
        commit: The commit that wrote it.
        line: Its 1-based number in that commit's version of the file.
        path: The file's path in that commit.
        parent: The commit before, or empty for the file's first commit.
        before: The file's path in the commit before.
    """

    commit: str
    line: int
    path: str
    parent: str
    before: str


def blank(line: str) -> str:
    """A line with its line ending and its citations' numbers removed.

    Args:
        line: One line of a test file.

    Returns:
        The line without its numbers, so a renumbered line equals its old self.
    """
    return CITATION.sub(r"\1:", line.rstrip("\r\n"))


def kin(before: str, after: str) -> float:
    """How alike two lines read: difflib's ratio, 0 to 1.

    Args:
        before: One line.
        after: Another.

    Returns:
        The ratio, or 0 when its cheap upper bound is already below `KIN`.
    """
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    if matcher.real_quick_ratio() < KIN or matcher.quick_ratio() < KIN:
        return 0.0
    return matcher.ratio()


def follow(
    before: list[str], after: list[str], first: int, start: int
) -> dict[int, int]:
    """Maps the lines of a block replaced by one of another size.

    Each old line, in order, goes to the most alike new line after the last
    one taken, when the two are at least `KIN` alike: the line was
    reworded, not replaced.

    Args:
        before: The old block's lines.
        after: The new block's lines.
        first: 0-based index of the old block in its file.
        start: 0-based index of the new block in its file.

    Returns:
        Old line number to new line number (1-based) for the lines followed.
    """
    found: dict[int, int] = {}
    floor = 0
    for index, line in enumerate(before):
        scores = [(kin(line, item), at) for at, item in enumerate(after)]
        best = max(scores[floor:], default=(0.0, 0))
        if best[0] >= KIN:
            found[first + index + 1] = start + best[1] + 1
            floor = best[1] + 1
    return found


def align(before: list[str], after: list[str]) -> dict[int, int]:
    """Maps 1-based line numbers from before to after.

    Unchanged lines follow their text; a block replaced by one of the same
    size was reworded in place, so its lines keep their position in it; in
    a block replaced by one of another size, a line follows its rewording
    (`follow`).

    Args:
        before: The lines of the earlier version.
        after: The lines of the later version.

    Returns:
        Old line number to new line number, for every line that has one.
    """
    found: dict[int, int] = {}
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    for tag, first, last, start, end in matcher.get_opcodes():
        if tag == "equal" or (tag == "replace" and last - first == end - start):
            for offset in range(last - first):
                found[first + offset + 1] = start + offset + 1
        elif tag == "replace":
            found.update(
                follow(before[first:last], after[start:end], first, start)
            )
    return found


def pair(before: str, line: str) -> list[tuple[re.Match[str], re.Match[str]]]:
    """The citations of two versions of a line, paired in order.

    Args:
        before: The earlier version.
        line: The later version, equal to it but for citation numbers.

    Returns:
        (earlier, later) citation matches.
    """
    return list(
        zip(CITATION.finditer(before), CITATION.finditer(line), strict=True)
    )


class Atlas:
    """Versions, blame and line maps of a repository, built once each.

    Attributes:
        root: Repository root.
        files: File lines by commit and path.
        charts: Doc line maps by origin commit and doc.
        blames: Line origins by commit and path.
        works: Per test file, each working line unchanged since HEAD to its
            line there.
        history: Whether a line's origin is the commit that wrote it
            (`--history`) rather than HEAD.
        base: A commit every citation, committed or not, was right against
            (`--base`), or empty.
        docs: The only docs whose citations move; empty for all.
    """

    def __init__(
        self,
        root: pathlib.Path,
        history: bool = False,
        base: str = "",
        docs: tuple[str, ...] = (),
    ) -> None:
        """Starts with nothing read.

        Args:
            root: Repository root.
            history: Whether to trace each line to the commit that wrote it.
            base: A commit every citation was right against, or empty.
            docs: The only docs whose citations move; empty for all.
        """
        self.root = root
        self.history = history
        self.base = base
        self.docs = docs
        self.files: dict[tuple[str, str], list[str] | None] = {}
        self.charts: dict[tuple[str, str], dict[int, int] | None] = {}
        self.blames: dict[tuple[str, str], dict[int, Origin]] = {}
        self.works: dict[str, dict[int, int]] = {}

    def git(self, *arguments: str) -> str | None:
        """The output of one git command, or None when it fails."""
        done = subprocess.run(
            ["git", *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=self.root,
            check=False,
        )
        return done.stdout if done.returncode == 0 else None

    def show(self, ref: str, path: str) -> list[str] | None:
        """The lines of a file at a commit.

        Args:
            ref: A commit.
            path: The file's path relative to the root.

        Returns:
            Its lines, or None when the file did not exist there.
        """
        key = (ref, path)
        if key not in self.files:
            text = self.git("show", f"{ref}:{path}")
            self.files[key] = None if text is None else text.splitlines()
        return self.files[key]

    def chart(self, ref: str, doc: str) -> dict[int, int] | None:
        """How a doc's lines moved from a commit to the working copy.

        Args:
            ref: The origin commit.
            doc: The doc's path relative to the root.

        Returns:
            Old line to new line; None when the doc is missing at the
            commit or now (a sample name in a test, or a deleted doc, which
            no move can mend), or not one of `docs`. An unchanged doc maps
            every line to itself.
        """
        if self.docs and doc not in self.docs:
            return None
        key = (ref, doc)
        if key not in self.charts:
            before = self.show(ref, doc)
            path = self.root / doc
            if before is None or not path.is_file():
                self.charts[key] = None
            else:
                after = path.read_text("utf-8").splitlines()
                self.charts[key] = align(before, after)
        return self.charts[key]

    def blame(self, ref: str, path: str) -> dict[int, Origin]:
        """Where each committed line of a file was last written.

        Args:
            ref: A commit.
            path: The file's path in it.

        Returns:
            1-based line number to its origin, for committed lines only;
            empty for a file git does not track.
        """
        key = (ref, path)
        if key in self.blames:
            return self.blames[key]
        text = self.git("blame", "--line-porcelain", ref, "--", path)
        found: dict[int, Origin] = {}
        fields: dict[str, str] = {}
        header = None
        for line in (text or "").splitlines():
            match = HEADER.match(line)
            if match:
                header, fields = match, {}
            elif line.startswith("\t") and header is not None:
                commit = header.group(1)
                if commit != UNCOMMITTED:
                    previous = fields.get("previous", "")
                    parent = previous.partition(" ")[0]
                    before = previous.partition(" ")[2]
                    found[int(header.group(3))] = Origin(
                        commit,
                        int(header.group(2)),
                        fields.get("filename", path),
                        parent,
                        before,
                    )
                header = None
            else:
                fields[line.partition(" ")[0]] = line.partition(" ")[2]
        self.blames[key] = found
        return found

    def renumbered(
        self, origin: Origin, text: str, previous: Origin, line: str
    ) -> bool:
        """Whether a commit only moved a line's citations with its docs.

        Args:
            origin: Where the line was written.
            text: The line in that commit.
            previous: Where the line before it was written.
            line: The line before it.

        Returns:
            True when the two differ only in citation numbers, every doc
            whose citation changed was edited in that commit too, and every
            old number maps from its own origin to the working copy (one
            that does not was re-cited by a person where no move could
            follow it).
        """
        if blank(text) != blank(line) or text == line:
            return False
        for old, new in pair(line, text):
            if old.group(0) == new.group(0):
                continue
            doc = new.group(1)
            if self.show(origin.parent, doc) == self.show(origin.commit, doc):
                return False
            mapping = self.chart(previous.commit, doc) or {}
            if any(int(n) not in mapping for n in NUMBER.findall(old.group(2))):
                return False
        return True

    def trace(self, path: str, number: int) -> tuple[str, str] | None:
        """The commit a working line's citations were written against.

        Args:
            path: A test file relative to the root.
            number: 1-based line number in the working copy.

        Returns:
            The origin commit and the line as it read there; None for a
            line not yet committed, or renumbered since the last commit
            (moved already, by this script or by hand).
        """
        if self.base:
            work = (self.root / path).read_text("utf-8").splitlines()
            return self.base, work[number - 1]
        head = self.show("HEAD", path)
        if head is None:
            return None
        if path not in self.works:
            work = (self.root / path).read_text("utf-8").splitlines()
            # Paired here and not by `git blame` of the working copy, which
            # can pair a renumbered line with a sibling that now reads the
            # same.
            paired = align(
                [blank(item) for item in work], [blank(item) for item in head]
            )
            self.works[path] = {
                line: match
                for line, match in paired.items()
                if work[line - 1] == head[match - 1]
            }
        match = self.works[path].get(number)
        if match is None:
            return None
        if not self.history:
            return "HEAD", head[match - 1]
        origin = self.blame("HEAD", path).get(match)
        if origin is None:
            return None
        return self.retrace(origin, head[match - 1])

    def retrace(self, origin: Origin, text: str) -> tuple[str, str]:
        """Follows a line back through the commits that only moved it.

        Args:
            origin: Where the line was last written.
            text: The line as it read there.

        Returns:
            The commit its citations were written against, and the line as
            it read there.
        """
        while origin.parent:
            older = self.show(origin.parent, origin.before)
            newer = self.show(origin.commit, origin.path)
            if older is None or newer is None:
                break
            match = align(
                [blank(item) for item in newer], [blank(item) for item in older]
            ).get(origin.line)
            if match is None:
                break
            previous = self.blame(origin.parent, origin.before).get(match)
            if previous is None or not self.renumbered(
                origin, text, previous, older[match - 1]
            ):
                break
            origin, text = previous, older[match - 1]
        return origin.commit, text


def renumber(numbers: str, targets: list[int]) -> str:
    """A citation's numbers part with each number replaced in turn.

    Args:
        numbers: `12`, `12-15` or `12-L15`.
        targets: One new number per number in it.

    Returns:
        The same form with the new numbers.
    """
    pieces = NUMBER.split(numbers)
    return pieces[0] + "".join(
        f"{target}{piece}"
        for target, piece in zip(targets, pieces[1:], strict=True)
    )


def restate(
    line: str, before: str, ref: str, chart: Chart
) -> tuple[str, int, list[str]]:
    """Restores the citations of one line to the lines they were written for.

    Args:
        line: The working line.
        before: The line as it read at its origin.
        ref: The origin commit.
        chart: `Atlas.chart` or a stand-in.

    Returns:
        The rewritten line, how many citations moved, and the citations
        whose line has no target (kept as they were).
    """
    moved = 0
    lost: list[str] = []
    pieces: list[str] = []
    position = 0
    for old, new in pair(before, line):
        mapping = chart(ref, new.group(1))
        numbers = [int(n) for n in NUMBER.findall(old.group(2))]
        if mapping is None:
            continue
        if any(n not in mapping for n in numbers):
            lost.append(new.group(0))
            continue
        tail = renumber(old.group(2), [mapping[n] for n in numbers])
        if tail != new.group(2):
            pieces += [line[position : new.start(2)], tail]
            position = new.end(2)
            moved += 1
    return "".join(pieces) + line[position:], moved, lost


def recite(
    text: str, trace: Trace, chart: Chart
) -> tuple[str, int, list[tuple[int, str]]]:
    """Moves the stale citations of one test file.

    Args:
        text: The working test file.
        trace: Line number to (origin commit, line at origin), or None for
            a line not yet committed (`Atlas.trace` for one file).
        chart: `Atlas.chart` or a stand-in.

    Returns:
        The rewritten file, how many citations moved, and each lost
        citation with its line number.
    """
    current = text.splitlines(keepends=True)
    moved = 0
    lost: list[tuple[int, str]] = []
    for index, line in enumerate(current):
        if not CITATION.search(line):
            continue
        origin = trace(index + 1)
        if origin is None or blank(origin[1]) != blank(line):
            continue
        ref, before = origin
        current[index], count, missing = restate(line, before, ref, chart)
        moved += count
        lost += [(index + 1, citation) for citation in missing]
    return "".join(current), moved, lost


def survey(atlas: Atlas, write: bool) -> tuple[int, list[str], list[str]]:
    """Recites every test file under `tests/`.

    Args:
        atlas: The repository, with how to find each line's origin.
        write: Whether to rewrite the files.

    Returns:
        How many citations moved (or would move), the files they are in,
        and the lost citations as `path:line: citation`.
    """
    root = atlas.root
    moved = 0
    files: list[str] = []
    lost: list[str] = []
    for path in sorted((root / "tests").rglob("*.py")):
        text = path.read_text("utf-8")
        if not CITATION.search(text):
            continue
        name = path.relative_to(root).as_posix()
        rewritten, count, missing = recite(
            text, functools.partial(atlas.trace, name), atlas.chart
        )
        moved += count
        lost += [f"{name}:{number}: {citation}" for number, citation in missing]
        if rewritten != text:
            files.append(name)
            if write:
                path.write_text(rewritten, "utf-8")
    return moved, files, lost


def main(arguments: list[str] | None = None) -> int:
    """Rewrites or names the stale citations.

    Args:
        arguments: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        0, or with `--check` 1 when a citation is stale or lost.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    origin = parser.add_mutually_exclusive_group()
    origin.add_argument("--history", action="store_true")
    origin.add_argument("--base", default="", metavar="REF")
    parser.add_argument("--docs", nargs="+", default=[], metavar="DOC")
    options = parser.parse_args(arguments)
    atlas = Atlas(ROOT, options.history, options.base, tuple(options.docs))
    moved, files, lost = survey(atlas, write=not options.check)
    verb = "stale" if options.check else "moved"
    print(f"{moved} citations {verb} in {len(files)} files")
    for name in files:
        print(f"  {name}")
    for item in lost:
        print(f"lost {item}")
    return 1 if options.check and (moved or lost) else 0


if __name__ == "__main__":
    sys.exit(main())
