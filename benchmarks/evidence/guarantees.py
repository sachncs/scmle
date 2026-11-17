"""C12 evidence: every documented guarantee linked to an enforcing test.

A documented guarantee is found mechanically and conservatively, so that the
count errs towards too many guarantees, never too few.

Where it is looked for: `README.md` and every `docs/*.md`. Fenced code
blocks, HTML comments, headings and table header rows are skipped. Each
remaining paragraph or list item is split into sentences; each table data
row is one unit, because its last cells usually name the enforcing test.

What makes a unit a guarantee: it contains one of `WORDS`, matched as a
whole word without regard to case, outside inline code and outside the
text of a Markdown link (which names another page). The list is

* the binding words of RFC 2119 (`must`, and the negation `cannot`), and not
  its optional ones (`should`, `may`), which promise nothing;
* universal statements of behaviour (`always`, `never`);
* the product's refusal vocabulary, since a refusal is the behaviour most
  docs promise (`refuse`, `refusal`, `block`, `reject`, `fail(s) closed`,
  each with its inflections);
* the explicit words of promise (`guarantee`, `ensure`, `enforce`).

`does not` and `only` are left out: they mostly describe limits and scope
and would swamp the list without naming a behaviour that a test can check.
A user instruction such as "you must set X" is kept, being a statement a
test could contradict.

What counts as a link (fuzzy word overlap never does):

1. The unit itself cites a path under `tests/` that exists (a file, a
   directory, or `path::name`). A cited path that does not exist is
   recorded under `missing` and does not link.
2. A Python file under `tests/` names the doc with a line or an anchor:
   `docs/X.md:12`, `docs/X.md:12-15`, `docs/X.md#L12`, `docs/X.md#L12-L15`
   or `docs/X.md#heading-slug`. The enclosing test function (or the file,
   for a module-level mention) is linked to every unit whose lines meet the
   range, or to every unit directly under that heading.
3. The spec-test convention: in a test file whose module docstring cites
   exactly one scanned doc, a function `test_l41_l43_...` or
   `test_l23_to_l39_...` links the units on lines 41 to 43 (or 23 to 39)
   of that doc. `tests/spec/test_product_spec.py` uses this convention for
   the design specification, which is not a scanned doc, so it links none.

Verdict: `pass` only when every guarantee has a link and there is at least
one guarantee; `fail` otherwise. The value gives linked over total.

The stored result goes stale when a scanned doc, the listing of `tests/`,
any test-side doc mention (form 2) or this module's code changes.

    python -m benchmarks.evidence.guarantees --write
    python -m benchmarks.evidence.guarantees --check
"""

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

from scmle.assurance.detectors import certificates
from scmle.contracts import errors

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = Path(__file__).with_name("guarantees.json")
WORDS = (
    "always",
    "never",
    "must",
    "cannot",
    "can't",
    "guarantee",
    "guarantees",
    "guaranteed",
    "ensure",
    "ensures",
    "enforce",
    "enforces",
    "enforced",
    "refuse",
    "refuses",
    "refused",
    "refusal",
    "refusals",
    "block",
    "blocks",
    "blocked",
    "blocking",
    "reject",
    "rejects",
    "rejected",
    "fails closed",
    "fail closed",
)
KEYWORDS = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in WORDS) + r")\b", re.IGNORECASE
)
FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
ITEM = re.compile(r"^\s*([-*+]|\d+[.)])\s+")
ROW = re.compile(r"^\s*\|")
RULE = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")
CODE = re.compile(r"`[^`]*`|\[[^\]]*\]\([^)]*\)")
BREAK = re.compile(r"(?<=[.!?])[\"')\]*_]*\s+(?=[A-Z0-9`*\[(\"_])")
CITATION = re.compile(r"(?<![\w/.-])tests/[\w./-]*\w(?:::\w+)?")
MENTION = re.compile(
    r"(?<![\w/.-])(README\.md|docs/[\w-]+\.md)"
    r"(?:#L(\d+)(?:-L?(\d+))?|:(\d+)(?:-(\d+))?\b|#([a-z0-9][a-z0-9-]*))"
)
SPEC = re.compile(r"^test_l(\d+)((?:_(?:to_)?l\d+)*)")
DOC_NAME = re.compile(r"(?<![\w/.-])(README\.md|docs/[\w-]+\.md)")


def slugify(heading: str) -> str:
    """The GitHub anchor of a heading.

    Args:
        heading: Heading text without the leading hashes.

    Returns:
        Lowercase text with punctuation removed and spaces as hyphens.
    """
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def hide(text: str) -> list[str]:
    """The lines of Markdown with code fences and HTML comments blanked.

    Args:
        text: The Markdown source.

    Returns:
        One string per source line; a line inside a fenced code block or
        an HTML comment (and the fence or comment line itself) is empty.
    """
    shown = []
    fenced = False
    comment = False
    for line in text.splitlines():
        if FENCE.match(line):
            fenced = not fenced
            shown.append("")
        elif comment:
            comment = "-->" not in line
            shown.append("")
        elif "<!--" in line:
            comment = "-->" not in line.split("<!--")[-1]
            shown.append("")
        else:
            shown.append("" if fenced else line)
    return shown


def chunk(text: str) -> list[dict[str, Any]]:
    """Splits Markdown into paragraphs, list items and table rows.

    Args:
        text: The Markdown source.

    Returns:
        Units with `start` and `end` (1-based lines), `section` (the slug
        of the nearest heading above), `kind` (`prose` or `row`) and the
        `lines` they hold.
    """
    units: list[dict[str, Any]] = []
    section = ""
    lines = hide(text)
    for number, line in enumerate(lines, start=1):
        heading = HEADING.match(line)
        following = lines[number] if number < len(lines) else ""
        if heading:
            section = slugify(heading.group(2))
        elif ROW.match(line):
            if not RULE.match(line) and not RULE.match(following):
                units.append(start(number, section, "row", line))
        elif line.strip():
            previous = units[-1] if units else None
            if (
                ITEM.match(line)
                or previous is None
                or previous["kind"] == "row"
                or previous["end"] != number - 1
                or previous["section"] != section
            ):
                units.append(start(number, section, "prose", line))
            else:
                previous["lines"].append(line)
                previous["end"] = number
    return units


def start(number: int, section: str, kind: str, line: str) -> dict[str, Any]:
    """A new unit holding one line.

    Args:
        number: The 1-based line number.
        section: Slug of the nearest heading above.
        kind: `prose` or `row`.
        line: The line's text.

    Returns:
        The unit.
    """
    return {
        "start": number,
        "end": number,
        "section": section,
        "kind": kind,
        "lines": [line],
    }


def split(unit: dict[str, Any]) -> list[dict[str, Any]]:
    """Splits one unit into sentences; a table row stays whole.

    Args:
        unit: A unit from `chunk`.

    Returns:
        Sentences with `start`, `end`, `section` and `text`.
    """
    if unit["kind"] == "row":
        cells = [
            c.strip() for c in unit["lines"][0].strip().strip("|").split("|")
        ]
        return [
            {
                "start": unit["start"],
                "end": unit["end"],
                "section": unit["section"],
                "text": " | ".join(cells),
            }
        ]
    offsets: list[tuple[int, int]] = []
    pieces: list[str] = []
    position = 0
    for index, line in enumerate(unit["lines"]):
        piece = line.strip()
        offsets.append((position, unit["start"] + index))
        pieces.append(piece)
        position += len(piece) + 1
    joined = " ".join(pieces)

    def line_at(offset: int) -> int:
        return max(n for start, n in offsets if start <= offset)

    found = []
    begin = 0
    for match in [*BREAK.finditer(joined), None]:
        stop = len(joined) if match is None else match.start()
        sentence = joined[begin:stop].strip()
        if sentence:
            found.append(
                {
                    "start": line_at(begin),
                    "end": line_at(max(begin, stop - 1)),
                    "section": unit["section"],
                    "text": sentence,
                }
            )
        if match is not None:
            begin = match.end()
    return found


def extract(text: str, name: str) -> list[dict[str, Any]]:
    """The guarantees in one doc, with the test paths they cite.

    Args:
        text: The Markdown source.
        name: The doc's path relative to the repository root.

    Returns:
        One record per guarantee: `doc`, `line`, `end`, `section`, `text`,
        `words` (the guarantee words found) and `cited` (paths under
        `tests/` named in it).
    """
    found = []
    for unit in chunk(text):
        for sentence in split(unit):
            words = KEYWORDS.findall(CODE.sub(" ", sentence["text"]))
            if not words:
                continue
            found.append(
                {
                    "doc": name,
                    "line": sentence["start"],
                    "end": sentence["end"],
                    "section": sentence["section"],
                    "text": sentence["text"],
                    "words": sorted({w.lower() for w in words}),
                    "cited": sorted(set(CITATION.findall(sentence["text"]))),
                }
            )
    return found


def enclosing(tree: ast.Module, line: int) -> str | None:
    """The innermost function whose source holds a line.

    Args:
        tree: The parsed test file.
        line: 1-based line number.

    Returns:
        The function name (with its class, `Class::name`), or None for a
        line outside every function.
    """
    owners = {
        id(child): node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        for child in node.body
    }
    best: tuple[int, str] | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        last = node.end_lineno or node.lineno
        if node.lineno <= line <= last and (
            best is None or node.lineno > best[0]
        ):
            owner = owners.get(id(node))
            label = f"{owner}::{node.name}" if owner else node.name
            best = (node.lineno, label)
    return None if best is None else best[1]


def mark(text: str, path: str, docs: set[str]) -> list[dict[str, Any]]:
    """Test-side mentions of a scanned doc with a line or an anchor.

    Args:
        text: Source of one test file.
        path: Its path relative to the repository root.
        docs: The scanned doc paths.

    Returns:
        One record per mention: `doc`, `test` (`path::function`, or the
        path), and either `lines` ([first, last]) or `anchor`.
    """
    hits = [m for m in MENTION.finditer(text) if m.group(1) in docs]
    if not hits:
        return []
    try:
        tree: ast.Module | None = ast.parse(text)
    except SyntaxError:
        tree = None
    found = []
    for hit in hits:
        line = text.count("\n", 0, hit.start()) + 1
        name = None if tree is None else enclosing(tree, line)
        record: dict[str, Any] = {
            "doc": hit.group(1),
            "test": f"{path}::{name}" if name else path,
        }
        first = hit.group(2) or hit.group(4)
        if first:
            last = hit.group(3) or hit.group(5) or first
            record["lines"] = [int(first), int(last)]
        else:
            record["anchor"] = hit.group(6)
        found.append(record)
    return found


def specify(text: str, path: str, docs: set[str]) -> list[dict[str, Any]]:
    """Links from the spec-test convention (`test_l41_l43_...`).

    Args:
        text: Source of one test file.
        path: Its path relative to the repository root.
        docs: The scanned doc paths.

    Returns:
        Records like those of `mark`, with `lines`; empty unless the module
        docstring cites exactly one scanned doc.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    cited = set(DOC_NAME.findall(ast.get_docstring(tree) or "")) & docs
    if len(cited) != 1:
        return []
    doc = cited.pop()
    found = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        spec = SPEC.match(node.name)
        if spec is None:
            continue
        numbers = [int(spec.group(1))]
        numbers += [int(n) for n in re.findall(r"\d+", spec.group(2))]
        found.append(
            {
                "doc": doc,
                "test": f"{path}::{node.name}",
                "lines": [min(numbers), max(numbers)],
            }
        )
    return found


def collect(root: Path) -> list[Path]:
    """The scanned docs: `README.md` and every `docs/*.md`.

    Args:
        root: Repository root.

    Returns:
        Existing paths in sorted order.

    Raises:
        InputError: If `root` holds neither `README.md` nor `docs/`.
    """
    found = sorted((root / "docs").glob("*.md"))
    if (root / "README.md").is_file():
        found.insert(0, root / "README.md")
    if not found:
        message = f"no README.md or docs/*.md under {root}"
        raise errors.InputError(message)
    return found


def survey(
    root: Path, docs: set[str]
) -> tuple[list[str], list[dict[str, Any]]]:
    """Lists `tests/` and gathers the test-side links.

    Args:
        root: Repository root.
        docs: The scanned doc paths.

    Returns:
        The sorted listing (paths relative to the root, without caches) and
        the link records of `mark` and `specify`.
    """
    base = root / "tests"
    listing = sorted(
        p.relative_to(root).as_posix()
        for p in base.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    )
    links: list[dict[str, Any]] = []
    for name in listing:
        if not name.endswith(".py"):
            continue
        text = (root / name).read_text(encoding="utf-8", errors="replace")
        if "README.md" not in text and "docs/" not in text:
            continue
        links += mark(text, name, docs)
        links += specify(text, name, docs)
    return listing, links


def link(
    guarantees: list[dict[str, Any]],
    links: list[dict[str, Any]],
    exists: set[str],
) -> None:
    """Fills each guarantee's `tests` and `missing` in place.

    Args:
        guarantees: Records from `extract`.
        links: Records from `mark` and `specify`.
        exists: Cited paths that exist (without any `::name`).
    """
    for item in guarantees:
        tests = {c for c in item["cited"] if c.split("::")[0] in exists}
        for record in links:
            if record["doc"] != item["doc"]:
                continue
            if "lines" in record:
                first, last = record["lines"]
                if first <= item["end"] and item["line"] <= last:
                    tests.add(record["test"])
            elif record["anchor"] == item["section"]:
                tests.add(record["test"])
        item["tests"] = sorted(tests)
        item["missing"] = sorted(
            c for c in item["cited"] if c.split("::")[0] not in exists
        )


def fingerprint(
    root: Path,
    paths: list[Path],
    listing: list[str],
    links: list[dict[str, Any]],
) -> str:
    """Hash of everything the result depends on.

    Args:
        root: Repository root.
        paths: The scanned docs.
        listing: The `tests/` listing.
        links: The test-side link records.

    Returns:
        64 lowercase hexadecimal characters.
    """
    digest = hashlib.sha256()
    digest.update(
        certificates.code_hash(
            slugify,
            hide,
            chunk,
            start,
            split,
            extract,
            enclosing,
            mark,
            specify,
            link,
            collect,
            survey,
            tally,
        ).encode("ascii")
    )
    digest.update(json.dumps(WORDS).encode("utf-8"))
    for path in paths:
        name = path.relative_to(root).as_posix()
        digest.update(name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    digest.update("\n".join(listing).encode("utf-8"))
    digest.update(json.dumps(links, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def tally(guarantees: list[dict[str, Any]]) -> dict[str, int]:
    """Counts guarantees, linked and unlinked.

    Args:
        guarantees: Linked records.

    Returns:
        The three counts.
    """
    linked = sum(1 for g in guarantees if g["tests"])
    return {
        "guarantees": len(guarantees),
        "linked": linked,
        "unlinked": len(guarantees) - linked,
    }


def measure(root: Path = ROOT) -> dict[str, Any]:
    """Finds every documented guarantee and its enforcing tests.

    Args:
        root: Repository root.

    Returns:
        The document written to `RESULTS_PATH`: the definition used, the
        counts overall and by doc, every guarantee with its links, the
        content hash and the duration.
    """
    started = time.monotonic()
    paths = collect(root)
    docs = {p.relative_to(root).as_posix() for p in paths}
    listing, links = survey(root, docs)
    guarantees: list[dict[str, Any]] = []
    for path in paths:
        name = path.relative_to(root).as_posix()
        guarantees += extract(path.read_text(encoding="utf-8"), name)
    cited = {c.split("::")[0] for g in guarantees for c in g["cited"]}
    exists = {c for c in cited if (root / c).exists()}
    link(guarantees, links, exists)
    unlinked = Counter(g["doc"] for g in guarantees if not g["tests"])
    total = Counter(g["doc"] for g in guarantees)
    return {
        "version": 1,
        "claim": "C12",
        "definition": {
            "docs": "README.md and docs/*.md",
            "words": list(WORDS),
            "links": [
                "the guarantee cites an existing path under tests/",
                "a test file names the doc with :line, #Lline or #anchor",
                (
                    "spec-test convention test_l<a>_l<b> in a file whose "
                    "docstring cites the doc"
                ),
            ],
        },
        "docs": sorted(docs),
        "counts": tally(guarantees),
        "by_doc": {
            doc: {
                "guarantees": total[doc],
                "unlinked": unlinked[doc],
            }
            for doc in sorted(total)
        },
        "test_links": len(links),
        "guarantees": guarantees,
        "content_hash": fingerprint(root, paths, listing, links),
        "seconds": round(time.monotonic() - started, 1),
    }


def load(document: dict[str, Any] | None) -> dict[str, Any] | None:
    """The given document, or the stored one (None when missing).

    Args:
        document: A results document, or None to read `RESULTS_PATH`.

    Returns:
        The document, or None when it is not given and not stored.

    Raises:
        InputError: If the document is not a mapping.
    """
    if document is None:
        if not RESULTS_PATH.exists():
            return None
        document = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        message = "guarantees: the document must be a JSON object"
        raise errors.InputError(message)
    return document


def problems(document: dict[str, Any] | None = None) -> list[str]:
    """Why the stored result is not for the current docs, tests and code.

    Args:
        document: The results document; read from `RESULTS_PATH` when None.

    Returns:
        A message for a missing file, a hash that differs from the current
        docs, `tests/` listing, test-side links or code, or counts that do
        not match the listed guarantees; empty when current.
    """
    found = load(document)
    if found is None:
        return ["guarantees.json is missing"]
    paths = collect(ROOT)
    docs = {p.relative_to(ROOT).as_posix() for p in paths}
    listing, links = survey(ROOT, docs)
    issues = []
    if found.get("content_hash") != fingerprint(ROOT, paths, listing, links):
        issues.append("guarantees: result is for other docs, tests or code")
    if found.get("counts") != tally(found.get("guarantees", [])):
        issues.append("guarantees: counts do not match the listed records")
    return issues


def summary(document: dict[str, Any] | None = None) -> tuple[str, str]:
    """The scorecard value text and verdict.

    Args:
        document: The results document; read from `RESULTS_PATH` when None.

    Returns:
        The value text (linked over total) and `pass`, `fail` or
        `not measured` (missing, stale or no guarantee found).
    """
    found = load(document)
    if found is None:
        return "guarantees.json is missing", "not measured"
    issues = problems(found)
    if issues:
        return "; ".join(issues), "not measured"
    counts = found["counts"]
    if counts["guarantees"] == 0:
        return "no documented guarantee found", "not measured"
    value = (
        f"{counts['linked']}/{counts['guarantees']} documented guarantees "
        "linked to an enforcing test"
    )
    return value, "pass" if counts["unlinked"] == 0 else "fail"


def main(arguments: list[str] | None = None) -> int:
    """Writes (`--write`) or checks (`--check`) the stored result.

    Args:
        arguments: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        0 for `--write`; for `--check`, 0 when the result is current and 1
        otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    options = parser.parse_args(arguments)
    if options.write:
        document = measure()
        RESULTS_PATH.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        value, verdict = summary(document)
        print(json.dumps({"value": value, "verdict": verdict}))
        return 0
    issues = problems()
    print(json.dumps({"ok": not issues, "problems": issues}, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
