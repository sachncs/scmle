"""C12 evidence: every documented guarantee linked to an enforcing test.

A documented guarantee is found mechanically and conservatively, so that the
count errs towards too many guarantees, never too few.

Where it is looked for: everything the claim covers, "every guarantee in
docs and site" (plan, C12): `README.md`, every Markdown file under `docs/`
(with `docs/preregistration/`), and the website source, every `.astro`
page and `.html` fragment under `website/src` (read by
`benchmarks.evidence.pages`). Fenced code blocks, HTML comments, headings
and table header rows are skipped. Each remaining paragraph or list item
is split into sentences; each table data row is one unit, because its last
cells usually name the enforcing test.

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

What counts as a link: a test that enforces that specific statement.
Fuzzy word overlap never does, and neither does a test file that merely
mentions the doc, nor a statement that names a whole test file:

1. The unit itself cites one test, `tests/path.py::name` (or
   `::Class::name`), and that test is defined in that file. A citation of
   a whole file or directory names no specific test: it is recorded under
   `vague` and does not link. A cited path or test that does not exist is
   recorded under `missing` and does not link.
2. A test function under `tests/` names the doc with a line or an anchor:
   `docs/X.md:12`, `docs/X.md:12-15`, `docs/X.md#L12`, `docs/X.md#L12-L15`
   or `docs/X.md#heading-slug` (site sources the same way, for example
   `website/src/pages/docs/cli.astro:40`). That function is linked to every
   unit whose lines meet the range. An anchor names a section, not a
   statement, so it links only a section that holds exactly one guarantee.
   A mention outside every test function (a module docstring) links
   nothing.
3. The spec-test convention: in a test file whose module docstring cites
   exactly one scanned doc, a function `test_l41_l43_...` or
   `test_l23_to_l39_...` links the units on lines 41 to 43 (or 23 to 39)
   of that doc. `tests/spec/test_product_spec.py` uses this convention for
   the design specification, which is not a scanned doc, so it links none.

Verdict: `pass` only when every guarantee has a link and there is at least
one guarantee; `fail` otherwise. The value gives linked over total.

The stored result goes stale when a scanned doc or page, the listing of
`tests/`, any test-side doc mention (form 2), a cited test file, or the
code of this module or of `pages` changes.

    python -m benchmarks.evidence.guarantees --list    # unlinked, no write
    python -m benchmarks.evidence.guarantees --write
    python -m benchmarks.evidence.guarantees --check
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

from benchmarks.evidence import links
from benchmarks.evidence import pages
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
CITATION = re.compile(r"(?<![\w/.-])tests/[\w./-]*\w(?:::\w+){0,2}")
SITE = ("*.astro", "*.html")


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

    single = len(unit["lines"]) == 1

    found = []
    begin = 0
    for match in [*BREAK.finditer(joined), None]:
        stop = len(joined) if match is None else match.start()
        sentence = joined[begin:stop].strip()
        if sentence:
            found.append(
                {
                    "start": line_at(begin),
                    "end": (
                        unit["end"] if single else line_at(max(begin, stop - 1))
                    ),
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
        text: The Markdown source, or a site page's source.
        name: The doc's path relative to the repository root; a name not
            ending in `.md` is read as a site page.

    Returns:
        One record per guarantee: `doc`, `line`, `end`, `section`, `text`,
        `words` (the guarantee words found) and `cited` (paths under
        `tests/` named in it).
    """
    found = []
    units = chunk(text) if name.endswith(".md") else pages.units(text, name)
    for unit in units:
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


def collect(root: Path) -> list[Path]:
    """The scanned docs: `README.md`, `docs/**/*.md` and the site source.

    Args:
        root: Repository root.

    Returns:
        Existing paths: `README.md` first, then every Markdown file under
        `docs/` (subdirectories included), then every `.astro` page and
        `.html` fragment under `website/src`, each group sorted.

    Raises:
        InputError: If `root` holds none of them.
    """
    found = sorted((root / "docs").rglob("*.md"))
    if (root / "README.md").is_file():
        found.insert(0, root / "README.md")
    site = root / "website" / "src"
    found += sorted(p for pattern in SITE for p in site.rglob(pattern))
    if not found:
        message = f"no README.md, docs/ or website/src under {root}"
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
    records: list[dict[str, Any]] = []
    for name in listing:
        if not name.endswith(".py"):
            continue
        text = (root / name).read_text(encoding="utf-8", errors="replace")
        if not any(key in text for key in ("README.md", "docs/", "website/")):
            continue
        records += links.mark(text, name, docs)
        records += links.specify(text, name, docs)
    return listing, records


def fingerprint(
    root: Path,
    paths: list[Path],
    listing: list[str],
    records: list[dict[str, Any]],
    exists: set[str],
) -> str:
    """Hash of everything the result depends on.

    Args:
        root: Repository root.
        paths: The scanned docs.
        listing: The `tests/` listing.
        records: The test-side link records.
        exists: The citations that resolve (so a renamed or deleted cited
            test makes the result stale).

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
            links.enclosing,
            links.mark,
            links.specify,
            links.link,
            collect,
            survey,
            links.defined,
            links.resolve,
            gather,
            tally,
        ).encode("ascii")
    )
    digest.update(Path(pages.__file__).read_bytes())
    digest.update(json.dumps(WORDS).encode("utf-8"))
    digest.update(json.dumps(sorted(exists)).encode("utf-8"))
    for path in paths:
        name = path.relative_to(root).as_posix()
        digest.update(name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    digest.update("\n".join(listing).encode("utf-8"))
    digest.update(json.dumps(records, sort_keys=True).encode("utf-8"))
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


def gather(
    root: Path,
) -> tuple[list[Path], list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Scans the docs and tests and links every guarantee.

    Args:
        root: Repository root.

    Returns:
        The scanned paths, the `tests/` listing, the test-side links and
        the guarantees found (not yet linked).
    """
    paths = collect(root)
    docs = {p.relative_to(root).as_posix() for p in paths}
    listing, records = survey(root, docs)
    guarantees: list[dict[str, Any]] = []
    for path in paths:
        name = path.relative_to(root).as_posix()
        guarantees += extract(path.read_text(encoding="utf-8"), name)
    return paths, listing, records, guarantees


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
    paths, listing, records, guarantees = gather(root)
    exists = links.resolve(root, {c for g in guarantees for c in g["cited"]})
    links.link(guarantees, records, exists)
    unlinked = Counter(g["doc"] for g in guarantees if not g["tests"])
    total = Counter(g["doc"] for g in guarantees)
    return {
        "version": 2,
        "claim": "C12",
        "definition": {
            "docs": (
                "README.md, docs/**/*.md (with docs/preregistration) and the "
                "site source website/src/**/*.astro and *.html"
            ),
            "words": list(WORDS),
            "links": [
                (
                    "the guarantee cites one test, tests/path.py::name, "
                    "defined in that file (a whole file is vague, no link)"
                ),
                (
                    "a test function names the doc with :line or #Lline; "
                    "#anchor only for a section holding one guarantee"
                ),
                (
                    "spec-test convention test_l<a>_l<b> in a file whose "
                    "docstring cites the doc"
                ),
            ],
        },
        "docs": sorted(p.relative_to(root).as_posix() for p in paths),
        "counts": tally(guarantees),
        "by_doc": {
            doc: {
                "guarantees": total[doc],
                "unlinked": unlinked[doc],
            }
            for doc in sorted(total)
        },
        "test_links": len(records),
        "guarantees": guarantees,
        "vague": sum(1 for g in guarantees if g["vague"] and not g["tests"]),
        "content_hash": fingerprint(root, paths, listing, records, exists),
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


def problems(
    document: dict[str, Any] | None = None, root: Path = ROOT
) -> list[str]:
    """Why the stored result is not for the current docs, tests and code.

    Args:
        document: The results document; read from `RESULTS_PATH` when None.
        root: Repository root whose docs and tests are compared.

    Returns:
        A message for a missing file, a hash that differs from the current
        docs, `tests/` listing, test-side links or code, or counts that do
        not match the listed guarantees; empty when current.
    """
    found = load(document)
    if found is None:
        return ["guarantees.json is missing"]
    paths, listing, records, guarantees = gather(root)
    exists = links.resolve(root, {c for g in guarantees for c in g["cited"]})
    issues = []
    current = fingerprint(root, paths, listing, records, exists)
    if found.get("content_hash") != current:
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
    """Lists unlinked guarantees, or writes or checks the stored result.

    Args:
        arguments: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        0 for `--list` and `--write`; for `--check`, 0 when the result is
        current and 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    options = parser.parse_args(arguments)
    if options.list:
        document = measure()
        for item in document["guarantees"]:
            if not item["tests"]:
                print(f"{item['doc']}:{item['line']}\t{item['text']}")
        print(json.dumps(document["counts"]))
        return 0
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
