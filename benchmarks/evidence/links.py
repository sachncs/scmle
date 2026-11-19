"""Links between documented guarantees and the tests that enforce them.

Part of the C12 evidence (`benchmarks.evidence.guarantees`, whose module
docstring states the rule): a guarantee is linked only to a test that
enforces that specific statement, named by the statement itself
(`tests/path.py::name`) or by a test function that names the statement's
lines.
"""

import ast
from collections import Counter
from pathlib import Path
import re
from typing import Any

SOURCE = r"README\.md|docs/[\w/-]+\.md|website/src/[\w/.-]+?\.(?:astro|html)"
MENTION = re.compile(
    r"(?<![\w/.-])(" + SOURCE + r")"
    r"(?:#L(\d+)(?:-L?(\d+))?|:(\d+)(?:-(\d+))?\b|#([a-z0-9][a-z0-9-]*))"
)
SPEC = re.compile(r"^test_l(\d+)((?:_(?:to_)?l\d+)*)")
DOC_NAME = re.compile(r"(?<![\w/.-])(" + SOURCE + r")")


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
        One record per mention inside a test function: `doc`, `test`
        (`path::function`), and either `lines` ([first, last]) or
        `anchor`. A mention outside every function names no test that
        enforces the statement and is left out.
    """
    hits = [m for m in MENTION.finditer(text) if m.group(1) in docs]
    if not hits:
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    found = []
    for hit in hits:
        line = text.count("\n", 0, hit.start()) + 1
        name = enclosing(tree, line)
        if name is None:
            continue
        record: dict[str, Any] = {
            "doc": hit.group(1),
            "test": f"{path}::{name}",
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


def defined(path: Path) -> set[str]:
    """The tests a Python file defines, as `name` and `Class::name`.

    Args:
        path: A test file.

    Returns:
        Names of its module-level functions and of the methods of its
        module-level classes; empty for a file that does not parse.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, ValueError):
        return set()
    names = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.update(
                f"{node.name}::{child.name}"
                for child in node.body
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
            )
    return names


def resolve(root: Path, cited: set[str]) -> set[str]:
    """The citations that name something real.

    Args:
        root: Repository root.
        cited: Citations from `extract` (`path` or `path::name`).

    Returns:
        Each `path::name` whose file defines that test, and each bare path
        that exists (a bare path is `vague`, never a link).
    """
    found = set()
    for citation in cited:
        path, _, name = citation.partition("::")
        target = root / path
        if not name:
            if target.exists():
                found.add(citation)
        elif target.is_file() and name in defined(target):
            found.add(citation)
    return found


def link(
    guarantees: list[dict[str, Any]],
    links: list[dict[str, Any]],
    exists: set[str],
) -> None:
    """Fills each guarantee's `tests`, `vague` and `missing` in place.

    Args:
        guarantees: Records from `extract`.
        links: Records from `mark` and `specify`.
        exists: Citations that name something real (from `resolve`).
    """
    single = Counter((g["doc"], g["section"]) for g in guarantees)
    for item in guarantees:
        tests = {c for c in item["cited"] if "::" in c and c in exists}
        for record in links:
            if record["doc"] != item["doc"]:
                continue
            if "lines" in record:
                first, last = record["lines"]
                if first <= item["end"] and item["line"] <= last:
                    tests.add(record["test"])
            elif (
                record["anchor"] == item["section"]
                and single[(item["doc"], item["section"])] == 1
            ):
                tests.add(record["test"])
        item["tests"] = sorted(tests)
        item["vague"] = sorted(
            c for c in item["cited"] if "::" not in c and c in exists
        )
        item["missing"] = sorted(c for c in item["cited"] if c not in exists)
