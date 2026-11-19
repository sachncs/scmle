"""Fails at write time on the owner rules that review should never read for.

Each rule is a function from a syntax node to a message (or `None`), and
`RULES` names the folders it reads:

- `compound`: a multi-word name (snake_case or CamelCase) for a function,
  class or class attribute, by `structure.compound`, less `structure.EXEMPT`.
- `underscore`: a leading-underscore name, defined or referenced.
- `alias`: an `import ... as` binding.
- `swallow`: a bare `except:` or one catching `Exception` or
  `BaseException`, unless its body re-raises (a bare `raise` or a typed
  error) or its `except` line carries a `# handled: <reason>` comment.
- `stub`: a function whose body is only `pass`, `...`, a docstring, a
  `NotImplementedError` or a placeholder return (`None`, `0`, `""`, an empty
  literal), and every TODO, FIXME or XXX comment. Abstract methods,
  overloads, `Protocol` members and a definition whose signature carries
  a `# handled: <reason>` comment (a no-op default a subclass overrides)
  are exempt.
- `document`: a public module-level or class-level definition with no
  docstring, or one in Sphinx or NumPy style rather than Google style.
- `echo`: a `print()` call in library code (not `scmle/cli` or
  `scmle/rehearsal/demo`, whose output is printing).

A rule in `STRICT` holds over the whole tree: the tree has no violation of
it. The others carry debt the renaming phases pay off, so they ratchet: a
changed file (against `--base`, default `HEAD`, plus untracked files) may
not hold a violation that the base versions of the changed files do not
already hold. Comparing against all changed files at once lets a definition
move between files without counting as new, so an old multi-word name fails
only once it is renamed. The tree-wide multi-word counts are also held to
`structure.json` (`compound_functions`, `compound_classes`,
`compound_attributes`), the ceiling `structure.py --write` lowers.

    uv run python scripts/rules.py                  # whole tree
    uv run python scripts/rules.py scmle/a.py ...   # only these files
    uv run python scripts/rules.py --base origin/master

Pre-commit runs it on the staged Python files; the conventions test runs it
on the whole tree.
"""

import argparse
import ast
import collections
from collections.abc import Callable
from collections.abc import Iterable
import importlib.util
import json
import pathlib
import re
import subprocess  # nosec B404
import sys
import types
from typing import NamedTuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
FOLDERS = ("scmle", "tests", "benchmarks", "scripts", "examples")
LIBRARY = ("scmle",)
PRINTERS = ("scmle/cli/", "scmle/rehearsal/demo/")
BROAD = frozenset({"Exception", "BaseException"})
MARKER = "# handled:"
COMMENT = re.compile(r"#.*\b(TODO|FIXME|XXX)\b")
FOREIGN = re.compile(
    r"^\s*(:param\b|:returns?:|:raises?\b|-{3,}\s*$)", re.MULTILINE
)
EXEMPT = frozenset({"abstractmethod", "overload"})
DUNDER = re.compile(r"^__\w+__$")
PLACEHOLDERS = frozenset(
    {
        "pass",
        "...",
        "raise NotImplementedError",
        "return",
        "return None",
        "return 0",
        "return ''",
        "return []",
        "return {}",
        "return ()",
    }
)


class Finding(NamedTuple):
    """One rule violation.

    Attributes:
        path: The file, relative to the repository root.
        line: The line the violation starts on.
        rule: The rule's name, a key of `RULES`.
        text: What is wrong; equal texts in one file are the same violation.
    """

    path: str
    line: int
    rule: str
    text: str

    def __str__(self) -> str:
        """The finding as `path:line rule: text`."""
        return f"{self.path}:{self.line} {self.rule}: {self.text}"


class Source(NamedTuple):
    """A parsed module and what the rules need around its nodes.

    Attributes:
        lines: The module's source lines.
        parents: Each node's enclosing node.
    """

    lines: list[str]
    parents: dict[ast.AST, ast.AST]


def sibling(name: str) -> types.ModuleType:
    """Loads another script of this folder, which is not a package.

    Args:
        name: The script's file stem.

    Returns:
        The loaded module.

    Raises:
        ImportError: The script cannot be loaded.
    """
    spec = importlib.util.spec_from_file_location(
        name, pathlib.Path(__file__).with_name(f"{name}.py")
    )
    if spec is None or spec.loader is None:
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STRUCTURE = sibling("structure")


def private(name: str) -> bool:
    """Whether a name starts with an underscore and is not a dunder."""
    return name.startswith("_") and not DUNDER.match(name)


def decorated(node: ast.AST) -> set[str]:
    """The last dotted part of each decorator a definition carries."""
    found = set()
    for decorator in getattr(node, "decorator_list", []):
        target = (
            decorator.func if isinstance(decorator, ast.Call) else decorator
        )
        found.add(ast.unparse(target).rpartition(".")[2])
    return found


def protocol(node: ast.AST | None) -> bool:
    """Whether a node is a class that derives from `Protocol` directly."""
    return isinstance(node, ast.ClassDef) and any(
        ast.unparse(base).rpartition(".")[2].partition("[")[0] == "Protocol"
        for base in node.bases
    )


def compound(node: ast.AST, source: Source) -> str | None:
    """Names a multi-word function, class or class attribute a node defines.

    Args:
        node: Any node; only definitions and `self` attribute stores count.
        source: The module the node is in.

    Returns:
        The message, or `None` when the node defines no multi-word name.
    """
    name = ""
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        name = node.name
    elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
        method = source.parents.get(node)
        while method is not None and not isinstance(method, ast.FunctionDef):
            method = source.parents.get(method)
        own = ""
        if isinstance(method, ast.FunctionDef) and method.args.args:
            own = method.args.args[0].arg
        if isinstance(node.value, ast.Name) and node.value.id == own:
            name = node.attr
    elif (
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Store)
        and isinstance(source.parents.get(source.parents[node]), ast.ClassDef)
    ):
        name = node.id
    if name and STRUCTURE.counted(name):
        return f"multi-word name {name}"
    return None


def underscore(node: ast.AST, source: Source) -> str | None:
    """Names a leading-underscore identifier a node defines or references.

    Args:
        node: Any node.
        source: The module the node is in (unused; every rule takes it).

    Returns:
        The message, or `None`.
    """
    del source
    names: list[str] = []
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        names = [node.name]
    elif isinstance(node, ast.Name):
        names = [node.id]
    elif isinstance(node, ast.Attribute):
        names = [node.attr]
    elif isinstance(node, ast.arg | ast.keyword):
        names = [node.arg or ""]
    elif isinstance(node, ast.alias):
        names = [node.name.rpartition(".")[2], node.asname or ""]
    elif isinstance(node, ast.Global | ast.Nonlocal):
        names = node.names
    found = [name for name in names if private(name)]
    return f"leading-underscore name {found[0]}" if found else None


def alias(node: ast.AST, source: Source) -> str | None:
    """Names an `import ... as` binding.

    Args:
        node: Any node.
        source: The module the node is in (unused; every rule takes it).

    Returns:
        The message, or `None`.
    """
    del source
    if isinstance(node, ast.alias) and node.asname:
        return f"import alias {node.name} as {node.asname}"
    return None


def reraises(handler: ast.ExceptHandler) -> bool:
    """Whether a handler's body raises again: bare, or a typed error."""
    for node in ast.walk(handler):
        if not isinstance(node, ast.Raise):
            continue
        if node.exc is None:
            return True
        target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
        if ast.unparse(target).rpartition(".")[2] not in BROAD:
            return True
    return False


def swallow(node: ast.AST, source: Source) -> str | None:
    """Names a broad handler that neither re-raises nor says why it is safe.

    Args:
        node: Any node.
        source: The module the node is in.

    Returns:
        The message, or `None`.
    """
    if not isinstance(node, ast.ExceptHandler):
        return None
    if node.type is None:
        caught = "bare except"
    else:
        kinds = (
            node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
        )
        broad = [
            ast.unparse(kind)
            for kind in kinds
            if ast.unparse(kind).rpartition(".")[2] in BROAD
        ]
        if not broad:
            return None
        caught = f"except {broad[0]}"
    header = source.lines[
        node.lineno - 1 : max(node.lineno, node.body[0].lineno - 1)
    ]
    if reraises(node) or any(MARKER in line for line in header):
        return None
    return f"{caught} without a typed re-raise or {MARKER} reason"


def placeholder(statement: ast.stmt) -> bool:
    """Whether a statement is a stub body: pass, `...`, or a fake result."""
    text = ast.unparse(statement)
    return text in PLACEHOLDERS or text.startswith("raise NotImplementedError(")


def stub(node: ast.AST, source: Source) -> str | None:
    """Names a function whose body only stands in for an implementation.

    Args:
        node: Any node.
        source: The module the node is in.

    Returns:
        The message, or `None`.
    """
    if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return None
    if decorated(node) & EXEMPT or protocol(source.parents.get(node)):
        return None
    header = source.lines[
        node.lineno - 1 : max(node.lineno, node.body[0].lineno - 1)
    ]
    if any(MARKER in line for line in header):
        return None
    body = node.body[1:] if ast.get_docstring(node) is not None else node.body
    if body and not (len(body) == 1 and placeholder(body[0])):
        return None
    return f"stub body in {node.name}"


def document(node: ast.AST, source: Source) -> str | None:
    """Names a public definition with no Google-style docstring.

    Only module-level definitions and those directly in a public class
    count; nested helpers and overloads do not.

    Args:
        node: Any node.
        source: The module the node is in.

    Returns:
        The message, or `None`.
    """
    if not isinstance(
        node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ):
        return None
    owner = source.parents.get(node)
    public = not node.name.startswith("_") or (
        node.name.startswith("__") and node.name.endswith("__")
    )
    placed = isinstance(owner, ast.Module) or (
        isinstance(owner, ast.ClassDef) and not owner.name.startswith("_")
    )
    dunder = node.name.startswith("__") and node.name != "__init__"
    if not public or not placed or dunder or "overload" in decorated(node):
        return None
    text = ast.get_docstring(node)
    if text is None:
        return f"no docstring on {node.name}"
    if FOREIGN.search(text):
        return f"docstring of {node.name} is not Google style"
    return None


def echo(node: ast.AST, source: Source) -> str | None:
    """Names a `print()` call.

    Args:
        node: Any node.
        source: The module the node is in (unused; every rule takes it).

    Returns:
        The message, or `None`.
    """
    del source
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ):
        return "print() in library code"
    return None


RULES: dict[
    str, tuple[Callable[[ast.AST, Source], str | None], tuple[str, ...]]
] = {
    "compound": (compound, ("scmle", "scripts")),
    "underscore": (underscore, ("scmle", "tests")),
    "alias": (alias, FOLDERS),
    "swallow": (swallow, ("scmle", "scripts")),
    "stub": (stub, LIBRARY),
    "document": (document, LIBRARY),
    "echo": (echo, LIBRARY),
}
STRICT = frozenset({"underscore", "alias"})


def applies(rule: str, path: str) -> bool:
    """Whether a rule reads a file, by its repository-relative path."""
    if "mutants" in path.split("/") or not path.endswith(".py"):
        return False
    if rule == "echo" and path.startswith(PRINTERS):
        return False
    return path.split("/", 1)[0] in RULES[rule][1]


def inspect(path: str, text: str, rules: Iterable[str]) -> list[Finding]:
    """Applies rules to one module's text.

    Args:
        path: The file, relative to the repository root.
        text: Its source.
        rules: The rule names to apply; those that do not read the file
            are skipped.

    Returns:
        Every violation, in line order.
    """
    chosen = [rule for rule in rules if applies(rule, path)]
    if not chosen:
        return []
    tree = ast.parse(text)
    parents = {
        child: node
        for node in ast.walk(tree)
        for child in ast.iter_child_nodes(node)
    }
    source = Source(text.splitlines(), parents)
    found = [
        Finding(path, getattr(node, "lineno", 1), rule, message)
        for node in ast.walk(tree)
        for rule in chosen
        if (message := RULES[rule][0](node, source))
    ]
    if "stub" in chosen:
        found += [
            Finding(path, number, "stub", f"{match.group(1)} comment")
            for number, line in enumerate(source.lines, 1)
            if (match := COMMENT.search(line))
        ]
    return sorted(found, key=lambda finding: (finding.line, finding.rule))


def git(*arguments: str) -> str | None:
    """Runs a read-only git command in the repository.

    Args:
        *arguments: The git arguments.

    Returns:
        Its standard output, or `None` when git fails or is absent.
    """
    try:
        done = subprocess.run(  # nosec B603 B607
            ["git", *arguments],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):  # handled: not a checkout
        return None
    return done.stdout


def changed(base: str) -> list[str]:
    """Python files changed against a base, untracked files included."""
    listed = (git("diff", "--name-only", base, "--") or "").splitlines()
    listed += (git("ls-files", "--others", "--exclude-standard") or "").split()
    return sorted({path for path in listed if path.endswith(".py")})


def ratchet(paths: list[str], base: str, rules: Iterable[str]) -> list[Finding]:
    """Violations the changed files hold beyond their base versions.

    Args:
        paths: Changed files, relative to the repository root.
        base: The git revision to compare with.
        rules: The rules to ratchet.

    Returns:
        Each violation whose rule and text the base versions of all the
        changed files hold fewer times than the files hold now.
    """
    rules = list(rules)
    before: collections.Counter[tuple[str, str]] = collections.Counter()
    now: list[Finding] = []
    for path in paths:
        old = git("show", f"{base}:{path}")
        if old is not None:
            before.update(
                (finding.rule, finding.text)
                for finding in inspect(path, old, rules)
            )
        if (ROOT / path).exists():
            now += inspect(path, (ROOT / path).read_text("utf-8"), rules)
    excess = collections.Counter(
        (finding.rule, finding.text) for finding in now
    )
    excess.subtract(before)
    found = []
    for finding in reversed(now):
        key = (finding.rule, finding.text)
        if excess[key] > 0:
            excess[key] -= 1
            found.append(finding)
    return found[::-1]


def tree() -> list[str]:
    """Every Python file the rules may read, relative to the root."""
    return sorted(
        path.relative_to(ROOT).as_posix()
        for folder in FOLDERS
        for path in (ROOT / folder).rglob("*.py")
    )


def ceiling() -> list[str]:
    """Multi-word counts above the ceiling `structure.json` records."""
    budget = json.loads(STRUCTURE.BUDGET.read_text(encoding="utf-8"))
    counts = dict.fromkeys(
        ("compound_functions", "compound_classes", "compound_attributes"), 0
    )
    for path in sorted(STRUCTURE.PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ClassDef):
                counts["compound_classes"] += STRUCTURE.counted(node.name)
                counts["compound_attributes"] += sum(
                    map(STRUCTURE.counted, STRUCTURE.attributes(node))
                )
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                counts["compound_functions"] += STRUCTURE.counted(node.name)
    return [
        f"structure.json: {key} is {value}, above {budget[key]}"
        for key, value in counts.items()
        if value > budget.get(key, value)
    ]


def check(paths: list[str] | None = None, base: str = "HEAD") -> list[str]:
    """Every violation of the owner rules.

    Args:
        paths: Files to check, relative to the root; `None` checks the
            whole tree and the `structure.json` ceiling.
        base: The git revision the ratcheted rules compare with.

    Returns:
        One line per violation, empty when the rules hold.
    """
    scope = tree() if paths is None else sorted(set(paths))
    found = [
        finding
        for path in scope
        if (ROOT / path).exists()
        for finding in inspect(
            path, (ROOT / path).read_text(encoding="utf-8"), STRICT
        )
    ]
    moved = changed(base)
    found += [
        finding
        for finding in ratchet(moved, base, set(RULES) - STRICT)
        if finding.path in scope
    ]
    lines = [str(finding) for finding in found]
    return lines + (ceiling() if paths is None else [])


def main(arguments: list[str] | None = None) -> int:
    """Checks the rules and prints each violation.

    Args:
        arguments: Command-line arguments; `sys.argv` when `None`.

    Returns:
        The exit status: 1 when a rule fails, else 0.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("paths", nargs="*", help="files to check")
    parser.add_argument("--base", default="HEAD", help="revision to ratchet on")
    options = parser.parse_args(arguments)
    paths = [
        pathlib.Path(path).resolve().relative_to(ROOT).as_posix()
        for path in options.paths
    ]
    lines = check(paths or None, options.base)
    for line in lines:
        print(line)
    return 1 if lines else 0


if __name__ == "__main__":
    sys.exit(main())
