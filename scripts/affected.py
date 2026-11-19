"""Selects and runs the tests a change can affect.

The selection walks a static dependency graph of every tracked file. A
Python file depends on the modules it imports at any depth (relative
imports resolved, every parent package included), on modules it names in
strings (`"scmle.cli:main"`, `monkeypatch.setattr("scmle.x.y", ...)`, a
formatted `f"scmle.models.{name}"` names the whole package) and on the
paths it names (`ROOT / "tests" / "fixtures" / "a.json"`, `"docs/*.md"`,
`f"data/{name}.pq"`). Any other text file only mentions what it names: a
reader of `docs/x.md` is affected when the page changes, not when a module
it names changes, unless that module was deleted, which breaks a check
that names exist. Import lines in such files (snippets that tests run)
still count as imports.

A file that lists the repository from its root (`ROOT.rglob`, `git
ls-files`) depends on everything, and one that lists folders from the root
treats each bare word it holds as a folder.

A test is affected when a changed file is reachable from it: by imports,
reads, or a console script named in `pyproject.toml`. A change the graph
cannot bound selects every test: `pyproject.toml`, `uv.lock`, `setup.cfg`,
any `conftest.py` (fixtures reach tests by name, not by import), and any
file other than Python that no file references by path. Matching is by
name, so a deleted module still reaches its old importers.

The selection errs towards running more. Facts per file are cached by
content hash in `.pytest_cache/affected/graph.json`.

    uv run python scripts/affected.py              # run, working tree vs HEAD
    uv run python scripts/affected.py --base main  # run, working tree vs main
    uv run python scripts/affected.py --list       # print the tests only
    uv run python scripts/affected.py -- -x -q     # pass options to pytest
"""

import argparse
import ast
import collections
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import hashlib
import importlib.util
import json
import pathlib
import re
import subprocess
import sys
import tomllib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = pathlib.Path(".pytest_cache") / "affected" / "graph.json"
VERSION = 6
# Changing these changes how every test runs.
GLOBAL = frozenset(
    {
        "pyproject.toml",
        "uv.lock",
        "setup.cfg",
        "pytest.ini",
        "tox.ini",
        "conftest.py",
    }
)
# Folders whose files only release tooling reads, never a test: a change
# there selects the tests that list the whole repository, not every test.
INERT = ("changelog.d/",)
# Folders on `sys.path` under pytest or when a script runs. Only import
# statements reach modules below the root, whose names carry a `~`, since a
# word in a string such as `"rules"` is rarely a module.
SEARCH = ("", "tests/", "scripts/")
LISTING = frozenset({"rglob", "glob", "iterdir", "walk", "listdir", "scandir"})
# Larger files are data, not references: the largest tracked source file is
# under 1 MiB, so 8 MiB leaves room and skips only datasets and binaries.
LIMIT = 8 * 1024 * 1024
# Longer string constants are messages or embedded documents, not paths:
# the longest tracked path is under 80 characters.
LONGEST = 300
# Configuration whose bare words may be commands, such as `run: scmle fit`.
CONFIG = frozenset({".yml", ".yaml", ".toml", ".cfg", ".ini", ".sh", ""})
WILD = re.compile(r"[*?\[{<%]")
EXTENSION = re.compile(r"\.[A-Za-z0-9]{1,8}$")
IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
DOTTED = re.compile(
    r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?::[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)?"
)
WORD = re.compile(r"[\w.\-]+")
TEXTPATH = re.compile(
    r"[\w.\-]+(?:/[\w.\-*{}]+)+/?|[\w\-]+\.[A-Za-z0-9]{1,8}\b"
)
TEXTMODULE = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
TEXTIMPORT = re.compile(r"^\s*(?:from|import)\s+([\w.]+)", re.MULTILINE)
TEXTRUN = re.compile(r"-m\s+([A-Za-z_][\w.]*)")
TEXTWORD = re.compile(r"(?<![\w.\-/])[A-Za-z_][\w\-]*(?![\w.\-/])")


class GitError(Exception):
    """A git command failed, so the change set is unknown."""


@dataclasses.dataclass(frozen=True)
class Facts:
    """What one file references.

    Attributes:
        imported: Dotted module names, with every parent package, and
            `package.*` for a formatted name inside a package.
        read: Normalised paths of files or folders, possibly partial.
        mentioned: Module names and `/`-prefixed paths a non-Python file
            names; they count only for deleted files.
        exhaustive: True when the file depends on every file.
    """

    imported: frozenset[str] = frozenset()
    read: frozenset[str] = frozenset()
    mentioned: frozenset[str] = frozenset()
    exhaustive: bool = False


def normalise(text: str) -> str:
    """Reduces a path-like string to the part that names a real path.

    Args:
        text: A string from source.

    Returns:
        The path without leading `./`, `../` or `/`, cut to its folder at
        the first wildcard or placeholder; empty when nothing remains.
    """
    text = text.replace("\\", "/")
    found = WILD.search(text)
    if found:
        text = text[: found.start()]
        text = text[: text.rfind("/") + 1] if "/" in text else ""
    parts = [part for part in text.split("/") if part not in {"", ".", ".."}]
    return "/".join(parts)


def expand(name: str) -> set[str]:
    """Lists a dotted name with every parent.

    Args:
        name: A dotted name, optionally `module:attribute`.

    Returns:
        `a`, `a.b` and `a.b.c` for `a.b:c`.
    """
    parts = name.replace(":", ".").split(".")
    return {".".join(parts[:end]) for end in range(1, len(parts) + 1)}


def statement(name: str) -> set[str]:
    """Lists the names an import statement may load.

    Args:
        name: The imported dotted name.

    Returns:
        Every parent of the name, plain and with the `~` of a module found
        from `tests/` or `scripts/`.
    """
    names = expand(name)
    return names | {"~" + part for part in names}


def classify(text: str, imported: set[str], read: set[str]) -> None:
    """Records a string constant as a module name, a command, a path.

    A bare word is also recorded with a leading `!`, the name of a console
    script it may run.

    Args:
        text: The constant.
        imported: Module names, extended in place.
        read: Paths, extended in place.
    """
    if len(text) > LONGEST or any(space in text for space in " \t\n\r"):
        return
    if DOTTED.fullmatch(text):
        imported.update(expand(text))
    if IDENTIFIER.fullmatch(text):
        imported.add("!" + text)
    if "/" in text or "\\" in text or EXTENSION.search(text):
        path = normalise(text)
        if path:
            read.add(path)


def package(path: str) -> list[str]:
    """Names the package a source file belongs to.

    Args:
        path: The file, relative to the repository.

    Returns:
        The package's dotted parts.
    """
    return path.removesuffix(".py").split("/")[:-1]


def chain(node: ast.expr) -> list[ast.expr]:
    """Flattens `a / b / c` into its operands.

    Args:
        node: The outermost division.

    Returns:
        The operands, left to right.
    """
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return [*chain(node.left), node.right]
    return [node]


def join(operands: Sequence[ast.expr]) -> list[str]:
    """Joins runs of string constants into paths.

    Args:
        operands: The parts of a path expression.

    Returns:
        One path per run of adjacent constants.
    """
    runs: list[list[str]] = [[]]
    for operand in operands:
        if isinstance(operand, ast.Constant) and isinstance(operand.value, str):
            runs[-1].append(operand.value)
        elif runs[-1]:
            runs.append([])
    return [normalise("/".join(run)) for run in runs if run]


def rooted(node: ast.expr, roots: set[str]) -> bool:
    """Tells whether an expression is a bare name bound to the root.

    Args:
        node: The expression.
        roots: Names bound from `__file__` without any constant.

    Returns:
        True for such a name.
    """
    return isinstance(node, ast.Name) and node.id in roots


def receiver(node: ast.Call) -> ast.expr | None:
    """Finds what a listing call lists.

    Args:
        node: A call.

    Returns:
        The listed expression, or None when the call does not list.
    """
    function = node.func
    if isinstance(function, ast.Attribute) and function.attr in LISTING:
        target = function.value
        if isinstance(target, ast.Name) and target.id == "os" and node.args:
            return node.args[0]
        return target
    if isinstance(function, ast.Name) and function.id in LISTING:
        return node.args[0] if node.args else ast.Constant("")
    return None


def bind(tree: ast.Module) -> set[str]:
    """Finds names bound to a path derived from `__file__` alone.

    Args:
        tree: The module.

    Returns:
        Names such as `ROOT = pathlib.Path(__file__).parents[2]`.
    """
    roots: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        inner = list(ast.walk(node.value))
        named = any(
            isinstance(item, ast.Name) and item.id == "__file__"
            for item in inner
        )
        constant = any(
            isinstance(item, ast.Constant) and isinstance(item.value, str)
            for item in inner
        )
        if named and not constant:
            roots.update(
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            )
    return roots


def search(text: str, path: str) -> Facts:
    """Finds references in a file that does not parse as Python.

    Args:
        text: The file's text.
        path: The file, relative to the repository.

    Returns:
        Module names and paths the text mentions.
    """
    imported: set[str] = set()
    read: set[str] = set()
    for name in TEXTMODULE.findall(text):
        imported.update(expand(name))
    for name in TEXTPATH.findall(text):
        found = normalise(name.rstrip(".-"))
        if found:
            read.add(found)
    folder = path.rpartition("/")[0]
    if folder and path.endswith(".py"):
        read.update(f"{folder}/{name.split('.')[0]}" for name in imported)
    statements: set[str] = set()
    for name in TEXTIMPORT.findall(text):
        statements.update(statement(name.strip(".")))
    for name in TEXTRUN.findall(text):
        imported.update(expand(name))
        imported.add(name + ".__main__")
    if pathlib.PurePosixPath(path).suffix in CONFIG:
        imported.update("!" + word for word in TEXTWORD.findall(text))
    if path.endswith(".py"):
        return Facts(frozenset(imported | statements), frozenset(read))
    mentioned = imported | {"/" + name for name in read}
    return Facts(frozenset(statements), mentioned=frozenset(mentioned))


class Scanner:
    """Collects the facts of one parsed Python module.

    Attributes:
        path: The module, relative to the repository.
        folder: The module's folder.
        roots: Names bound to the repository root.
        inner: Ids of divisions inside a longer path chain.
        imported: Module names found so far.
        read: Paths found so far.
        words: String constants that could be folder or module names.
        listing: True once the module lists a folder.
        exhaustive: True once the module lists the repository root.
    """

    def __init__(self, path: str, tree: ast.Module) -> None:
        """Prepares a scan of one module.

        Args:
            path: The module, relative to the repository.
            tree: Its syntax tree.
        """
        self.path = path
        self.folder = path.rpartition("/")[0]
        self.roots = bind(tree)
        self.inner = {
            id(node.left)
            for node in ast.walk(tree)
            if isinstance(node, ast.BinOp) and isinstance(node.left, ast.BinOp)
        }
        self.imported: set[str] = set()
        self.read: set[str] = set()
        self.words: set[str] = set()
        self.listing = False
        self.exhaustive = False
        handlers: dict[type[ast.AST], Callable[[Any], None]] = {
            ast.Import: self.bring,
            ast.ImportFrom: self.pull,
            ast.Constant: self.note,
            ast.JoinedStr: self.format,
            ast.BinOp: self.divide,
            ast.Call: self.call,
            ast.Name: self.mention,
            ast.Attribute: self.mention,
        }
        for node in ast.walk(tree):
            handler = handlers.get(type(node))
            if handler:
                handler(node)

    def sibling(self, name: str) -> None:
        """Records a bare import that may be a module in the same folder.

        Args:
            name: The imported dotted name.
        """
        if self.folder:
            self.read.add(f"{self.folder}/{name.split('.', maxsplit=1)[0]}")

    def bring(self, node: ast.Import) -> None:
        """Records `import a.b`.

        Args:
            node: The statement.
        """
        for alias in node.names:
            self.imported.update(statement(alias.name))
            self.sibling(alias.name)

    def pull(self, node: ast.ImportFrom) -> None:
        """Records `from a import b`, resolving relative imports.

        Args:
            node: The statement.
        """
        base = package(self.path) if node.level else []
        base = base[: len(base) - node.level + 1] if node.level else base
        stem = ".".join([*base, *([node.module] if node.module else [])])
        if stem:
            self.imported.update(statement(stem))
        for alias in node.names:
            if alias.name != "*":
                name = f"{stem}.{alias.name}".strip(".")
                self.imported.update(statement(name))
        if not node.level and node.module:
            self.sibling(node.module)

    def note(self, node: ast.Constant) -> None:
        """Records a string constant.

        Args:
            node: The constant.
        """
        if not isinstance(node.value, str):
            return
        classify(node.value, self.imported, self.read)
        self.exhaustive = self.exhaustive or "ls-files" in node.value
        if WORD.fullmatch(node.value):
            self.words.add(node.value)

    def format(self, node: ast.JoinedStr) -> None:
        """Records the fixed head of an f-string as a folder or package.

        Args:
            node: The f-string.
        """
        first = node.values[0] if node.values else None
        if not isinstance(first, ast.Constant) or not isinstance(
            first.value, str
        ):
            return
        head = first.value
        if "/" in head:
            found = normalise(head[: head.rfind("/") + 1])
            if found:
                self.read.add(found)
        if head.endswith(".") and DOTTED.fullmatch(head.rstrip(".")):
            self.imported.update(expand(head.rstrip(".")))
            self.imported.add(head + "*")

    def divide(self, node: ast.BinOp) -> None:
        """Records the paths a `root / "a" / "b"` chain builds.

        Args:
            node: The division; only the outermost of a chain counts.
        """
        if not isinstance(node.op, ast.Div) or id(node) in self.inner:
            return
        operands = chain(node)
        for index, found in enumerate(join(operands)):
            first = index == 0 and rooted(operands[0], self.roots)
            if found and (first or "/" in found or EXTENSION.search(found)):
                self.read.add(found)

    def call(self, node: ast.Call) -> None:
        """Records folder listings and `joinpath` or `os.path.join` paths.

        Args:
            node: The call.
        """
        listed = receiver(node)
        if listed is not None:
            self.listing = True
            self.exhaustive = self.exhaustive or rooted(listed, self.roots)
        function = node.func
        if not isinstance(function, ast.Attribute):
            return
        if function.attr not in {"joinpath", "join"}:
            return
        head = function.value
        if function.attr == "join" and node.args:
            head = node.args[0]
        for found in join(node.args):
            if found and ("/" in found or rooted(head, self.roots)):
                self.read.add(found)

    def mention(self, node: ast.Name | ast.Attribute) -> None:
        """Records use of `entry_points`, which loads every plugin.

        Args:
            node: The name or attribute.
        """
        label = node.id if isinstance(node, ast.Name) else node.attr
        if label == "entry_points":
            self.imported.add("!entry_points")

    def summarise(self) -> Facts:
        """Finishes the scan.

        Returns:
            The module's facts.
        """
        read = set(self.read)
        if self.listing and self.roots:
            read.update(filter(None, map(normalise, self.words)))
        imported = set(self.imported)
        if "-m" in self.words:
            imported.update(
                word + ".__main__"
                for word in self.words
                if DOTTED.fullmatch(word)
            )
        return Facts(
            frozenset(imported), frozenset(read), exhaustive=self.exhaustive
        )


def scan(text: str, path: str) -> Facts:
    """Finds what one file references.

    Args:
        text: The file's text.
        path: The file, relative to the repository.

    Returns:
        The file's facts.
    """
    if not path.endswith(".py"):
        return search(text, path)
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return search(text, path)
    return Scanner(path, tree).summarise()


def identify(path: str, aliases: Mapping[str, Iterable[str]]) -> set[str]:
    """Lists every name under which other files can reference a file.

    Args:
        path: The file, relative to the repository.
        aliases: Module name to the other names it answers to, such as a
            console script.

    Returns:
        Its path, every trailing part of its path and of its folders (each
        with a leading `/`, apart from module names), and for Python its
        module names, `package.*` for each parent package
        and its aliases.
    """
    parts = path.split("/")
    names = {"/" + "/".join(parts[start:]) for start in range(len(parts))}
    for end in range(1, len(parts)):
        names.update("/" + "/".join(parts[start:end]) for start in range(end))
    if path.endswith(".py"):
        for prefix in SEARCH:
            if not path.startswith(prefix):
                continue
            dotted = path.removeprefix(prefix).removesuffix(".py").split("/")
            if dotted[-1] == "__init__":
                dotted = dotted[:-1]
            if not dotted or not all(IDENTIFIER.fullmatch(p) for p in dotted):
                continue
            module = ".".join(dotted)
            if prefix:
                names.add("~" + module)
                continue
            names.add(module)
            names.update("!" + name for name in aliases.get(module, ()))
            names.update(
                ".".join(dotted[:end]) + ".*" for end in range(1, len(dotted))
            )
    return names


def digest(data: bytes) -> str:
    """Hashes file content for the cache.

    Args:
        data: The content.

    Returns:
        A hex digest.
    """
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def alias(root: pathlib.Path) -> dict[str, set[str]]:
    """Reads console scripts and entry points from `pyproject.toml`.

    Args:
        root: The repository.

    Returns:
        Module name to the names that run or load it.
    """
    path = root / "pyproject.toml"
    if not path.is_file():
        return {}
    try:
        project = tomllib.loads(path.read_text("utf-8")).get("project", {})
    except tomllib.TOMLDecodeError:
        return {}
    aliases: dict[str, set[str]] = collections.defaultdict(set)
    for name, target in project.get("scripts", {}).items():
        aliases[str(target).split(":")[0]].add(name)
    for group in project.get("entry-points", {}).values():
        for target in group.values():
            aliases[str(target).split(":")[0]].add("entry_points")
    return dict(aliases)


def tested(path: str) -> bool:
    """Tells whether pytest collects a file.

    Args:
        path: The file, relative to the repository.

    Returns:
        True for `tests/**/test_*.py` and `tests/**/*_test.py`.
    """
    name = path.rpartition("/")[2]
    return (
        path.startswith("tests/")
        and name.endswith(".py")
        and (name.startswith("test_") or name.endswith("_test.py"))
    )


class Graph:
    """The reverse dependency graph of a repository's files.

    Attributes:
        root: The repository.
        files: Paths relative to the root.
        facts: Each readable file's facts.
    """

    def __init__(
        self, root: pathlib.Path, files: Sequence[str], cache: pathlib.Path
    ) -> None:
        """Scans the files, reusing cached facts for unchanged content.

        Args:
            root: The repository.
            files: Paths relative to the root.
            cache: The cache file.
        """
        self.root = root
        self.files = list(files)
        self.facts = self.load(cache)
        self.aliases = alias(root)
        self.index: dict[str, set[str]] = collections.defaultdict(set)
        self.mentions: dict[str, set[str]] = collections.defaultdict(set)
        self.exhaustive = set()
        for path, facts in self.facts.items():
            for name in facts.imported:
                self.index[name].add(path)
            for name in facts.read:
                self.index["/" + name].add(path)
            for name in facts.mentioned:
                self.mentions[name].add(path)
            if facts.exhaustive:
                self.exhaustive.add(path)

    def load(self, cache: pathlib.Path) -> dict[str, Facts]:
        """Scans every file, through the cache.

        Args:
            cache: The cache file.

        Returns:
            Path to facts, for files that exist.
        """
        stored: dict[str, list[object]] = {}
        try:
            document = json.loads(cache.read_text("utf-8"))
            if document.get("version") == VERSION:
                stored = document["files"]
        except (OSError, ValueError, KeyError, AttributeError):
            stored = {}
        facts: dict[str, Facts] = {}
        fresh: dict[str, list[object]] = {}
        for path in self.files:
            file = self.root / path
            try:
                if not file.is_file() or file.stat().st_size > LIMIT:
                    continue
                data = file.read_bytes()
            except OSError:
                continue
            key = digest(data)
            entry = stored.get(path)
            if entry and entry[0] == key:
                found = Facts(
                    frozenset(entry[1]),  # type: ignore[call-overload]
                    frozenset(entry[2]),  # type: ignore[call-overload]
                    frozenset(entry[3]),  # type: ignore[call-overload]
                    bool(entry[4]),
                )
            else:
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    text = ""
                found = scan(text, path) if "\0" not in text else Facts()
            facts[path] = found
            fresh[path] = [
                key,
                sorted(found.imported),
                sorted(found.read),
                sorted(found.mentioned),
                found.exhaustive,
            ]
        if fresh != stored:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps({"version": VERSION, "files": fresh}), "utf-8"
                )
                temporary.replace(cache)
            except OSError:
                pass
        return facts

    def depend(self, path: str, deleted: set[str]) -> set[str]:
        """Finds the files that reference one file directly.

        Args:
            path: The file, relative to the root.
            deleted: Changed files that no longer exist, whose mentions in
                prose count.

        Returns:
            The direct dependents.
        """
        dependents: set[str] = set()
        for name in identify(path, self.aliases):
            dependents.update(self.index.get(name, ()))
            if path in deleted:
                dependents.update(self.mentions.get(name, ()))
        return dependents

    def reach(self, changed: set[str]) -> set[str]:
        """Finds every file a change can affect.

        Args:
            changed: Changed, added or deleted paths, relative to the root.

        Returns:
            The changed files, the files that list the whole repository,
            and every file that depends on them, transitively.
        """
        deleted = {p for p in changed if not (self.root / p).exists()}
        reached = set(changed) | self.exhaustive
        queue = collections.deque(reached)
        while queue:
            for dependent in self.depend(queue.popleft(), deleted) - reached:
                reached.add(dependent)
                queue.append(dependent)
        return reached

    def select(self, changed: Iterable[str]) -> list[str]:
        """Finds the tests a change can affect.

        Args:
            changed: Changed, added or deleted paths, relative to the root.

        Returns:
            The affected test files that exist, sorted.
        """
        paths = {normalise(path) for path in changed}
        if not paths:
            return []
        present = {
            path
            for path in set(self.files) | paths
            if tested(path) and (self.root / path).is_file()
        }
        if any(self.unclassified(path, paths) for path in paths):
            return sorted(present)
        return sorted(self.reach(paths) & present)

    def unclassified(self, path: str, changed: set[str]) -> bool:
        """Tells whether the graph cannot bound a changed file's effect.

        Args:
            path: A changed path, relative to the root.
            changed: Every changed path, to tell deleted files.

        Returns:
            True for global configuration, any `conftest.py`, and a file
            other than Python that no file references by path.
        """
        name = path.rpartition("/")[2]
        if path in GLOBAL or name in GLOBAL:
            return True
        if path.endswith(".py") or path.startswith(INERT):
            return False
        deleted = {p for p in changed if not (self.root / p).exists()}
        return not self.depend(path, deleted)


def git(root: pathlib.Path, *arguments: str) -> list[str]:
    """Runs a read-only git command.

    Args:
        root: The repository.
        *arguments: The command's arguments.

    Returns:
        The NUL-separated output.

    Raises:
        GitError: When git fails.
    """
    try:
        result = subprocess.run(
            ["git", arguments[0], "-z", *arguments[1:]],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        message = f"git {' '.join(arguments)} failed: {detail.strip()}"
        raise GitError(message) from error
    return [line for line in result.stdout.split("\0") if line]


def track(root: pathlib.Path) -> list[str]:
    """Lists tracked and untracked, unignored files.

    Args:
        root: The repository.

    Returns:
        Paths relative to the root.
    """
    return sorted(set(git(root, "ls-files", "-co", "--exclude-standard")))


def compare(root: pathlib.Path, base: str | None) -> list[str]:
    """Lists the paths that differ from a revision, including new files.

    Args:
        root: The repository.
        base: The revision; HEAD when None.

    Returns:
        Old and new paths of every changed file.
    """
    revision = base or "HEAD"
    changed = git(root, "diff", "--name-only", "--no-renames", revision, "--")
    untracked = git(root, "ls-files", "--others", "--exclude-standard")
    return sorted(set(changed) | set(untracked))


def main(argv: Sequence[str] | None = None) -> int:
    """Runs or lists the affected tests.

    Args:
        argv: Command-line arguments; after `--`, options for pytest.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", help="revision to diff against (HEAD)")
    parser.add_argument("--list", action="store_true", help="print, not run")
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("extra", nargs="*", help="options for pytest")
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    try:
        changed = compare(root, arguments.base)
        files = track(root)
    except GitError as error:
        print(error, file=sys.stderr)
        return 2
    tests = Graph(root, files, root / CACHE).select(changed)
    if arguments.list:
        print("\n".join(tests))
        return 0
    if not tests:
        print("No affected tests.", file=sys.stderr)
        return 0
    command = [sys.executable, "-m", "pytest", *tests, *arguments.extra]
    if importlib.util.find_spec("xdist") is not None:
        command[3:3] = ["-n", "auto"]
    else:
        print(
            "pytest-xdist is not installed; running serially.", file=sys.stderr
        )
    return subprocess.run(command, cwd=root, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
