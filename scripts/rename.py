"""Renames definitions by a map, and only the references that resolve to them.

A map is `scripts/renames/<lane>.toml`, one `dotted.old.name = "new"` entry
per definition: a module, class, function, module-level value, method or
attribute. Nothing is renamed by matching the word. A site changes only when
name resolution says it refers to the definition:

- names and attribute chains, by libcst's scope and fully-qualified-name
  providers (imports, relative imports and shadowing included), so a local
  that shadows a function, or `pandas.read_json`, is left alone;
- the names an import binds, `__all__` entries, and the re-exports of a
  package `__init__`, followed to every alias;
- attributes on a receiver whose type mypy infers (`self.`, typed
  parameters, `super()`, `type[...]`), for a method or attribute and every
  class that overrides it, plus keyword arguments to the class's
  constructor;
- string literals, comments and prose that spell the full dotted path
  (`scmle.models.engines.Classifier`, also `module:name`), which is how
  pickled and worker-started paths are recorded;
- backtick citations in docstrings, comments, tests and `docs/`
  (`` `name` ``, `` `module.name` ``, `` `Class.method()` ``), resolved from
  the enclosing class and the module's bindings; prose outside Python
  resolves a citation only when its dotted form names one definition.

It cannot follow `getattr` with a computed name, a receiver mypy types as
`Any` (printed as a `review:` line), or a structural `Protocol` that matches
by shape; mypy strict in `--verify` reports what those miss. A map is
refused when a new name is not one word (the naming rule of `structure.py`),
when an old name is exempt there, when a method overrides one of a class
outside the repository, or when a new name is already bound where a renamed
site is. History (`CHANGELOG.md`, `docs/RENAMES.md`, `plan/`) and data files
are never rewritten; the renames are appended to `docs/RENAMES.md`.

`--verify` is the proof, as in `move.py`: with the map applied, no site
resolves to an old name, every new definition exists, and ruff, mypy strict
on the touched package modules and import-linter pass.

    uv run python scripts/rename.py data --dry-run
    uv run python scripts/rename.py data
    uv run python scripts/rename.py data --verify
"""

import argparse
import ast
import collections.abc
import dataclasses
import keyword
import pathlib
import re
import subprocess
import sys
import tomllib

import libcst
from libcst import metadata
from mypy import build
from mypy import nodes
from mypy import types
from mypy.modulefinder import BuildSource
from mypy.options import Options
from mypy.server import subexpr
import structure

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAPS = pathlib.Path(__file__).with_name("renames")
RENAMES = "docs/RENAMES.md"
HEADING = "## Name renames"
HISTORY = ("CHANGELOG.md", RENAMES, "plan/", "scripts/renames/")
PROSE = (".md", ".rst", ".txt", ".astro", ".html", ".mjs", ".ts")
CONFIG = (".toml", ".cfg", ".ini", ".yml", ".yaml")
SPAN = re.compile(r"`([A-Za-z_][\w.]*)(\(\))?`")


@dataclasses.dataclass(frozen=True)
class Target:
    """One map entry, resolved.

    Attributes:
        old: The dotted path searched for.
        new: The new last segment.
        kind: `module`, `class`, `function`, `value` or `member`.
        names: Every dotted path that refers to it: `old`, its re-exports,
            and for a member the same member of every overriding class.
        family: For a member, the classes that define it together.
    """

    old: str
    new: str
    kind: str
    names: frozenset[str]
    family: frozenset[str] = frozenset()

    @property
    def word(self) -> str:
        """The old last segment."""
        return self.old.rpartition(".")[2]


@dataclasses.dataclass(frozen=True)
class Edit:
    """One site that changes.

    Attributes:
        path: The file, relative to the root.
        line: Its line.
        before: The text before.
        after: The text after.
        reason: How the site resolved.
    """

    path: str
    line: int
    before: str
    after: str
    reason: str


@dataclasses.dataclass
class Plan:
    """Everything a map changes, computed before anything is written.

    Attributes:
        edits: Every site that changes.
        texts: New content per changed file.
        moves: Module files or packages to move, old path to new.
        refusals: Why the map cannot be applied; empty when it can.
        reviews: Sites the tool could not resolve, for a person to check.
    """

    edits: list[Edit] = dataclasses.field(default_factory=list)
    texts: dict[str, str] = dataclasses.field(default_factory=dict)
    moves: dict[str, str] = dataclasses.field(default_factory=dict)
    refusals: list[str] = dataclasses.field(default_factory=list)
    reviews: list[str] = dataclasses.field(default_factory=list)


def read(path: pathlib.Path) -> dict[str, str]:
    """Reads a map, flattening TOML's dotted keys back to dotted paths.

    Args:
        path: The map file.

    Returns:
        Old dotted path to new name.

    Raises:
        ValueError: When a value is not a string.
    """

    def flatten(table: dict[str, object], prefix: str) -> dict[str, str]:
        found: dict[str, str] = {}
        for key, value in table.items():
            name = f"{prefix}{key}"
            if isinstance(value, dict):
                found.update(flatten(value, f"{name}."))
            elif isinstance(value, str):
                found[name] = value
            else:
                raise ValueError(f"{name}: the new name must be a string")
        return found

    return flatten(tomllib.loads(path.read_text(encoding="utf-8")), "")


def listing(root: pathlib.Path) -> list[str]:
    """Files under the root, relative and sorted; tracked ones in a checkout."""
    if (root / ".git").exists():
        names = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split("\n")
        return sorted(name for name in names if (root / name).is_file())
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.parts)
        and "__pycache__" not in path.parts
    )


def dotted(path: str) -> str:
    """The module name of a Python file, relative to the root."""
    parts = pathlib.PurePosixPath(path).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def resolve(module: str, package: bool, level: int, name: str) -> str:
    """The absolute module an import names.

    Args:
        module: The importing module.
        package: Whether the importing file is a package `__init__`.
        level: The number of leading dots.
        name: The module after the dots; empty when there is none.

    Returns:
        The dotted module.
    """
    if not level:
        return name
    parts = module.split(".") if package else module.split(".")[:-1]
    base = parts[: len(parts) - level + 1]
    return ".".join([*base, name] if name else base)


def flat(body: list[ast.stmt]) -> collections.abc.Iterator[ast.stmt]:
    """Statements of a block, with the bodies of `if` and `try` inlined."""
    for node in body:
        if isinstance(node, ast.If):
            yield from flat(node.body + node.orelse)
        elif isinstance(node, ast.Try):
            yield from flat(node.body + node.orelse + node.finalbody)
            for handler in node.handlers:
                yield from flat(handler.body)
        else:
            yield node


def bound(node: ast.stmt) -> list[str]:
    """Plain names an assignment statement binds."""
    if isinstance(node, ast.Assign):
        return [item.id for item in node.targets if isinstance(item, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


class Index:
    """Definitions, bindings and re-exports, read from the syntax trees.

    Attributes:
        root: The repository root.
        package: The top-level package renames are confined to.
        files: Every file, relative to the root.
        modules: Module name to file, for every Python file.
        kinds: Dotted path to kind, for every definition in the package.
        bindings: Module name to its top-level names and what they resolve
            to.
        exports: Dotted alias to the dotted path it imports, for the
            package's modules.
    """

    def __init__(self, root: pathlib.Path, package: str) -> None:
        """Reads every module of the package."""
        self.root = root
        self.package = package
        self.files = listing(root)
        self.modules = {
            dotted(name): name for name in self.files if name.endswith(".py")
        }
        self.kinds: dict[str, str] = {}
        self.bindings: dict[str, dict[str, str]] = {}
        self.exports: dict[str, str] = {}
        for module, name in self.modules.items():
            if self.inside(module):
                self.scan(module, name)

    def inside(self, module: str) -> bool:
        """Whether a dotted path is in the package."""
        return module == self.package or module.startswith(f"{self.package}.")

    def binds(self, module: str) -> dict[str, str]:
        """The top-level names a module binds, read on first use."""
        if module not in self.bindings and module in self.modules:
            self.scan(module, self.modules[module])
        return self.bindings.get(module, {})

    def scan(self, module: str, name: str) -> None:
        """Records a module's bindings, and its definitions in the package."""
        tree = ast.parse((self.root / name).read_text(encoding="utf-8"))
        package = name.endswith("__init__.py")
        names = self.bindings.setdefault(module, {})
        inside = self.inside(module)
        if inside:
            self.kinds[module] = "module"
        for node in flat(tree.body):
            if isinstance(node, ast.ImportFrom):
                base = resolve(module, package, node.level, node.module or "")
                for alias in node.names:
                    local = alias.asname or alias.name
                    names[local] = f"{base}.{alias.name}"
                    if inside and alias.name != "*":
                        self.exports[f"{module}.{local}"] = names[local]
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    head = alias.name.split(".")[0]
                    names[alias.asname or head] = (
                        alias.name if alias.asname else head
                    )
            for item in self.defines(node, module):
                names[item] = f"{module}.{item}"

    def defines(self, node: ast.stmt, module: str) -> list[str]:
        """Records what one top-level statement defines.

        Returns:
            The names it binds.
        """
        found = bound(node)
        kind = "value"
        if isinstance(node, ast.ClassDef):
            found, kind = [node.name], "class"
            if self.inside(module):
                self.members(node, f"{module}.{node.name}")
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            found, kind = [node.name], "function"
        if self.inside(module):
            self.kinds.update({f"{module}.{item}": kind for item in found})
        return found

    def members(self, node: ast.ClassDef, prefix: str) -> None:
        """Records a class's methods, attributes and nested classes."""
        for item in flat(node.body):
            if isinstance(item, ast.ClassDef):
                self.kinds[f"{prefix}.{item.name}"] = "class"
                self.members(item, f"{prefix}.{item.name}")
            elif isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                self.kinds[f"{prefix}.{item.name}"] = "member"
                own = item.args.args[0].arg if item.args.args else ""
                self.kinds.update(
                    {
                        f"{prefix}.{inner.attr}": "member"
                        for inner in ast.walk(item)
                        if isinstance(inner, ast.Attribute)
                        and isinstance(inner.ctx, ast.Store)
                        and isinstance(inner.value, ast.Name)
                        and inner.value.id == own
                    }
                )
            for name in bound(item):
                self.kinds[f"{prefix}.{name}"] = "member"

    def aliases(self, path: str) -> set[str]:
        """A dotted path and every path that re-exports it, at any depth.

        A module bound by an import (`from scmle import data`) is not
        followed: `scmle.x.data.y` would be a path nobody writes, and a
        module that imports its own package would grow it without end.
        """
        found = {path}
        work = [path]
        exports = {
            alias: origin
            for alias, origin in self.exports.items()
            if self.kinds.get(origin) != "module"
        }
        while work:
            name = work.pop()
            for alias, origin in exports.items():
                if name == origin or name.startswith(f"{origin}."):
                    extra = alias + name[len(origin) :]
                    if extra not in found:
                        found.add(extra)
                        work.append(extra)
        return found


def inner(proper: types.ProperType) -> types.Type | None:
    """The type a wrapper stands for: `type[X]`, a bound, a tuple's class."""
    if isinstance(proper, types.TypeType):
        return proper.item
    if isinstance(proper, types.TypeVarType):
        return proper.upper_bound
    if isinstance(proper, types.TupleType):
        return proper.partial_fallback
    return None


def owners(kind: types.Type | None) -> set[str]:
    """The full names of the classes whose members an expression offers."""
    if kind is None:
        return set()
    proper = types.get_proper_type(kind)
    if isinstance(proper, types.UnionType):
        return {name for item in proper.items for name in owners(item)}
    if inner(proper) is not None:
        return owners(inner(proper))
    info = None
    if isinstance(proper, types.Instance):
        info = proper.type
    elif isinstance(proper, types.CallableType | types.Overloaded) and (
        proper.is_type_obj()
    ):
        info = proper.type_object()
    return {item.fullname for item in info.mro} if info else set()


class Types:
    """The types mypy infers, for resolving attribute receivers.

    Attributes:
        result: The mypy build, with every expression's type exported.
        classes: Every class the repository's Python files define.
    """

    def __init__(self, index: Index) -> None:
        """Type-checks every Python file of the repository, in memory."""
        options = Options()
        options.export_types = True
        options.preserve_asts = True
        options.incremental = False
        options.follow_imports = "silent"
        options.check_untyped_defs = True
        options.ignore_missing_imports = True
        sources = [
            BuildSource(str(index.root / name), module)
            for module, name in index.modules.items()
        ]
        self.result = build.build(sources, options)
        self.classes: dict[str, nodes.TypeInfo] = {}
        for module in index.modules:
            if module in self.result.files:
                self.collect(self.result.files[module].names, module)

    def collect(self, table: nodes.SymbolTable, prefix: str) -> None:
        """Records the classes a symbol table defines, nested ones too."""
        for name, symbol in table.items():
            node = symbol.node
            if isinstance(node, nodes.TypeInfo) and (
                node.fullname == f"{prefix}.{name}"
            ):
                self.classes[node.fullname] = node
                self.collect(node.names, node.fullname)

    def family(self, owner: str, member: str) -> tuple[set[str], str]:
        """The classes that define a member together, through overriding.

        Args:
            owner: The full name of a class that defines the member.
            member: The member's name.

        Returns:
            The classes, and a refusal when the member overrides one of a
            class outside the repository (a framework calls it by name).
        """
        if owner not in self.classes:
            return set(), f"{owner}: mypy found no such class"
        group = {owner}
        size = 0
        while size != len(group):
            size = len(group)
            for name in sorted(group):
                for base in self.classes[name].mro[1:]:
                    if member not in base.names:
                        continue
                    if base.fullname not in self.classes:
                        return group, (
                            f"{owner}.{member} overrides {base.fullname}"
                        )
                    group.add(base.fullname)
            group.update(
                name
                for name, info in self.classes.items()
                if member in info.names
                and any(base.fullname in group for base in info.mro[1:])
            )
        return group, ""

    def below(self, group: set[str]) -> list[nodes.TypeInfo]:
        """Every class that is, or inherits from, a class of a group."""
        return [
            info
            for info in self.classes.values()
            if any(base.fullname in group for base in info.mro)
        ]

    def receivers(
        self, module: str, wanted: dict[str, list[Target]]
    ) -> tuple[dict[tuple[str, int], list[Target | None]], list[int]]:
        """Which attribute accesses of a member's word resolve to it.

        Args:
            module: The module to read.
            wanted: Member targets by their old word.

        Returns:
            Per word and line, one entry per access in column order: the
            target whose family holds the receiver's type, or None; and the
            lines of accesses whose receiver mypy could not type.
        """
        found: dict[tuple[str, int], list[tuple[int, Target | None]]] = {}
        unknown = []
        tree = self.result.files.get(module)
        for node in subexpr.get_subexpressions(tree) if tree else []:
            if not isinstance(node, nodes.MemberExpr | nodes.SuperExpr):
                continue
            if node.name not in wanted:
                continue
            line = node.end_line or node.line
            if isinstance(node, nodes.SuperExpr):
                mro = node.info.mro if node.info else []
                classes = {item.fullname for item in mro}
            else:
                classes = owners(self.result.types.get(node.expr))
                if not classes and not node.fullname:
                    unknown.append(line)
            hit = next(
                (t for t in wanted[node.name] if classes & t.family), None
            )
            found.setdefault((node.name, line), []).append((node.column, hit))
        ordered = {
            key: [item[1] for item in sorted(items, key=lambda item: item[0])]
            for key, items in found.items()
        }
        return ordered, unknown


def single(name: str) -> str:
    """Why a new name breaks the naming rule; empty when it does not."""
    if not name.isidentifier() or keyword.iskeyword(name):
        return "is not an identifier"
    if name.startswith("_"):
        return "has a leading underscore"
    if structure.compound(name):
        return "is not one word"
    return ""


def remap(path: str, entries: dict[str, str]) -> str:
    """A dotted path after the map renames its prefixes."""
    parts = path.split(".")
    for size in range(len(parts), 0, -1):
        head = ".".join(parts[:size])
        if head in entries:
            parts[size - 1] = entries[head]
    return ".".join(parts)


def unique(parts: list[str], known: set[str]) -> str:
    """The head of a prose citation whose dotted form names one definition.

    A bare word never resolves; `Class.method` resolves when one definition
    ends that way, or, once renamed, when one class is called `Class`.

    Returns:
        The head's dotted path, or empty.
    """
    for size in range(len(parts), 1 if len(parts) == 1 else 0, -1):
        tail = "." + ".".join(parts[:size])
        matches = [name for name in known if name.endswith(tail)]
        if len(matches) == 1:
            return matches[0][: len(matches[0]) - len(tail)] + f".{parts[0]}"
    return ""


def expand(
    parts: list[str], known: set[str], index: Index, scope: tuple[str, str]
) -> str:
    """The dotted path the head of a citation resolves to.

    Args:
        parts: The citation's segments.
        known: Every dotted path a citation may resolve to.
        index: The repository index.
        scope: The module and the class the citation is written in; both
            empty in prose outside Python.

    Returns:
        The head's dotted path, or empty when it does not resolve.
    """
    module, owner = scope
    head = parts[0]
    local = f"{module}.{head}"
    member = f"{owner}.{head}"
    choices = (
        member if owner and member in known else "",
        index.binds(module).get(head, "") if module else "",
        head if head == index.package else "",
        f"{index.package}.{head}"
        if index.kinds.get(f"{index.package}.{head}") == "module"
        else "",
        (local if local in known else "") if module else unique(parts, known),
    )
    return next((item for item in choices if item), "")


def respell(
    parts: list[str], base: str, chosen: list[Target]
) -> tuple[list[str], list[str]]:
    """A dotted reference with every renamed segment replaced.

    Args:
        parts: The segments as written.
        base: The dotted path the first segment resolves to.
        chosen: The targets.

    Returns:
        The new segments, and the reason for each change.
    """
    full = [*base.split("."), *parts[1:]]
    offset = len(full) - len(parts)
    result = list(parts)
    reasons = []
    for target in chosen:
        for name in target.names:
            size = name.count(".") + 1
            at = size - 1 - offset
            if (
                at >= 0
                and full[:size] == name.split(".")
                and parts[at] == target.word
            ):
                result[at] = target.new
                reasons.append(f"resolves to {name}")
    return result, reasons


def cite(
    text: str,
    index: Index,
    chosen: list[Target],
    scope: tuple[str, str] | None,
    known: set[str],
) -> tuple[str, list[str], list[str]]:
    """Text with its citations and full dotted paths renamed.

    Args:
        text: A string literal, comment or file.
        index: The repository index.
        chosen: The targets.
        scope: The module and class the text is written in (both empty in
            prose); None to rename only full dotted paths.
        known: Every dotted path a citation may resolve to.

    Returns:
        The new text, one reason per change, and the citations of an old
        word that did not resolve.
    """
    words = {target.word for target in chosen}
    reasons: list[str] = []
    unresolved: list[str] = []

    def span(match: re.Match[str]) -> str:
        parts = match.group(1).split(".")
        if scope is None or not words & set(parts):
            return match.group(0)
        base = expand(parts, known, index, scope)
        if not base:
            unresolved.append(match.group(1))
            return match.group(0)
        result, why = respell(parts, base, chosen)
        reasons.extend(why)
        return f"`{'.'.join(result)}{match.group(2) or ''}`"

    def path(match: re.Match[str]) -> str:
        parts = re.split(r"[.:]", match.group(0))
        if not words & set(parts):
            return match.group(0)
        result, why = respell(parts, index.package, chosen)
        reasons.extend(why)
        marks = re.findall(r"[.:]", match.group(0))
        return result[0] + "".join(
            mark + part for mark, part in zip(marks, result[1:], strict=True)
        )

    qualified = re.compile(
        rf"(?<![\w.]){re.escape(index.package)}(?:[.:][A-Za-z_]\w*)+"
    )
    text = qualified.sub(path, SPAN.sub(span, text))
    return text, reasons, unresolved


class Writer(libcst.CSTTransformer):
    """Replaces the value of the nodes a plan changes."""

    def __init__(self, changes: dict[libcst.CSTNode, str]) -> None:
        """Keeps the changes, node to its new value."""
        super().__init__()
        self.changes = changes

    def on_leave(  # type: ignore[override]
        self, original: libcst.CSTNode, updated: libcst.CSTNode
    ) -> libcst.CSTNode:
        """Swaps in the new value of a changed node."""
        if original in self.changes:
            return updated.with_changes(value=self.changes[original])
        return updated


def chain(node: libcst.BaseExpression | None) -> list[libcst.Name]:
    """The names of a dotted expression, left to right."""
    if isinstance(node, libcst.Attribute):
        return [*chain(node.value), node.attr]
    return [node] if isinstance(node, libcst.Name) else []


class Source:
    """One Python file's syntax tree with its resolved metadata.

    Attributes:
        path: The file, relative to the root.
        module: Its module name.
        wrapper: The libcst tree with metadata.
        names: Node to the fully qualified names it may refer to.
        lines: Node to its first line.
        columns: Node to its first column.
        scopes: Node to its scope.
        parents: Node to its parent.
        builtins: The built-in names the file uses.
    """

    def __init__(
        self, path: str, manager: metadata.FullRepoManager, root: pathlib.Path
    ) -> None:
        """Parses a file and resolves its names, scopes and positions."""
        self.path = path
        self.module = dotted(path)
        self.wrapper = manager.get_metadata_wrapper_for_path(str(root / path))
        self.names = self.wrapper.resolve(metadata.FullyQualifiedNameProvider)
        positions = self.wrapper.resolve(metadata.PositionProvider)
        self.lines = {node: at.start.line for node, at in positions.items()}
        self.columns = {node: at.start.column for node, at in positions.items()}
        self.scopes = self.wrapper.resolve(metadata.ScopeProvider)
        self.parents = self.wrapper.resolve(metadata.ParentNodeProvider)
        self.builtins = {
            item.name.removeprefix("builtins.")
            for found in self.names.values()
            for item in found
            if item.name.startswith("builtins.")
        }

    def qualified(self, node: libcst.CSTNode) -> set[str]:
        """The fully qualified names a node may refer to."""
        return {item.name for item in self.names.get(node, set())}

    def owner(self, node: libcst.CSTNode) -> str:
        """The full name of the class a node is written in, or empty."""
        while node in self.parents:
            node = self.parents[node]
            if isinstance(node, libcst.ClassDef):
                found = self.qualified(node.name)
                return min(found) if found else ""
        return ""

    def taken(self, node: libcst.CSTNode, name: str) -> bool:
        """Whether a name is already bound where a node is.

        A built-in counts only when the file uses it, since a new binding
        would then shadow it.
        """
        scope = self.scopes.get(node)
        if scope is None:
            return False
        found = scope[name]
        own = [
            item
            for item in found
            if not isinstance(item, metadata.BuiltinAssignment)
        ]
        return bool(own) or (bool(found) and name in self.builtins)


class Sites:
    """The changes to one Python file.

    Attributes:
        source: The file.
        chosen: The targets.
        words: The targets by their old word.
        changes: Node to its new value.
        edits: One edit per changed line.
        refusals: New names already bound where a site is.
        reviews: Sites that did not resolve.
    """

    def __init__(self, source: Source, chosen: list[Target]) -> None:
        """Starts with no changes."""
        self.source = source
        self.chosen = chosen
        self.words: dict[str, list[Target]] = {}
        for target in chosen:
            self.words.setdefault(target.word, []).append(target)
        self.changes: dict[libcst.CSTNode, str] = {}
        self.edits: list[Edit] = []
        self.refusals: list[str] = []
        self.reviews: list[str] = []

    def change(self, node: libcst.CSTNode, after: str, reason: str) -> None:
        """Records a node's new value and the lines it changes."""
        before = str(getattr(node, "value", ""))
        self.changes[node] = after
        line = self.source.lines.get(node, 0)
        pairs = zip(before.splitlines(), after.splitlines(), strict=False)
        self.edits += [
            Edit(self.source.path, line + at, old.strip(), new.strip(), reason)
            for at, (old, new) in enumerate(pairs)
            if old != new
        ]

    def refuse(self, node: libcst.CSTNode, why: str) -> None:
        """Records a refusal at a node."""
        line = self.source.lines.get(node, 0)
        self.refusals.append(f"{self.source.path}:{line}: {why}")

    def match(self, word: str, found: set[str]) -> Target | None:
        """The target a reference's resolved names all point to."""
        return next(
            (t for t in self.words.get(word, []) if found and found <= t.names),
            None,
        )

    def name(self, node: libcst.Name) -> None:
        """Renames a name that resolves to a target."""
        found = self.source.qualified(node)
        target = self.match(node.value, found)
        if target:
            self.change(node, target.new, f"resolves to {min(found)}")
            if target.kind != "member" and self.source.taken(node, target.new):
                self.refuse(node, f"{target.new} is already bound here")
        elif any(found & t.names for t in self.words[node.value]):
            self.refuse(node, f"{node.value} may bind to several definitions")

    def attribute(self, node: libcst.Attribute, typed: Target | None) -> None:
        """Renames an attribute that resolves, by name or by receiver type.

        Args:
            node: The attribute.
            typed: The target mypy resolved the receiver to, if any.
        """
        found = self.source.qualified(node)
        target = self.match(node.attr.value, found)
        if target:
            self.change(node.attr, target.new, f"resolves to {min(found)}")
        elif typed is not None:
            self.change(node.attr, typed.new, "typed receiver")

    def segments(self, written: list[libcst.Name], name: str) -> None:
        """Renames the module segments of an import.

        Args:
            written: The module's names as written, after any leading dots.
            name: The absolute module they name.
        """
        parts = name.split(".")
        prefix = len(parts) - len(written)
        for target in self.chosen:
            for alias in target.names if target.kind == "module" else ():
                size = alias.count(".") + 1
                at = size - 1 - prefix
                if (
                    0 <= at < len(written)
                    and parts[:size] == alias.split(".")
                    and written[at].value == target.word
                ):
                    self.change(written[at], target.new, f"imports {alias}")

    def imports(self, node: libcst.ImportFrom) -> None:
        """Renames what a `from` import names."""
        written = chain(node.module)
        base = resolve(
            self.source.module,
            self.source.path.endswith("__init__.py"),
            len(node.relative),
            ".".join(item.value for item in written),
        )
        self.segments(written, base)
        if isinstance(node.names, libcst.ImportStar):
            return
        for alias in node.names:
            word = getattr(alias.name, "value", "")
            target = self.match(word, {f"{base}.{word}"})
            if not target:
                continue
            self.change(alias.name, target.new, "import")
            if alias.asname is None and self.source.taken(node, target.new):
                self.refuse(node, f"{target.new} is already bound here")

    def calls(self, node: libcst.Call) -> None:
        """Renames the keyword arguments of a family class's constructor."""
        called = self.source.qualified(node.func)
        for target in self.chosen:
            classes = {name.rpartition(".")[0] for name in target.names}
            if target.kind != "member" or not called & classes:
                continue
            for argument in node.args:
                if argument.keyword and argument.keyword.value == target.word:
                    self.change(argument.keyword, target.new, "keyword")

    def exports(self, node: libcst.Assign) -> None:
        """Renames the entries of a module's `__all__`."""
        if not any(
            getattr(item.target, "value", "") == "__all__"
            for item in node.targets
        ) or not isinstance(node.value, libcst.List | libcst.Tuple):
            return
        for element in node.value.elements:
            value = element.value
            if not isinstance(value, libcst.SimpleString):
                continue
            word = value.value[1:-1]
            target = self.match(word, {f"{self.source.module}.{word}"})
            if target:
                quote = value.value[0]
                self.change(value, f"{quote}{target.new}{quote}", "__all__")

    def text(
        self,
        node: libcst.SimpleString | libcst.Comment,
        index: Index,
        known: set[str],
    ) -> None:
        """Renames the citations and dotted paths in a string or comment."""
        if isinstance(node, libcst.SimpleString) and "b" in node.prefix.lower():
            return
        scope = (self.source.module, self.source.owner(node))
        after, reasons, unresolved = cite(
            node.value, index, self.chosen, scope, known
        )
        if after != node.value:
            self.change(node, after, "; ".join(sorted(set(reasons))))
        line = self.source.lines.get(node, 0)
        self.reviews += [
            f"{self.source.path}:{line}: unresolved citation `{item}`"
            for item in unresolved
        ]


def receivers(
    found: Sites, typed: Types | None
) -> dict[libcst.Name, Target | None]:
    """What mypy resolves each attribute of a member's old word to.

    mypy and libcst both list the accesses; within one line they are paired
    in column order, which both keep. An access mypy cannot type is a
    review line only in a file that names a class of the member's family.

    Args:
        found: The file's sites, for its targets and review lines.
        typed: The mypy types, when a target is a member.

    Returns:
        Attribute name node to the target its receiver's type belongs to.
    """
    source = found.source
    members = {
        word: [t for t in targets if t.kind == "member"]
        for word, targets in found.words.items()
    }
    members = {word: targets for word, targets in members.items() if targets}
    if typed is None or not members:
        return {}
    hits, unknown = typed.receivers(source.module, members)
    classes = {
        c for targets in members.values() for t in targets for c in t.family
    }
    mentioned = any(
        item.name in classes or item.name.rpartition(".")[0] in classes
        for names in source.names.values()
        for item in names
    )
    found.reviews += [
        f"{source.path}:{line}: mypy cannot type a receiver of an old name"
        for line in (unknown if mentioned else [])
    ]
    written: dict[tuple[str, int], list[libcst.Name]] = {}
    for node in source.lines:
        if isinstance(node, libcst.Attribute) and node.attr.value in members:
            key = (node.attr.value, source.lines[node.attr])
            written.setdefault(key, []).append(node.attr)
    return {
        name: hit
        for key, names in written.items()
        for name, hit in zip(
            sorted(names, key=source.columns.__getitem__),
            hits.get(key, []),
            strict=False,
        )
    }


def sites(
    source: Source,
    index: Index,
    chosen: list[Target],
    typed: Types | None,
    known: set[str],
) -> Sites:
    """The changes to one Python file.

    Args:
        source: The file.
        index: The repository index.
        chosen: The targets.
        typed: The mypy types, when a target is a member.
        known: Every dotted path a citation may resolve to.

    Returns:
        Its changes, refusals and review lines.
    """
    found = Sites(source, chosen)
    inferred = receivers(found, typed)
    for node in source.lines:
        if isinstance(node, libcst.Name) and node.value in found.words:
            found.name(node)
        elif isinstance(node, libcst.Attribute) and (
            node.attr.value in found.words
        ):
            found.attribute(node, inferred.get(node.attr))
        elif isinstance(node, libcst.ImportFrom):
            found.imports(node)
        elif isinstance(node, libcst.Import):
            for alias in node.names:
                names = chain(alias.name)
                found.segments(names, ".".join(item.value for item in names))
        elif isinstance(node, libcst.Call):
            found.calls(node)
        elif isinstance(node, libcst.Assign):
            found.exports(node)
        elif isinstance(node, libcst.SimpleString | libcst.Comment):
            found.text(node, index, known)
    return found


def resolved(
    index: Index,
    entries: dict[str, str],
    after: bool,
    typed: collections.abc.Callable[[], Types],
) -> tuple[list[Target], list[str]]:
    """Resolves every map entry to a target.

    Args:
        index: The repository index.
        entries: The map, old dotted path to new name.
        after: Whether the map is already applied (for `--verify`): the
            definition is then found under its new name, and the target is
            what an old reference would still say.
        typed: Builds the mypy types, on first need.

    Returns:
        The targets, and refusals.
    """
    chosen = []
    refusals = []
    for old, new in sorted(entries.items()):
        parent, word = old.rpartition(".")[::2]
        home = remap(parent, entries) if after else parent
        defined = f"{home}.{new}" if after else old
        kind = index.kinds.get(defined)
        why = single(new)
        if why:
            why = f"{old} -> {new}: the new name {why}"
        elif word in structure.EXEMPT:
            why = f"{old}: exempt, {structure.EXEMPT[word]}"
        elif kind is None:
            why = f"{old}: no such definition ({defined})"
        elif not after and (
            f"{parent}.{new}" in index.kinds or new in index.binds(parent)
        ):
            why = f"{old} -> {new}: {parent} already binds {new}"
        if why or kind is None:
            refusals.append(why)
            continue
        names = index.aliases(f"{home}.{word}")
        family: set[str] = set()
        if kind == "member":
            family, why = typed().family(home, new if after else word)
            refusals += [why] if why else []
            refusals += [
                f"{old} -> {new}: {info.fullname} has {new}"
                for info in ([] if after else typed().below(family))
                if info.get(new) is not None
            ]
            names = {
                alias
                for owner in family
                for alias in index.aliases(f"{owner}.{word}")
            }
        chosen.append(
            Target(
                f"{home}.{word}", new, kind, frozenset(names), frozenset(family)
            )
        )
    return chosen, refusals


def candidates(index: Index, chosen: list[Target]) -> dict[str, str]:
    """The files that spell an old word, which any reference must.

    Returns:
        Path to content, history and data files left out.
    """
    words = re.compile(r"\b(?:" + "|".join(t.word for t in chosen) + r")\b")
    found = {}
    for name in index.files:
        if name.startswith(HISTORY) or not name.endswith(
            (".py", *PROSE, *CONFIG)
        ):
            continue
        try:
            text = (index.root / name).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if words.search(text):
            found[name] = text
    return found


def prose(
    name: str, text: str, index: Index, chosen: list[Target], known: set[str]
) -> tuple[str, list[Edit], list[str]]:
    """The changes to one file that is not Python.

    Returns:
        The new text, one edit per changed line, and review lines.
    """
    scope = ("", "") if name.endswith(PROSE) else None
    new, reasons, unresolved = cite(text, index, chosen, scope, known)
    reason = "; ".join(sorted(set(reasons)))
    lines = zip(text.splitlines(), new.splitlines(), strict=True)
    edits = [
        Edit(name, number, old.strip(), line.strip(), reason)
        for number, (old, line) in enumerate(lines, start=1)
        if old != line
    ]
    reviews = [f"{name}: unresolved citation `{item}`" for item in unresolved]
    return new, edits, reviews


def plan(
    root: pathlib.Path, package: str, entries: dict[str, str], after: bool
) -> Plan:
    """Computes every change a map makes, or would still make.

    Args:
        root: The repository root.
        package: The top-level package.
        entries: The map.
        after: Whether the map is already applied (for `--verify`).

    Returns:
        The plan; nothing is written.
    """
    index = Index(root, package)
    cache: list[Types] = []

    def typed() -> Types:
        if not cache:
            cache.append(Types(index))
        return cache[0]

    chosen, refusals = resolved(index, entries, after, typed)
    result = Plan(refusals=refusals)
    if not chosen:
        return result
    known = set(index.kinds) | {name for item in chosen for name in item.names}
    found = candidates(index, chosen)
    manager = metadata.FullRepoManager(
        str(root),
        [str(root / name) for name in found if name.endswith(".py")],
        {metadata.FullyQualifiedNameProvider},
    )
    checker = typed() if any(t.kind == "member" for t in chosen) else None
    for name, text in found.items():
        if not name.endswith(".py"):
            new, edits, reviews = prose(name, text, index, chosen, known)
            result.edits += edits
            result.reviews += reviews
            if new != text:
                result.texts[name] = new
            continue
        source = Source(name, manager, root)
        changed = sites(source, index, chosen, checker, known)
        result.edits += changed.edits
        result.refusals += changed.refusals
        result.reviews += changed.reviews
        if changed.changes:
            tree = source.wrapper.module.visit(Writer(changed.changes))
            result.texts[name] = tree.code
    for target in chosen if not after else []:
        if target.kind == "module":
            path = pathlib.PurePosixPath(index.modules[target.old])
            folder = path.name == "__init__.py"
            old = path.parent if folder else path
            moved = old.with_name(target.new + ("" if folder else ".py"))
            result.moves[str(old)] = str(moved)
    return result


def tidy(root: pathlib.Path, result: Plan) -> None:
    """Formats the changed Python files where they now are."""
    moved = {
        old.removesuffix(".py"): new.removesuffix(".py")
        for old, new in result.moves.items()
    }
    touched = []
    for name in result.texts:
        path = name
        for old, new in moved.items():
            if path.startswith(f"{old}/") or path == f"{old}.py":
                path = new + path[len(old) :]
        if path.endswith(".py"):
            touched.append(path)
    for command in (
        ["format", "-q"],
        ["check", "-q", "--select", "I", "--fix"],
    ):
        if touched:
            subprocess.run(
                [sys.executable, "-m", "ruff", *command, *touched],
                cwd=root,
                check=False,
            )


def write(root: pathlib.Path, result: Plan, entries: dict[str, str]) -> None:
    """Writes a plan: the files, the module moves and the record."""
    for name, text in result.texts.items():
        (root / name).write_text(text, encoding="utf-8")
    for old, new in sorted(
        result.moves.items(), key=lambda item: -len(item[0])
    ):
        (root / old).rename(root / new)
    record = root / RENAMES
    if record.exists():
        text = record.read_text(encoding="utf-8")
        if HEADING not in text:
            text = text.rstrip("\n") + f"\n\n{HEADING}\n\n"
        lines = "".join(
            f"- `{old}` -> `{new}`\n" for old, new in entries.items()
        )
        record.write_text(text + lines, encoding="utf-8")
    tidy(root, result)


def verify(root: pathlib.Path, package: str, entries: dict[str, str]) -> int:
    """Proves a map is fully applied and the tree still checks.

    The files checked are the Python files that spell a new name; that
    choice only selects what ruff and mypy read.

    Returns:
        0 when nothing resolves to an old name and every check passes.
    """
    left = plan(root, package, entries, after=True)
    failures = list(left.refusals)
    failures += [
        f"{edit.path}:{edit.line}: still names an old definition:"
        f" {edit.before} ({edit.reason})"
        for edit in left.edits
    ]
    print(*(f"review: {line}" for line in left.reviews), sep="\n")
    pattern = re.compile(r"\b(?:" + "|".join(sorted(entries.values())) + r")\b")
    touched = [
        name
        for name in listing(root)
        if name.endswith(".py")
        and not name.startswith(HISTORY)
        and pattern.search((root / name).read_text(encoding="utf-8"))
    ]
    modules = [name for name in touched if name.startswith(f"{package}/")]
    checks = {
        "ruff": [sys.executable, "-m", "ruff", "check", "-q", *touched],
        "mypy": [sys.executable, "-m", "mypy", "--strict", *modules],
    }
    if not modules:
        checks.pop("mypy")
    pyproject = root / "pyproject.toml"
    if pyproject.exists() and "importlinter" in pyproject.read_text("utf-8"):
        lint = pathlib.Path(sys.executable).with_name("lint-imports")
        checks["import-linter"] = [str(lint)]
    for tool, command in checks.items():
        done = subprocess.run(
            command, cwd=root, capture_output=True, text=True, check=False
        )
        if done.returncode:
            failures.append(f"{tool} failed:\n{done.stdout}{done.stderr}")
    print(*failures, sep="\n")
    return 1 if failures else 0


def main(arguments: list[str] | None = None) -> int:
    """Applies, previews or verifies a rename map.

    Args:
        arguments: Command-line arguments; `sys.argv` when None.

    Returns:
        0 on success; 1 when the map is refused or the proof fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lane", nargs="?")
    parser.add_argument("--map", type=pathlib.Path)
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--package", default="scmle")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    options = parser.parse_args(arguments)
    if options.map is None and options.lane is None:
        parser.error("name a lane or pass --map")
    entries = read(options.map or MAPS / f"{options.lane}.toml")
    root = options.root.resolve()
    if options.verify:
        return verify(root, options.package, entries)
    result = plan(root, options.package, entries, after=False)
    for edit in result.edits:
        print(
            f"{edit.path}:{edit.line}: {edit.before} -> {edit.after}"
            f"  ({edit.reason})"
        )
    for old, new in result.moves.items():
        print(f"move {old} -> {new}")
    print(*(f"review: {line}" for line in result.reviews), sep="\n")
    if result.refusals:
        print(*(f"refused: {line}" for line in result.refusals), sep="\n")
        return 1
    if not options.dry_run:
        write(root, result, entries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
