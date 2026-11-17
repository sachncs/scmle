"""Moves modules by the layer map and proves the change is rename-only.

`scripts/layers.toml` records the moves of register phase 3 (DES-13), one
step per target package. This applies one step: it moves the files (`git
mv` in a checkout), creates the new packages, deletes the packages a step
empties, and rewrites every reference to a moved module in tracked text:
imports at any depth, dotted paths in code and strings, file paths, config
(`pyproject.toml`, `setup.cfg`, `gates.toml`, workflows, `Makefile`),
`docs/` and `website/`. It then formats the touched Python files and
appends the moves to `docs/RENAMES.md`.

Data files (`*.json`, `*.jsonl`) are records their tools regenerate, and
history (`CHANGELOG.md`, `docs/RENAMES.md`, `plan/`) and the map itself are
never rewritten.

`--verify` is the proof. For every changed Python file it applies the map
to the committed file's syntax tree (import targets, the names imports
bind, dotted paths in strings) and requires the result to equal the new
tree, comparing imports as sets per scope because sorting reorders them.
Every other changed file must equal the committed text with the map
applied. A file that differs is a hand edit and fails the proof unless it
is named with `--hand`. A module that binds one name to two imports fails.

    uv run python scripts/move.py runtime
    uv run python scripts/move.py runtime --verify [--hand FILE ...]
    uv run python scripts/move.py --all --dry-run --into DIR
"""

import argparse
import ast
import collections
import pathlib
import re
import shutil
import subprocess
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAP = pathlib.Path(__file__).with_name("layers.toml")
RENAMES = "docs/RENAMES.md"
HEADING = "## Module moves"
HISTORY = (
    "CHANGELOG.md",
    RENAMES,
    "plan/",
    "tests/fixtures/public_api.json",
    "scripts/layers.toml",
)
DATA = (".json", ".jsonl", ".pq", ".png", ".svg", ".csv", ".lock", ".typed")
# Prose formats, where a package-relative dotted path such as `core.retry`
# names a module; in code it may be an attribute chain, so code rewrites
# only the full `scmle.` form.
PROSE = (".md", ".astro", ".html", ".txt", ".mjs", ".ts")


class ConflictError(Exception):
    """A rename the codemod cannot make safely; the file needs a hand edit."""


def read(
    path: pathlib.Path,
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Reads the layer map.

    Args:
        path: The map file.

    Returns:
        Step name to its moves (old dotted path to new), in map order, and
        new package name to its docstring.
    """
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    steps = {
        step["name"]: dict(step["moves"]) for step in document.get("step", [])
    }
    return steps, dict(document.get("packages", {}))


def rename(module: str, moves: dict[str, str]) -> str:
    """Applies the longest matching move to a dotted name."""
    matches = [
        old for old in moves if module == old or module.startswith(f"{old}.")
    ]
    if not matches:
        return module
    old = max(matches, key=len)
    return moves[old] + module[len(old) :]


def forms(
    moves: dict[str, str], prose: bool
) -> list[tuple[re.Pattern[str], str]]:
    """The textual forms of each move, longest first.

    Args:
        moves: Old dotted path to new.
        prose: Whether package-relative dotted paths are rewritten too.

    Returns:
        Pattern and replacement pairs.
    """
    found = []
    for old in sorted(moves, key=len, reverse=True):
        new = moves[old]
        relative, target = old.split(".", 1)[1], new.split(".", 1)[1]
        parent, base = old.rsplit(".", 1)
        home, name = new.rsplit(".", 1)
        if base == name:
            # An import written as text (a script in a string, a code block
            # in a doc); a real import statement was already rewritten. In
            # an unparsed string the line break before it reads `\n`.
            text = rf"from {re.escape(parent)} import {re.escape(base)}"
            found.append(
                (
                    rf"(?:(?<![\w.])|(?<=\\n)){text}(?![\w,])",
                    f"from {home} import {name}",
                )
            )
        found.append((rf"(?<![\w.]){re.escape(old)}(?!\w)", new))
        slash, moved = old.replace(".", "/"), new.replace(".", "/")
        found.append((rf"(?<![\w./]){re.escape(slash)}(?=\.py\b|/)", moved))
        loose, placed = relative.replace(".", "/"), target.replace(".", "/")
        found.append((rf"(?<![\w./-]){re.escape(loose)}(?=\.py\b|/)", placed))
        if prose and "." in relative:
            found.append((rf"(?<![\w./]){re.escape(relative)}(?!\w)", target))
    return [
        (re.compile(pattern), replacement) for pattern, replacement in found
    ]


def translate(text: str, moves: dict[str, str], *, prose: bool) -> str:
    """Rewrites every textual reference to a moved module.

    All forms are matched in one pass, so a rewritten path is never matched
    again by a shorter form.
    """
    patterns = forms(moves, prose=prose)
    combined = re.compile(
        "|".join(f"({pattern.pattern})" for pattern, _ in patterns)
    )

    def swap(match: re.Match[str]) -> str:
        return patterns[match.lastindex - 1][1] if match.lastindex else match[0]

    return combined.sub(swap, text)


def modules(paths: list[str]) -> set[str]:
    """Dotted module names of the package's Python files."""
    found = set()
    for path in paths:
        if path.startswith("scmle/") and path.endswith(".py"):
            parts = path.removesuffix(".py").split("/")
            found.add(
                ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
            )
    return found


def bound(tree: ast.AST) -> collections.Counter[str]:
    """How many times each name is bound other than by an import."""
    found: collections.Counter[str] = collections.Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            found[node.id] += 1
        elif isinstance(node, ast.arg):
            found[node.arg] += 1
        elif isinstance(
            node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        ):
            found[node.name] += 1
    return found


def pairs(
    node: ast.Import | ast.ImportFrom, moves: dict[str, str], known: set[str]
) -> list[tuple[str, str]]:
    """The (target, bound name) pairs of an import, after the moves.

    `from a import b` targets `a.b` whether `b` is a module or a name; when
    `a.b` is a moved module the bound name follows its new basename.
    """
    if isinstance(node, ast.Import):
        return [
            (rename(alias.name, moves), alias.name.split(".")[0])
            for alias in node.names
        ]
    found = []
    for alias in node.names:
        full = f"{node.module}.{alias.name}" if node.module else alias.name
        target = rename(full, moves)
        name = target.rsplit(".", 1)[-1] if full in known else alias.name
        found.append((target, name))
    return found


def renamed(
    tree: ast.AST, moves: dict[str, str], known: set[str]
) -> dict[str, str]:
    """Names whose import binds a different name after the moves, old to new."""
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for (_, name), alias in zip(
                pairs(node, moves, known), node.names, strict=True
            ):
                if name != alias.name:
                    found[alias.name] = name
    return found


def edits(text: str, moves: dict[str, str], known: set[str]) -> str:
    """Rewrites the `from ... import` statements that name a moved module.

    Args:
        text: Python source.
        moves: Old dotted path to new.
        known: Every module of the package before the moves.

    Returns:
        The source with each changed statement replaced, split by new parent
        module.
    """
    lines = text.splitlines(keepends=True)
    changes = []
    for node in ast.walk(ast.parse(text)):
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] == "scmle"
        ):
            continue
        groups: dict[str, list[str]] = {}
        for target, name in pairs(node, moves, known):
            groups.setdefault(target.rsplit(".", 1)[0], []).append(name)
        if list(groups) == [node.module]:
            continue
        first, last = node.lineno - 1, int(node.end_lineno or node.lineno) - 1
        indent = lines[first][: len(lines[first]) - len(lines[first].lstrip())]
        tail = (
            lines[last]
            .encode()[int(node.end_col_offset or 0) :]
            .decode()
            .rstrip("\n")
        )
        statements = [
            f"{indent}from {parent} import {', '.join(names)}{tail}\n"
            for parent, names in groups.items()
        ]
        changes.append((first, last, statements))
    for first, last, statements in sorted(changes, reverse=True):
        lines[first : last + 1] = statements
    return "".join(lines)


def rebind(text: str, renames: dict[str, str], path: str) -> str:
    """Renames the uses of every name an import now binds differently.

    Raises:
        ConflictError: When the old name is also bound some other way, or the
            new name is already in use.
    """
    if not renames:
        return text
    tree = ast.parse(text)
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    others = bound(tree)
    for old, new in renames.items():
        if others[old] or new in used or others[new]:
            raise ConflictError(
                f"{path}: renaming {old} to {new} needs a hand edit"
            )
    lines = [line.encode() for line in text.splitlines(keepends=True)]
    targets = sorted(
        (
            (node.lineno - 1, node.col_offset, node.id)
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id in renames
        ),
        reverse=True,
    )
    for row, column, name in targets:
        line = lines[row]
        lines[row] = (
            line[:column]
            + renames[name].encode()
            + line[column + len(name.encode()) :]
        )
    return b"".join(lines).decode()


def git(root: pathlib.Path, *arguments: str) -> str:
    """Runs git in a checkout and returns its output."""
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def listing(root: pathlib.Path) -> list[str]:
    """Tracked files of a checkout, or every file of a plain copy."""
    if (root / ".git").exists():
        return git(root, "ls-files").split()
    skip = {".git", ".venv", "__pycache__", "node_modules", "runs"}
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and not skip & set(path.relative_to(root).parts)
    )


def eligible(path: str) -> bool:
    """Whether the codemod rewrites a file."""
    return not path.startswith(HISTORY) and not path.endswith(DATA)


def location(root: pathlib.Path, dotted: str) -> pathlib.Path:
    """The file or folder of a dotted module path."""
    folder = root / dotted.replace(".", "/")
    return folder if folder.is_dir() else folder.with_suffix(".py")


def prune(
    root: pathlib.Path, folders: set[pathlib.Path], *, tracked: bool
) -> None:
    """Deletes the packages a step left with nothing but a docstring."""
    for folder in folders:
        rest = [
            item.name for item in folder.iterdir() if item.name != "__pycache__"
        ]
        init = folder / "__init__.py"
        if rest != ["__init__.py"]:
            continue
        if not docstring(str(init), init.read_text(encoding="utf-8")):
            raise SystemExit(f"{init} has code; delete the package by hand")
        if tracked:
            git(root, "rm", "-q", str(init.relative_to(root)))
        shutil.rmtree(folder)


def relocate(
    root: pathlib.Path, moves: dict[str, str], notes: dict[str, str]
) -> list[str]:
    """Moves the files of one step and creates and deletes packages.

    Returns:
        Files the step created, relative to the root.
    """
    tracked = (root / ".git").exists()
    created = []
    emptied = set()
    for old, new in moves.items():
        source = location(root, old)
        target = root / new.replace(".", "/")
        target = target if source.is_dir() else target.with_suffix(".py")
        if target.exists():
            raise SystemExit(f"{target} exists; {old} cannot move there")
        for parent in reversed(target.relative_to(root).parents[:-1]):
            init = root / parent / "__init__.py"
            if not init.exists():
                init.parent.mkdir(exist_ok=True)
                note = notes.get(parent.name, parent.name)
                init.write_text(f'"""{note}"""\n', encoding="utf-8")
                created.append(str(init.relative_to(root)))
        if tracked:
            git(
                root,
                "mv",
                str(source.relative_to(root)),
                str(target.relative_to(root)),
            )
        else:
            shutil.move(source, target)
        emptied.add(source.parent)
    prune(root, emptied, tracked=tracked)
    if tracked and created:
        git(root, "add", *created)
    return created


def record(root: pathlib.Path, moves: dict[str, str]) -> None:
    """Appends the moves to the renames file."""
    path = root / RENAMES
    text = path.read_text(encoding="utf-8")
    if HEADING not in text:
        text = text.rstrip("\n") + (
            f"\n\n{HEADING}\n\nModules moved between packages by the layer map "
            "(`scripts/layers.toml`, register phase 3). Each entry is "
            "`old -> new`; the names inside a module are unchanged.\n"
        )
    lines = "".join(f"- `{old}` -> `{new}`\n" for old, new in moves.items())
    path.write_text(text.rstrip("\n") + "\n" + lines, encoding="utf-8")


def rewrite(
    name: str, text: str, moves: dict[str, str], known: set[str]
) -> str:
    """One file with every reference to a moved module rewritten.

    Raises:
        ConflictError: When a rename would bind one name twice or clash.
    """
    result = text
    if name.endswith(".py"):
        renames = renamed(ast.parse(text), moves, known)
        result = rebind(edits(text, moves, known), renames, name)
    result = translate(result, moves, prose=name.endswith(PROSE))
    if name.endswith(".py") and result != text:
        double = set(doubles(ast.parse(result))) - set(doubles(ast.parse(text)))
        if double:
            raise ConflictError(f"{name}: binds {sorted(double)} twice")
    return result


def apply(
    root: pathlib.Path, moves: dict[str, str], notes: dict[str, str]
) -> tuple[list[str], list[str]]:
    """Applies one step to a tree.

    Returns:
        The Python files the step changed, and one line per file it left
        unchanged because the rename needs a hand edit.
    """
    before = set(listing(root))
    known = modules(sorted(before))
    relocate(root, moves, notes)
    touched = []
    refused = []
    for name in listing(root):
        path = root / name
        if not eligible(name) or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        try:
            result = rewrite(name, text, moves, known)
        except ConflictError as error:
            refused.append(str(error))
            continue
        if result == text:
            continue
        if name.endswith(".py"):
            touched.append(name)
        path.write_text(result, encoding="utf-8")
    record(root, moves)
    if touched:
        for command in (
            ["format", "-q"],
            ["check", "-q", "--select", "I", "--fix"],
        ):
            subprocess.run(
                [sys.executable, "-m", "ruff", *command, *touched],
                cwd=root,
                check=False,
            )
    return touched, refused


def scopes(
    tree: ast.AST, moves: dict[str, str], known: set[str]
) -> dict[str, collections.Counter[tuple[str, str]]]:
    """Imports per enclosing definition, after the moves, as multisets."""
    found: dict[str, collections.Counter[tuple[str, str]]] = {}

    def visit(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Import | ast.ImportFrom):
                bucket = found.setdefault(scope, collections.Counter())
                bucket.update(pairs(child, moves, known))
            inner = scope
            if isinstance(
                child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
            ):
                inner = f"{scope}.{child.name}"
            visit(child, inner)

    visit(tree, "")
    return found


def strip(tree: ast.AST, renames: dict[str, str]) -> str:
    """The tree without imports and with renamed names, as source."""
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if isinstance(block, list) and block:
                kept = [
                    item
                    for item in block
                    if not isinstance(item, ast.Import | ast.ImportFrom)
                ]
                setattr(node, field, kept or [ast.Pass()])
        if isinstance(node, ast.Name) and node.id in renames:
            node.id = renames[node.id]
    return ast.unparse(tree)


def doubles(tree: ast.AST) -> list[str]:
    """Names one scope binds to two different imports."""
    found = []
    for scope, bucket in scopes(tree, {}, set()).items():
        targets: dict[str, set[str]] = {}
        for target, name in bucket:
            targets.setdefault(name, set()).add(target)
        found += [
            f"{scope or '<module>'}: {name}"
            for name, items in targets.items()
            if len(items) > 1 and name != "scmle"
        ]
    return found


def same(old: str, new: str, moves: dict[str, str], known: set[str]) -> bool:
    """Whether a new Python file is the old one with the map applied."""
    before, after = ast.parse(old), ast.parse(new)
    renames = renamed(before, moves, known)
    if scopes(before, moves, known) != scopes(after, {}, set()):
        return False
    text = translate(strip(before, renames), moves, prose=False)
    return text == strip(after, {})


def source(name: str, moves: dict[str, str]) -> str:
    """The committed path of a file, before the moves."""
    path = pathlib.PurePosixPath(name)
    dotted = str(path.with_suffix("")).replace("/", ".")
    back = [
        old
        for old, new in moves.items()
        if dotted == new or dotted.startswith(f"{new}.")
    ]
    if not name.startswith("scmle/") or not back:
        return name
    old = max(back, key=lambda item: len(moves[item]))
    return (old + dotted[len(moves[old]) :]).replace(".", "/") + path.suffix


def docstring(name: str, text: str) -> bool:
    """Whether a file is a package marker holding at most a docstring."""
    if not name.endswith("__init__.py"):
        return False
    body = ast.parse(text).body
    return not body or (len(body) == 1 and isinstance(body[0], ast.Expr))


def check(
    root: pathlib.Path,
    name: str,
    old: str,
    moves: dict[str, str],
    known: set[str],
) -> list[str]:
    """Why one file is not its committed version with the map applied.

    Returns:
        One line per failure; empty when the file is a rename.
    """
    try:
        text = (root / name).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    committed = subprocess.run(
        ["git", "show", f"HEAD:{old}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if committed.returncode:
        return [] if docstring(name, text) else [f"{name}: new file"]
    before = committed.stdout
    if before == text:
        return []
    if not name.endswith(".py"):
        moved = translate(before, moves, prose=name.endswith(PROSE))
        return [] if moved == text else [f"{name}: not a rename"]
    double = set(doubles(ast.parse(text))) - set(doubles(ast.parse(before)))
    failures = [f"{name}: binds {item} twice" for item in sorted(double)]
    if not same(before, text, moves, known):
        failures.append(f"{name}: not a rename")
    return failures


def verify(root: pathlib.Path, moves: dict[str, str], hand: set[str]) -> int:
    """Proves the working tree is HEAD with one step applied.

    Returns:
        0 when every change is the map applied (or a named hand edit).
    """
    committed = git(root, "ls-tree", "-r", "--name-only", "HEAD").split()
    known = modules(committed)
    origin = {name: source(name, moves) for name in listing(root)}
    failures = []
    for name, old in origin.items():
        if eligible(name) and name != RENAMES:
            failures += check(root, name, old, moves, known)
    for old in sorted(set(committed) - set(origin.values())):
        if not docstring(old, git(root, "show", f"HEAD:{old}")):
            failures.append(f"{old}: deleted")
    unexcused = [line for line in failures if line.split(":")[0] not in hand]
    for line in failures:
        print(("hand edit: " if line not in unexcused else "") + line)
    return 1 if unexcused else 0


def main(arguments: list[str] | None = None) -> int:
    """Applies or verifies steps of the layer map.

    Returns:
        0 on success, 1 when the proof fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", nargs="?")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--hand", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--into", type=pathlib.Path)
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--map", type=pathlib.Path, default=MAP)
    options = parser.parse_args(arguments)
    steps, notes = read(options.map)
    chosen = list(steps) if options.all else [options.step]
    if any(step not in steps for step in chosen):
        parser.error(f"choose a step from: {', '.join(steps)}")
    root = options.root
    if options.dry_run:
        if options.into is None:
            parser.error("--dry-run needs --into DIR")
        git(root, "checkout-index", "-a", f"--prefix={options.into.resolve()}/")
        root = options.into
    if options.verify:
        return verify(root, steps[options.step], set(options.hand))
    failed = 0
    for step in chosen:
        touched, refused = apply(root, steps[step], notes)
        print(f"{step}: {len(touched)} Python files rewritten")
        print(*(f"  hand edit needed: {line}" for line in refused), sep="\n")
        failed |= bool(refused)
    return failed


if __name__ == "__main__":
    sys.exit(main())
