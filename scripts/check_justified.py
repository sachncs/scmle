"""Lists functions and classes that nothing justifies.

Usage:
    python scripts/check_justified.py [--json PATH] [--strict]
        [--fail-on SHAPE ...]

A static, name-based check of the rule in plan.md section 3a. It finds:

* unreferenced: defined in `scmle/`, named nowhere else in `scmle/` or the
  other source trees, and not a dunder method or an override;
* tested only: named nowhere in `scmle/` but named in tests, benchmarks,
  examples or scripts;
* passthrough: a function whose whole body forwards its own arguments to one
  call;
* getter: a method whose whole body returns one attribute of `self`;
* lone method: a class with no base, no state and exactly one public method;
* lone interface: an abstract class or protocol with at most one subclass in
  `scmle/`;
* single use: a short module-level function called from exactly one place.

Limits, so the output is read correctly. The check is by name, so two
unrelated symbols that share a name hide each other: it under-reports
unreferenced symbols and never over-reports them. It cannot see names built at
run time (`getattr`, registries, plugin entry points); a symbol found only
through a string constant counts as referenced. A definition with a decorator
other than `staticmethod` or `classmethod` (pydantic validators, registries,
fixtures) is treated as called by its framework and is not reported as
unreferenced. The protocol check counts a class as an implementer if it defines
every public method name the protocol declares, which is crude. Every result is
a candidate for
a human or a builder to judge, not a verdict. `--strict` exits 1 if any
candidate remains.
"""

import argparse
import ast
import collections
import fnmatch
import json
import pathlib
import sys

PACKAGE = "scmle"
OTHER = ("tests", "benchmarks", "examples", "scripts")
SHAPES = (
    "unreferenced",
    "tested only",
    "passthrough",
    "getter",
    "lone method",
    "lone interface",
    "single use",
)
INTERFACE_BASES = frozenset({"ABC", "Protocol"})
SINGLE_USE_LINES = 5
MAXIMUM_NAME_REFERENCE_LENGTH = 120
ENTRY_FIELDS = 3
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def parse(path: pathlib.Path) -> ast.Module | None:
    """Parses one source file.

    Args:
        path: File to parse.

    Returns:
        The syntax tree, or None if the file cannot be parsed.
    """
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return None


def trees(root: pathlib.Path, folder: str) -> dict[pathlib.Path, ast.Module]:
    """Parses every Python file under a folder.

    Args:
        root: Repository root.
        folder: Folder name below the root.

    Returns:
        Mapping from path to syntax tree for the files that parse.
    """
    found = {}
    for path in sorted((root / folder).rglob("*.py")):
        tree = parse(path)
        if tree is not None:
            found[path] = tree
    return found


def count(sources: dict[pathlib.Path, ast.Module]) -> collections.Counter:
    """Counts how often each name is used.

    Counts loaded names, attribute names, imported names and string constants
    that look like identifiers or dotted paths. Definitions do not count.

    Args:
        sources: Parsed files.

    Returns:
        Mapping from name to use count.
    """
    used: collections.Counter = collections.Counter()
    for tree in sources.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used[node.id] += 1
            elif isinstance(node, ast.Attribute):
                used[node.attr] += 1
            elif isinstance(node, ast.alias):
                used[node.name.split(".")[-1]] += 1
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                text = node.value.strip()
                tail = text.replace(":", ".").split(".")[-1]
                if (
                    tail.isidentifier()
                    and len(text) < MAXIMUM_NAME_REFERENCE_LENGTH
                ):
                    used[tail] += 1
    return used


def strip(body: list[ast.stmt]) -> list[ast.stmt]:
    """Drops a leading docstring from a body.

    Args:
        body: Statements of a function or class.

    Returns:
        The statements after any docstring.
    """
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def forwards(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether the body only forwards the function's own arguments.

    Args:
        node: Function to test.

    Returns:
        True if the body is one call whose positional arguments are the
        function's own parameters, in order, and whose keywords only repeat
        a parameter under its own name.
    """
    body = strip(node.body)
    if len(body) != 1 or not isinstance(body[0], ast.Return | ast.Expr):
        return False
    call = body[0].value
    if not isinstance(call, ast.Call):
        return False
    parameters = [
        argument.arg
        for argument in node.args.posonlyargs + node.args.args
        if argument.arg not in ("self", "cls")
    ]
    passed = [
        argument.id for argument in call.args if isinstance(argument, ast.Name)
    ]
    plain = all(
        keyword.arg is None
        or (
            isinstance(keyword.value, ast.Name)
            and keyword.value.id == keyword.arg
        )
        for keyword in call.keywords
    )
    return (
        plain
        and len(passed) == len(call.args)
        and len(passed) >= 1
        and passed == parameters[: len(passed)]
    )


def returns(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether a method only returns one attribute of `self`.

    Args:
        node: Method to test.

    Returns:
        True for `return self.name` as the whole body.
    """
    body = strip(node.body)
    return (
        len(body) == 1
        and isinstance(body[0], ast.Return)
        and isinstance(body[0].value, ast.Attribute)
        and isinstance(body[0].value.value, ast.Name)
        and body[0].value.value.id == "self"
    )


def name(node: ast.expr) -> str:
    """Names a base class or decorator expression.

    Args:
        node: Expression from a class's base list or decorator list.

    Returns:
        The last name in the expression, or an empty string.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return name(node.value)
    if isinstance(node, ast.Call):
        return name(node.func)
    return ""


def declared(path: pathlib.Path) -> list[tuple[str, str]]:
    """Reads the names declared as used through run-time-built names.

    Each non-comment line is `MODULE_GLOB NAME_GLOB  reason`. The reason is
    required so every entry says where the name is built.

    Args:
        path: File to read; a missing file means no declarations.

    Returns:
        Pairs of module glob and name glob.
    """
    if not path.exists():
        return []
    patterns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split(None, ENTRY_FIELDS - 1)
        if line.startswith("#") or not parts:
            continue
        if len(parts) < ENTRY_FIELDS:
            raise SystemExit(f"{path}: entry without a reason: {line!r}")
        patterns.append((parts[0], parts[1]))
    return patterns


def survey(
    package: dict[pathlib.Path, ast.Module],
) -> tuple[collections.Counter, dict[str, set[str]]]:
    """Counts subclasses and records which classes define each method name.

    Args:
        package: Parsed files of the package.

    Returns:
        Subclass counts by base name, and method name to defining classes.
    """
    subclasses: collections.Counter = collections.Counter()
    methods: dict[str, set[str]] = collections.defaultdict(set)
    for tree in package.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for parent in node.bases:
                    subclasses[name(parent)] += 1
                for member in node.body:
                    if isinstance(member, FUNCTIONS):
                        methods[member.name].add(node.name)
    return subclasses, methods


class Finder:
    """Collects candidates from the package under the 3a rule."""

    def __init__(
        self,
        root: pathlib.Path,
        production: collections.Counter,
        elsewhere: collections.Counter,
        subclasses: collections.Counter,
        methods: dict[str, set[str]],
        patterns: list[tuple[str, str]],
    ) -> None:
        """Stores the counts the tests need.

        Args:
            root: Repository root, for relative paths.
            production: Name use counts inside the package.
            elsewhere: Name use counts in the other source trees.
            subclasses: Subclass counts by base name.
            methods: Method name to the classes that define it.
            patterns: Module and name globs declared as dynamically used.
        """
        self.root = root
        self.patterns = patterns
        self.production = production
        self.elsewhere = elsewhere
        self.subclasses = subclasses
        self.methods = methods
        self.found: list[dict] = []

    def add(
        self, shape: str, path: pathlib.Path, node: ast.AST, kind: str
    ) -> None:
        """Records one candidate.

        Args:
            shape: One of `SHAPES`.
            path: File that holds the definition.
            node: Definition node.
            kind: "function", "method" or "class".
        """
        self.found.append(
            {
                "shape": shape,
                "module": str(path.relative_to(self.root)),
                "name": node.name,
                "kind": kind,
                "line": node.lineno,
                "production_refs": self.production[node.name],
                "test_refs": self.elsewhere[node.name],
            }
        )

    def implementers(self, node: ast.ClassDef) -> int:
        """Counts classes that satisfy a protocol structurally.

        A class satisfies the protocol if it defines every public method the
        protocol declares. The protocol itself is not counted.

        Args:
            node: The protocol class.

        Returns:
            Number of classes that define all of its public method names.
        """
        wanted = {
            m.name
            for m in node.body
            if isinstance(m, FUNCTIONS) and not m.name.startswith("__")
        }
        if not wanted:
            return 0
        classes = set.intersection(*(self.methods[label] for label in wanted))
        return len(classes - {node.name})

    def dynamic(self, path: pathlib.Path, label: str) -> bool:
        """Whether a name is declared as used by a name built at run time.

        Args:
            path: File that holds the definition.
            label: Defined name.

        Returns:
            True if a line of the dynamic-use file matches the module and name.
        """
        relative = str(path.relative_to(self.root))
        return any(
            fnmatch.fnmatch(relative, module) and fnmatch.fnmatch(label, glob)
            for module, glob in self.patterns
        )

    def judge(self, node: ast.AST, kind: str, path: pathlib.Path) -> None:
        """Applies the reference and forwarding tests to one definition.

        Args:
            node: Function, method or class definition.
            kind: "function", "method" or "class".
            path: File that holds the definition.
        """
        label = node.name
        if label.startswith("__") and label.endswith("__"):
            return
        override = kind == "method" and len(self.methods[label]) > 1
        decorators = {name(item) for item in node.decorator_list}
        registered = bool(decorators - {"staticmethod", "classmethod"})
        unused = self.production[label] == 0 and not override and not registered
        if unused and label != "main" and not self.dynamic(path, label):
            shape = "tested only" if self.elsewhere[label] else "unreferenced"
            self.add(shape, path, node, kind)
        if kind != "class" and forwards(node):
            self.add("passthrough", path, node, kind)

    def visit(self, path: pathlib.Path, tree: ast.Module) -> None:
        """Examines the top-level definitions of one file.

        Args:
            path: File being examined.
            tree: Its syntax tree.
        """
        for node in tree.body:
            if isinstance(node, FUNCTIONS):
                self.judge(node, "function", path)
                short = len(strip(node.body)) <= SINGLE_USE_LINES
                if self.production[node.name] == 1 and short:
                    self.add("single use", path, node, "function")
            elif isinstance(node, ast.ClassDef):
                self.inspect(node, path)

    def inspect(self, node: ast.ClassDef, path: pathlib.Path) -> None:
        """Examines one class and its methods.

        Args:
            node: Class definition.
            path: File that holds it.
        """
        self.judge(node, "class", path)
        members = [m for m in strip(node.body) if not isinstance(m, ast.Pass)]
        functions = [m for m in members if isinstance(m, FUNCTIONS)]
        stateless = all(not m.name.startswith("__") for m in functions)
        if (
            not node.bases
            and not node.decorator_list
            and len(functions) == 1
            and len(functions) == len(members)
            and stateless
        ):
            self.add("lone method", path, node, "class")
        bases = {name(parent) for parent in node.bases}
        metaclass = any(name(item.value) == "ABCMeta" for item in node.keywords)
        if "Protocol" in bases:
            if self.implementers(node) <= 1:
                self.add("lone interface", path, node, "class")
        elif (bases & INTERFACE_BASES or metaclass) and (
            self.subclasses[node.name] <= 1
        ):
            self.add("lone interface", path, node, "class")
        for member in functions:
            self.judge(member, "method", path)
            properties = {name(item) for item in member.decorator_list}
            if returns(member) and "property" not in properties:
                self.add("getter", path, member, "method")


def main() -> int:
    """Runs the check and prints a summary.

    Returns:
        1 if `--strict` is set and any candidate remains, or if a candidate
        of a shape named by `--fail-on` remains; otherwise 0.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--json", type=pathlib.Path, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--fail-on",
        action="append",
        default=[],
        choices=SHAPES,
        help="exit 1 if any candidate of this shape remains (repeatable)",
    )
    parser.add_argument(
        "--dynamic",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("justified_dynamic.txt"),
    )
    options = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    package = trees(root, PACKAGE)
    others: dict[pathlib.Path, ast.Module] = {}
    for folder in OTHER:
        others.update(trees(root, folder))
    subclasses, methods = survey(package)
    finder = Finder(
        root,
        count(package),
        count(others),
        subclasses,
        methods,
        declared(options.dynamic),
    )
    for path, tree in package.items():
        finder.visit(path, tree)
    tally = collections.Counter(item["shape"] for item in finder.found)
    for shape in SHAPES:
        print(f"{shape:16} {tally[shape]:6}")
    print(f"{'total':16} {len(finder.found):6}")
    if options.json is not None:
        options.json.write_text(
            json.dumps(finder.found, indent=1) + "\n", encoding="utf-8"
        )
    blocking = [
        item for item in finder.found if item["shape"] in options.fail_on
    ]
    return 1 if (options.strict and finder.found) or blocking else 0


if __name__ == "__main__":
    sys.exit(main())
