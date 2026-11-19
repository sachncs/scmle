"""Every deserialisation call site in a package, and whether it is safe.

Unpickling executes code, so a byte that reaches a deserialiser must first be
authenticated (SEC-1). This module finds every call to a deserialiser
(`Sink`) in the source of a package, statically, without importing it, and
gives each call (`Site`) one verdict:

* `authenticated`: the call reads an in-memory buffer (`io.BytesIO(x)`) of
  bytes `x` that the same function took from an authenticating reader
  (`READERS`): `envelope.read` (HMAC over a digest), `envelope.fetch` and
  `bundle.archive.resolve` (a digest the caller's signed manifest records).
  The bytes checked are the bytes loaded, so nothing is re-read from disk.
* `forwarded`: the buffer wraps a parameter, and every call of the function
  in the package passes bytes from an authenticating reader (one level).
* `roundtrip`: the call loads a buffer the same function created empty and
  dumped into; the bytes never left the process.
* `unauthenticated`: anything else, such as a path, a file handle or bytes
  of unknown origin. Each is a defect.

A call the library itself makes safe is not a site: `numpy.load` without
`allow_pickle=True`, `torch.load(weights_only=True)`, `yaml.load` with a
safe loader and `keras` loading with `safe_mode` left on.

The static check proves where the bytes come from; that each reader refuses
unsigned or tampered input is proved per path by
`tests/unit/test_deserialisation_paths.py`.

    python -m scmle.security.deserialisation scmle
"""

import argparse
import ast
from collections.abc import Iterator
import dataclasses
import json
import pathlib
import sys

AUTHENTICATED = "authenticated"
FORWARDED = "forwarded"
ROUNDTRIP = "roundtrip"
UNAUTHENTICATED = "unauthenticated"
VERDICTS = (AUTHENTICATED, FORWARDED, ROUNDTRIP, UNAUTHENTICATED)
BUFFER = "io.BytesIO"
# Authenticating readers, and which element of the result holds the bytes
# (None: the result itself).
READERS: dict[str, int | None] = {
    "scmle.envelope.read": 1,
    "scmle.envelope.fetch": None,
    "scmle.artifacts.bundle.archive.resolve": None,
}
DUMPS = ("joblib.dump", "pickle.dump", "cloudpickle.dump", "dill.dump")
SAFE_LOADERS = ("SafeLoader", "CSafeLoader", "BaseLoader")
MODULE = "<module>"
SOURCES = ("filename", "file", "data", "f", "path", "stream")


def value(node: ast.expr) -> object:
    """A constant's value, or the last name of an attribute or name.

    Args:
        node: The expression.

    Returns:
        The literal value, the trailing identifier, or None.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


@dataclasses.dataclass(frozen=True)
class Sink:
    """A function that turns bytes into objects and may run code doing so.

    Attributes:
        name: The qualified name, such as `joblib.load`.
        keyword: A keyword argument that can make the call safe, if any.
        safe: The values of `keyword` that make it safe.
        default: Whether the call is safe when it omits `keyword`.
    """

    name: str
    keyword: str = ""
    safe: tuple[object, ...] = ()
    default: bool = False

    def harmless(self, call: ast.Call) -> bool:
        """Whether this call is made safe by its own arguments.

        Args:
            call: The call.

        Returns:
            True when `keyword` is given a safe value, or omitted and the
            library default is safe.
        """
        if not self.keyword:
            return False
        for each in call.keywords:
            if each.arg == self.keyword:
                return value(each.value) in self.safe
        return self.default


SINKS = {
    each.name: each
    for each in (
        Sink("pickle.load"),
        Sink("pickle.loads"),
        Sink("pickle.Unpickler"),
        Sink("cloudpickle.load"),
        Sink("cloudpickle.loads"),
        Sink("dill.load"),
        Sink("dill.loads"),
        Sink("joblib.load"),
        Sink("marshal.load"),
        Sink("marshal.loads"),
        Sink("shelve.open"),
        Sink("jsonpickle.decode"),
        Sink("pandas.read_pickle"),
        Sink("torch.load", "weights_only", (True,)),
        Sink("yaml.load", "Loader", SAFE_LOADERS),
        Sink("yaml.unsafe_load"),
        Sink("yaml.full_load"),
        Sink("numpy.load", "allow_pickle", (False,), default=True),
        Sink("keras.models.load_model", "safe_mode", (True,), default=True),
        Sink("keras.saving.load_model", "safe_mode", (True,), default=True),
    )
}


@dataclasses.dataclass(frozen=True)
class Site:
    """One call to a deserialiser.

    Attributes:
        module: The dotted module name.
        function: The enclosing function, `Class.method` for a method.
        line: The line of the call.
        sink: The deserialiser called.
        verdict: One of `VERDICTS`.
        reason: Why the verdict was given.
    """

    module: str
    function: str
    line: int
    sink: str
    verdict: str
    reason: str

    @property
    def key(self) -> str:
        """The site's stable name, `module:function`."""
        return f"{self.module}:{self.function}"


class Module:
    """One parsed source file and the names its imports bind.

    Attributes:
        name: The dotted module name.
        tree: The parsed source.
        names: Each local name bound by an import or a top-level definition,
            mapped to the qualified name it stands for.
    """

    def __init__(self, name: str, tree: ast.Module) -> None:
        """Reads the imports and top-level definitions.

        Args:
            name: The dotted module name.
            tree: The parsed source.
        """
        self.name = name
        self.tree = tree
        self.names: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        self.names[alias.asname] = alias.name
                    else:
                        root = alias.name.split(".")[0]
                        self.names[root] = root
            elif isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    self.names[alias.asname or alias.name] = (
                        f"{node.module}.{alias.name}"
                    )
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.ClassDef):
                self.names.setdefault(node.name, f"{name}.{node.name}")

    def resolve(self, node: ast.expr) -> str:
        """The qualified name an expression such as `a.b.c` stands for.

        Args:
            node: A name or an attribute chain.

        Returns:
            The qualified name, or "" when the root is not a known name.
        """
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name) or node.id not in self.names:
            return ""
        return ".".join([self.names[node.id], *reversed(parts)])

    def functions(self) -> Iterator[tuple[str, ast.FunctionDef]]:
        """Every function and method, with its local name.

        Yields:
            `(name, node)`, `Class.method` for methods.
        """
        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef):
                yield node.name, node
            elif isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        yield f"{node.name}.{item.name}", item

    def sink(self, call: ast.Call) -> Sink | None:
        """The deserialiser a call reaches, unless its arguments make it safe.

        Args:
            call: The call.

        Returns:
            The sink, or None for any other call or a harmless one.
        """
        found = SINKS.get(self.resolve(call.func))
        if found is None or found.harmless(call):
            return None
        return found

    def reader(self, expression: ast.expr) -> str:
        """The authenticating reader an expression takes its bytes from.

        Args:
            expression: A bound value, possibly `call(...)[index]`.

        Returns:
            The reader's qualified name, or "" when it is not one.
        """
        index = None
        if isinstance(expression, ast.Subscript) and isinstance(
            expression.slice, ast.Constant
        ):
            index = expression.slice.value
            expression = expression.value
        if not isinstance(expression, ast.Call):
            return ""
        name = self.resolve(expression.func)
        if name in READERS and READERS[name] == index:
            return name
        return ""

    def authentic(self, function: ast.AST, name: str) -> str:
        """The reader every binding of `name` in a function comes from.

        Args:
            function: The function.
            name: The local name.

        Returns:
            The reader when every binding is from one reader, else "".
        """
        found = {self.reader(each) for each in binds(function, name)}
        if len(found) == 1 and "" not in found:
            return found.pop()
        return ""

    def wrapped(self, node: ast.expr | None) -> str:
        """The name an `io.BytesIO(name)` buffer wraps.

        Args:
            node: The deserialiser's argument.

        Returns:
            The wrapped name, or "" when the argument is not such a buffer.
        """
        if (
            isinstance(node, ast.Call)
            and self.resolve(node.func) == BUFFER
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
        ):
            return node.args[0].id
        return ""

    def roundtrip(self, function: ast.AST, name: str) -> bool:
        """Whether `name` is an empty buffer the function itself dumped into.

        Args:
            function: The function.
            name: The deserialiser's argument, a local name.

        Returns:
            True when every binding of `name` is `io.BytesIO()` and a dump
            writes to it in the same function.
        """
        bound = list(binds(function, name))
        empty = bool(bound) and all(
            isinstance(each, ast.Call)
            and self.resolve(each.func) == BUFFER
            and not each.args
            for each in bound
        )
        dumped = any(
            isinstance(node, ast.Call)
            and self.resolve(node.func) in DUMPS
            and len(node.args) > 1
            and isinstance(node.args[1], ast.Name)
            and node.args[1].id == name
            for node in ast.walk(function)
        )
        return empty and dumped


def binds(function: ast.AST, name: str) -> Iterator[ast.expr]:
    """The values assigned to `name` in a function, element-wise.

    A tuple target yields `value[index]` as an `ast.Subscript`, so that a
    reader's result position can be checked.

    Args:
        function: The function.
        name: The local name.

    Yields:
        Each value expression bound to `name`.
    """
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                yield node.value
            elif isinstance(target, ast.Tuple):
                for index, element in enumerate(target.elts):
                    if isinstance(element, ast.Name) and element.id == name:
                        yield ast.Subscript(
                            node.value, ast.Constant(index), ast.Load()
                        )


def source(call: ast.Call) -> ast.expr | None:
    """The argument a deserialiser reads from: the first, or a keyword.

    Args:
        call: The deserialiser call.

    Returns:
        The argument expression, or None when there is none.
    """
    if call.args:
        return call.args[0]
    for each in call.keywords:
        if each.arg in SOURCES:
            return each.value
    return None


def parameters(function: ast.AST) -> list[str]:
    """A function's positional parameter names, `self` and `cls` dropped.

    Args:
        function: The function, or a module, which has none.

    Returns:
        The names, in order.
    """
    if not isinstance(function, ast.FunctionDef):
        return []
    names = [a.arg for a in function.args.posonlyargs + function.args.args]
    return [n for n in names if n not in {"self", "cls"}]


def argument(call: ast.Call, position: int, name: str) -> ast.expr | None:
    """The expression a call passes for one parameter.

    Args:
        call: The call.
        position: The parameter's position, `self` excluded.
        name: The parameter's name.

    Returns:
        The argument, or None when the call does not pass it.
    """
    for each in call.keywords:
        if each.arg == name:
            return each.value
    if position < len(call.args):
        return call.args[position]
    return None


class Package:
    """Every module of a package, parsed, and the sites found in them.

    Attributes:
        root: The package directory.
        modules: The parsed modules by dotted name.
    """

    def __init__(self, root: pathlib.Path) -> None:
        """Parses every Python file under `root`.

        Args:
            root: The package directory, such as `scmle/`.
        """
        self.root = root
        self.modules: dict[str, Module] = {}
        for path in sorted(root.rglob("*.py")):
            parts = path.relative_to(root.parent).with_suffix("").parts
            dotted = ".".join(p for p in parts if p != "__init__")
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
            self.modules[dotted] = Module(dotted, tree)

    def sites(self) -> list[Site]:
        """Every deserialisation call site, with its verdict.

        A call outside any function or method, at module level or in a
        nested class, is judged in the scope of the whole module.

        Returns:
            The sites in module and line order.
        """
        found = []
        for module in self.modules.values():
            seen: set[int] = set()
            scopes: list[tuple[str, ast.AST]] = [*module.functions()]
            scopes.append((MODULE, module.tree))
            for local, scope in scopes:
                for node in ast.walk(scope):
                    if not isinstance(node, ast.Call) or id(node) in seen:
                        continue
                    seen.add(id(node))
                    called = module.sink(node)
                    if called is not None:
                        verdict, reason = self.judge(module, local, scope, node)
                        found.append(
                            Site(
                                module.name,
                                local,
                                node.lineno,
                                called.name,
                                verdict,
                                reason,
                            )
                        )
        return sorted(found, key=lambda s: (s.module, s.line))

    def judge(
        self, module: Module, local: str, scope: ast.AST, call: ast.Call
    ) -> tuple[str, str]:
        """Gives one call its verdict.

        Args:
            module: The module.
            local: The enclosing function's local name, or `MODULE`.
            scope: The enclosing function, or the module.
            call: The deserialiser call.

        Returns:
            The verdict and why.
        """
        given = source(call)
        name = module.wrapped(given)
        if name:
            origin = module.authentic(scope, name)
            if origin:
                return AUTHENTICATED, f"bytes from {origin}"
            if name in parameters(scope):
                return self.forward(module, local, scope, name)
            return UNAUTHENTICATED, f"bytes {name} of unknown origin"
        if isinstance(given, ast.Name) and module.roundtrip(scope, given.id):
            return ROUNDTRIP, "an in-process buffer the function dumped"
        text = "nothing" if given is None else ast.unparse(given)
        return UNAUTHENTICATED, f"reads {text}, not authenticated bytes"

    def callers(
        self, target: str, position: int, name: str
    ) -> list[tuple[str, str]]:
        """Every call of a function, and the reader of the bytes it passes.

        Args:
            target: The function's qualified name.
            position: The parameter's position.
            name: The parameter's name.

        Returns:
            `(module:function, reader)` per call; the reader is "" when the
            bytes passed are not from an authenticating reader.
        """
        found = []
        for caller in self.modules.values():
            for outer, enclosing in caller.functions():
                for node in ast.walk(enclosing):
                    if not (
                        isinstance(node, ast.Call)
                        and caller.resolve(node.func) == target
                    ):
                        continue
                    given = argument(node, position, name)
                    origin = (
                        caller.authentic(enclosing, given.id)
                        if isinstance(given, ast.Name)
                        else ""
                    )
                    found.append((f"{caller.name}:{outer}", origin))
        return found

    def forward(
        self, module: Module, local: str, scope: ast.AST, name: str
    ) -> tuple[str, str]:
        """Judges a buffer of a parameter by every call of the function.

        Args:
            module: The module defining the function.
            local: The function's local name.
            scope: The function.
            name: The parameter the buffer wraps.

        Returns:
            `forwarded` when the function is called and every call passes
            bytes from an authenticating reader, else `unauthenticated`;
            and why.
        """
        found = self.callers(
            f"{module.name}.{local}", parameters(scope).index(name), name
        )
        if not found:
            return UNAUTHENTICATED, f"{name} comes from no caller"
        unproven = [where for where, origin in found if not origin]
        if unproven:
            return (
                UNAUTHENTICATED,
                f"{name} is not authenticated in {', '.join(unproven)}",
            )
        readers = ", ".join(sorted({pair[1] for pair in found}))
        return FORWARDED, f"{len(found)} calls pass bytes from {readers}"


def scan(root: pathlib.Path) -> list[Site]:
    """Every deserialisation call site in a package.

    Args:
        root: The package directory.

    Returns:
        The sites with their verdicts.
    """
    return Package(root).sites()


def main(arguments: list[str] | None = None) -> int:
    """Prints every site as JSON; exits 1 when any is unauthenticated.

    Args:
        arguments: Command-line arguments; None reads `sys.argv`.

    Returns:
        0 when every site is authenticated, forwarded or a round trip.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=pathlib.Path)
    options = parser.parse_args(arguments)
    found = scan(options.package)
    print(json.dumps([dataclasses.asdict(s) for s in found], indent=2))  # noqa: T201 - the report is the output
    return int(any(s.verdict == UNAUTHENTICATED for s in found))


if __name__ == "__main__":
    sys.exit(main())
