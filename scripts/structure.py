"""Measures the package's structure and fails when a number gets worse.

The counts are the plan's structure budgets (register task 0.1, claim C10)
and the naming rule: Google style, no leading-underscore names, and
single-word names, with snake_case only where one word would be unclear.
Size limits come from `acceptance.toml` (`module_lines_max`,
`class_lines_max`, `function_lines_max`). `structure.json` records the
counts last accepted; a count above its recorded value fails, and a count
below it is a ratchet the next `--write` records.

    uv run python scripts/structure.py            # check against the budget
    uv run python scripts/structure.py --write    # record current counts
    uv run python scripts/structure.py --simulate scripts/layers.toml
    uv run python scripts/structure.py --external # list external collisions

Three counts serve the naming phase (register phase 6). `external_names`
counts the multi-word names the package defines that a library it imports
also offers (a module name or a class attribute), so `pandas.read_json`
and an scmle `read_json` read alike; `--external` lists them. The libraries
are the standard library and the dependency set `uv.lock` declares, not
whatever is installed: `--write` records the collisions in `external.json`,
and a declared library absent here keeps its recorded ones.
`import_aliases` counts `import ... as` bindings, which the naming rule
forbids. `compound_attributes` counts multi-word attribute names per class,
in the class body and on `self`. `EXEMPT` lists the names no count
includes, with the reason each keeps its spelling.

`--simulate` applies a layer map (`scripts/layers.toml`: the layer order and
the module moves) to the package's import graph, including function-level
and `TYPE_CHECKING` imports, and prints every import that would break the
order and the largest package cycle left. While a map is recorded there,
`layer_breaks` counts such imports against the budget. Phase 3 (DES-13)
ended with its map applied and deleted; import-linter's layers contract in
`pyproject.toml` is now the gate, and a later phase records its own map.
"""

import argparse
import ast
import importlib
import importlib.metadata
import inspect
import json
import pathlib
import re
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "scmle"
BUDGET = pathlib.Path(__file__).with_name("structure.json")
# The external collisions found at the last `--write`, by name, with the
# library that offers each: a declared library not installed here keeps its
# recorded collisions, so the count is the same on every machine.
COLLISIONS = pathlib.Path(__file__).with_name("external.json")
LOCK = ROOT / "uv.lock"
LOOSE = re.compile(r"dict\[str, *(Any|object)\]")
CAPITALS = re.compile(r"[A-Z][a-z0-9]*")
BARE = frozenset({"ValueError", "TypeError", "RuntimeError"})
LAYERS = pathlib.Path(__file__).with_name("layers.toml")
# Multi-word names no count includes, each with the reason it must keep its
# spelling: another library calls the first seventeen by name, and a
# certificate hashes the source of the last.
SERVER = "http.server: BaseHTTPRequestHandler calls it by name"
SOCKET = "socketserver: BaseServer calls it by name for each request"
ESTIMATOR = "scikit-learn: the estimator API calls it by name"
EXEMPT = {
    "do_GET": SERVER + " (do_<method> per request)",
    "do_HEAD": SERVER + " (do_<method> per request)",
    "do_POST": SERVER + " (do_<method> per request)",
    "log_message": SERVER,
    "version_string": SERVER,
    "process_request": SOCKET,
    "process_request_thread": SOCKET + " (ThreadingMixIn per request)",
    "http_request": "urllib.request: OpenerDirector calls <scheme>_request",
    "redirect_request": "urllib.request: HTTPRedirectHandler calls it",
    "get_params": ESTIMATOR + " (clone, search)",
    "set_params": ESTIMATOR + " (clone, search)",
    "get_feature_names_out": ESTIMATOR + " (pipelines, set_output)",
    "get_n_splits": "scikit-learn: cross-validation asks a splitter for it",
    "inverse_transform": ESTIMATOR + " (pipelines, target transforms)",
    "predict_proba": ESTIMATOR + " (scorers, calibration, pipelines)",
    "state_dict": "torch: nn.Module checkpoints call it by name",
    "load_state_dict": "torch: nn.Module checkpoints call it by name",
    "shrunk_covariance": (
        "the uncertainty certificate hashes reconcile.shrunk_covariance;"
        " renaming it invalidates the certificate"
    ),
}


def limits() -> dict[str, int]:
    """Reads the size limits from the thresholds file.

    Returns:
        `module`, `class` and `function` line limits.
    """
    document = tomllib.loads((ROOT / "acceptance.toml").read_text("utf-8"))
    return {
        kind: int(document[f"{kind}_lines_max"]["value"])
        for kind in ("module", "class", "function")
    }


def span(node: ast.AST) -> int:
    """Lines a definition occupies, decorators excluded."""
    return int(getattr(node, "end_lineno", 0)) - int(node.lineno) + 1


def compound(name: str) -> bool:
    """Whether a name joins several words (snake_case or CamelCase).

    The `Error` suffix Google style requires on exception names is a marker,
    not a word of the name, so `ConflictError` counts as one word.
    """
    if name.startswith("__") and name.endswith("__"):
        return False
    if "_" in name.strip("_"):
        return True
    return len(CAPITALS.findall(name.removesuffix("Error") or name)) > 1


def bare(node: ast.AST) -> bool:
    """Whether a node raises a built-in error a gate cannot tell from a bug.

    `ValueError`, `TypeError` and `RuntimeError` mix bad input with defects;
    input problems raise `errors.InputError` instead (ERR-20).
    """
    if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
        return False
    target = node.exc.func
    name = target.attr if isinstance(target, ast.Attribute) else ""
    name = name or (target.id if isinstance(target, ast.Name) else "")
    return name in BARE


def catches(node: ast.AST) -> bool:
    """Whether a node is a handler that catches a bare `ValueError`."""
    if not isinstance(node, ast.ExceptHandler) or node.type is None:
        return False
    kinds = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
    return any(ast.unparse(kind) == "ValueError" for kind in kinds)


def counted(name: str) -> bool:
    """Whether a name is multi-word and not exempt."""
    return compound(name) and name not in EXEMPT


def attributes(node: ast.ClassDef) -> set[str]:
    """Attribute names a class defines: in its body and on `self`."""
    found = set()
    for item in node.body:
        if isinstance(item, ast.AnnAssign | ast.AugAssign):
            found.add(getattr(item.target, "id", ""))
        elif isinstance(item, ast.Assign):
            found.update(getattr(target, "id", "") for target in item.targets)
        elif isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
            own = item.args.args[0].arg if item.args.args else ""
            found.update(
                inner.attr
                for inner in ast.walk(item)
                if isinstance(inner, ast.Attribute)
                and isinstance(inner.ctx, ast.Store)
                and isinstance(inner.value, ast.Name)
                and inner.value.id == own
            )
    return found - {""}


def aliases(tree: ast.AST) -> int:
    """Counts the import bindings renamed with `as`.

    Code strings a module runs in a child process are read too.
    """
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            count += sum(alias.asname is not None for alias in node.names)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "import " in node.value
        ):
            try:
                inner = ast.parse(node.value)
            except SyntaxError:
                continue
            count += aliases(inner)
    return count


def locked() -> set[str]:
    """The distributions of the declared dependency set.

    Returns:
        The normalized name of every package `uv.lock` resolves for the
        project's dependencies and extras.
    """
    packages = tomllib.loads(LOCK.read_text(encoding="utf-8"))["package"]
    return {normal(package["name"]) for package in packages}


def normal(name: str) -> str:
    """A distribution name normalized as PEP 503 does."""
    return re.sub(r"[-_.]+", "-", name).lower()


def external(imported: set[str], defined: set[str]) -> dict[str, str]:
    """Defined names that a declared library the package imports offers.

    A library's public names are its module's names and the attributes of
    the classes it exposes. The libraries are the standard library and the
    declared dependency set (`locked`), not whatever is installed: a
    declared library installed here is read, and one that is not keeps the
    collisions recorded for it at the last `--write` (`COLLISIONS`), so the
    count is the same on every machine with the same lock file.

    Args:
        imported: Modules the package imports.
        defined: Multi-word names the package defines.

    Returns:
        Each colliding name and the first module that also offers it.
    """
    declared = locked()
    installed = importlib.metadata.packages_distributions()
    found: dict[str, str] = {}
    for module in sorted(imported):
        top = module.partition(".")[0]
        owners = {normal(owner) for owner in installed.get(top, [])}
        if top not in sys.stdlib_module_names and not owners & declared:
            continue
        try:
            loaded = importlib.import_module(module)
        except Exception:  # noqa: BLE001 - an optional library is absent
            continue
        offered = set(dir(loaded))
        for value in vars(loaded).values():
            if inspect.isclass(value):
                offered.update(dir(value))
        for name in sorted(defined & offered):
            found.setdefault(name, module)
    recorded = (
        json.loads(COLLISIONS.read_text(encoding="utf-8"))
        if COLLISIONS.exists()
        else {}
    )
    for name, module in sorted(recorded.items()):
        if name in defined and module.partition(".")[0] not in installed:
            found.setdefault(name, module)
    return dict(sorted(found.items()))


def measure() -> dict[str, int]:
    """Counts every budgeted quantity over the package.

    Returns:
        Quantity name to its count.
    """
    return scan()[0]


def scan() -> tuple[dict[str, int], dict[str, str]]:
    """Counts every budgeted quantity, and names the external collisions.

    Returns:
        Quantity name to its count, and each multi-word name the package
        defines that an imported library also offers, with that library.
    """
    bound = limits()
    found = dict.fromkeys(
        (
            "modules_over",
            "classes_over",
            "functions_over",
            "compound_functions",
            "compound_classes",
            "compound_attributes",
            "private_names",
            "loose_dicts",
            "largest_module",
            "bare_raises",
            "value_catches",
            "import_aliases",
            "external_names",
        ),
        0,
    )
    if LAYERS.exists():
        found["layer_breaks"] = len(simulate(load(LAYERS))[0])
    imported: set[str] = set()
    defined: set[str] = set()
    root = PACKAGE.name
    for path in sorted(PACKAGE.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        lines = text.count("\n")
        found["modules_over"] += lines > bound["module"]
        found["largest_module"] = max(found["largest_module"], lines)
        found["loose_dicts"] += len(LOOSE.findall(text))
        tree = ast.parse(text)
        found["import_aliases"] += aliases(tree)
        for node in ast.walk(tree):
            found["bare_raises"] += bare(node)
            found["value_catches"] += catches(node)
            if isinstance(node, ast.Import | ast.ImportFrom):
                names = [alias.name for alias in node.names]
                if isinstance(node, ast.ImportFrom):
                    # `from http import server` imports the module
                    # `http.server`; a name that is not a module fails to
                    # import and is skipped.
                    module = node.module if not node.level else None
                    names = [module] if module else []
                    names += [f"{module}.{alias.name}" for alias in node.names]
                    names = names if module else []
                imported.update(
                    name for name in names if name.split(".")[0] != root
                )
                continue
            if isinstance(node, ast.ClassDef):
                found["classes_over"] += span(node) > bound["class"]
                found["compound_classes"] += counted(node.name)
                found["compound_attributes"] += sum(
                    map(counted, attributes(node))
                )
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                found["functions_over"] += span(node) > bound["function"]
                found["compound_functions"] += counted(node.name)
            else:
                continue
            if counted(node.name):
                defined.add(node.name)
            found["private_names"] += node.name.startswith("_") and not (
                node.name.endswith("__")
            )
    collisions = external(imported, defined)
    found["external_names"] = len(collisions)
    return found, collisions


def load(path: pathlib.Path) -> tuple[list[list[str]], dict[str, str]]:
    """Reads a layer map.

    Args:
        path: The map file.

    Returns:
        The layers (each a list of sibling packages, top first) and the
        moves (old dotted path to new, every step in order).
    """
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    layers = [
        [name.strip() for name in row.split("|")] for row in document["layers"]
    ]
    moves = {
        old: new
        for step in document.get("step", [])
        for old, new in step["moves"]
    }
    return layers, moves


def name(path: pathlib.Path) -> str:
    """The dotted module name of a file under the package."""
    parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def imports(tree: ast.AST, known: set[str]) -> set[str]:
    """Every package module a parsed module imports, at any depth.

    Args:
        tree: The parsed module.
        known: Dotted names of every module in the package.

    Returns:
        The imported modules; `from a import b` names `a.b` when that is a
        module and `a` otherwise.
    """
    root = PACKAGE.name
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif (
            isinstance(node, ast.ImportFrom) and node.module and not node.level
        ):
            for alias in node.names:
                inner = f"{node.module}.{alias.name}"
                found.add(inner if inner in known else node.module)
    return {item for item in found if item.startswith(f"{root}.")}


def rename(module: str, moves: dict[str, str]) -> str:
    """Applies the longest matching move to a dotted name.

    Args:
        module: A dotted module name.
        moves: Old dotted path to new.

    Returns:
        The name after every move.
    """
    for _ in range(len(moves) + 1):
        matches = [
            old
            for old in moves
            if module == old or module.startswith(f"{old}.")
        ]
        if not matches:
            return module
        old = max(matches, key=len)
        module = moves[old] + module[len(old) :]
    raise ValueError(f"the moves loop on {module}")


def package(module: str) -> str:
    """The top-level package (or module) a dotted name belongs to."""
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else ""


def reach(graph: dict[str, set[str]], start: str) -> set[str]:
    """Every node reachable from a start node, the start included."""
    seen = {start}
    work = [start]
    while work:
        for step in graph.get(work.pop(), set()) - seen:
            seen.add(step)
            work.append(step)
    return seen


def cycle(edges: set[tuple[str, str]]) -> int:
    """Size of the largest strongly connected component of a graph.

    The graph has one node per package, so reachability from every node is
    cheap and plain.

    Args:
        edges: Directed edges.

    Returns:
        The number of nodes in the largest component; 1 when acyclic.
    """
    graph: dict[str, set[str]] = {}
    for source, target in edges:
        graph.setdefault(source, set()).add(target)
    nodes = {node for edge in edges for node in edge}
    closure = {node: reach(graph, node) for node in nodes}
    return max(
        (
            sum(node in closure[other] for other in closure[node])
            for node in nodes
        ),
        default=1,
    )


def simulate(
    layout: tuple[list[list[str]], dict[str, str]],
) -> tuple[list[str], int]:
    """Applies a layer map to the package's import graph.

    Args:
        layout: A map from `load`.

    Returns:
        One line per import that breaks the layer order after the moves
        (upward, between siblings, or to or from an unplaced package), and the
        size of the largest package cycle after the moves.
    """
    layers, moves = layout
    rank = {item: depth for depth, row in enumerate(layers) for item in row}
    files = sorted(PACKAGE.rglob("*.py"))
    known = {name(path) for path in files}
    breaks: list[str] = []
    edges: set[tuple[str, str]] = set()
    for path in files:
        source = name(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for target in sorted(imports(tree, known)):
            above, below = (
                package(rename(source, moves)),
                package(rename(target, moves)),
            )
            if not above or not below or above == below:
                continue
            edges.add((above, below))
            if above not in rank or below not in rank:
                verdict = "unplaced"
            elif rank[above] > rank[below]:
                verdict = "upward"
            elif rank[above] == rank[below]:
                verdict = "sibling"
            else:
                continue
            breaks.append(
                f"{source} -> {target} ({above} -> {below}, {verdict})"
            )
    return breaks, cycle(edges)


def compare(found: dict[str, int], budget: dict[str, int]) -> list[str]:
    """Lists every count that is worse than its budget.

    Args:
        found: Current counts.
        budget: Accepted counts.

    Returns:
        One line per regression; a quantity missing from the budget is a
        regression too.
    """
    return [
        f"{name}: {count} > {budget.get(name, 0)}"
        for name, count in found.items()
        if count > budget.get(name, 0)
    ]


def main(arguments: list[str] | None = None) -> int:
    """Checks or records the structure budget.

    Args:
        arguments: Command-line arguments; `sys.argv` when None.

    Returns:
        0 when no count is worse than the budget, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--simulate", type=pathlib.Path)
    parser.add_argument("--external", action="store_true")
    options = parser.parse_args(arguments)
    if options.external:
        for name, module in sorted(scan()[1].items()):
            print(f"{name}  {module}")
        return 0
    if options.simulate:
        breaks, largest = simulate(load(options.simulate))
        print(*breaks, sep="\n")
        print(f"breaks: {len(breaks)}, largest package cycle: {largest}")
        return 1 if breaks else 0
    found, collisions = scan()
    print(json.dumps(found, indent=2))
    if options.write:
        BUDGET.write_text(json.dumps(found, indent=2) + "\n", encoding="utf-8")
        COLLISIONS.write_text(
            json.dumps(collisions, indent=2) + "\n", encoding="utf-8"
        )
        return 0
    worse = compare(found, json.loads(BUDGET.read_text(encoding="utf-8")))
    for line in worse:
        print(f"worse than budget: {line}", file=sys.stderr)
    return 1 if worse else 0


if __name__ == "__main__":
    sys.exit(main())
