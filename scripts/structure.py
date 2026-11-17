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

`--simulate` applies a layer map (`scripts/layers.toml`: the layer order and
the module moves) to the package's import graph, including function-level
and `TYPE_CHECKING` imports, and prints every import that would break the
order and the largest package cycle left. `layer_breaks` in the budget is
the number of such imports for the recorded map (register phase 3, DES-13).
"""

import argparse
import ast
import json
import pathlib
import re
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "scmle"
BUDGET = pathlib.Path(__file__).with_name("structure.json")
LOOSE = re.compile(r"dict\[str, *(Any|object)\]")
CAPITALS = re.compile(r"[A-Z][a-z0-9]*")
BARE = frozenset({"ValueError", "TypeError", "RuntimeError"})
LAYERS = pathlib.Path(__file__).with_name("layers.toml")


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


def measure() -> dict[str, int]:
    """Counts every budgeted quantity over the package.

    Returns:
        Quantity name to its count.
    """
    bound = limits()
    found = dict.fromkeys(
        (
            "modules_over",
            "classes_over",
            "functions_over",
            "compound_functions",
            "compound_classes",
            "private_names",
            "loose_dicts",
            "largest_module",
            "bare_raises",
            "value_catches",
        ),
        0,
    )
    if LAYERS.exists():
        found["layer_breaks"] = len(simulate(load(LAYERS))[0])
    for path in sorted(PACKAGE.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        lines = text.count("\n")
        found["modules_over"] += lines > bound["module"]
        found["largest_module"] = max(found["largest_module"], lines)
        found["loose_dicts"] += len(LOOSE.findall(text))
        for node in ast.walk(ast.parse(text)):
            found["bare_raises"] += bare(node)
            found["value_catches"] += catches(node)
            if isinstance(node, ast.ClassDef):
                found["classes_over"] += span(node) > bound["class"]
                found["compound_classes"] += compound(node.name)
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                found["functions_over"] += span(node) > bound["function"]
                found["compound_functions"] += compound(node.name)
            else:
                continue
            found["private_names"] += node.name.startswith("_") and not (
                node.name.endswith("__")
            )
    return found


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
    options = parser.parse_args(arguments)
    if options.simulate:
        breaks, largest = simulate(load(options.simulate))
        print(*breaks, sep="\n")
        print(f"breaks: {len(breaks)}, largest package cycle: {largest}")
        return 1 if breaks else 0
    found = measure()
    print(json.dumps(found, indent=2))
    if options.write:
        BUDGET.write_text(json.dumps(found, indent=2) + "\n", encoding="utf-8")
        return 0
    worse = compare(found, json.loads(BUDGET.read_text(encoding="utf-8")))
    for line in worse:
        print(f"worse than budget: {line}", file=sys.stderr)
    return 1 if worse else 0


if __name__ == "__main__":
    sys.exit(main())
