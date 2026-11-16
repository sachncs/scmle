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
    """Whether a name joins several words (snake_case or CamelCase)."""
    if name.startswith("__") and name.endswith("__"):
        return False
    if "_" in name.strip("_"):
        return True
    return len(CAPITALS.findall(name)) > 1


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
        ),
        0,
    )
    for path in sorted(PACKAGE.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        lines = text.count("\n")
        found["modules_over"] += lines > bound["module"]
        found["largest_module"] = max(found["largest_module"], lines)
        found["loose_dicts"] += len(LOOSE.findall(text))
        for node in ast.walk(ast.parse(text)):
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
    options = parser.parse_args(arguments)
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
