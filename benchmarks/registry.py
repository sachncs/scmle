"""The gate registry: every check that can stop a run, declared (claim C1).

`gates.toml` at the repository root lists each gate with the claim it
serves, the function that raises, the condition under which it passes, the
condition under which it blocks, how a user proceeds anyway, the variables
its decision reads and, where one decides it, its certificate. `scan` finds
every function in the package that raises `BlockedError` or `LeakageError`;
a site that raises and is not registered, or a registered site that no
longer raises, fails CI, so no gate exists that the scorecard cannot see.

A variable names a configuration field by its stored key. The keys renamed
to single words (`tasks.renamed`) are still read in their old spelling for
one release, with a warning, and stored under the new one.

    python -m benchmarks.registry        # compare the registry with the code
"""

import ast
import dataclasses
import json
import pathlib
import sys
import tomllib
import warnings

from scmle.contracts import tasks

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATH = ROOT / "gates.toml"
PACKAGE = ROOT / "scmle"
ERRORS = frozenset({"BlockedError", "LeakageError"})
CLAIMS = frozenset(f"C{n}" for n in range(1, 15))
TEXTS = ("site", "error", "claim", "passes", "blocks", "override")
RENAMED = tasks.renamed()


@dataclasses.dataclass(frozen=True)
class Gate:
    """One check that can stop a run.

    Attributes:
        name: Its key in `gates.toml`.
        site: `module:qualname` of the function that raises.
        error: The error it raises.
        claims: The scorecard claims it serves.
        passes: When the run continues.
        blocks: When it raises.
        override: How a user proceeds anyway, or `none`.
        variables: Configuration fields or constants its decision reads.
        certificate: The certificate store and detector that decide it, or
            an empty string.
    """

    name: str
    site: str
    error: str
    claims: tuple[str, ...]
    passes: str
    blocks: str
    override: str
    variables: tuple[str, ...] = ()
    certificate: str = ""


def parse(name: str, table: dict[str, object]) -> Gate:
    """Builds one gate from its table, refusing anything malformed.

    Args:
        name: The table's key.
        table: Its fields.

    Returns:
        The gate.

    Raises:
        ValueError: For an unknown or missing field, empty text, an error
            other than the blocking ones, an unknown claim or a site that
            is not `module:qualname`.
    """
    known = {*TEXTS, "variables", "certificate"}
    unknown = set(table) - known
    missing = set(TEXTS) - set(table)
    if unknown or missing:
        raise ValueError(
            f"{name}: unknown fields {sorted(unknown)}, "
            f"missing fields {sorted(missing)}"
        )
    for key in TEXTS:
        text = table[key]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{name}: {key} must be non-empty text")
    if table["error"] not in ERRORS:
        raise ValueError(f"{name}: error must be one of {sorted(ERRORS)}")
    claims = tuple(c.strip() for c in str(table["claim"]).split(","))
    if not set(claims) <= CLAIMS:
        raise ValueError(f"{name}: unknown claim in {claims}")
    if str(table["site"]).count(":") != 1:
        raise ValueError(f"{name}: site must be module:qualname")
    variables = table.get("variables", [])
    if not isinstance(variables, list):
        raise ValueError(f"{name}: variables must be a list")
    for old in [str(v) for v in variables if v in RENAMED]:
        warnings.warn(
            f"{name}: variable {old} is deprecated and will be removed in "
            f"the next release; write {RENAMED[old]}",
            FutureWarning,
            stacklevel=2,
        )
    return Gate(
        name,
        str(table["site"]),
        str(table["error"]),
        claims,
        str(table["passes"]),
        str(table["blocks"]),
        str(table["override"]),
        tuple(RENAMED.get(str(v), str(v)) for v in variables),
        str(table.get("certificate", "")),
    )


def load(path: pathlib.Path = PATH) -> dict[str, Gate]:
    """Reads every gate in the registry.

    Nested TOML tables (`[split.group]`) are flattened to dotted names.

    Args:
        path: The registry file.

    Returns:
        Gate name to gate.

    Raises:
        ValueError: If a gate is malformed or two gates share a site.
    """
    found: dict[str, Gate] = {}

    def visit(prefix: str, table: dict[str, object]) -> None:
        if "site" in table:
            found[prefix] = parse(prefix, table)
            return
        for key, value in table.items():
            if not isinstance(value, dict):
                raise ValueError(f"{prefix}.{key}: expected a table")
            visit(f"{prefix}.{key}" if prefix else key, value)

    visit("", tomllib.loads(path.read_text(encoding="utf-8")))
    sites = [g.site for g in found.values()]
    doubled = sorted({s for s in sites if sites.count(s) > 1})
    if doubled:
        raise ValueError(f"sites registered twice: {doubled}")
    return found


def raised(node: ast.Raise) -> str:
    """The name of the error a `raise` statement raises, or ''."""
    target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
    if isinstance(target, ast.Attribute):
        return target.attr
    return target.id if isinstance(target, ast.Name) else ""


def scan(package: pathlib.Path = PACKAGE) -> dict[str, str]:
    """Finds every function in the package that raises a blocking error.

    Args:
        package: The package folder.

    Returns:
        `module:qualname` to the error it raises (the first, if several).
    """
    found: dict[str, str] = {}
    for path in sorted(package.rglob("*.py")):
        module = ".".join(
            path.relative_to(package.parent).with_suffix("").parts
        )
        stack: list[tuple[ast.AST, str]] = [
            (ast.parse(path.read_text(encoding="utf-8")), "")
        ]
        while stack:
            node, scope = stack.pop()
            for child in ast.iter_child_nodes(node):
                inner = scope
                if isinstance(
                    child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
                ):
                    inner = f"{scope}.{child.name}" if scope else child.name
                elif isinstance(child, ast.Raise) and raised(child) in ERRORS:
                    found.setdefault(f"{module}:{scope}", raised(child))
                stack.append((child, inner))
    return found


def compare(gates: dict[str, Gate], sites: dict[str, str]) -> list[str]:
    """Lists every disagreement between the registry and the code.

    Args:
        gates: As returned by `load`.
        sites: As returned by `scan`.

    Returns:
        One line per unregistered site, stale entry or wrong error.
    """
    registered = {g.site: g for g in gates.values()}
    problems = [
        f"unregistered gate: {site} raises {error}"
        for site, error in sorted(sites.items())
        if site not in registered
    ]
    problems += [
        f"registered gate no longer raises: {g.name} ({site})"
        for site, g in sorted(registered.items())
        if site not in sites
    ]
    problems += [
        f"{g.name}: registered as {g.error}, raises {sites[site]}"
        for site, g in sorted(registered.items())
        if site in sites and sites[site] != g.error
    ]
    return problems


def main() -> int:
    """Compares the registry with the code.

    Returns:
        0 when they agree, 1 otherwise.
    """
    problems = compare(load(), scan())
    print(json.dumps({"gates": len(load()), "problems": problems}, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
