"""Each mapped new name fits the kind of definition it renames.

The naming rule asks a verb of a function, a noun of a class and the
`Error` suffix of an exception. These checks find each mapped definition
in the source tree and hold its new name to that rule. They also make
sure no handler that a class finds by a computed `getattr` name is mapped,
because rename.py cannot follow that name. Entries whose definition is
gone are left to rename.py's dry run, which refuses them.
"""

import ast
import functools
import pathlib
import re
import tomllib
import typing

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
MAPS = ROOT / "scripts" / "renames"
LANES = ("data_assurance", "models", "pipeline", "rest")

# Every verb a mapped function may take. A new function name has to be
# here, so adding a word is a reviewed act and a noun or adjective cannot
# slip through the way `fisher`, `detectable` and `centred` once did.
VERBS = frozenset(
    [
        "abstain",
        "accept",
        "access",
        "account",
        "accumulate",
        "act",
        "adapt",
        "add",
        "aggregate",
        "allot",
        "amend",
        "anchor",
        "annex",
        "append",
        "apply",
        "appraise",
        "approximate",
        "assay",
        "assemble",
        "assess",
        "associate",
        "attest",
        "audit",
        "autocorrelate",
        "average",
        "balance",
        "batch",
        "block",
        "bound",
        "build",
        "bundle",
        "calculate",
        "calibrate",
        "cap",
        "catalog",
        "centre",
        "certify",
        "chart",
        "check",
        "checksum",
        "choose",
        "claim",
        "classify",
        "collate",
        "combine",
        "commit",
        "compare",
        "confine",
        "conflict",
        "constrain",
        "consult",
        "contrast",
        "control",
        "convert",
        "correct",
        "corroborate",
        "corrupt",
        "count",
        "cover",
        "declare",
        "decode",
        "decompose",
        "deconfound",
        "default",
        "depict",
        "describe",
        "detect",
        "diagnose",
        "digest",
        "disconnect",
        "discriminate",
        "dispatch",
        "download",
        "draft",
        "draw",
        "duplicate",
        "embed",
        "emit",
        "encode",
        "enforce",
        "escalate",
        "estimate",
        "evaluate",
        "examine",
        "exclude",
        "execute",
        "exhaust",
        "expire",
        "explain",
        "explore",
        "export",
        "extract",
        "featurize",
        "feed",
        "fetch",
        "fill",
        "fingerprint",
        "fit",
        "flag",
        "flush",
        "fold",
        "forecast",
        "forge",
        "frame",
        "freeze",
        "gate",
        "gauge",
        "govern",
        "group",
        "hedge",
        "identify",
        "imprint",
        "index",
        "infer",
        "insist",
        "inspect",
        "inventory",
        "join",
        "judge",
        "key",
        "kill",
        "label",
        "load",
        "lock",
        "log",
        "lose",
        "materialize",
        "measure",
        "mismatch",
        "move",
        "mutate",
        "narrow",
        "nest",
        "normalize",
        "observe",
        "order",
        "orphan",
        "overlap",
        "parse",
        "permute",
        "persist",
        "perturb",
        "pin",
        "plan",
        "poll",
        "post",
        "predict",
        "prepare",
        "price",
        "process",
        "profile",
        "project",
        "provision",
        "qualify",
        "quantify",
        "rank",
        "ratchet",
        "rate",
        "read",
        "reassess",
        "rebind",
        "recall",
        "recheck",
        "recognize",
        "recompute",
        "record",
        "register",
        "rehearse",
        "reject",
        "relabel",
        "release",
        "renew",
        "reopen",
        "repeat",
        "replay",
        "report",
        "require",
        "resample",
        "resolve",
        "restore",
        "resume",
        "revalidate",
        "reweight",
        "rotate",
        "route",
        "run",
        "sample",
        "save",
        "scaffold",
        "scale",
        "scan",
        "score",
        "scramble",
        "screen",
        "seed",
        "select",
        "separate",
        "sequence",
        "serialize",
        "sideline",
        "sign",
        "size",
        "skip",
        "snapshot",
        "spawn",
        "stamp",
        "standardize",
        "starve",
        "step",
        "stratify",
        "support",
        "survey",
        "sweep",
        "tabulate",
        "tailor",
        "tally",
        "temper",
        "test",
        "timestamp",
        "trace",
        "train",
        "transform",
        "truncate",
        "unregister",
        "update",
        "validate",
        "value",
        "verify",
        "vet",
        "watch",
        "watermark",
        "weigh",
        "withhold",
        "write",
    ]
)
ATTRIBUTES = frozenset(
    {"property", "cached_property", "setter", "getter", "deleter"}
)
FAILURES = ("Error", "Exception")
DISPATCH = re.compile(r"""getattr\(\s*(?:self|cls)\s*,\s*f["']([a-z_]+)\{""")


Definition = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


class Site(typing.NamedTuple):
    """A mapped entry's new name, its definition and its enclosing class."""

    new: str
    node: Definition | None
    owner: ast.ClassDef | None


def entries(lane: str) -> dict[str, str]:
    """The mapped entries of a lane, without its review table."""
    table = tomllib.loads((MAPS / f"{lane}.toml").read_text("utf-8"))
    table.pop("review", None)
    found: dict[str, str] = {}

    def flatten(part: dict[str, typing.Any], prefix: str) -> None:
        for key, value in part.items():
            if isinstance(value, dict):
                flatten(value, f"{prefix}{key}.")
            else:
                found[f"{prefix}{key}"] = value

    flatten(table, "")
    return found


@functools.cache
def parse(file: pathlib.Path) -> ast.Module:
    """The syntax tree of a source file."""
    return ast.parse(file.read_text("utf-8"))


def walk(
    body: list[ast.stmt], names: list[str]
) -> tuple[Definition | None, ast.ClassDef | None]:
    """Follows class and def names down nested bodies.

    Returns:
        `(node, owner)`: the definition and the class it sits in, or
        `(None, None)` when a name is not a def or class of its body.
    """
    node: Definition | None = None
    owner: ast.ClassDef | None = None
    for name in names:
        found = [
            item
            for item in body
            if isinstance(
                item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
            and item.name == name
        ]
        if not found:
            return None, None
        owner = node if isinstance(node, ast.ClassDef) else None
        node = found[-1]
        body = node.body
    return node, owner


def locate(path: str) -> tuple[Definition | None, ast.ClassDef | None]:
    """The definition a dotted path names and its enclosing class, if any.

    An attribute, a module, or a definition that is gone gives
    `(None, None)`.
    """
    parts = path.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        base = ROOT.joinpath(*parts[:cut])
        for file in (base.with_suffix(".py"), base / "__init__.py"):
            if file.is_file():
                return walk(parse(file).body, parts[cut:])
    return None, None


def tail(node: ast.expr) -> str:
    """The last name of a decorator or base expression."""
    while isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    return node.id if isinstance(node, ast.Name) else ""


def kind(node: Definition | ast.stmt | None) -> str | None:
    """`function`, `attribute` (a property), `exception`, `class` or None."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        names = {tail(item) for item in node.decorator_list}
        return "attribute" if names & ATTRIBUTES else "function"
    if isinstance(node, ast.ClassDef):
        bases = [tail(item) for item in node.bases]
        failing = any(name.endswith(FAILURES) for name in bases)
        return "exception" if failing else "class"
    return None


def survey(lane: str, wanted: str) -> dict[str, Site]:
    """Mapped entries of a lane whose definition is of the wanted kind."""
    found = {}
    for old, new in entries(lane).items():
        node, owner = locate(old)
        if kind(node) == wanted:
            found[old] = Site(new, node, owner)
    return found


@pytest.mark.parametrize("lane", LANES)
def test_every_mapped_function_takes_a_verb(lane: str) -> None:
    functions = survey(lane, "function")
    assert functions
    bad = {
        old: site.new
        for old, site in functions.items()
        if site.new not in VERBS
    }
    assert bad == {}


@pytest.mark.parametrize("lane", LANES)
def test_every_mapped_exception_keeps_the_error_suffix(lane: str) -> None:
    exceptions = survey(lane, "exception")
    bad = {
        old: site.new
        for old, site in exceptions.items()
        if not site.new.endswith("Error")
    }
    assert bad == {}


@pytest.mark.parametrize("lane", LANES)
def test_no_handler_found_by_a_computed_name_is_mapped(lane: str) -> None:
    bad = []
    for old, site in survey(lane, "function").items():
        if site.owner is None or site.node is None:
            continue
        prefixes = DISPATCH.findall(ast.unparse(site.owner))
        if any(site.node.name.startswith(prefix) for prefix in prefixes):
            bad.append(old)
    assert bad == []


def test_the_vocabulary_holds_only_verbs_in_use() -> None:
    used = {
        site.new for lane in LANES for site in survey(lane, "function").values()
    }
    assert sorted(VERBS - used) == []


def test_a_noun_given_to_a_function_is_caught() -> None:
    assert "fisher" not in VERBS
    assert "detectable" not in VERBS
    assert "centred" not in VERBS


def test_a_property_counts_as_an_attribute() -> None:
    tree = ast.parse(
        "class A:\n    @property\n    def size_bytes(self):\n        return 1\n"
    )
    node, owner = walk(tree.body, ["A", "size_bytes"])
    assert kind(node) == "attribute"
    assert owner is tree.body[0]


def test_an_exception_is_told_from_a_class() -> None:
    tree = ast.parse(
        "class A(errors.ScmleError):\n    pass\nclass B(Base):\n    pass\n"
    )
    assert kind(tree.body[0]) == "exception"
    assert kind(tree.body[1]) == "class"


def test_a_computed_handler_prefix_is_found() -> None:
    source = (
        "class Loop:\n"
        "    def execute(self, args):\n"
        '        return getattr(self, f"run_{args.action}")(args)\n'
    )
    owner = ast.parse(source).body[0]
    assert DISPATCH.findall(ast.unparse(owner)) == ["run_"]


def test_a_definition_that_is_gone_is_left_to_the_dry_run() -> None:
    assert locate("scmle.pipeline.context.Context.check_cancelled") == (
        None,
        None,
    )
