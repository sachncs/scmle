"""The beginner documents stay runnable, linked and complete.

These tests are cheap: they train nothing. They check that every command in a
runnable shell block parses with the real command-line parser, that every
runnable Python block parses and only uses names that exist, that the
glossary covers the terms a newcomer meets and that each is used elsewhere,
and that the documents are linked from the maps.
"""

import ast
import contextlib
import importlib
import inspect
import io
import json
import pathlib
import re
import shlex

import pytest

from scmle import cli

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
BEGINNER = (
    "START_HERE.md",
    "GLOSSARY.md",
    "DECISIONS.md",
    "READING_RESULTS.md",
    "WHEN_THINGS_GO_WRONG.md",
    "WHAT_SCMLE_CANNOT_DO.md",
    "LEAKAGE.md",
)
FENCE = re.compile(r"```(\w+)([^\n]*)\n(.*?)```", re.S)
REQUIRED_TERMS = (
    "target",
    "feature",
    "metric",
    "roc auc",
    "overfitting",
    "leakage",
    "held-out data",
    "cross-validation",
    "calibration",
    "threshold",
    "drift",
    "charter",
    "custody",
    "tier",
    "e-value",
    "conformal",
    "causal claim",
    "closed loop",
    "escalation",
    "bundle",
    "signature",
    "sbom",
    "provenance",
    "registry",
    "canary",
    "rto",
    "rpo",
    "unverified",
    "attestation",
    "confirmed",
)


def blocks(name, language, marker=None):
    """Returns the fenced blocks of a document in a language."""
    text = (DOCS / name).read_text(encoding="utf-8")
    found = []
    for lang, info, body in FENCE.findall(text):
        if lang != language:
            continue
        if marker is not None and marker not in info.split():
            continue
        found.append(body)
    return found


def commands(block):
    """Splits a shell block into commands, joining continued lines."""
    found = []
    pending = ""
    for line in block.splitlines():
        pending = (pending + " " + line).strip() if pending else line.strip()
        if pending.endswith("\\"):
            pending = pending[:-1]
            continue
        try:
            shlex.split(pending)
        except ValueError:
            continue
        if pending and not pending.startswith("#"):
            found.append(pending)
        pending = ""
    assert not pending, f"unterminated command: {pending}"
    return found


def scmle_arguments(command):
    """Returns the arguments after `scmle`, or None for another program."""
    tokens = shlex.split(command)
    while tokens and re.fullmatch(r"\w+=.*", tokens[0]):
        tokens = tokens[1:]
    if tokens and tokens[0] == "scmle":
        return tokens[1:]
    return None


def all_scmle_commands():
    found = []
    for name in BEGINNER:
        for body in blocks(name, "bash"):
            for command in commands(body):
                arguments = scmle_arguments(command)
                if arguments is not None:
                    found.append((name, command, arguments))
    return found


def test_there_are_runnable_commands_to_check():
    assert len(all_scmle_commands()) >= 20


@pytest.mark.parametrize(
    ("name", "command", "arguments"),
    all_scmle_commands(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_every_scmle_command_in_the_docs_parses(name, command, arguments):
    parser = cli.build_parser()
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        try:
            parser.parse_args(arguments)
        except SystemExit as stop:
            if stop.code == 0:
                return
            pytest.fail(f"{name}: `{command}` is refused: {stderr.getvalue()}")


def test_start_here_marks_every_block_with_a_command_as_runnable():
    text = (DOCS / "START_HERE.md").read_text(encoding="utf-8")
    for lang, info, body in FENCE.findall(text):
        if lang == "bash" and "runnable" not in info.split():
            for command in commands(body):
                assert scmle_arguments(command) is None, command


def aliases(tree):
    """Maps imported names to their modules or objects."""
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                found[item.asname or item.name.split(".")[0]] = (
                    importlib.import_module(item.name.split(".")[0])
                )
        elif isinstance(node, ast.ImportFrom) and node.module:
            module = importlib.import_module(node.module)
            for item in node.names:
                try:
                    found[item.asname or item.name] = getattr(module, item.name)
                except AttributeError:
                    found[item.asname or item.name] = importlib.import_module(
                        f"{node.module}.{item.name}"
                    )
    return found


def resolve(node, known):
    """Resolves a dotted chain rooted in an imported name, else None."""
    names = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name) or node.id not in known:
        return None
    target = known[node.id]
    for attribute in reversed(names):
        target = getattr(target, attribute)
    return target


def python_blocks():
    found = []
    for name in BEGINNER:
        for index, body in enumerate(blocks(name, "python", "runnable")):
            found.append((name, index, body))
    return found


def test_there_are_runnable_python_blocks():
    assert len(python_blocks()) >= 5


@pytest.mark.parametrize(
    ("name", "index", "body"),
    python_blocks(),
    ids=lambda value: (
        value if isinstance(value, str) and len(value) < 40 else None
    ),
)
def test_runnable_python_blocks_use_names_that_exist(name, index, body):
    tree = ast.parse(body)
    known = aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            resolve(node, known)
        if isinstance(node, ast.Call):
            target = resolve(node.func, known)
            if target is None or not callable(target):
                continue
            try:
                parameters = inspect.signature(target).parameters
            except (TypeError, ValueError):
                continue
            if any(p.kind is p.VAR_KEYWORD for p in parameters.values()):
                continue
            for keyword in node.keywords:
                assert keyword.arg in parameters, (
                    f"{name} block {index}: {keyword.arg} is not an "
                    f"argument of {ast.unparse(node.func)}"
                )


@pytest.mark.parametrize(
    ("name", "index", "body"),
    python_blocks(),
    ids=lambda value: (
        value if isinstance(value, str) and len(value) < 40 else None
    ),
)
def test_scripts_that_train_keep_the_main_guard(name, index, body):
    if ".run()" in body:
        assert 'if __name__ == "__main__":' in body, (
            f"{name} block {index} trains without a main guard"
        )


def glossary_terms():
    text = (DOCS / "GLOSSARY.md").read_text(encoding="utf-8")
    terms = []
    for heading in re.findall(r"^### (.+)$", text, re.M):
        plain = re.sub(r"\([^)]*\)", "", heading)
        for part in plain.split(","):
            part = part.strip()
            if len(part) >= 3:
                terms.append(part)
    return terms


def test_the_glossary_has_every_required_term():
    text = (DOCS / "GLOSSARY.md").read_text(encoding="utf-8").lower()
    headings = " ".join(re.findall(r"^### (.+)$", text, re.M))
    missing = [t for t in REQUIRED_TERMS if t not in headings and t not in text]
    assert not missing, missing
    for term in ("custody", "tier", "rto", "rpo", "canary", "e-value"):
        assert term in headings, term


def test_every_glossary_entry_has_meaning_example_and_reason():
    text = (DOCS / "GLOSSARY.md").read_text(encoding="utf-8")
    sections = re.split(r"^### ", text, flags=re.M)[1:]
    for section in sections:
        title = section.splitlines()[0]
        for label in ("**Meaning.**", "**Example.**", "**Why it matters.**"):
            assert label in section, f"{title} lacks {label}"


def test_every_glossary_term_is_used_somewhere_else():
    others = ""
    for path in [*DOCS.glob("*.md"), ROOT / "README.md"]:
        if path.name != "GLOSSARY.md":
            others += path.read_text(encoding="utf-8").lower() + "\n"
    unused = [t for t in glossary_terms() if t.lower() not in others]
    assert not unused, unused


LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


@pytest.mark.parametrize("name", BEGINNER)
def test_links_in_the_beginner_documents_resolve(name):
    path = DOCS / name
    text = path.read_text(encoding="utf-8")
    for target in LINK.findall(text):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        assert (path.parent / target).resolve().exists(), f"{name}: {target}"


@pytest.mark.parametrize("name", BEGINNER)
def test_the_maps_link_every_beginner_document(name):
    for path in (ROOT / "README.md", DOCS / "README.md"):
        assert name in path.read_text(encoding="utf-8"), (path.name, name)


@pytest.mark.parametrize("path", [ROOT / "README.md", DOCS / "README.md"])
def test_the_maps_list_start_here_before_the_reference_pages(path):
    text = path.read_text(encoding="utf-8")
    assert text.index("START_HERE.md") < text.index("TUTORIAL.md")


MEASUREMENTS = ROOT / "benchmarks" / "leakage_screen_measurements.json"


def measurements():
    return json.loads(MEASUREMENTS.read_text(encoding="utf-8"))


def test_leakage_doc_quotes_the_committed_measurements():
    text = (DOCS / "LEAKAGE.md").read_text(encoding="utf-8")
    record = measurements()
    certified = record["certified"]
    assert (
        f"**{certified['null_worlds_with_a_suspect']} of "
        f"{certified['null_replicates']}**"
    ) in text
    low, high = certified["null_false_alarm_wilson_95"]
    assert f"{low:.0f} to {high * 100:.2f} percent" in text
    per_type = certified["plant_replicates_per_type"]
    for found in certified["detected"].values():
        assert f"{found} of {per_type}" in text
    assert certified["skill_ceiling"] == 0.95
    assert "AUC 0.95" in text
    planted = record["planted"]
    detected = sum(row["detected"] for row in planted.values())
    total = sum(row["tables"] for row in planted.values())
    assert f"**{detected} of {total}**" in text
    flagged = sum(row["real_columns_flagged"] for row in planted.values())
    assert flagged == 5 * 7
    row = next(iter(planted.values()))
    assert (
        f"{row['real_columns_flagged']}\n  of the {row['real_columns_screened']}"
    ) in text


def test_leakage_doc_table_matches_the_measured_flags():
    text = (DOCS / "LEAKAGE.md").read_text(encoding="utf-8")
    for table in measurements()["tables"]:
        name = table["table"]
        for column in table["flagged"]:
            assert column in text, (name, column)
        count = f"({len(table['false_flags'])} of {table['columns']}"
        if table["false_flags"] and not table["leaks_found"]:
            assert count in text, (name, count)
    breast = next(
        t for t in measurements()["tables"] if t["table"] == "breast_cancer"
    )
    assert f"{breast['false_flag_rate'] * 100:.1f} percent" in text


def test_the_screen_is_reported_descriptive_for_multiclass_tables():
    for table in measurements()["tables"]:
        if table["classes"] > 2:
            assert table["gating"] == ["descriptive"]
            assert table["would_block"] == []
