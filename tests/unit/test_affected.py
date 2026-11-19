"""The affected-test selector never misses a test a change can reach."""

import importlib.util
import pathlib
import subprocess
import sys
from typing import Any

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "affected.py"
SPEC = importlib.util.spec_from_file_location("affected", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
affected = importlib.util.module_from_spec(SPEC)
sys.modules["affected"] = affected
SPEC.loader.exec_module(affected)

FILES = {
    "pyproject.toml": '[project.scripts]\ntool = "pkg.cli:main"\n',
    "pkg/__init__.py": "",
    "pkg/low.py": "VALUE = 1\n",
    "pkg/mid.py": "from pkg import low\n",
    "pkg/top.py": "from . import mid\n",
    "pkg/lazy.py": "def load():\n    import pkg.deep.leaf\n",
    "pkg/deep/__init__.py": "",
    "pkg/deep/leaf.py": "X = 2\n",
    "pkg/cli.py": "def main():\n    return 0\n",
    "pkg/plugins/__init__.py": "",
    "pkg/plugins/one.py": "",
    "pkg/registry.py": (
        "import importlib\n\n\n"
        "def find(name):\n"
        '    return importlib.import_module(f"pkg.plugins.{name}")\n'
    ),
    "pkg/data.json": "{}\n",
    "pkg/reader.py": (
        "import pathlib\n\n"
        'DATA = pathlib.Path(__file__).with_name("data.json")\n'
    ),
    "docs/guide.md": "See pkg/low.py.\n",
    "tests/conftest.py": "",
    "tests/support.py": "import pkg.low\n",
    "tests/fixtures/sample.csv": "a\n1\n",
    "tests/unit/conftest.py": "",
    "tests/unit/test_low.py": "import support\n",
    "tests/unit/test_top.py": "from pkg.top import mid\n",
    "tests/unit/test_lazy.py": "def test():\n    from pkg import lazy\n",
    "tests/unit/test_reader.py": "from pkg import reader\n",
    "tests/unit/test_registry.py": "import pkg.registry\n",
    "tests/unit/test_string.py": (
        'def test(monkeypatch):\n    monkeypatch.setattr("pkg.cli.main", 1)\n'
    ),
    "tests/unit/test_cli.py": (
        'import subprocess\n\nsubprocess.run(["tool", "--help"], check=False)\n'
    ),
    "tests/unit/test_fixture.py": (
        "import pathlib\n\n"
        "ROOT = pathlib.Path(__file__).parents[2]\n"
        'SAMPLE = ROOT / "tests" / "fixtures" / "sample.csv"\n'
    ),
    "tests/unit/test_docs.py": (
        "import pathlib\n\n"
        "ROOT = pathlib.Path(__file__).parents[2]\n"
        'PAGES = sorted((ROOT / "docs").glob("*.md"))\n'
    ),
    "tests/unit/test_walk.py": (
        "import pathlib\n\n"
        "ROOT = pathlib.Path(__file__).parents[2]\n"
        'FILES = list(ROOT.rglob("*.py"))\n'
    ),
    "tests/unit/test_folders.py": (
        "import pathlib\n\n"
        "ROOT = pathlib.Path(__file__).parents[2]\n"
        'FOLDERS = ("pkg",)\n'
        "FILES = [p for f in FOLDERS for p in (ROOT / f).rglob('*.py')]\n"
    ),
    "tests/unit/test_alone.py": "def test():\n    assert True\n",
    "tests/other/test_other.py": "import json\n",
}
EVERYWHERE = {"tests/unit/test_walk.py"}


def build(root: pathlib.Path, files: dict[str, str] = FILES) -> Any:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return affected.Graph(root, sorted(files), root / "cache.json")


@pytest.fixture
def graph(tmp_path: pathlib.Path) -> Any:
    return build(tmp_path)


def test_scan_reads_imports_at_any_depth_and_relative() -> None:
    facts = affected.scan(
        "import a.b\nfrom c import d\nfrom . import e\n"
        "def f():\n    from .g import h\n",
        "pkg/sub/mod.py",
    )
    assert {"a", "a.b", "c", "c.d", "pkg.sub", "pkg.sub.e"} <= facts.imported
    assert {"pkg.sub.g", "pkg.sub.g.h"} <= facts.imported


def test_scan_reads_paths_from_strings_chains_and_formats() -> None:
    facts = affected.scan(
        "import pathlib\n"
        "ROOT = pathlib.Path(__file__).parent\n"
        'A = ROOT / "tests" / "fixtures" / "a.json"\n'
        'B = "./docs/guide.md"\n'
        'C = f"data/frames/{name}.pq"\n'
        'D = "website/src/*.astro"\n'
        'E = "results/{}.json"\n',
        "tests/unit/test_x.py",
    )
    assert {
        "tests/fixtures/a.json",
        "docs/guide.md",
        "data/frames",
        "website/src",
        "results",
    } <= facts.read


def test_scan_reads_dotted_strings_and_formatted_packages() -> None:
    facts = affected.scan(
        'A = "pkg.mod:func"\nB = f"pkg.plugins.{n}"\n', "x.py"
    )
    assert {"pkg", "pkg.mod", "pkg.mod.func"} <= facts.imported
    assert "pkg.plugins.*" in facts.imported


def test_scan_names_console_scripts_and_run_modules() -> None:
    facts = affected.scan('ARGS = ["tool", "-m", "pkg.cli"]\n', "t.py")
    assert {"!tool", "pkg.cli.__main__"} <= facts.imported
    assert "pkg.__main__" not in facts.imported
    assert (
        "pkg.__main__" not in affected.scan('A = "pkg.cli"\n', "t.py").imported
    )
    config = affected.scan("run: tool fit\nother: python -m pkg\n", "ci.yml")
    assert {"!tool", "pkg.__main__"} <= config.mentioned


def test_scan_keeps_bare_imports_apart_from_strings() -> None:
    assert "~support" in affected.scan("import support\n", "t.py").imported
    assert "~support" not in affected.scan('A = "support"\n', "t.py").imported


def test_scan_marks_listing_from_the_root_as_exhaustive() -> None:
    assert affected.scan(FILES["tests/unit/test_walk.py"], "t.py").exhaustive
    assert affected.scan('ARGS = ["git", "ls-files"]\n', "t.py").exhaustive
    assert not affected.scan(
        FILES["tests/unit/test_docs.py"], "t.py"
    ).exhaustive


def test_scan_tolerates_syntax_errors_and_text_files() -> None:
    broken = affected.scan("def (:\n import pkg.low\n", "bad.py")
    assert "pkg.low" in broken.imported
    text = affected.scan("module = pkg.low see docs/guide.md\n", "a.toml")
    assert {"pkg.low", "/docs/guide.md"} <= text.mentioned
    assert not text.read
    assert "pkg.low" not in text.imported
    snippet = affected.scan("```python\nimport pkg.low\n```\n", "a.md")
    assert "pkg.low" in snippet.imported


def test_identify_names_modules_paths_and_folders() -> None:
    names = affected.identify("tests/support.py", {})
    assert {"~support", "tests.support", "/tests/support.py", "/tests"} <= names
    assert "support" not in names
    names = affected.identify("pkg/deep/leaf.py", {"pkg.deep.leaf": {"tool"}})
    assert {"pkg.deep.leaf", "pkg.deep.*", "pkg.*", "!tool"} <= names
    assert {"/leaf.py", "/deep/leaf.py", "/pkg/deep", "/deep"} <= names
    assert "deep" not in names
    assert "pkg.deep" in affected.identify("pkg/deep/__init__.py", {})


def expect(graph: Any, changed: list[str], tests: set[str]) -> None:
    assert set(graph.select(changed)) == set(tests) | EVERYWHERE


def test_transitive_import_selects_only_dependent_tests(graph: Any) -> None:
    expect(
        graph,
        ["pkg/low.py"],
        {
            "tests/unit/test_low.py",
            "tests/unit/test_top.py",
            "tests/unit/test_folders.py",
        },
    )


def test_prose_mentions_count_only_for_deleted_files(
    graph: Any, tmp_path: pathlib.Path
) -> None:
    assert "tests/unit/test_docs.py" not in graph.select(["pkg/low.py"])
    (tmp_path / "pkg" / "low.py").unlink()
    assert "tests/unit/test_docs.py" in graph.select(["pkg/low.py"])


def test_package_init_change_reaches_submodule_importers(graph: Any) -> None:
    selected = set(graph.select(["pkg/deep/__init__.py"]))
    assert "tests/unit/test_lazy.py" in selected
    assert "tests/unit/test_alone.py" not in selected


def test_data_read_by_an_imported_module_selects_the_test(graph: Any) -> None:
    expect(
        graph,
        ["pkg/data.json"],
        {"tests/unit/test_reader.py", "tests/unit/test_folders.py"},
    )


def test_fixture_and_doc_paths_select_their_readers(graph: Any) -> None:
    expect(graph, ["tests/fixtures/sample.csv"], {"tests/unit/test_fixture.py"})
    expect(graph, ["docs/guide.md"], {"tests/unit/test_docs.py"})


def test_formatted_import_selects_every_module_of_the_package(
    graph: Any,
) -> None:
    selected = set(graph.select(["pkg/plugins/one.py"]))
    assert "tests/unit/test_registry.py" in selected


def test_dotted_string_and_console_script_select_their_tests(
    graph: Any,
) -> None:
    selected = set(graph.select(["pkg/cli.py"]))
    assert {"tests/unit/test_string.py", "tests/unit/test_cli.py"} <= selected


def test_conftest_and_global_files_select_all(graph: Any) -> None:
    every = {name for name in FILES if affected.tested(name)}
    for changed in (
        "tests/unit/conftest.py",
        "tests/conftest.py",
        "pyproject.toml",
        "uv.lock",
        "pkg/conftest.py",
    ):
        assert set(graph.select([changed])) == every, changed
    assert set(graph.select(["pkg/low.py", "uv.lock"])) == every


def test_changed_and_deleted_files(graph: Any, tmp_path: pathlib.Path) -> None:
    expect(graph, ["tests/unit/test_alone.py"], {"tests/unit/test_alone.py"})
    (tmp_path / "pkg" / "low.py").unlink()
    graph = affected.Graph(tmp_path, sorted(FILES), tmp_path / "cache.json")
    selected = set(graph.select(["pkg/low.py"]))
    assert "tests/unit/test_low.py" in selected
    assert graph.select([]) == []


def test_unreferenced_data_file_falls_back_to_every_test(graph: Any) -> None:
    every = {name for name in FILES if affected.tested(name)}
    assert set(graph.select(["notes/elsewhere.txt"])) == every
    assert set(graph.select([".github/workflows/ci.yml"])) == every


def test_changelog_fragment_selects_only_exhaustive_tests(graph: Any) -> None:
    expect(graph, ["changelog.d/entry.md"], set())


def test_unreferenced_python_file_selects_only_exhaustive_tests(
    graph: Any,
) -> None:
    expect(graph, ["scripts/orphan.py"], set())


def test_selection_covers_every_test_that_imports_a_changed_module(
    graph: Any,
) -> None:
    for module in ("pkg/low.py", "pkg/mid.py", "pkg/deep/leaf.py"):
        name = module.removesuffix(".py").replace("/", ".")
        importers = {
            path
            for path, facts in graph.facts.items()
            if affected.tested(path) and name in facts.imported
        }
        assert importers <= set(graph.select([module])), module


def test_cache_reuses_facts_for_unchanged_files(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build(tmp_path)
    calls = []
    original = affected.scan

    def counted(text: str, path: str) -> Any:
        calls.append(path)
        return original(text, path)

    monkeypatch.setattr(affected, "scan", counted)
    (tmp_path / "pkg" / "low.py").write_text("VALUE = 3\n", encoding="utf-8")
    graph = affected.Graph(tmp_path, sorted(FILES), tmp_path / "cache.json")
    assert calls == ["pkg/low.py"]
    assert "tests/unit/test_low.py" in graph.select(["pkg/low.py"])


def test_corrupt_cache_is_rebuilt(tmp_path: pathlib.Path) -> None:
    (tmp_path / "cache.json").write_text("{not json", encoding="utf-8")
    graph = build(tmp_path)
    assert "tests/unit/test_low.py" in graph.select(["pkg/low.py"])


def git(root: pathlib.Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True
    )


def test_cli_lists_tests_for_the_working_tree(tmp_path: pathlib.Path) -> None:
    build(tmp_path)
    ignored = "cache.json\n.pytest_cache/\n"
    (tmp_path / ".gitignore").write_text(ignored, "utf-8")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "-A")
    git(
        tmp_path,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-qm",
        "base",
    )
    (tmp_path / "pkg" / "deep" / "leaf.py").write_text("X = 3\n", "utf-8")
    (tmp_path / "tests" / "unit" / "test_new.py").write_text("", "utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--list", "--root", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    listed = set(result.stdout.split())
    assert (
        listed
        == {
            "tests/unit/test_lazy.py",
            "tests/unit/test_new.py",
            "tests/unit/test_folders.py",
        }
        | EVERYWHERE
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--list",
            "--root",
            str(tmp_path),
            "--base",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert set(result.stdout.split()) == listed


def test_cli_rejects_unknown_base(tmp_path: pathlib.Path) -> None:
    build(tmp_path)
    git(tmp_path, "init", "-q")
    with pytest.raises(affected.GitError):
        affected.compare(tmp_path, "nonexistent-ref")
