"""The owner-rule checker flags each rule's violations and nothing else."""

import importlib.util
import json
import pathlib
import textwrap

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "rules.py"


def load():
    """Imports the script as a module, since `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location("rules", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rules = load()


def found(source, rule, path="scmle/sample.py"):
    """The messages one rule finds in a dedented source."""
    text = textwrap.dedent(source)
    return [item.text for item in rules.inspect(path, text, [rule])]


@pytest.mark.parametrize(
    "source",
    [
        "def read_json():\n    return 1\n",
        "class FooBar:\n    pass\n",
        "class Box:\n    max_depth = 3\n",
        "class Box:\n    def grow(self):\n        self.max_depth = 3\n",
    ],
)
def test_compound_flags_multi_word_definitions(source):
    assert found(source, "compound")


@pytest.mark.parametrize(
    "source",
    [
        "def read():\n    return 1\n",
        "class ConflictError(Exception):\n    pass\n",
        "class Box:\n    def predict_proba(self):\n        return 1\n",
        "class Box:\n    def grow(self, other):\n        other.max_depth = 3\n",
        "some_value = 3\n",
    ],
)
def test_compound_passes_single_words_exemptions_and_foreign_stores(source):
    assert not found(source, "compound")


def test_compound_reads_the_package_and_scripts_only():
    source = "def test_read_json():\n    return 1\n"

    assert not found(source, "compound", "tests/unit/test_x.py")
    assert found(source, "compound", "scripts/x.py")


@pytest.mark.parametrize(
    "source",
    ["def _a():\n    return 1\n", "x = y._z\n", "def f(_a):\n    return 1\n"],
)
def test_underscore_flags_leading_underscores(source):
    assert found(source, "underscore")


def test_underscore_passes_dunders():
    assert not found(
        "class A:\n    def __init__(self):\n        x = 1\n", "underscore"
    )


def test_alias_flags_renamed_imports_only():
    assert found("import numpy as np\n", "alias") == [
        "import alias numpy as np"
    ]
    assert found("from os import path as p\n", "alias")
    assert not found("import numpy\nfrom os import path\n", "alias")


@pytest.mark.parametrize(
    "source",
    [
        "try:\n    f()\nexcept:\n    g()\n",
        "try:\n    f()\nexcept Exception:\n    log()\n",
        "try:\n    f()\nexcept (KeyError, BaseException):\n    log()\n",
        "try:\n    f()\nexcept Exception as e:\n    raise Exception(e)\n",
        "try:\n    f()\nexcept Exception: pass\n",
    ],
)
def test_swallow_flags_broad_handlers_that_carry_on(source):
    assert found(source, "swallow")


@pytest.mark.parametrize(
    "source",
    [
        "try:\n    f()\nexcept ValueError:\n    g()\n",
        "try:\n    f()\nexcept Exception:\n    log()\n    raise\n",
        (
            "try:\n    f()\nexcept Exception as e:\n"
            "    raise errors.InputError('x') from e\n"
        ),
        "try:\n    f()\nexcept Exception:  # handled: probe only\n    g()\n",
        "try:\n    f()\nexcept Exception: pass  # handled: probe only\n",
    ],
)
def test_swallow_passes_typed_reraises_markers_and_narrow_handlers(source):
    assert not found(source, "swallow")


@pytest.mark.parametrize(
    "body",
    [
        "pass",
        "...",
        "'''Only a docstring.'''",
        "raise NotImplementedError",
        "raise NotImplementedError('later')",
        "return",
        "return None",
        "return 0",
        "return ''",
        "return []",
        "return {}",
        "return ()",
    ],
)
def test_stub_flags_placeholder_bodies(body):
    assert found(f"def f():\n    {body}\n", "stub") == ["stub body in f"]


@pytest.mark.parametrize(
    "source",
    [
        "def f():\n    return False\n",
        "def f():\n    return 1\n",
        "def f(x):\n    '''Doc.'''\n    return [x]\n",
        "@abc.abstractmethod\ndef f():\n    ...\n",
        "@overload\ndef f(x: int) -> int: ...\n",
        "class P(typing.Protocol):\n    def f(self):\n        ...\n",
        "def close(self):  # handled: nothing to release\n    pass\n",
    ],
)
def test_stub_passes_real_bodies_and_declarations(source):
    assert not found(source, "stub")


def test_stub_flags_todo_comments_and_reads_the_package_only():
    source = "x = 1  # TODO: finish\n# FIXME later\n"

    assert found(source, "stub") == ["TODO comment", "FIXME comment"]
    assert not found("def f():\n    pass\n", "stub", "tests/unit/test_x.py")


def test_document_flags_missing_and_foreign_docstrings():
    source = '''
    def f():
        return 1

    def g(x):
        """Sphinx.

        :param x: a value
        """
        return x

    def h(x):
        """NumPy.

        Parameters
        ----------
        x : int
        """
        return x

    class A:
        def method(self):
            return 1
    '''

    assert found(source, "document") == [
        "no docstring on f",
        "docstring of g is not Google style",
        "docstring of h is not Google style",
        "no docstring on A",
        "no docstring on method",
    ]


def test_document_passes_google_private_nested_and_overloads():
    source = '''
    def f(x):
        """Google.

        Args:
            x: A value.
        """

        def inner():
            return x

        return inner

    class Box:
        """A box."""

        def __repr__(self):
            return "Box"

    @overload
    def g(x: int) -> int: ...
    '''

    assert not found(source, "document")


def test_echo_flags_print_outside_the_printing_packages():
    assert found("print(1)\n", "echo") == ["print() in library code"]
    assert not found("print(1)\n", "echo", "scmle/cli/main.py")
    assert not found("print(1)\n", "echo", "scripts/x.py")


def test_inspect_skips_generated_mutants_and_non_python_files():
    assert not rules.inspect("mutants/scmle/a.py", "import a as b\n", ["alias"])
    assert not rules.inspect("scmle/a.txt", "import a as b\n", ["alias"])


@pytest.fixture
def repository(tmp_path, monkeypatch):
    """A fake checkout: base versions in a dict, current files on disk."""
    base = {}

    def git(*arguments):
        if arguments[0] == "show":
            return base.get(arguments[1].partition(":")[2])
        return ""

    monkeypatch.setattr(rules, "ROOT", tmp_path)
    monkeypatch.setattr(rules, "git", git)
    (tmp_path / "scmle").mkdir()
    return tmp_path, base


def test_ratchet_lets_old_names_stay_and_fails_new_ones(repository):
    root, base = repository
    base["scmle/a.py"] = '"""A."""\n\n\ndef read_json():\n    """R."""\n'
    (root / "scmle" / "a.py").write_text(
        base["scmle/a.py"] + '\n\ndef write_json():\n    """W."""\n',
        encoding="utf-8",
    )

    new = rules.ratchet(["scmle/a.py"], "HEAD", ["compound"])

    assert [item.text for item in new] == ["multi-word name write_json"]


def test_ratchet_lets_a_definition_move_between_changed_files(repository):
    root, base = repository
    base["scmle/a.py"] = "def read_json():\n    return 1\n"
    base["scmle/b.py"] = ""
    (root / "scmle" / "a.py").write_text("", encoding="utf-8")
    (root / "scmle" / "b.py").write_text(
        "def read_json():\n    return 1\n", encoding="utf-8"
    )

    assert not rules.ratchet(["scmle/a.py", "scmle/b.py"], "HEAD", ["compound"])


def test_ratchet_fails_a_rename_to_another_multi_word_name(repository):
    root, base = repository
    base["scmle/a.py"] = "def read_json():\n    return 1\n"
    (root / "scmle" / "a.py").write_text(
        "def load_json():\n    return 1\n", encoding="utf-8"
    )

    new = rules.ratchet(["scmle/a.py"], "HEAD", ["compound"])

    assert [item.text for item in new] == ["multi-word name load_json"]


def test_ratchet_counts_repeated_violations(repository):
    root, base = repository
    handler = "try:\n    f()\nexcept Exception:\n    g()\n"
    base["scmle/a.py"] = handler
    (root / "scmle" / "a.py").write_text(handler * 2, encoding="utf-8")

    new = rules.ratchet(["scmle/a.py"], "HEAD", ["swallow"])

    assert [item.line for item in new] == [7]


def test_check_fails_strict_rules_on_untouched_files(repository, capsys):
    root = repository[0]
    (root / "scmle" / "a.py").write_text("import os as o\n", encoding="utf-8")

    status = rules.main([str(root / "scmle" / "a.py")])

    assert status == 1
    assert "scmle/a.py:1 alias: import alias os as o" in capsys.readouterr().out


def test_check_passes_a_clean_file(repository):
    root = repository[0]
    (root / "scmle" / "a.py").write_text("import os\n", encoding="utf-8")

    assert rules.main([str(root / "scmle" / "a.py")]) == 0


def test_ceiling_fails_a_multi_word_count_above_the_budget(
    tmp_path, monkeypatch
):
    package = tmp_path / "scmle"
    package.mkdir()
    (package / "a.py").write_text(
        "def read_json():\n    return 1\n", encoding="utf-8"
    )
    budget = tmp_path / "structure.json"
    budget.write_text(json.dumps({"compound_functions": 0}), encoding="utf-8")
    monkeypatch.setattr(rules.STRUCTURE, "PACKAGE", package)
    monkeypatch.setattr(rules.STRUCTURE, "BUDGET", budget)

    assert rules.ceiling() == [
        "structure.json: compound_functions is 1, above 0"
    ]
