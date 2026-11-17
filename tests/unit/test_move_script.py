"""The layer-map codemod moves modules and proves the change is rename-only."""

import importlib.util
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "move.py"
MAP = """
[packages]
low = "The bottom layer."

[[step]]
name = "low"
moves = [["scmle.alpha.retry", "scmle.low.retry"]]
"""
FILES = {
    "scmle/__init__.py": '"""Package."""\n',
    "scmle/alpha/__init__.py": '"""Alpha."""\n',
    "scmle/alpha/retry.py": '"""Retries."""\n\n\ndef again():\n    return 1\n',
    "scmle/alpha/user.py": (
        '"""Uses retry in every form."""\n\n'
        "import scmle.alpha.retry\n"
        "from scmle.alpha import retry\n\n"
        'SITE = "scmle.alpha.retry:again"\n'
        'FILE = "scmle/alpha/retry.py"\n\n\n'
        "def run():\n"
        "    from scmle.alpha.retry import again\n\n"
        "    return retry.again() + scmle.alpha.retry.again() + again()\n"
    ),
    "gates.toml": 'site = "scmle.alpha.retry:again"\n',
    "docs/RENAMES.md": "# Renames\n",
    "docs/GUIDE.md": (
        "See `alpha.retry` and `scmle/alpha/retry.py`.\n"
        "from scmle.alpha import retry\n"
    ),
}


def load():
    """Imports the script as a module, since `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location("move", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(root, *arguments):
    subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True
    )


@pytest.fixture
def tree(tmp_path):
    """A committed toy package with one module to move."""
    for name, text in FILES.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/layers.toml").write_text(MAP, encoding="utf-8")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(
        tmp_path,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-qm",
        "start",
    )
    return tmp_path


def run(module, tree, *extra):
    map_path = str(tree / "scripts/layers.toml")
    return module.main(["low", "--root", str(tree), "--map", map_path, *extra])


def test_rename_follows_the_longest_move():
    module = load()
    moves = {"scmle.a": "scmle.b", "scmle.a.c": "scmle.d"}
    assert module.rename("scmle.a.c.e", moves) == "scmle.d.e"
    assert module.rename("scmle.a.x", moves) == "scmle.b.x"
    assert module.rename("scmle.ab", moves) == "scmle.ab"


def test_step_rewrites_every_reference_and_verifies(tree):
    module = load()
    assert run(module, tree) == 0
    assert (tree / "scmle/low/retry.py").exists()
    assert not (tree / "scmle/alpha/retry.py").exists()
    assert (tree / "scmle/low/__init__.py").read_text() == (
        '"""The bottom layer."""\n'
    )
    user = (tree / "scmle/alpha/user.py").read_text()
    assert "import scmle.low.retry" in user
    assert "from scmle.low import retry" in user
    assert "from scmle.low.retry import again" in user
    assert '"scmle.low.retry:again"' in user
    assert '"scmle/low/retry.py"' in user
    assert "scmle.low.retry.again()" in user
    assert "scmle.alpha.retry" not in user
    assert (
        tree / "gates.toml"
    ).read_text() == 'site = "scmle.low.retry:again"\n'
    guide = (tree / "docs/GUIDE.md").read_text()
    assert guide == (
        "See `low.retry` and `scmle/low/retry.py`.\nfrom scmle.low import retry\n"
    )
    assert "`scmle.alpha.retry` -> `scmle.low.retry`" in (
        (tree / "docs/RENAMES.md").read_text()
    )
    assert run(module, tree, "--verify") == 0


def test_the_map_itself_is_never_rewritten(tree):
    module = load()
    run(module, tree)
    assert (tree / "scripts/layers.toml").read_text() == MAP


def test_verify_rejects_a_planted_edit(tree, capsys):
    module = load()
    run(module, tree)
    path = tree / "scmle/alpha/user.py"
    path.write_text(
        path.read_text().replace("return retry", "return 2 + retry")
    )
    assert run(module, tree, "--verify") == 1
    assert "scmle/alpha/user.py: not a rename" in capsys.readouterr().out
    assert run(module, tree, "--verify", "--hand", "scmle/alpha/user.py") == 0


def test_verify_rejects_a_double_binding(tree, capsys):
    module = load()
    run(module, tree)
    path = tree / "scmle/alpha/user.py"
    text = path.read_text()
    path.write_text(
        text.replace('"""\n\n', '"""\n\nfrom scmle import retry\n', 1)
    )
    assert run(module, tree, "--verify") == 1
    assert "binds <module>: retry twice" in capsys.readouterr().out


def test_a_rename_that_would_clash_is_refused():
    module = load()
    moves = {"scmle.alpha.ranking.wrapper": "scmle.beta.ranking"}
    known = {"scmle.alpha.ranking", "scmle.alpha.ranking.wrapper"}
    text = (
        "from scmle.alpha import ranking\n"
        "from scmle.alpha.ranking import wrapper\n\n"
        "VALUE = (ranking, wrapper)\n"
    )
    with pytest.raises(module.ConflictError):
        module.rewrite("test.py", text, moves, known)
