"""What a page finds does not depend on where the snippet check runs (C12).

The same docs once ran 211/237 snippets in a scratch clone and 222/237 in
the working tree: the pages ran the `python` of whichever environment
started the check (an editable install of another checkout, with whatever
extras it held) and found tools from the caller's `PATH`. Each page now
gets its own environment built from the working-tree snapshot and only the
tools of `TOOLS`.
"""

import os
import pathlib
import shutil
import stat
import subprocess
import textwrap

import pytest

from benchmarks.evidence import snippets
from scmle.contracts import errors

uv = shutil.which("uv")
needs = pytest.mark.skipif(uv is None, reason="the bench needs uv")


def project(folder):
    """A one-module project with a lock file, committed like a snapshot."""
    folder.mkdir()
    (folder / "pyproject.toml").write_text(
        textwrap.dedent(
            """\
            [project]
            name = "probe"
            version = "1.0"
            requires-python = ">=3.11"
            dependencies = []

            [build-system]
            requires = ["hatchling"]
            build-backend = "hatchling.build"
            """
        ),
        encoding="utf-8",
    )
    package = folder / "probe"
    package.mkdir()
    (package / "__init__.py").write_text("NAME = 'probe'\n", encoding="utf-8")
    subprocess.run(
        [uv, "lock", "--quiet"], cwd=folder, check=True, capture_output=True
    )
    return folder


def page(text):
    """Snippets of a one-block page."""
    return snippets.extract(f"```bash\n{text}\n```\n", "docs/PAGE.md")


@needs
def test_a_page_finds_only_its_environment_the_tools_and_the_system(
    tmp_path, monkeypatch
):
    # A tool only the caller has must stay out of reach of every page.
    hidden = tmp_path / "caller"
    hidden.mkdir()
    tool = hidden / "only-here"
    tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{hidden}{os.pathsep}{os.environ['PATH']}")
    source = project(tmp_path / "source")
    bench = snippets.furnish(source, tmp_path)
    found = page(
        "command -v uv && command -v git && command -v pip\n"
        "! command -v only-here\n"
        'python -c "import probe, sys; '
        "assert 'source' not in probe.__file__, probe.__file__; "
        'assert sys.prefix != sys.base_prefix"'
    )

    (record,) = snippets.follow(found, tmp_path, 120.0, bench=bench)

    assert record["status"] == "ran", record["reason"]


@needs
def test_each_page_gets_a_fresh_environment(tmp_path):
    source = project(tmp_path / "source")
    bench = snippets.furnish(source, tmp_path)
    found = page(
        'test ! -e "$(python -c "import sys; print(sys.prefix)")/marked"\n'
        'touch "$(python -c "import sys; print(sys.prefix)")/marked"'
    )

    first = snippets.follow(found, tmp_path, 120.0, bench=bench)
    second = snippets.follow(found, tmp_path, 120.0, bench=bench)

    assert [r["status"] for r in first + second] == ["ran", "ran"]


def test_a_missing_tool_is_an_input_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))

    with pytest.raises(errors.InputError, match="need uv"):
        snippets.furnish(tmp_path, tmp_path)


@needs
def test_a_snapshot_that_cannot_install_is_an_input_error(tmp_path):
    broken = tmp_path / "source"
    broken.mkdir()
    (broken / "pyproject.toml").write_text("[project\n", encoding="utf-8")
    bench = snippets.furnish(broken, tmp_path)

    with pytest.raises(errors.InputError, match="cannot install"):
        snippets.install(bench, tmp_path / "environment")


def test_the_definition_names_the_bench():
    assert snippets.TOOLS == ("uv", "git")
    text = pathlib.Path(snippets.__file__).read_text(encoding="utf-8")
    assert "--no-editable" in text
    assert "sys.executable" in text  # only for a terminal without a bench
