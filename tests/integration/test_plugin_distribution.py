"""An installed distribution's `scmle.models` entry point is discovered.

The distribution is built and installed with pip in a temporary directory,
through a standard-library build backend shipped inside it, so the test needs
no network and no build tool beyond pip.
"""

import importlib
import subprocess
import sys
import textwrap

import pytest

from scmle.models import registry
from scmle.runtime import plugins

BACKEND = '''
"""A PEP 517 backend that writes a pure-Python wheel with zipfile."""

import base64
import hashlib
import pathlib
import zipfile

NAME, VERSION = "scmle_tiny_plugin", "0.1"
INFO = f"{NAME}-{VERSION}.dist-info"
FILES = {
    "tinyplugin.py": pathlib.Path("tinyplugin.py").read_text(),
    f"{INFO}/METADATA": (
        f"Metadata-Version: 2.1\\nName: {NAME}\\nVersion: {VERSION}\\n"
    ),
    f"{INFO}/WHEEL": (
        "Wheel-Version: 1.0\\nGenerator: tiny\\nRoot-Is-Purelib: true\\n"
        "Tag: py3-none-any\\n"
    ),
    f"{INFO}/entry_points.txt": pathlib.Path("entry_points.txt").read_text(),
}


def digest(text):
    raw = hashlib.sha256(text.encode()).digest()
    return "sha256=" + base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def build_wheel(directory, config_settings=None, metadata_directory=None):
    name = f"{NAME}-{VERSION}-py3-none-any.whl"
    record = "".join(
        f"{path},{digest(text)},{len(text.encode())}\\n"
        for path, text in FILES.items()
    ) + f"{INFO}/RECORD,,\\n"
    with zipfile.ZipFile(pathlib.Path(directory) / name, "w") as wheel:
        for path, text in FILES.items():
            wheel.writestr(path, text)
        wheel.writestr(f"{INFO}/RECORD", record)
    return name
'''

PLUGIN = '''
"""A plugin advertising one conforming adapter and one shadowing a built-in."""

from sklearn import naive_bayes

from scmle.models import custom

adapter = custom.rebuild(
    custom.Registration("tiny_bayes", naive_bayes.GaussianNB, None)
)
shadow = custom.rebuild(
    custom.Registration("random_forest", naive_bayes.GaussianNB, None)
)
'''

ENTRY_POINTS = """
[scmle.models]
tiny_bayes = tinyplugin:adapter
tiny_shadow = tinyplugin:shadow
"""


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """Builds and installs the distribution, then puts it on the path."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [build-system]
            requires = []
            build-backend = "backend"
            backend-path = ["."]
            """
        )
    )
    (source / "backend.py").write_text(BACKEND)
    (source / "tinyplugin.py").write_text(PLUGIN)
    (source / "entry_points.txt").write_text(ENTRY_POINTS)
    target = tmp_path / "site"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-index",
            "--no-deps",
            "--no-build-isolation",
            "--disable-pip-version-check",
            "--target",
            str(target),
            str(source),
        ],
        check=True,
        timeout=300,
    )
    monkeypatch.syspath_prepend(str(target))
    importlib.invalidate_caches()
    yield target
    plugins.REGISTRY.withdraw(plugins.Kind.ESTIMATORS, "tiny_bayes")
    sys.modules.pop("tinyplugin", None)


def test_pip_installed_entry_points_are_discovered(installed):
    found = plugins.REGISTRY.discover(plugins.Kind.ESTIMATORS)

    names = {adapter.name for adapter in found.items}
    assert {"tiny_bayes", "random_forest"} <= names
    assert (installed / "tinyplugin.py").exists()


def test_the_catalog_registers_the_plugin_and_refuses_a_shadowed_builtin(
    installed,
):
    catalog = registry.Catalog.stock()
    builtin = catalog.get("random_forest")

    failures = catalog.load_plugins()

    assert catalog.get("tiny_bayes").family == "custom"
    assert "tiny_bayes" in plugins.REGISTRY.entries(plugins.Kind.ESTIMATORS)
    assert catalog.get("random_forest") is builtin
    assert "random_forest" in failures
    assert "built-in" in catalog.plugin_failures["random_forest"]
    assert "random_forest" not in plugins.REGISTRY.entries(
        plugins.Kind.ESTIMATORS
    )
