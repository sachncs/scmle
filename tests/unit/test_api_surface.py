"""The supported top-level API does not disappear accidentally."""

from importlib import metadata
import types

import pandas

import scmle
import scmle.api.project
from scmle.governance import vault


def test_the_package_root_re_exports_nothing():
    """Every public name lives in the module that defines it, and only there."""
    leaked = [
        name
        for name, value in vars(scmle).items()
        if isinstance(value, type | types.FunctionType)
    ]

    assert leaked == []
    assert not hasattr(scmle, "__all__")


def test_version_is_a_nonempty_release_string():
    assert scmle.__version__
    assert " " not in scmle.__version__


def test_package_documentation_points_to_standalone_site():
    """Package metadata must lead developers to the standalone docs site."""
    project_urls = metadata.metadata("scmle").get_all("Project-URL")

    assert "Documentation, https://sachncs.github.io/scmle/" in (
        project_urls or []
    )


def test_public_project_exposes_an_explicit_dataset_modality():
    """The user-facing API carries modality declarations to the contract."""
    project = scmle.api.project.Project(
        data={"feature": [1, 2], "y": [0, 1]},
        target="y",
        modality="image",
    )

    assert project.spec.data.modality == "image"


def test_plan_closes_the_preview_test_set(monkeypatch):
    """A side-effect-free preview must not leave its vault subprocess alive."""
    closed = []
    close = vault.SealedTestSet.close

    def record_close(instance):
        closed.append(instance.process.pid)
        close(instance)

    monkeypatch.setattr(vault.SealedTestSet, "close", record_close)
    project = scmle.api.project.Project(
        data=pandas.DataFrame({"feature": list(range(80)), "y": [0, 1] * 40}),
        target="y",
        time_budget=30,
    )

    project.plan()

    assert closed
