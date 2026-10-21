"""The supported top-level API does not disappear accidentally."""

from importlib import metadata

import pandas as pd

import scml
from scml.governance import vault

PUBLIC_API = {
    "MLProject",
    "ModalityCurationProject",
    "ModalityHttpServing",
    "ModalityProject",
    "ModalityPredictor",
    "ModalitySampleDecoder",
    "ModelBundle",
    "RunResult",
    "__version__",
    "load_model",
    "predict",
    "register_estimator",
    "register_metric",
    "unregister_estimator",
}


def test_top_level_exports_are_the_documented_public_api():
    assert set(scml.__all__) == PUBLIC_API
    assert all(hasattr(scml, name) for name in PUBLIC_API)


def test_version_is_a_nonempty_release_string():
    assert scml.__version__
    assert " " not in scml.__version__


def test_package_documentation_points_to_standalone_site():
    """Package metadata must lead developers to the standalone docs site."""
    project_urls = metadata.metadata("super-charge-ml").get_all("Project-URL")

    assert "Documentation, https://sachncs.github.io/super-charge-ml/" in (
        project_urls or []
    )


def test_public_project_exposes_an_explicit_dataset_modality():
    """The user-facing API carries modality declarations to the contract."""
    project = scml.MLProject(
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
    project = scml.MLProject(
        data=pd.DataFrame({"feature": list(range(80)), "y": [0, 1] * 40}),
        target="y",
        time_budget=30,
    )

    project.plan()

    assert closed
