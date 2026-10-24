"""The supported top-level API does not disappear accidentally."""

from importlib import metadata

import pandas as pd

import scmle
from scmle.governance import vault

PUBLIC_API = {
    "BayesianPrior",
    "DatasetCatalog",
    "EstimandSpec",
    "EstimationPolicy",
    "EstimationResult",
    "ExtensionCurationAdapter",
    "LifecycleCancellation",
    "LifecycleCancelledError",
    "LifecycleCheckpoint",
    "LongitudinalEstimand",
    "LongitudinalEstimationPolicy",
    "LongitudinalEstimationResult",
    "LongitudinalIPWEstimator",
    "MLProject",
    "ModalityCurationAdapter",
    "ModalityCurationProject",
    "ModalityCurationResult",
    "ModalityHttpServing",
    "ModalityProject",
    "ModalityPredictor",
    "ModalitySampleDecoder",
    "ModelBundle",
    "NonTabularModelCapability",
    "NonTabularModelProject",
    "NonTabularModelResult",
    "ParameterEstimate",
    "PeriodATE",
    "RunResult",
    "SequentialEstimand",
    "SequentialEstimationPolicy",
    "SequentialEstimationProject",
    "SequentialEstimationResult",
    "SequentialIPWEstimator",
    "SequentialPeriodDiagnostics",
    "SequentialRegimeEstimate",
    "UnsupportedNonTabularModelError",
    "__version__",
    "estimate_parameters",
    "load_model",
    "predict",
    "register_estimator",
    "register_metric",
    "unregister_estimator",
}


def test_top_level_exports_are_the_documented_public_api():
    assert set(scmle.__all__) == PUBLIC_API
    assert all(hasattr(scmle, name) for name in PUBLIC_API)


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
    project = scmle.MLProject(
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
    project = scmle.MLProject(
        data=pd.DataFrame({"feature": list(range(80)), "y": [0, 1] * 40}),
        target="y",
        time_budget=30,
    )

    project.plan()

    assert closed
