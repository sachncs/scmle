"""The supported top-level API does not disappear accidentally."""

import scml

PUBLIC_API = {
    "MLProject",
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


def test_public_project_exposes_an_explicit_dataset_modality():
    """The user-facing API carries modality declarations to the contract."""
    project = scml.MLProject(
        data={"feature": [1, 2], "y": [0, 1]},
        target="y",
        modality="image",
    )

    assert project.spec.data.modality == "image"
