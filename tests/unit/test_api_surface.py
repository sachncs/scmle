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
