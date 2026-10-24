"""JEPA contract tests that do not require the optional torch extra."""

import pytest

from scmle.models.deeplearning.jepa import JepaEstimator
from scmle.models.deeplearning.jepa_engine import JepaEngine


def test_jepa_is_a_self_supervised_representation_contract():
    """The estimator exposes only explicit tabular representation settings."""
    estimator = JepaEstimator(
        hidden_size=32,
        embedding_size=16,
        mask_probability=0.25,
        epochs=3,
        batch_size=32,
        random_state=7,
    )

    params = estimator.get_params()

    assert params["embedding_size"] == 16
    assert params["mask_probability"] == 0.25
    assert "architecture" not in params
    assert hasattr(estimator, "fit_transform")


@pytest.mark.skipif(
    pytest.importorskip("importlib.util").find_spec("torch") is None,
    reason="torch is not installed",
)
def test_tabular_jepa_trains_and_encodes():
    """The optional worker path produces finite, reproducible embeddings."""
    import numpy as np

    rng = np.random.default_rng(5)
    features = rng.normal(size=(48, 4)).astype(np.float32)
    estimator = JepaEstimator(
        hidden_size=16,
        embedding_size=8,
        epochs=2,
        batch_size=24,
        random_state=5,
    )

    estimator.fit(features)
    embeddings = estimator.transform(features[:5])

    assert embeddings.shape == (5, 8)
    assert np.isfinite(embeddings).all()


@pytest.mark.skipif(
    pytest.importorskip("importlib.util").find_spec("torch") is None,
    reason="torch is not installed",
)
@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("epochs", 0, "epochs must be a positive integer"),
        ("batch_size", 0, "batch_size must be auto or a positive integer"),
        ("learning_rate", 0.0, "learning_rate must be finite and positive"),
    ],
)
def test_invalid_training_configuration_has_actionable_diagnostic(
    parameter, value, message
):
    """Invalid training settings fail at the JEPA contract boundary."""
    import numpy as np

    rng = np.random.default_rng(11)
    features = rng.normal(size=(8, 3)).astype(np.float32)
    settings = {parameter: value}

    with pytest.raises(ValueError, match=message):
        JepaEngine(**settings).fit(features)


@pytest.mark.skipif(
    pytest.importorskip("importlib.util").find_spec("torch") is None,
    reason="torch is not installed",
)
def test_non_numeric_features_have_actionable_diagnostic():
    """String data is rejected as outside the numeric-tabular JEPA contract."""
    with pytest.raises(ValueError, match="numeric 2-D feature matrix"):
        JepaEngine(epochs=1).fit([["a"], ["b"], ["c"], ["d"]] * 2)
