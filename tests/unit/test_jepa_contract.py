"""JEPA contract tests that do not require the optional torch extra."""

import pytest

from scml.models.deeplearning.jepa import JepaEstimator


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
