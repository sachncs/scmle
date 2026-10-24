"""Quantile intervals cover as promised and widen where the data is noisy."""

import numpy as np
import pandas as pd
import pytest
import support

import scmle
from scmle.contracts import errors


def heteroscedastic(rows, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, rows)
    noise = 0.2 + 0.6 * x
    return pd.DataFrame({"x": x, "y": 2 * x + rng.normal(0, 1, rows) * noise})


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return support.make_project(
        heteroscedastic(1500, 0),
        "y",
        time_budget=300,
        output_dir=tmp_path_factory.mktemp("quantile"),
    ).run()


def test_quantile_intervals_reach_the_requested_coverage(result):
    fresh = heteroscedastic(2000, 1)

    band = result.bundle.predict_interval(
        fresh[["x"]], coverage=0.9, method="quantile"
    )

    covered = ((fresh.y >= band.lower) & (fresh.y <= band.upper)).mean()
    assert covered >= 0.86


def test_quantile_intervals_are_wider_where_the_noise_is(result):
    grid = pd.DataFrame({"x": [0.5, 9.5]})

    band = result.bundle.predict_interval(grid, coverage=0.9, method="quantile")
    width = (band.upper - band.lower).to_numpy()

    assert width[1] > 2 * width[0]


def test_the_default_method_keeps_one_width_everywhere(result):
    grid = pd.DataFrame({"x": [0.5, 9.5]})

    band = result.bundle.predict_interval(grid, coverage=0.9)
    width = (band.upper - band.lower).to_numpy()

    assert width[0] == pytest.approx(width[1])


def test_quantile_intervals_survive_saving_and_loading(result, tmp_path):
    grid = pd.DataFrame({"x": [1.0, 5.0, 9.0]})
    expected = result.bundle.predict_interval(grid, method="quantile")

    result.bundle.save(tmp_path / "copy")
    loaded = scmle.ModelBundle.load(tmp_path / "copy", trusted=True)

    pd.testing.assert_frame_equal(
        loaded.predict_interval(grid, method="quantile"), expected
    )


def test_unknown_methods_are_refused(result):
    with pytest.raises(errors.ArtifactError, match="unknown interval method"):
        result.bundle.predict_interval(
            pd.DataFrame({"x": [1.0]}), method="bootstrap"
        )
