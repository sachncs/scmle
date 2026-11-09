"""CV+ fold models are stored in the bundle, served, signed and checked."""

import numpy
import pandas
import pytest
import support

import scmle
from scmle.artifacts import bundle
from scmle.contracts import errors


def world(rows, seed):
    rng = numpy.random.default_rng(seed)
    x = rng.uniform(0, 10, rows)
    return pandas.DataFrame({"x": x, "y": 2 * x + rng.normal(0, 1, rows)})


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return support.make_project(
        world(400, 0),
        "y",
        time_budget=300,
        output_dir=tmp_path_factory.mktemp("cvplus"),
    ).run()


def test_a_small_regression_run_stores_the_fold_models(result):
    ensemble = result.bundle.cv_plus
    assert ensemble is not None
    assert len(ensemble.models) >= 2
    assert len(ensemble.residuals) == len(ensemble.fold_of_row)


def test_cv_plus_intervals_cover_new_rows(result):
    fresh = world(2000, 1)
    band = result.bundle.predict_interval(
        fresh[["x"]], coverage=0.9, method="cv_plus"
    )
    covered = ((fresh.y >= band.lower) & (fresh.y <= band.upper)).mean()
    assert covered >= 0.86


def test_cv_plus_survives_saving_and_loading(result, tmp_path):
    grid = pandas.DataFrame({"x": [1.0, 5.0, 9.0]})
    expected = result.bundle.predict_interval(grid, method="cv_plus")
    result.bundle.save(tmp_path / "copy")
    loaded = scmle.ModelBundle.load(tmp_path / "copy", trusted=True)
    pandas.testing.assert_frame_equal(
        loaded.predict_interval(grid, method="cv_plus"), expected
    )


def test_a_changed_fold_model_file_is_refused(result, tmp_path):
    result.bundle.save(tmp_path / "copy")
    path = tmp_path / "copy" / bundle.CV_PLUS_FILE
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(errors.ArtifactError, match="CV\\+ fold models"):
        scmle.ModelBundle.load(tmp_path / "copy", trusted=True)


def test_a_bundle_without_the_models_says_so(result):
    plain = scmle.ModelBundle(
        result.bundle.pipeline, result.bundle.schema, result.bundle.info
    )
    with pytest.raises(errors.ArtifactError, match="no CV\\+ fold models"):
        plain.predict_interval(pandas.DataFrame({"x": [1.0]}), method="cv_plus")
