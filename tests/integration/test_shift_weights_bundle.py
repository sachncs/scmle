"""Shift weights attached to a bundle: weighted intervals, refusal, tamper."""

import numpy
import pandas
import pytest
import support

import scmle
from scmle.artifacts import bundle
from scmle.contracts import errors
from scmle.evaluation import shift_weights


def world(rows, seed, centre=0.0):
    rng = numpy.random.default_rng(seed)
    x = rng.normal(centre, 1.0, rows)
    return pandas.DataFrame({"x": x, "y": 2 * x + rng.normal(0, 0.5, rows)})


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return support.make_project(
        world(500, 0),
        "y",
        time_budget=300,
        output_dir=tmp_path_factory.mktemp("shift"),
    ).run()


def calibration_rows(result):
    count = len(result.bundle.conformal.scores)
    return world(count, 5)


def test_rows_must_match_the_stored_scores(result):
    with pytest.raises(errors.ArtifactError, match="conformal scores"):
        result.bundle.attach_shift_weights(world(10, 1), world(100, 2))


def test_weighted_intervals_need_attached_weights(result):
    plain = scmle.ModelBundle(
        result.bundle.pipeline, result.bundle.schema, result.bundle.info
    )
    with pytest.raises(errors.ArtifactError, match="conformal"):
        plain.predict_interval(world(5, 3)[["x"]], shift=True)
    with pytest.raises(errors.ArtifactError, match="no shift weights"):
        result.bundle.shift_weights = None
        result.bundle.predict_interval(world(5, 3)[["x"]], shift=True)


def test_a_mild_shift_gives_weighted_intervals_that_round_trip(
    result, tmp_path
):
    rows = calibration_rows(result)
    stored = result.bundle.attach_shift_weights(
        rows[["x"]], world(400, 6, centre=0.3)[["x"]]
    )
    assert stored.columns == ("x",)
    fresh = world(1500, 7, centre=0.3)
    band = result.bundle.predict_interval(
        fresh[["x"]], coverage=0.9, shift=True
    )
    covered = ((fresh.y >= band.lower) & (fresh.y <= band.upper)).mean()
    assert covered >= 0.85
    result.bundle.save(tmp_path / "copy")
    loaded = scmle.ModelBundle.load(tmp_path / "copy", trusted=True)
    pandas.testing.assert_frame_equal(
        loaded.predict_interval(fresh[["x"]], coverage=0.9, shift=True), band
    )


def test_a_changed_weights_file_is_refused(result, tmp_path):
    rows = calibration_rows(result)
    result.bundle.attach_shift_weights(
        rows[["x"]], world(400, 6, centre=0.3)[["x"]]
    )
    result.bundle.save(tmp_path / "copy")
    path = tmp_path / "copy" / bundle.SHIFT_WEIGHTS_FILE
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(errors.ArtifactError, match="shift weights"):
        scmle.ModelBundle.load(tmp_path / "copy", trusted=True)


def test_an_extreme_shift_is_refused_and_nothing_is_stored(result):
    result.bundle.shift_weights = None
    rows = calibration_rows(result)
    with pytest.raises(shift_weights.WeightRefusedError):
        result.bundle.attach_shift_weights(
            rows[["x"]], world(400, 8, centre=6.0)[["x"]]
        )
    assert result.bundle.shift_weights is None


def test_residual_is_the_only_method_with_weights(result):
    with pytest.raises(errors.ArtifactError, match="residual"):
        result.bundle.predict_interval(
            world(5, 3)[["x"]], method="quantile", shift=True
        )
