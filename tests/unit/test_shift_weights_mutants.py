"""Behaviours of the shift weights that the main suite left open."""

import inspect

import numpy
import pandas
import pytest
from sklearn import linear_model

from scmle.evaluation import shift_weights


class Discriminator:
    """Returns fixed probabilities and records what it was asked about."""

    def __init__(self, probability):
        self.probability = numpy.asarray(probability, dtype=float)
        self.seen = None

    def predict_proba(self, features):
        self.seen = features
        return numpy.column_stack([1.0 - self.probability, self.probability])


def sample(seed=0, rows=(300, 200), shift=0.5):
    rng = numpy.random.default_rng(seed)
    return (
        rng.standard_normal((rows[0], 2)),
        rng.standard_normal((rows[1], 2)) + shift,
    )


def test_the_defaults_are_the_documented_ones():
    parameters = inspect.signature(shift_weights.estimate_weights).parameters
    assert shift_weights.MIN_EFFECTIVE_SIZE == 50.0
    assert shift_weights.DEFAULT_CLIP == 20.0
    assert parameters["clip"].default == 20.0
    assert parameters["minimum"].default == 50.0
    bare = shift_weights.ShiftWeights(numpy.ones(1), numpy.ones(1), 1.0, 2.0)
    assert bare.discriminator is None
    assert bare.scale == 1.0


def test_the_weights_are_the_clipped_size_corrected_odds_of_the_fit():
    calibration, target = sample()
    result = shift_weights.estimate_weights(calibration, target, clip=3.0)
    matrix = numpy.vstack([calibration, target])
    probability = result.discriminator.predict_proba(matrix)[:, 1]
    expected = numpy.minimum(probability / (1.0 - probability) * 1.5, 3.0)
    assert len(result.calibration) == 300
    assert len(result.target) == 200
    numpy.testing.assert_allclose(result.calibration, expected[:300])
    numpy.testing.assert_allclose(result.target, expected[300:])
    assert result.calibration.max() <= 3.0
    assert result.target.max() == pytest.approx(3.0)
    assert result.clip == 3.0
    assert result.scale == 1.5


def test_the_discriminator_separates_calibration_from_target():
    calibration, target = sample(seed=1, shift=2.0)
    result = shift_weights.estimate_weights(
        calibration, target, clip=50.0, minimum=0.0
    )
    model = result.discriminator
    assert isinstance(model, linear_model.LogisticRegression)
    assert model.max_iter == 1000
    assert list(model.classes_) == [0.0, 1.0]
    assert result.target.mean() > result.calibration.mean()
    ahead = model.predict_proba(numpy.array([[4.0, 4.0]]))[0, 1]
    behind = model.predict_proba(numpy.array([[-4.0, -4.0]]))[0, 1]
    assert ahead > 0.9 > 0.1 > behind


def test_the_effective_size_is_that_of_the_calibration_weights():
    calibration, target = sample(seed=2)
    result = shift_weights.estimate_weights(calibration, target, minimum=1.0)
    assert result.effective_size == pytest.approx(
        shift_weights.effective_sample_size(result.calibration)
    )
    assert result.effective_size != pytest.approx(
        shift_weights.effective_sample_size(result.target)
    )
    assert isinstance(result.effective_size, float)


def test_a_perfectly_separated_target_is_pinned_at_the_clip():
    calibration = numpy.linspace(-3.0, -1.0, 200).reshape(-1, 1)
    target = numpy.linspace(1.0, 3.0, 200).reshape(-1, 1)
    result = shift_weights.estimate_weights(
        calibration, target, clip=7.0, minimum=0.0
    )
    assert result.target.max() == 7.0
    assert result.target[-1] == 7.0
    assert result.calibration.min() >= 0.0


def test_the_minimum_is_inclusive_and_the_refusal_says_both_sizes():
    calibration, target = sample(seed=3)
    size = shift_weights.estimate_weights(
        calibration, target, minimum=0.0
    ).effective_size
    kept = shift_weights.estimate_weights(calibration, target, minimum=size)
    assert kept.effective_size == size
    with pytest.raises(shift_weights.WeightRefusedError) as raised:
        shift_weights.estimate_weights(calibration, target, minimum=size + 1)
    assert str(raised.value) == (
        f"effective sample size {size:.1f} is below {size + 1:.0f}; "
        "weighted conformal is refused"
    )


def test_the_default_minimum_refuses_a_handful_of_effective_rows():
    calibration, target = sample(seed=4, rows=(30, 30), shift=0.0)
    with pytest.raises(shift_weights.WeightRefusedError, match="below 50;"):
        shift_weights.estimate_weights(calibration, target)


def test_the_effective_size_of_known_weights():
    assert shift_weights.effective_sample_size([1, 1, 1, 1]) == 4.0
    assert shift_weights.effective_sample_size([1.0, 2.0, 3.0]) == (36.0 / 14.0)
    assert shift_weights.effective_sample_size(numpy.array([5.0, 0.0])) == 1.0
    assert shift_weights.effective_sample_size(numpy.zeros(3)) == 0.0
    assert isinstance(shift_weights.effective_sample_size([1, 2]), float)


def test_ratio_is_the_clipped_scaled_odds_of_the_second_column():
    stub = Discriminator([0.5, 0.9, 1.0])
    weights = shift_weights.ShiftWeights(
        numpy.ones(2), numpy.ones(2), 2.0, 3.0, stub, 2.0
    )
    features = [[1, 2], [3, 4], [5, 6]]
    out = weights.ratio(features)
    numpy.testing.assert_allclose(out, [2.0, 3.0, 3.0])
    assert stub.seen.dtype == float
    numpy.testing.assert_array_equal(stub.seen, numpy.array(features))


def test_ratio_uses_the_small_floor_not_one_for_certain_rows():
    stub = Discriminator([1.0])
    weights = shift_weights.ShiftWeights(
        numpy.ones(1), numpy.ones(1), 1.0, 1e15, stub, 1.0
    )
    assert weights.ratio(numpy.zeros((1, 1)))[0] == pytest.approx(1e12)


def test_the_stored_weights_read_the_named_columns_in_order():
    stub = Discriminator([0.5, 0.5])
    weights = shift_weights.ShiftWeights(
        numpy.ones(1), numpy.ones(1), 1.0, 5.0, stub, 1.0
    )
    stored = shift_weights.StoredShiftWeights(weights, ("b", "a"))
    frame = pandas.DataFrame({"a": [1, 2], "b": [3, 4], "c": [9, 9]})
    out = stored.test_weights(frame)
    numpy.testing.assert_allclose(out, [1.0, 1.0])
    numpy.testing.assert_array_equal(
        stub.seen, numpy.array([[3.0, 1.0], [4.0, 2.0]])
    )
    assert stub.seen.dtype == float


def test_integer_weights_are_summed_as_floats_and_small_ones_count():
    huge = numpy.array([3_000_000_000, 3_000_000_000])
    assert shift_weights.effective_sample_size(huge) == 2.0
    assert shift_weights.effective_sample_size([0.5, 0.5]) == 2.0
