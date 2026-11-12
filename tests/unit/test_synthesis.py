"""Tests for synthetic tabular generation."""

import numpy
import pandas
import pytest

from scmle.sampling import synthesis


def correlated_frame(rows=1500, seed=0):
    rng = numpy.random.default_rng(seed)
    base = rng.normal(size=rows)
    frame = pandas.DataFrame(
        {
            "income": numpy.exp(base + rng.normal(scale=0.3, size=rows)),
            "age": 30 + 8 * base + rng.normal(size=rows),
            "noise": rng.uniform(size=rows),
            "tier": rng.choice(["a", "b", "c"], rows, p=[0.6, 0.3, 0.1]),
        }
    )
    frame.loc[::20, "age"] = numpy.nan
    return frame


def test_generator_preserves_marginals_dependence_and_categories():
    real = correlated_frame()
    generator = synthesis.GaussianCopulaGenerator().fit(real)
    fake = generator.sample(4000, numpy.random.default_rng(1))
    assert list(fake.columns) == list(real.columns) and len(fake) == 4000
    numeric = ["income", "age", "noise"]
    real_corr = real[numeric].corr(method="spearman").to_numpy()
    fake_corr = fake[numeric].astype(float).corr(method="spearman").to_numpy()
    assert numpy.nanmax(numpy.abs(real_corr - fake_corr)) < 0.08


def test_missingness_and_dtypes_are_reproduced():
    real = correlated_frame()
    fake = (
        synthesis.GaussianCopulaGenerator()
        .fit(real)
        .sample(3000, numpy.random.default_rng(3))
    )
    assert fake["age"].isna().mean() == pytest.approx(
        real["age"].isna().mean(), abs=0.02
    )
    assert fake["income"].min() >= real["income"].min() - 1e-9
    assert fake["income"].max() <= real["income"].max() + 1e-9
    assert set(fake["tier"]) <= {"a", "b", "c"}


def test_sampling_is_seeded():
    real = correlated_frame()
    generator = synthesis.GaussianCopulaGenerator().fit(real)
    first = generator.sample(50, numpy.random.default_rng(7))
    second = generator.sample(50, numpy.random.default_rng(7))
    pandas.testing.assert_frame_equal(first, second)


def test_oversampler_balances_classes_with_new_minority_rows_only():
    real = correlated_frame()
    target = pandas.Series(numpy.where(numpy.arange(len(real)) % 10 == 0, 1, 0))
    features, labels = synthesis.CopulaOverSampler().resample(real, target, 0)
    assert labels.value_counts().to_dict() == {0: 1350, 1: 1350}
    minority = features[labels == 1]
    assert len(minority) == 1350


def test_tiny_classes_are_left_alone():
    real = correlated_frame(rows=60)
    target = pandas.Series([1] * 3 + [0] * 57)
    labels = synthesis.CopulaOverSampler().resample(real, target, 0)[1]
    assert labels.value_counts().to_dict() == {
        0: 57,
        1: 3,
    }  # too few rows to model


def test_generators_are_pluggable():
    class Constant(synthesis.SyntheticGenerator):
        name = "constant"

        def fit(self, features):
            self.row = features.iloc[[0]]
            return self

        def sample(self, rows, rng):
            return pandas.concat([self.row] * rows, ignore_index=True)

    class ConstantOverSampler(synthesis.GeneratorOverSampler):
        generator_class = Constant

    real = correlated_frame(rows=200)
    target = pandas.Series([1] * 20 + [0] * 180)
    labels = ConstantOverSampler().resample(real, target, 0)[1]
    assert labels.value_counts().to_dict() == {0: 180, 1: 180}


def test_generator_resampler_implements_the_resample_contract_directly():
    """Synthetic rows use the general contract, not source row indices."""
    assert issubclass(
        synthesis.GeneratorOverSampler, synthesis.resampling.Resampler
    )
    assert not hasattr(synthesis.GeneratorOverSampler, "indices")
