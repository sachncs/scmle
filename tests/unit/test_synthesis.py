"""Tests for synthetic tabular generation and its evaluator."""

import numpy as np
import pandas as pd
import pytest

from mml.models import synthesis


def correlated_frame(rows=1500, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=rows)
    frame = pd.DataFrame(
        {
            "income": np.exp(base + rng.normal(scale=0.3, size=rows)),
            "age": 30 + 8 * base + rng.normal(size=rows),
            "noise": rng.uniform(size=rows),
            "tier": rng.choice(["a", "b", "c"], rows, p=[0.6, 0.3, 0.1]),
        }
    )
    frame.loc[::20, "age"] = np.nan
    return frame


def test_generator_preserves_marginals_dependence_and_categories():
    real = correlated_frame()
    generator = synthesis.GaussianCopulaGenerator().fit(real)
    fake = generator.sample(4000, np.random.default_rng(1))
    assert list(fake.columns) == list(real.columns) and len(fake) == 4000
    report = synthesis.FidelityEvaluator().evaluate(real, fake)
    assert report.max_marginal_ks < 0.06
    assert report.max_category_distance < 0.04
    assert report.correlation_error < 0.08
    assert report.duplicate_share == 0.0  # genuinely new rows, not copies


def test_a_naive_independent_shuffle_has_worse_dependence_fidelity():
    real = correlated_frame()
    rng = np.random.default_rng(2)
    shuffled = real.apply(
        lambda column: column.sample(
            frac=1, random_state=int(rng.integers(1e6))
        ).to_numpy()
    )
    copula = (
        synthesis.GaussianCopulaGenerator().fit(real).sample(len(real), rng)
    )
    evaluator = synthesis.FidelityEvaluator()
    assert (
        evaluator.evaluate(real, copula).correlation_error
        < 0.5 * evaluator.evaluate(real, shuffled).correlation_error
    )


def test_missingness_and_dtypes_are_reproduced():
    real = correlated_frame()
    fake = (
        synthesis.GaussianCopulaGenerator()
        .fit(real)
        .sample(3000, np.random.default_rng(3))
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
    first = generator.sample(50, np.random.default_rng(7))
    second = generator.sample(50, np.random.default_rng(7))
    pd.testing.assert_frame_equal(first, second)


def test_oversampler_balances_classes_with_new_minority_rows_only():
    real = correlated_frame()
    target = pd.Series(np.where(np.arange(len(real)) % 10 == 0, 1, 0))
    features, labels = synthesis.CopulaOverSampler().resample(real, target, 0)
    assert labels.value_counts().to_dict() == {0: 1350, 1: 1350}
    minority = features[labels == 1]
    assert len(minority) == 1350
    real_minority = real[target == 1]
    report = synthesis.FidelityEvaluator().evaluate(real_minority, minority)
    assert report.max_marginal_ks < 0.15


def test_tiny_classes_are_left_alone():
    real = correlated_frame(rows=60)
    target = pd.Series([1] * 3 + [0] * 57)
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
            return pd.concat([self.row] * rows, ignore_index=True)

    class ConstantOverSampler(synthesis.GeneratorOverSampler):
        generator_class = Constant

    real = correlated_frame(rows=200)
    target = pd.Series([1] * 20 + [0] * 180)
    _features, labels = ConstantOverSampler().resample(real, target, 0)
    assert labels.value_counts().to_dict() == {0: 180, 1: 180}
