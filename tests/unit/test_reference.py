"""Tests for reference distributions and PSI."""

import numpy as np
import pandas as pd
import pytest

from mml.contracts import data as data_contracts
from mml.monitoring import reference


def test_psi_is_zero_for_identical_and_grows_with_shift():
    even = np.full(10, 0.1)
    assert reference.population_stability_index(even, even) == pytest.approx(
        0.0
    )
    shifted = np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02])
    assert reference.population_stability_index(even, shifted) > 0.25


def test_numeric_reference_detects_a_mean_shift_but_not_resampling():
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(size=5000))
    ref = reference.NumericReference.fit(base)
    assert ref.psi(pd.Series(rng.normal(size=2000))) < 0.05
    assert ref.psi(pd.Series(rng.normal(loc=1.5, size=2000))) > 0.25
    assert len(ref.edges) == 9 and sum(ref.fractions) == pytest.approx(1.0)


def test_numeric_reference_tracks_missingness_and_coerces_text():
    values = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    ref = reference.NumericReference.fit(values)
    assert ref.missing_rate == pytest.approx(0.1)
    assert ref.proportions(pd.Series(["x", "y"])).sum() == 0.0


def test_categorical_reference_and_unseen_categories():
    rng = np.random.default_rng(1)
    base = pd.Series(rng.choice(["a", "b", "c"], 3000, p=[0.5, 0.3, 0.2]))
    ref = reference.CategoricalReference.fit(base)
    assert (
        ref.psi(pd.Series(rng.choice(["a", "b", "c"], 1000, p=[0.5, 0.3, 0.2])))
        < 0.05
    )
    drifted = pd.Series(
        rng.choice(["a", "b", "c", "new"], 1000, p=[0.2, 0.2, 0.2, 0.4])
    )
    assert ref.psi(drifted) > 0.25
    assert ref.unseen_rate(drifted) == pytest.approx(0.4, abs=0.05)
    assert ref.unseen_rate(pd.Series([], dtype=object)) == 0.0


def test_high_cardinality_collapses_into_other():
    values = pd.Series([f"c{i % 80}" for i in range(2000)])
    ref = reference.CategoricalReference.fit(values)
    assert len(ref.frequencies) == reference.MAX_CATEGORIES + 1
    assert sum(ref.frequencies.values()) == pytest.approx(1.0)


def test_profile_roundtrips_through_json():
    rng = np.random.default_rng(2)
    frame = pd.DataFrame(
        {"n": rng.normal(size=500), "c": rng.choice(["x", "y"], 500)}
    )
    plan = data_contracts.FeaturePlan(("n",), ("c",))
    scores = pd.Series(rng.random(500))
    profile = reference.ReferenceProfile.build(
        frame, plan, scores, "p", {"roc_auc": 0.9}
    )
    clone = reference.ReferenceProfile.from_dict(profile.to_dict())
    assert clone == profile
    labels = reference.ReferenceProfile.build(
        frame, plan, pd.Series(rng.choice(["u", "v"], 500)), "class", {}
    )
    assert isinstance(labels.output, reference.CategoricalReference)
    assert reference.ReferenceProfile.from_dict(labels.to_dict()) == labels
