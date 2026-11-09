"""Tests for reference distributions and PSI."""

import numpy
import pandas
import pytest

from scmle.contracts import data
from scmle.monitoring import reference


def test_psi_is_zero_for_identical_and_grows_with_shift():
    even = numpy.full(10, 0.1)
    assert reference.population_stability_index(even, even) == pytest.approx(
        0.0
    )
    shifted = numpy.array(
        [0.3, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02]
    )
    assert reference.population_stability_index(even, shifted) > 0.25


def test_numeric_reference_detects_a_mean_shift_but_not_resampling():
    rng = numpy.random.default_rng(0)
    base = pandas.Series(rng.normal(size=5000))
    ref = reference.NumericReference.fit(base)
    assert ref.psi(pandas.Series(rng.normal(size=2000))) < 0.05
    assert ref.psi(pandas.Series(rng.normal(loc=1.5, size=2000))) > 0.25
    assert len(ref.edges) == 9 and sum(ref.fractions) == pytest.approx(1.0)


def test_numeric_reference_tracks_missingness_and_coerces_text():
    values = pandas.Series(
        [1.0, 2.0, numpy.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    )
    ref = reference.NumericReference.fit(values)
    assert ref.missing_rate == pytest.approx(0.1)
    assert ref.proportions(pandas.Series(["x", "y"])).sum() == 0.0


def test_categorical_reference_and_unseen_categories():
    rng = numpy.random.default_rng(1)
    base = pandas.Series(rng.choice(["a", "b", "c"], 3000, p=[0.5, 0.3, 0.2]))
    ref = reference.CategoricalReference.fit(base)
    assert (
        ref.psi(
            pandas.Series(rng.choice(["a", "b", "c"], 1000, p=[0.5, 0.3, 0.2]))
        )
        < 0.05
    )
    drifted = pandas.Series(
        rng.choice(["a", "b", "c", "new"], 1000, p=[0.2, 0.2, 0.2, 0.4])
    )
    assert ref.psi(drifted) > 0.25
    assert ref.unseen_rate(drifted) == pytest.approx(0.4, abs=0.05)
    assert ref.unseen_rate(pandas.Series([], dtype=object)) == 0.0


def test_high_cardinality_collapses_into_other():
    values = pandas.Series([f"c{i % 80}" for i in range(2000)])
    ref = reference.CategoricalReference.fit(values)
    assert len(ref.frequencies) == reference.MAX_CATEGORIES + 1
    assert sum(ref.frequencies.values()) == pytest.approx(1.0)


def test_profile_roundtrips_through_json():
    rng = numpy.random.default_rng(2)
    frame = pandas.DataFrame(
        {"n": rng.normal(size=500), "c": rng.choice(["x", "y"], 500)}
    )
    plan = data.FeaturePlan(("n",), ("c",))
    scores = pandas.Series(rng.random(500))
    profile = reference.ReferenceProfile.build(
        frame, plan, scores, "p", {"roc_auc": 0.9}
    )
    clone = reference.ReferenceProfile.from_dict(profile.to_dict())
    assert clone == profile
    labels = reference.ReferenceProfile.build(
        frame, plan, pandas.Series(rng.choice(["u", "v"], 500)), "class", {}
    )
    assert isinstance(labels.output, reference.CategoricalReference)
    assert reference.ReferenceProfile.from_dict(labels.to_dict()) == labels


def test_noise_floor_shrinks_with_sample_size():
    rng = numpy.random.default_rng(5)
    small = reference.NumericReference.fit(pandas.Series(rng.normal(size=100)))
    large = reference.NumericReference.fit(
        pandas.Series(rng.normal(size=10000))
    )
    assert small.rows == 100 and large.rows == 10000
    assert small.noise_floor(500) > 5 * large.noise_floor(500)
    assert small.noise_floor(0) == 0.0


def fresh_reference(rng, rows=120):
    return reference.NumericReference.fit(pandas.Series(rng.normal(size=rows)))


def test_noise_floor_tracks_simulated_psi_without_drift():
    rng = numpy.random.default_rng(7)
    simulated = [
        fresh_reference(rng).psi(pandas.Series(rng.normal(size=600)))
        for trial in range(60)
    ]
    floor = fresh_reference(rng).noise_floor(600)
    assert numpy.mean(simulated) < floor < 3 * numpy.mean(simulated)
    assert numpy.mean([value > floor for value in simulated]) < 0.15


def test_adjusted_psi_cuts_false_alarms_but_keeps_real_drift():
    rng = numpy.random.default_rng(6)
    pairs = [
        (fresh_reference(rng), pandas.Series(rng.normal(size=600)))
        for trial in range(60)
    ]
    raw_alarms = sum(ref.psi(new) >= 0.1 for ref, new in pairs)
    adjusted_alarms = sum(ref.adjusted_psi(new) >= 0.1 for ref, new in pairs)
    assert raw_alarms > 15 and adjusted_alarms <= 2
    shifted = pandas.Series(rng.normal(loc=2.0, size=600))
    assert fresh_reference(rng).adjusted_psi(shifted) > 0.25
