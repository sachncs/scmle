"""Label-noise detection and representativeness against a reference."""

import numpy as np
import pandas as pd
import pytest

from scml.data.profiling import label_noise
from scml.data.profiling import representativeness


def population(rows=600, seed=0):
    return labelled(rows, seed)[0]


def labelled(rows=600, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {"a": rng.normal(size=rows), "b": rng.normal(size=rows)}
    )
    return frame, pd.Series((frame.a + frame.b > 0).astype(int))


def test_flipped_labels_are_found_first():
    features, labels = labelled()
    flipped = np.arange(0, 600, 20)
    noisy = labels.copy()
    noisy.iloc[flipped] = 1 - noisy.iloc[flipped]

    report = label_noise.find_label_noise(features, noisy)

    found = set(report.suspects[:30])
    assert len(found & set(flipped.tolist())) >= 20
    assert all(
        label == 1 - noisy.iloc[index]
        for index, label in zip(report.suspects, report.suggested, strict=True)
    )


def test_clean_labels_flag_few_rows():
    features, labels = labelled()
    assert label_noise.find_label_noise(features, labels).rate < 0.06


def test_label_noise_needs_data_and_two_classes():
    features, labels = labelled(30)
    with pytest.raises(ValueError, match="rows"):
        label_noise.find_label_noise(features, labels)
    features = population()
    with pytest.raises(ValueError, match="two classes"):
        label_noise.find_label_noise(features, pd.Series(np.zeros(600)))


def test_a_sample_from_the_same_population_is_representative():
    sample = population(500, seed=1)
    reference = population(500, seed=2)

    report = representativeness.compare_to_reference(sample, reference)

    assert report.representative and report.auc < 0.6


def test_a_skewed_sample_is_flagged_with_the_shifted_column():
    sample = population(500, seed=1)
    reference = population(500, seed=2)
    sample["a"] = sample["a"] + 1.5

    report = representativeness.compare_to_reference(sample, reference)

    assert not report.representative
    assert report.gaps[0].column == "a" and report.gaps[0].p_value < 0.001


def test_categorical_gaps_and_mismatched_columns():
    sample = pd.DataFrame({"c": ["x"] * 80 + ["y"] * 20})
    reference = pd.DataFrame({"c": ["x"] * 40 + ["y"] * 60})
    report = representativeness.compare_to_reference(sample, reference)
    assert report.gaps[0].statistic == pytest.approx(0.4)
    with pytest.raises(ValueError, match="same columns"):
        representativeness.compare_to_reference(
            sample, reference.rename(columns={"c": "d"})
        )
