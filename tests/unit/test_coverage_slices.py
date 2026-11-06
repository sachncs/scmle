"""Coverage and set size per declared slice, descriptive only."""

import numpy as np
import pandas as pd

from scmle.evaluation import coverage_slices


def test_categorical_levels_are_the_labels_and_a_new_level_is_kept():
    reference = pd.Series(["a", "b", "a", "b"])
    test = pd.Series(["a", "c", "b"])
    labels = coverage_slices.labels_for(test, reference)
    assert list(labels) == ["a", "c", "b"]


def test_numeric_bands_come_from_the_reference_rows_only():
    reference = pd.Series(np.arange(100.0))
    test = pd.Series([0.0, 30.0, 60.0, 99.0, 500.0, -5.0])
    labels = coverage_slices.labels_for(test, reference)
    assert list(labels) == [
        "band 1",
        "band 2",
        "band 3",
        "band 4",
        "band 4",
        "band 1",
    ]


def test_too_many_levels_is_not_assessable():
    reference = pd.Series([f"u{i}" for i in range(50)])
    assert coverage_slices.labels_for(reference, reference) is None


def test_the_summary_gives_coverage_interval_and_size_and_no_flags():
    rng = np.random.default_rng(0)
    labels = np.array(["x"] * 200 + ["y"] * 200 + ["z"] * 5)
    inside = rng.random(405) < 0.9
    sizes = np.where(labels == "x", 1.0, 2.0)

    found = coverage_slices.summary(inside, sizes, labels, 0.9)

    assert found["descriptive"] is True
    assert found["skipped_for_support"] == 1
    by_name = {s["slice"]: s for s in found["slices"]}
    assert set(by_name) == {"x", "y"}
    assert by_name["x"]["mean_size"] == 1.0 and by_name["y"]["mean_size"] == 2.0
    assert (
        by_name["x"]["lower"]
        <= by_name["x"]["coverage"]
        <= by_name["x"]["upper"]
    )
    assert all(
        "flagged" not in s and "p_value" not in s for s in found["slices"]
    )
    assert "not a test" in found["note"]


def test_infinite_sizes_give_no_mean():
    inside = np.ones(40, dtype=bool)
    sizes = np.full(40, np.inf)
    found = coverage_slices.summary(inside, sizes, np.array(["a"] * 40), 0.9)
    assert found["slices"][0]["mean_size"] is None
