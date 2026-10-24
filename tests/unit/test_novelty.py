"""Tests for out-of-distribution detection."""

import numpy as np
import pandas as pd

from scmle.inference import novelty


def training_frame(rows=1500, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.normal(size=rows),
            "b": rng.normal(size=rows),
            "c": rng.choice(["x", "y"], rows),
        }
    )


def test_rows_far_from_the_training_data_are_flagged_and_typical_rows_are_not():
    train = training_frame()
    detector = novelty.NoveltyDetector.fit(train, ("a", "b"), ("c",))
    typical = training_frame(rows=500, seed=1)
    strange = pd.DataFrame(
        {
            "a": [12.0, -15.0, 20.0],
            "b": [-14.0, 18.0, 16.0],
            "c": ["x", "y", "x"],
        }
    )

    assert detector.score(strange)["out_of_distribution"].all()
    assert detector.score(typical)["out_of_distribution"].mean() < 0.06
    scores = detector.score(pd.concat([typical.head(3), strange]))
    assert (
        scores["novelty_score"].iloc[3:].min()
        > scores["novelty_score"].iloc[:3].max()
    )


def test_unseen_categories_do_not_break_scoring():
    detector = novelty.NoveltyDetector.fit(training_frame(), ("a", "b"), ("c",))
    odd = pd.DataFrame({"a": [0.0], "b": [0.0], "c": ["never seen"]})

    assert len(detector.score(odd)) == 1


def test_there_is_nothing_to_learn_without_numeric_or_categorical_columns():
    assert novelty.NoveltyDetector.fit(training_frame(), (), ()) is None


def test_a_single_wild_value_is_flagged_and_the_feature_is_named():
    detector = novelty.NoveltyDetector.fit(training_frame(), ("a", "b"), ("c",))
    row = pd.DataFrame({"a": [0.1], "b": [500.0], "c": ["x"]})

    scored = detector.score(row)

    assert bool(scored["out_of_distribution"].iloc[0]) is True
    assert scored["out_of_range_features"].iloc[0] == ["b"]
    inside = detector.score(pd.DataFrame({"a": [0.1], "b": [0.2], "c": ["x"]}))
    assert inside["out_of_range_features"].iloc[0] == []
