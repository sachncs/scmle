"""Tests for user-facing recommendations."""

import numpy as np
import pandas as pd

from scml.contracts import data as data_contracts
from scml.contracts import tasks
from scml.data.profiling import profiler
from scml.data.validation import target as target_module
from scml.intelligence import recommendations


def profile_of(frame, **spec_fields):
    spec = tasks.ProjectSpec(
        data=tasks.DataSpec(uri="<in-memory>", target="y"), **spec_fields
    )
    task = target_module.TaskResolver(spec).resolve(frame["y"])
    manifest = data_contracts.DatasetManifest(
        "d", "s", "h", len(frame), frame.shape[1], "y"
    )
    return spec, profiler.DatasetProfiler(spec).profile(
        frame, manifest, task, 0.0
    )


def rare_frame(rows=800):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.normal(size=rows)})
    frame["y"] = (rng.random(rows) < 0.04).astype(int)
    return frame


def test_severe_imbalance_suggests_metric_and_threshold():
    spec, profile = profile_of(rare_frame())

    topics = {
        (r.topic, r.setting) for r in recommendations.recommend(profile, spec)
    }

    assert ("imbalance", "metric") in topics
    assert ("imbalance", "decision.threshold_objective") in topics


def test_a_chosen_threshold_objective_is_not_suggested_again():
    spec, profile = profile_of(
        rare_frame(),
        decision=tasks.DecisionSpec(threshold_objective="f1"),
    )

    settings = {r.setting for r in recommendations.recommend(profile, spec)}

    assert "decision.threshold_objective" not in settings


def test_balanced_clean_data_gets_no_imbalance_or_missing_advice():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"a": rng.normal(size=900)})
    frame["y"] = (frame.a + rng.normal(size=900) > 0).astype(int)
    spec, profile = profile_of(frame)

    topics = {r.topic for r in recommendations.recommend(profile, spec)}

    assert topics.isdisjoint({"imbalance", "missing values", "leakage"})


def test_small_data_and_duplicates_are_flagged():
    frame = rare_frame(120)
    frame = pd.concat([frame, frame.head(60)], ignore_index=True)
    spec, profile = profile_of(frame)

    topics = {r.topic for r in recommendations.recommend(profile, spec)}

    assert {"small data", "duplicates"} <= topics
