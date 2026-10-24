"""Inner cross-validation keeps entities together and time in order."""

import itertools

import numpy as np
import pandas as pd

from scmle.contracts import data as data_contracts
from scmle.contracts import tasks
from scmle.data.splitting import inner_cv
from scmle.models import registry as registry_module
from scmle.sampling import strategies

BINARY = tasks.TaskType.BINARY


def grouped_frame(groups=12, per_group=15, seed=0):
    rng = np.random.default_rng(seed)
    rows = groups * per_group
    frame = pd.DataFrame(
        {
            "unit": np.repeat([f"u{i}" for i in range(groups)], per_group),
            "day": np.tile(
                pd.date_range("2024-01-01", periods=per_group), groups
            ),
            "a": rng.normal(size=rows),
            "b": rng.normal(size=rows),
        }
    )
    target = pd.Series(
        (frame.a + rng.normal(scale=0.5, size=rows) > 0).astype(int)
    )
    return frame, target


def test_groups_never_straddle_inner_folds():
    frame, target = grouped_frame()
    splitter = inner_cv.StructuredKFold(3, group_column="unit")
    seen = set()
    for train, validation in splitter.split(frame, target):
        assert not set(frame.unit.iloc[train]) & set(
            frame.unit.iloc[validation]
        )
        seen |= set(validation)
    assert seen == set(range(len(frame)))  # a partition: every row once


def test_time_folds_are_contiguous_periods():
    frame, target = grouped_frame()
    shuffled = frame.sample(frac=1, random_state=1).reset_index(drop=True)
    splitter = inner_cv.StructuredKFold(3, time_column="day")
    spans = []
    for split in splitter.split(shuffled, target):
        days = shuffled.day.iloc[split[1]]
        spans.append((days.min(), days.max()))
    spans.sort()
    for earlier, later in itertools.pairwise(spans):
        assert earlier[1] <= later[0]  # periods do not interleave


def test_unstructured_data_falls_back_to_stratified_folds():
    frame, target = grouped_frame()
    splitter = inner_cv.StructuredKFold(3, classification=True, seed=1)
    for split in splitter.split(frame[["a", "b"]], target):
        assert 0.2 < target.iloc[split[1]].mean() < 0.8


def plan(**kwargs):
    return data_contracts.FeaturePlan(
        numeric=("a", "b"), categorical=(), **kwargs
    )


def test_calibration_strategy_uses_the_structured_splitter():
    registry = registry_module.ModelRegistry.with_builtins()
    strategy = strategies.get_strategy("calibrated")
    candidate = strategy.build(
        plan(group_column="unit"), registry.get("random_forest"), BINARY, 0
    )
    assert isinstance(candidate.cv, inner_cv.StructuredKFold)
    assert candidate.cv.group_column == "unit"
    frame, target = grouped_frame()
    candidate.fit(frame, target)
    assert candidate.predict_proba(frame).shape == (len(frame), 2)


def test_stack_members_are_full_pipelines_cross_fitted_by_group():
    registry = registry_module.ModelRegistry.with_builtins()
    params = {"members": [{"model": "decision_tree"}, {"model": "extra_trees"}]}
    from scmle.data.curation import preprocessing

    stack = preprocessing.build_pipeline(
        plan(group_column="unit"), registry.get("stacking"), BINARY, 0, params
    )
    assert isinstance(stack.cv, inner_cv.StructuredKFold)
    frame, target = grouped_frame()
    stack.fit(frame, target)
    assert stack.predict_proba(frame).shape == (len(frame), 2)
    # Raw frames (with the group column) go in; members preprocess themselves.
    assert all(hasattr(member, "named_steps") for member in stack.estimators_)
