"""Tests that tell informative gaps from random ones."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from scmle.contracts import data as data_contracts
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.curation import preprocessing
from scmle.data.profiling import missingness
from scmle.models import registry as registry_module


def sample(rows=1500, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "informative": rng.normal(size=rows),
            "random": rng.normal(size=rows),
            "sparse": rng.normal(size=rows),
            "complete": rng.normal(size=rows),
        }
    )
    target = pd.Series(rng.integers(0, 2, rows))
    frame.loc[(target == 1) & (rng.random(rows) < 0.5), "informative"] = np.nan
    frame.loc[rng.random(rows) < 0.2, "random"] = np.nan
    frame.loc[rng.choice(rows, 5, replace=False), "sparse"] = np.nan
    return frame, target


def test_gaps_that_follow_the_class_are_flagged_and_random_gaps_are_not():
    frame, target = sample()

    findings = missingness.assess(frame, list(frame), target, True)

    by_name = {f.column: f for f in findings}
    assert by_name["informative"].informative
    assert not by_name["random"].informative
    assert "sparse" not in by_name and "complete" not in by_name
    assert findings[0].column == "informative"


def test_numeric_targets_use_a_rank_test():
    frame = sample()[0]
    value = frame["informative"].isna() * 3.0 + np.random.default_rng(1).normal(
        size=len(frame)
    )

    findings = missingness.assess(
        frame, ["informative", "random"], value, False
    )

    by_name = {f.column: f for f in findings}
    assert by_name["informative"].test == "mann-whitney"
    assert by_name["informative"].informative
    assert not by_name["random"].informative


def test_many_columns_raise_the_bar_for_significance():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            f"c{i}": np.where(rng.random(800) < 0.3, np.nan, 1.0)
            for i in range(40)
        }
    )
    target = pd.Series(rng.integers(0, 2, 800))

    findings = missingness.assess(frame, list(frame), target, True)

    assert len(findings) == 40
    assert not any(f.informative for f in findings)


def test_indicators_are_added_only_for_flagged_columns():
    frame, target = sample()
    plan = data_contracts.FeaturePlan(
        ("informative", "random", "complete"),
        (),
        informative_missing=("informative",),
    )
    adapter = registry_module.ModelRegistry.with_builtins().get(
        "logistic_regression"
    )
    options = data_contracts.PreprocessingOptions(
        scale=True, impute=True, missing="indicator"
    )

    fitted = preprocessing.build_pipeline(
        plan, adapter, tasks.TaskType.BINARY, 0, options=options
    ).fit(frame, target)

    names = list(fitted.steps[0][1].get_feature_names_out())
    flags = [name for name in names if "missingindicator" in name]
    assert len(flags) == 1 and "informative" in flags[0]


def test_monotonic_signs_that_missing_indicators_would_misalign_are_refused():
    plan = data_contracts.FeaturePlan(
        ("informative", "random", "complete"),
        (),
        informative_missing=("informative",),
        monotonic={"random": 1},
    )
    adapter = registry_module.ModelRegistry.with_builtins().get(
        "hist_gradient_boosting"
    )
    options = data_contracts.PreprocessingOptions(
        scale=False, impute=True, missing="indicator"
    )

    with pytest.raises(errors.ConfigError, match="monotonic"):
        preprocessing.build_pipeline(
            plan, adapter, tasks.TaskType.BINARY, 0, options=options
        )
    with pytest.raises(errors.ConfigError, match="monotonic"):
        preprocessing.build_pipeline(
            plan,
            adapter,
            tasks.TaskType.BINARY,
            0,
            options=dataclasses.replace(
                options, missing="median", select=True
            ),
        )
