"""Tests for slice evaluation."""

import dataclasses

import numpy as np
import pandas as pd

from scml.contracts import data as data_contracts
from scml.contracts import tasks
from scml.data.curation import preprocessing
from scml.evaluation import diagnostics
from scml.evaluation import metrics
from scml.evaluation import slices
from scml.models import registry as registry_module

REGISTRY = registry_module.ModelRegistry.with_builtins()
PLAN = data_contracts.FeaturePlan(("signal",), ("region",))


def make_inputs(rows=1200):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "signal": rng.normal(size=rows),
            "region": rng.choice(["north", "south", "east"], rows),
        }
    )
    # In the south the label ignores the signal, so a model fitted on the
    # whole data must do worse there.
    clean = (frame.signal + rng.normal(0, 0.3, rows) > 0).astype(int)
    noisy = rng.integers(0, 2, rows)
    target = pd.Series(np.where(frame.region == "south", noisy, clean))
    half = rows // 2
    model = preprocessing.build_pipeline(
        PLAN, REGISTRY.get("logistic_regression"), tasks.TaskType.BINARY, 0
    ).fit(frame.iloc[:half], target.iloc[:half])
    metric = metrics.CATALOG.resolve(None, tasks.TaskType.BINARY)
    return diagnostics.DiagnosticInput(
        model,
        frame.iloc[half:].reset_index(drop=True),
        target.iloc[half:].reset_index(drop=True),
        tasks.TaskType.BINARY,
        metric,
        PLAN,
        0,
    )


def test_the_worst_slice_is_the_region_the_model_cannot_learn():
    report = slices.SliceReport().compute(make_inputs())

    worst = report["worst"][0]
    assert (worst["feature"], worst["slice"]) == ("region", "south")
    assert worst["gap"] > 0.15


def test_numeric_bands_are_used_for_metrics_that_are_not_rank_based():
    data = make_inputs()
    accuracy = dataclasses.replace(
        data, metric=metrics.CATALOG.resolve("accuracy", tasks.TaskType.BINARY)
    )
    report = slices.SliceReport().compute(accuracy)

    features = {item["feature"] for item in report["slices"]}
    assert features == {"region", "signal"}
    assert sum(1 for s in report["slices"] if s["feature"] == "signal") == 4


def test_rank_metrics_slice_by_category_only():
    report = slices.SliceReport().compute(make_inputs())

    assert {item["feature"] for item in report["slices"]} == {"region"}


def test_small_slices_are_skipped_not_scored():
    report = slices.SliceReport().compute(make_inputs(rows=120))

    assert report["skipped_for_support"] > 0
    assert all(item["rows"] >= slices.MIN_SUPPORT for item in report["slices"])
