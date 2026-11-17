"""Tests for slice evaluation."""

import dataclasses

import numpy
import pandas

from scmle.contracts import tasks
import scmle.contracts.data
from scmle.data.curation import preprocessing
from scmle.evaluation import diagnostics
from scmle.evaluation import metrics
from scmle.evaluation import slices
from scmle.models import registry

REGISTRY = registry.Catalog.stock()
PLAN = scmle.contracts.data.Blueprint(("signal",), ("region",))


def make_inputs(rows=1200):
    rng = numpy.random.default_rng(0)
    frame = pandas.DataFrame(
        {
            "signal": rng.normal(size=rows),
            "region": rng.choice(["north", "south", "east"], rows),
        }
    )
    # In the south the label ignores the signal, so a model fitted on the
    # whole data must do worse there.
    clean = (frame.signal + rng.normal(0, 0.3, rows) > 0).astype(int)
    noisy = rng.integers(0, 2, rows)
    target = pandas.Series(numpy.where(frame.region == "south", noisy, clean))
    half = rows // 2
    model = preprocessing.instantiate(
        PLAN, REGISTRY.get("logistic_regression"), tasks.Task.BINARY, 0
    ).fit(frame.iloc[:half], target.iloc[:half])
    metric = metrics.CATALOG.resolve(None, tasks.Task.BINARY)
    return diagnostics.DiagnosticInput(
        model,
        frame.iloc[half:].reset_index(drop=True),
        target.iloc[half:].reset_index(drop=True),
        tasks.Task.BINARY,
        metric,
        PLAN,
        0,
    )


def test_the_worst_slice_is_the_region_the_model_cannot_learn():
    report = slices.Slices().compute(make_inputs())

    worst = report["worst"][0]
    assert (worst["feature"], worst["slice"]) == ("region", "south")
    assert worst["gap"] > 0.15


def test_numeric_bands_are_used_for_metrics_that_are_not_rank_based():
    data = make_inputs()
    accuracy = dataclasses.replace(
        data, metric=metrics.CATALOG.resolve("accuracy", tasks.Task.BINARY)
    )
    report = slices.Slices().compute(accuracy)

    features = {item["feature"] for item in report["slices"]}
    assert features == {"region", "signal"}
    assert sum(1 for s in report["slices"] if s["feature"] == "signal") == 4


def test_rank_metrics_slice_by_category_only():
    report = slices.Slices().compute(make_inputs())

    assert {item["feature"] for item in report["slices"]} == {"region"}


def test_small_slices_are_skipped_not_scored():
    report = slices.Slices().compute(make_inputs(rows=120))

    assert report["skipped_for_support"] > 0
    assert all(item["rows"] >= slices.MIN_SUPPORT for item in report["slices"])


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_slices_that_cannot_fit_are_counted_not_scored(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(diagnostics, "clock", clock)
    data = make_inputs()
    score = data.metric.score

    class Ticking:
        name = data.metric.name
        greater_is_better = True

        def score(self, *args):
            clock.now += 10.0
            return score(*args)

    report = slices.Slices().compute(
        dataclasses.replace(data, metric=Ticking(), deadline=clock.now + 25)
    )

    assert len(report["slices"]) == 1
    assert report["skipped_for_time"] == 2
    assert report["truncated"] and report["reason"]


def test_slices_are_skipped_once_the_deadline_has_passed():
    data = dataclasses.replace(
        make_inputs(), deadline=diagnostics.clock() - 1.0
    )

    report = slices.Slices().compute(data)

    assert report["skipped"] and report["available"] is False
