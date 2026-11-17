"""Preparation work fits a time allowance from its fitted cost models."""

import numpy
import pandas
import pytest

from scmle.assurance.admissibility import screen
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.profiling import label_noise
from scmle.data.profiling import profiler
from scmle.data.readers import tabular
from scmle.pipeline import fingerprint


def labelled(rows=2000, seed=0):
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame(
        {"a": rng.normal(size=rows), "b": rng.normal(size=rows)}
    )
    return frame, pandas.Series((frame.a + frame.b > 0).astype(int))


def test_label_noise_cost_follows_the_fitted_model():
    expected = label_noise.MARGIN * (
        label_noise.BASE_SECONDS
        + label_noise.CLASS_SECONDS
        + label_noise.CELL_SECONDS * 1000 * 10
    )
    assert label_noise.cost(1000, 10, 2) == pytest.approx(expected)
    assert label_noise.cost(1000, 10, 8) > label_noise.cost(1000, 10, 2)


def test_contradict_checks_every_row_it_is_given():
    # Fitting the time budget is inspect's job (it subsamples or skips on
    # record); contradict itself never refuses for time.
    features, labels = labelled()
    report = label_noise.contradict(features, labels, 0)
    assert report.rows == 2000
    assert all(0 <= index < 2000 for index in report.suspects)


def test_running_out_of_time_skips_the_check_instead_of_blocking():
    features, labels = labelled()
    observation = label_noise.inspect(features, labels, 0, allowance=0.0)
    assert observation.evidence["status"] == "skipped"


def test_inspect_records_a_skip_with_its_reason():
    features, labels = labelled()
    observation = label_noise.inspect(features, labels, 0, allowance=0.01)
    assert observation.evidence["status"] == "skipped"
    assert "allowance" in observation.evidence["metrics"]["reason"]
    assert observation.note is not None
    assert observation.note.startswith("label noise not checked")


def test_inspect_bounds_the_rows_by_the_allowance():
    features, labels = labelled()
    allowance = label_noise.cost(400, 2, 2)
    observation = label_noise.inspect(features, labels, 0, allowance=allowance)
    assert observation.evidence["sample_bounds"] == {
        "input_rows": 2000,
        "rows_checked": 400,
        "maximum_rows": 400,
        "sampled": True,
    }


def test_inspect_applies_the_thread_cap(monkeypatch):
    seen = []
    original = label_noise.threadpoolctl.threadpool_limits

    def spy(limits=None):
        seen.append(limits)
        return original(limits=limits)

    monkeypatch.setattr(label_noise.threadpoolctl, "threadpool_limits", spy)
    features, labels = labelled(300)
    label_noise.inspect(features, labels, 0, threads=1)
    assert seen == [1]


def portrait(frame, budget=1800, cpu=None):
    spec = tasks.Config(
        data=tasks.Dataset(uri="m", target="y"),
        resources=tasks.Resources(time_budget_seconds=budget, cpu_limit=cpu),
    )
    manifest = tabular.manifested(frame, "y")
    return spec, profiler.Profiler(spec).profile(
        frame, manifest, tasks.Task.BINARY, 0.25
    )


def xor(rows=3000):
    rng = numpy.random.default_rng(0)
    frame = pandas.DataFrame(
        {"a": rng.normal(size=rows), "b": rng.normal(size=rows)}
    )
    frame["y"] = (frame.a * frame.b > 0).astype(int)
    return frame


def test_ml_difficulty_cost_follows_the_fitted_model():
    expected = fingerprint.MARGIN * (
        fingerprint.BASE_SECONDS
        + fingerprint.CELL_SECONDS * 2000 * 10 * 8**fingerprint.CLASS_POWER
    )
    assert fingerprint.cost(5000, 10, 8) == pytest.approx(expected)
    assert fingerprint.cost(2000, 10, 2) < fingerprint.cost(2000, 10, 3)


def test_ml_difficulty_subsamples_to_fit_its_allowance():
    frame = xor()
    spec, profile = portrait(frame)
    plan = profile.feature_plan
    allowance = fingerprint.cost(500, len(plan.features), 2)
    result = fingerprint.ml_difficulty(
        frame, spec, tasks.Task.BINARY, plan, allowance
    )
    assert result["available"] and result["rows"] == 500
    full = fingerprint.ml_difficulty(frame, spec, tasks.Task.BINARY, plan)
    assert full["rows"] == fingerprint.DIFFICULTY_ROWS


def test_ml_difficulty_skips_with_a_reason_when_nothing_fits():
    frame = xor()
    spec, profile = portrait(frame)
    result = fingerprint.ml_difficulty(
        frame, spec, tasks.Task.BINARY, profile.feature_plan, 0.01
    )
    assert result["available"] is False and result["skipped"] is True
    assert "allowance" in result["reason"]


def test_the_profiler_gives_ml_difficulty_its_share_of_the_budget():
    frame = xor()
    small = portrait(frame, budget=2)[1]
    stats = small.dataset_statistics["ml_difficulty"]
    assert stats["available"] is False and stats["skipped"] is True
    assert any(
        note.startswith("ml_difficulty skipped")
        for note in small.quality_findings
    )
    large = portrait(frame)[1]
    assert large.dataset_statistics["ml_difficulty"]["available"]


def test_the_profiler_applies_the_thread_cap(monkeypatch):
    seen = []
    original = profiler.threadpoolctl.threadpool_limits

    def spy(limits=None):
        seen.append(limits)
        return original(limits=limits)

    monkeypatch.setattr(profiler.threadpoolctl, "threadpool_limits", spy)
    portrait(xor(300), cpu=1)
    assert seen == [1]


def test_estimate_bounds_every_part_of_preparation_by_its_share():
    budget = 60
    rows, columns, classes = 1600, 30, 2
    total = profiler.estimate(rows, columns, classes, budget)
    shares = (
        screen.SCREEN_SHARE + fingerprint.SHARE + label_noise.SHARE
    ) * budget
    assert 0 < total <= profiler.rest(rows) + shares
    assert profiler.estimate(rows, columns, 0, budget) < total
    tiny = profiler.estimate(100, 2, 2, budget)
    assert tiny < total
    assert profiler.estimate(rows, columns, classes, 1800) >= total


def test_estimate_rejects_an_impossible_shape():
    with pytest.raises(errors.InputError):
        profiler.estimate(-1, 3, 2, 60)
