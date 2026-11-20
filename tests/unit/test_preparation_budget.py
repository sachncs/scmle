"""Preparation work fits a time allowance from its fitted cost models."""

import numpy
import pandas
import pytest

from scmle.assurance.admissibility import screen
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.profiling import label_noise
from scmle.data.readers import tabular
from scmle.pipeline import fingerprint
from scmle.pipeline import profiler


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
    plan = profile.plan
    allowance = fingerprint.cost(500, len(plan.features), 2)
    result = fingerprint.gauge(frame, spec, tasks.Task.BINARY, plan, allowance)
    assert result["available"] and result["rows"] == 500
    full = fingerprint.gauge(frame, spec, tasks.Task.BINARY, plan)
    assert full["rows"] == fingerprint.DIFFICULTY_ROWS


def test_ml_difficulty_skips_with_a_reason_when_nothing_fits():
    frame = xor()
    spec, profile = portrait(frame)
    result = fingerprint.gauge(
        frame, spec, tasks.Task.BINARY, profile.plan, 0.01
    )
    assert result["available"] is False and result["skipped"] is True
    assert "allowance" in result["reason"]


def test_the_profiler_gives_ml_difficulty_its_share_of_the_budget():
    frame = xor()
    small = portrait(frame, budget=2)[1]
    stats = small.overview["ml_difficulty"]
    assert stats["available"] is False and stats["skipped"] is True
    assert any(
        note.startswith("ml_difficulty skipped")
        for note in small.quality_findings
    )
    large = portrait(frame)[1]
    assert large.overview["ml_difficulty"]["available"]


def test_the_profiler_applies_the_thread_cap(monkeypatch):
    seen = []
    original = profiler.threadpoolctl.threadpool_limits

    def spy(limits=None):
        seen.append(limits)
        return original(limits=limits)

    monkeypatch.setattr(profiler.threadpoolctl, "threadpool_limits", spy)
    portrait(xor(300), cpu=1)
    assert seen == [1]


def test_estimate_bounds_the_capped_parts_by_their_share():
    budget = 60
    rows, columns, classes = 1600, 30, 2
    total = profiler.estimate(rows, columns, classes, budget)
    shares = (screen.SCREEN_SHARE + fingerprint.SHARE) * budget
    checked = profiler.noise(rows, columns, classes, budget)
    assert 0 < total <= profiler.rest(rows, columns) + shares + checked
    assert profiler.estimate(rows, columns, 0, budget) < total
    tiny = profiler.estimate(100, 2, 2, budget)
    assert tiny < total
    assert profiler.estimate(rows, columns, classes, 1800) >= total


def test_estimate_rejects_an_impossible_shape():
    with pytest.raises(errors.InputError):
        profiler.estimate(-1, 3, 2, 60)


@pytest.mark.parametrize(
    ("rows", "columns", "classes", "budget"),
    [
        (2522, 970, 2, 30),
        (2522, 970, 2, 60),
        (3000, 1776, 2, 60),
        (1274, 256, 10, 60),
        (1600, 30, 2, 60),
        (100, 2, 2, 60),
    ],
)
def test_the_label_noise_check_is_priced_by_its_own_cost_model(
    rows, columns, classes, budget
):
    # One model decides both how many rows the check admits and what the
    # preparation estimate charges for it, so the two cannot disagree.
    allowance = budget * label_noise.SHARE
    admitted = label_noise.affordable(columns, classes, allowance)
    checked = min(rows, label_noise.MAX_ROWS, admitted)
    priced = profiler.noise(rows, columns, classes, budget)
    assert priced == pytest.approx(label_noise.cost(checked, columns, classes))
    assert priced <= allowance * (1 + 1e-9)


def test_the_label_noise_cost_bounds_the_measured_checks():
    # Measured 2026-11-19 (OMP_NUM_THREADS=1, 60 s budget): gina's check
    # took 2.91 s on 638 of 970-column rows, Bioresponse's 2.21 s on 348 of
    # 1,776; gina took 8.8 s on all 2,522 rows.
    assert label_noise.cost(638, 970, 2) >= 2.91
    assert label_noise.cost(348, 1776, 2) >= 2.21
    assert label_noise.cost(2522, 970, 2) >= 8.8
    assert profiler.noise(2522, 970, 2, 60) >= 2.91


def test_a_skipped_label_noise_check_is_free():
    assert profiler.noise(2522, 970, 10, 30) == 0.0  # no MIN_ROWS rows fit
    assert profiler.noise(1274, 256, 10, 60) > profiler.noise(1274, 256, 2, 60)


def test_the_rest_grows_with_the_columns():
    assert profiler.rest(2522, 970) > 2 * profiler.rest(2522, 0)


@pytest.mark.parametrize(
    ("rows", "columns", "measured"),
    [(2522, 970, 13.7), (3000, 1776, 7.4), (2623, 1558, 12.9)],
)
def test_wide_preparation_is_estimated_within_its_measured_seconds(
    rows, columns, measured
):
    # Seconds to the first trial at a 60 s budget, measured 2026-11-19
    # (OMP_NUM_THREADS=1, beside a mutation run) with the label-noise check
    # sampled to its allowance: gina, Bioresponse, Internet-Advertisements.
    assert profiler.estimate(rows, columns, 2, 60) >= measured
