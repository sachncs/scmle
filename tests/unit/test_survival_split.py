"""Survival splits keep the event rate steady and every fold rankable."""

import numpy
import pandas
import pytest

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.splitting import splitters


def spec():
    return tasks.ProjectSpec(
        data=tasks.DataSpec(uri="x", target="time", event_column="event"),
        task="survival",
    )


def cohort(rows=1000, event_rate=0.12, seed=0):
    rng = numpy.random.default_rng(seed)
    return pandas.DataFrame(
        {
            "x": rng.normal(size=rows),
            "time": rng.exponential(1.0, rows) + 0.01,
            "event": (rng.random(rows) < event_rate).astype(int),
        }
    )


def test_auto_picks_event_stratification_for_survival():
    assert (
        splitters.resolve_strategy_name(spec(), tasks.TaskType.SURVIVAL)
        == "stratified_events"
    )


def test_the_event_rate_is_steady_across_test_and_folds():
    frame = cohort()

    split = splitters.SplitManager(spec(), tasks.TaskType.SURVIVAL).split(frame)

    overall = frame.event.mean()
    test = frame.event.iloc[list(split.manifest.test_ids)].mean()
    assert test == pytest.approx(overall, abs=0.02)
    for fold in split.manifest.folds:
        rate = frame.event.iloc[list(fold[1])].mean()
        assert 0 < rate < 1
        assert rate == pytest.approx(overall, abs=0.06)


def test_folds_shrink_when_events_are_scarce():
    frame = cohort(rows=400, event_rate=0.0)
    frame.loc[:4, "event"] = 1  # five events in 400 rows

    split = splitters.SplitManager(spec(), tasks.TaskType.SURVIVAL).split(frame)

    assert len(split.manifest.folds) < 5
    assert any("reduced" in note for note in split.manifest.notes)
    for fold in split.manifest.folds:
        assert frame.event.iloc[list(fold[1])].sum() >= 1


def test_a_single_event_cannot_be_split():
    frame = cohort(rows=300, event_rate=0.0)
    frame.loc[0, "event"] = 1

    with pytest.raises(errors.BlockedError, match="fewer than 2"):
        splitters.SplitManager(spec(), tasks.TaskType.SURVIVAL).split(frame)
