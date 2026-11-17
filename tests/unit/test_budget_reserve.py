"""Tests that post-training work fits the reserve or is skipped on record."""

import numpy
import pandas
import pytest

import scmle.api.project
from scmle.contracts import errors


def frame(rows=200, seed=0):
    rng = numpy.random.default_rng(seed)
    features = pandas.DataFrame(
        rng.normal(size=(rows, 4)), columns=["a", "b", "c", "d"]
    )
    return features.assign(y=(features["a"] > 0).astype(int))


def test_a_skip_is_recorded_in_the_events_and_the_report_list():
    context = scmle.api.project.Project(frame(), "y").orchestrator.new_context()
    context.forgo("threshold", "needs 9.0 s but 1.0 s remain")
    assert context.skipped == [
        {"work": "threshold", "reason": "needs 9.0 s but 1.0 s remain"}
    ]


def test_a_budget_too_small_for_the_minimum_run_is_refused(tmp_path):
    with pytest.raises(errors.BlockedError, match="smallest budget") as caught:
        scmle.api.project.Project(
            frame(), "y", time_budget=1, output_dir=tmp_path
        ).run()
    assert caught.value.details["smallest_budget_seconds"] > 1


def timed(budget, folds=5):
    from types import SimpleNamespace

    from scmle.contracts import tasks
    from scmle.core import limits

    recorded = []
    context = SimpleNamespace(
        regulator=limits.Governor(tasks.Resources(time_budget_seconds=budget)),
        split=SimpleNamespace(
            manifest=SimpleNamespace(folds=[([0], [1])] * folds)
        ),
        spec=SimpleNamespace(
            resources=tasks.Resources(time_budget_seconds=budget)
        ),
        forgo=lambda work, reason: recorded.append((work, reason)),
    )
    return context, recorded


def trial(name, seconds):
    from scmle.contracts import experiments

    return experiments.Trial(
        name,
        name,
        experiments.Termination.OK,
        "accuracy",
        0.9,
        0.0,
        [0.9] * 5,
        seconds,
    )


def test_selection_falls_back_to_a_candidate_whose_refit_fits():
    from scmle.core import stages

    context, recorded = timed(10)
    slow, quick = trial("slow", 100.0), trial("quick", 1.0)
    assert (
        stages.refitcost(context, slow) > 10 > stages.refitcost(context, quick)
    )
    assert stages.affordable(context, [slow, quick]) is quick
    assert recorded[0][0] == "selecting slow/default"
    only, recorded = timed(10)
    assert stages.affordable(only, [slow]) is slow  # nothing fits: cheapest
    assert recorded


def test_diagnostics_get_at_most_what_is_left_before_packaging():
    from scmle.core import stages

    context = timed(20)[0]
    context.regulator.reserve.parts = {"package": 5.0}
    assert stages.FinalEvaluationStage.ration(context) <= 15.0
    late = timed(1)[0]
    late.regulator.started_at -= 10
    assert stages.FinalEvaluationStage.ration(late) == 0.0


def test_a_package_fit_that_cannot_end_in_time_is_halted_on_record():
    from scmle.core import stages

    context, recorded = timed(10)
    assert stages.halt(context, trial("quick", 1.0), "learning curve") is None
    reason = stages.halt(context, trial("slow", 100.0), "learning curve")
    assert "before the run's end" in reason
    assert recorded == [("learning curve", reason)]


def test_package_fits_leave_the_packaging_tail_its_time():
    from scmle.core import stages

    context = timed(10)[0]
    # A 9 s fold fit would end before the 10 s finish, but not before the
    # tail that saves the bundle and writes the report.
    assert stages.foldcost(trial("near", 30.0)) == pytest.approx(9.0)
    assert stages.halt(context, trial("near", 30.0), "conformal") is not None
